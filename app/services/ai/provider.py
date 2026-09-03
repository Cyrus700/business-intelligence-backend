import asyncio
import logging
import re
import time
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass, field
from typing import Any

from app.core.config import get_settings
from app.services.ai.circuit import (
    CircuitState,
    estimate_cost_usd,
    get_circuit,
)

logger = logging.getLogger(__name__)

# Every figure in an answer comes from a tool result or the live snapshot, so
# sampling temperature only shapes wording. 0.4 made replies read like a filled
# -in template; this keeps them varied while leaving the numbers untouched.
RESPONSE_TEMPERATURE = 0.6


@dataclass
class ToolCall:
    id: str
    name: str
    arguments: dict


@dataclass
class AIMessage:
    """One turn in the conversation.

    Tool turns need more than role+content: the OpenAI/Groq protocol requires
    the assistant turn that requested tools to carry ``tool_calls``, and each
    result turn to reference the call it answers via ``tool_call_id``. Sending
    a bare ``{"role": "tool", "content": ...}`` is rejected by the API, which
    silently collapses the whole tool loop into a plain, tool-less answer.
    """

    role: str
    content: str
    tool_calls: list[ToolCall] | None = None
    tool_call_id: str | None = None
    name: str | None = None


@dataclass
class ToolResponse:
    content: str
    tool_calls: list[ToolCall] = field(default_factory=list)

    @property
    def has_tool_calls(self) -> bool:
        return bool(self.tool_calls)


def _json_loads(raw: Any) -> dict:
    import json

    if isinstance(raw, dict):
        return raw
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {}
    except (TypeError, ValueError):
        return {}


class BaseAIProvider(ABC):
    """Provider interface; every call is guarded by a circuit breaker."""

    circuit_name = "provider"

    @abstractmethod
    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        """Returns the full assistant reply."""

    async def chat_stream(self, messages: list[AIMessage], system_prompt: str | None = None) -> AsyncIterator[str]:
        """Yields incremental text chunks. Default: single chunk from chat()."""
        yield await self.chat(messages, system_prompt)

    @abstractmethod
    async def chat_with_tools(
        self,
        messages: list[AIMessage],
        tools: list[dict],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> ToolResponse:
        """Full assistant turn; may contain tool calls to execute by the caller.

        ``tool_choice="required"`` forces at least one call — used when the
        question names a date the snapshot cannot cover, where a model that
        declines to call a tool can only guess.
        """

    @abstractmethod
    def model_id(self) -> str:
        """Stable identity for circuit tracking / cost accounting."""

    def _circuit(self) -> CircuitState:
        return get_circuit(self.circuit_name, self.model_id())

    async def _circuit_open(self) -> bool:
        state = self._circuit()
        if not state.is_open:
            return False
        logger.warning("%s circuit OPEN (cooldown until %s)", self.circuit_name, state.circuit_open_until)
        return True

    def _record_success(self, latency_ms: int | None, input_text: str, output_text: str) -> None:
        state = self._circuit()
        cost = estimate_cost_usd(self.model_id(), input_text, output_text)
        state.record_success(latency_ms, cost)

    def _record_failure(self) -> None:
        self._circuit().record_failure()


class GroqProvider(BaseAIProvider):
    circuit_name = "groq"

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model or "openai/gpt-oss-120b"

    def model_id(self) -> str:
        return self.model

    def _msgs(self, messages: list[AIMessage], system_prompt: str | None) -> list[dict]:
        import json

        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            if m.role == "tool":
                # Must reference the call it answers, or the API 400s.
                msgs.append(
                    {
                        "role": "tool",
                        "content": m.content,
                        "tool_call_id": m.tool_call_id or "",
                        **({"name": m.name} if m.name else {}),
                    }
                )
                continue
            entry: dict[str, Any] = {"role": m.role, "content": m.content or ""}
            if m.tool_calls:
                entry["tool_calls"] = [
                    {
                        "id": c.id,
                        "type": "function",
                        "function": {"name": c.name, "arguments": json.dumps(c.arguments)},
                    }
                    for c in m.tool_calls
                ]
                # assistant turns that carry tool_calls must not send content=""
                entry["content"] = m.content or None
            msgs.append(entry)
        return msgs

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("groq circuit open")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            started = time.monotonic()
            resp = await client.chat.completions.create(
                model=self.model,
                messages=self._msgs(messages, system_prompt),
                temperature=RESPONSE_TEMPERATURE,
                max_tokens=2048,
            )
            reply = resp.choices[0].message.content or ""
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                reply,
            )
            return reply
        except Exception as e:
            self._record_failure()
            logger.warning("Groq API error: %s", e)
            raise

    async def chat_with_tools(
        self,
        messages: list[AIMessage],
        tools: list[dict],
        system_prompt: str | None = None,
        tool_choice: str = "auto",
    ) -> ToolResponse:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("Groq circuit open")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            started = time.monotonic()
            resp = await client.chat.completions.create(
                model=self.model,
                messages=self._msgs(messages, system_prompt),
                tools=tools or None,
                tool_choice=tool_choice if tools else None,
                temperature=RESPONSE_TEMPERATURE,
                max_tokens=2048,
            )
            msg = resp.choices[0].message
            reply_text = msg.content or ""
            calls = []
            for idx, t in enumerate(msg.tool_calls or []):
                fn = getattr(t, "function", None)
                calls.append(
                    ToolCall(
                        id=getattr(t, "id", "") or f"call_{idx}",
                        name=(fn.name if fn else ""),
                        # the API hands arguments back as a JSON *string*;
                        # dispatch_tool splats them as kwargs, so parse here
                        arguments=_json_loads(fn.arguments if fn else "{}"),
                    )
                )
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                reply_text,
            )
            return ToolResponse(content=reply_text, tool_calls=calls)
        except Exception as e:
            self._record_failure()
            logger.warning("Groq tool-call error: %s", e)
            raise

    async def chat_stream(self, messages: list[AIMessage], system_prompt: str | None = None) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("Groq circuit open")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            stream = await client.chat.completions.create(
                model=self.model,
                messages=self._msgs(messages, system_prompt),
                temperature=RESPONSE_TEMPERATURE,
                max_tokens=2048,
                stream=True,
            )
            started = time.monotonic()
            async for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                "",
            )
        except Exception as e:
            self._record_failure()
            logger.warning("Groq stream error: %s", e)
            raise


class GeminiProvider(BaseAIProvider):
    circuit_name = "gemini"

    ROLE_MAP = {"assistant": "model", "system": "user"}

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model or "gemini-2.0-flash"

    def model_id(self) -> str:
        return self.model

    @staticmethod
    def _contents(messages: list[AIMessage]) -> list:
        from google.genai import types

        out = []
        for m in messages:
            text = m.content or ""
            if m.role == "tool":
                # Gemini has no dedicated tool role in this call shape; fold the
                # result into a user turn so the model still sees the data
                # rather than dropping it.
                text = f"Tool result ({m.name or 'tool'}):\n{text}"
                role = "user"
            else:
                role = GeminiProvider.ROLE_MAP.get(m.role, m.role)
            if not text.strip():
                continue  # empty parts are rejected
            out.append(types.Content(role=role, parts=[types.Part.from_text(text=text)]))
        return out

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("Gemini circuit open")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                temperature=RESPONSE_TEMPERATURE,
            )
            started = time.monotonic()
            resp = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents=self._contents(messages),
                    config=config,
                ),
            )
            reply = resp.text or ""
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                reply,
            )
            return reply
        except Exception as e:
            self._record_failure()
            logger.warning("Gemini API error: %s", e)
            raise

    async def chat_with_tools(
        self,
        messages: list[AIMessage],
        tools: list[dict],
        system_prompt: str | None = None,
        tool_choice: str = "auto",  # noqa: ARG002 - Gemini decides on its own
    ) -> ToolResponse:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("Gemini circuit open")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                temperature=RESPONSE_TEMPERATURE,
                tools=[
                    {
                        "function_declarations": [
                            {
                                "name": t["function"]["name"],
                                "description": t["function"]["description"],
                                "parameters": t["function"].get("parameters"),
                            }
                            for t in tools
                        ]
                    }
                ],
            )
            started = time.monotonic()
            resp = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: client.models.generate_content(
                    model=self.model,
                    contents=self._contents(messages),
                    config=config,
                ),
            )
            reply = resp.text or ""
            calls = []
            for idx, part in enumerate(getattr(getattr(resp, "candidates", [None])[0], "function_calls", []) or []):
                args = _json_loads(getattr(part, "args", None))
                calls.append(ToolCall(id=f"c{idx}", name=part.name, arguments=args))
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                reply,
            )
            return ToolResponse(content=reply, tool_calls=calls)
        except Exception as e:
            self._record_failure()
            logger.warning("Gemini tool-call error: %s", e)
            raise

    async def chat_stream(self, messages: list[AIMessage], system_prompt: str | None = None) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        if await self._circuit_open():
            raise RuntimeError("Gemini circuit open")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                temperature=RESPONSE_TEMPERATURE,
            )
            started = time.monotonic()
            stream = await asyncio.get_running_loop().run_in_executor(
                None,
                lambda: list(
                    client.models.generate_content_stream(
                        model=self.model,
                        contents=self._contents(messages),
                        config=config,
                    )
                ),
            )
            for resp in stream:
                if getattr(resp, "text", None):
                    yield resp.text
            self._record_success(
                int((time.monotonic() - started) * 1000),
                " ".join(m.content for m in messages),
                "",
            )
        except Exception as e:
            self._record_failure()
            logger.warning("Gemini stream error: %s", e)
            raise


AIProvider = BaseAIProvider

SYSTEM_PROMPT_DASHBOARD = (
    "You are InsightFlow AI, a senior business-intelligence analyst for a retail dashboard. "
    "You MUST handle EVERY question the user asks — no matter how it is phrased or what data it concerns. "
    "You are measured on accuracy, brevity, precision and that you never ignore a question.\n\n"
    "Rules:\n"
    "0. COVERAGE: You answer EACH AND EVERY question. Never say you cannot answer, never return "
    "'I could not compose...' — instead CALL THE TOOLS to get live data. If no direct tool exists, "
    "call describe_catalog to discover all tables that exist right now, then sample_table to read live rows "
    "from the relevant table (new tables appear automatically). Every table is queryable live with no code change.\n"
    "1. Every figure you state must come from either the LIVE BUSINESS DATA snapshot below "
    "or a tool result in this conversation. Never invent, estimate, extrapolate or "
    "'approximately' a number. Tool results always win over the snapshot when they differ, "
    "because the snapshot only covers the last 30 days. If a figure is in neither, say it "
    "is not available and name the dashboard panel that has it.\n"
    "1b. DATES: the snapshot covers only the last 30 days. The moment a question names a "
    "specific day, month, year, or says today/yesterday/last week, you MUST call the tools "
    "with that date range instead of answering from the snapshot — and call "
    "get_data_coverage so you can tell a true zero from a date that was never loaded. "
    "Never report 0 for a date outside the loaded range; say the data is not loaded. "
    "Always state the exact date range your numbers cover.\n"
    "1c. UNIVERSAL: For 'whats the update', 'summary', 'overview', or any vague/status question, "
    "call query_kpis + get_data_coverage + get_anomalies + get_inventory + get_platform_stats (if super-admin) "
    "and compose a concise live digest across all domains. For 'how many business/businesses/organizations' ALWAYS call "
    "get_platform_stats (with status='approved' / 'rejected' / 'pending' when filter is present; handle typos aprrved/rejeect/regisgtered). "
    "For 'what tables/data do you have' ALWAYS call describe_catalog. For any question about "
    "a table you don't recognise, call describe_catalog first, then sample_table.\n"
    "1c2. LIST intent: When user says LIST / SHOW / DISPLAY / GET / FETCH + business/organization ('list business', 'list the businesses', 'show all businesses', 'list approved businesses', 'how many approved' with list), "
    "you MUST call get_platform_stats with detail=true (and status if present). Return the live table the tool gives you — do not summarize as just the count. "
    "Examples: 'list business' → get_platform_stats(detail=true, limit=15); 'list approved businesses' → get_platform_stats(status='approved', detail=true, limit=15). "
    "Always analyze the user's prompt for intent: COUNT vs LIST, and for status filter (approved/pending/rejected) with typo tolerance, then choose the accurate tool params.\n"
    "1d. PRECISION FOR COUNTS: For any count question (how many / total / count / number of business/approved/rejected/pending), "
    "answer with that ONE number first (live, bold), then at most 3 bullet points of breakdown. NEVER add revenue/orders/top-products/expense categories unless the question explicitly asks for them. Keep count answers under 120 words. "
    "For LIST questions, use the table format.\n"
    "2. ANSWER THE ACTUAL QUESTION FIRST, in one sentence, before any list. Then add only "
    "the figures that bear on it — never a standard KPI dump unless the question is 'whats the update'/'overview'. Shape the reply to the "
    "question: a single number for a single-number question; a markdown table when "
    "comparing items or periods; a short ordered list for a 'why' or 'how' question. "
    "Vary your wording and headings between answers; do not reuse a fixed template. "
    "Clean Markdown, scannable, under ~200 words unless depth is requested.\n"
    "3. State exact figures (e.g. रू 40,76,608, +17.6%) and always name the period they "
    "cover as real dates, not vague words — '1 Jun – 30 Jun 2026', not 'recently'. "
    "Distinguish the current period from the previous period when comparing.\n"
    "3b. GO ONE LEVEL DEEPER THAN THE QUESTION. A number on its own is a lookup, not "
    "analysis. When a figure has moved, say what moved it (which product, category, "
    "channel or region, and whether it was order volume or order value). When you are "
    "asked about the current month or quarter, say where it is on track to land. When "
    "revenue leans on one or two names, say so. Use the analysis tools for this — "
    "explain_change, revenue_bridge, analyse_concentration, project_period_end and "
    "simulate_scenario each return a finished decomposition. Never compute a "
    "contribution, a bridge, a projection or a scenario in your head; the tools exist "
    "because that arithmetic is where answers go wrong.\n"
    "4. End with '**Suggested action:**' and ONE specific next step that follows from the "
    "numbers you just gave — name the product, region, category or metric involved and "
    "why it matters. Make it decidable: what to do, to which thing, and what it is worth "
    "in rupees or percent based on the figures above. It must be different when the data "
    "is different; never generic advice like 'monitor your dashboard'. Skip it for casual "
    "chat or a pure lookup.\n"
    "5. Currency is Nepali rupees — write amounts like रू 40,76,608 (Indian digit "
    "grouping: last three digits, then groups of two).\n"
    "6. Be direct and professional: no filler, no disclaimers, no emoji unless the user "
    "uses them. Every answer must be understandable: use short sentences, bullet lists, tables for comparisons.\n"
    "7. If the user only greets or chats casually (hi, hello, how are you, thanks), "
    "reply conversationally in 1-2 sentences and do NOT dump data.\n"
    "8. If the question is ambiguous (no period, no metric named), pick the most useful "
    "reading, say which reading you used, and answer it — do not stall by asking a "
    "clarifying question first. For UNKNOWN phrasing, still answer with a live digest rather than saying you don't understand.\n\n"
    "Tone and depth adapt to the question. Two examples, deliberately different:\n\n"
    "Q: 'What was revenue on 10 June?'\n"
    "Revenue on 10 Jun 2026 was रू 7,040 across 2 orders — about a third of the "
    "daily average for that week.\n\n"
    "**Suggested action:** Wai Wai drove रू 6,080 of that day on a single wholesale "
    "order; check whether that account reorders this week.\n\n"
    "Q: 'Compare our channels last month.'\n"
    "Online overtook store in Jul 2026, taking 54% of revenue.\n\n"
    "| Channel | Revenue | Share | vs Jun |\n"
    "|---|---|---|---|\n"
    "| Online | रू 22,14,300 | 54.0% | +18.2% |\n"
    "| Store | रू 18,86,100 | 46.0% | -4.1% |\n\n"
    "**Suggested action:** Store revenue fell 4.1% while online grew — shift the "
    "marketing spend behind the store channel to online fulfilment.\n\n"
    "Q: 'Why is revenue down this month?'\n"
    "Revenue is down रू 3,12,400 (-8.1%) on last month, and it is a volume problem: "
    "orders fell 11.2% while the average order value actually rose 3.5%.\n\n"
    "- **Wai Wai** accounts for रू 1,84,000 of the drop (59% of the movement)\n"
    "- **Bhaktapur** stopped ordering entirely after 12 Jul\n"
    "- Everything else is flat to slightly up\n\n"
    "At the current run rate the month lands near रू 35,40,000 (95% range रू 32,10,000 – "
    "रू 38,70,000), about रू 2,60,000 short of July.\n\n"
    "**Suggested action:** Wai Wai and Bhaktapur together explain the whole shortfall — "
    "call the Bhaktapur account first, it went from रू 96,000/month to zero with no "
    "partial decline, which usually means a lost account rather than soft demand."
)


def _polish_line(line: str) -> str:
    """Upgrade one line of a model reply to consistent markdown.

    - Normalises '-', '+' and '*' bullets to '-'.
    - Bolds short 'Label:' prefixes (unless the line already has markdown).
    """
    m = re.match(r"^(\s*)[-*+]\s+(.*)$", line)
    if m:
        line = f"{m.group(1)}- {m.group(2)}"
    if "**" in line or "|" in line or "://" in line or not line.strip():
        return line
    m = re.match(r"^(\s*(?:-\s+)?)(\S[^:]{0,39}):\s*(.*)$", line)
    if m:
        rest = m.group(3)
        return f"{m.group(1)}**{m.group(2)}:** {rest}" if rest else f"{m.group(1)}**{m.group(2)}:**"
    return line


def polish_reply(text: str) -> str:
    """Deterministically restyle a model reply as clean, scannable markdown.

    Guarantees the same professional look the local engine produces, no
    matter how loosely the LLM followed the formatting rules.
    """
    return "\n".join(_polish_line(line) for line in text.split("\n"))


def repair_mojibake(text: str) -> str:
    """Repair Latin-1-mis-decoded UTF-8 runs, e.g. 'à¤°à¥' → 'रू'.

    LLMs occasionally emit the UTF-8 bytes of Devanagari text decoded as
    Latin-1. This rewrites each run of non-ASCII characters through the
    latin-1 → utf-8 round trip. Runs that are already valid Unicode are
    left untouched.
    """
    out: list[str] = []
    i = 0
    n = len(text)
    while i < n:
        if text[i] <= "\x7f":
            out.append(text[i])
            i += 1
            continue
        j = i
        while j < n and text[j] > "\x7f":
            j += 1
        run = text[i:j]
        try:
            fixed = run.encode("latin-1").decode("utf-8")
        except (UnicodeEncodeError, UnicodeDecodeError):
            fixed = run
        out.append(fixed)
        i = j
    return "".join(out)


async def get_ai_response(
    messages: list[AIMessage],
    system_prompt: str | None = None,
) -> str:
    system_prompt = system_prompt or SYSTEM_PROMPT_DASHBOARD
    providers = _providers()
    if not providers:
        return ""
    last_error: Exception | None = None
    for provider in providers:
        if await provider._circuit_open():  # noqa: SLF001
            continue
        try:
            reply = await provider.chat(messages, system_prompt)
            return polish_reply(repair_mojibake(reply))
        except Exception as e:
            last_error = e
            logger.warning("Provider %s failed, trying next: %s", type(provider).__name__, e)
            continue
    if last_error:
        logger.error("All AI providers failed: %s", last_error)
    return ""


async def get_ai_stream(
    messages: list[AIMessage],
    system_prompt: str | None = None,
) -> AsyncIterator[str]:
    system_prompt = system_prompt or SYSTEM_PROMPT_DASHBOARD
    providers = _providers()
    if not providers:
        return
    last_error: Exception | None = None
    for provider in providers:
        if await provider._circuit_open():  # noqa: SLF001
            continue
        try:
            # NOTE: chunks are streamed raw (only mojibake-repaired); full
            # markdown polishing is applied to the complete reply before it
            # is persisted (see api/v1/ai.py), matching what the client
            # renders.
            async for chunk in provider.chat_stream(messages, system_prompt):
                yield repair_mojibake(chunk)
            return
        except Exception as e:
            last_error = e
            logger.warning("Provider %s stream failed, trying next: %s", type(provider).__name__, e)
            continue
    if last_error:
        logger.error("All AI providers failed streaming: %s", last_error)


def _providers() -> list[BaseAIProvider]:
    settings = get_settings()
    providers: list[BaseAIProvider] = []
    if settings.groq_api_key:
        providers.append(GroqProvider())
    if settings.gemini_api_key:
        providers.append(GeminiProvider())
    return providers
