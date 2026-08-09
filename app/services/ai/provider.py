import asyncio
import logging
import re
from abc import ABC, abstractmethod
from collections.abc import AsyncIterator
from dataclasses import dataclass

from app.core.config import get_settings

logger = logging.getLogger(__name__)


@dataclass
class AIMessage:
    role: str
    content: str


class BaseAIProvider(ABC):
    @abstractmethod
    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        """Returns the full assistant reply."""

    async def chat_stream(
        self, messages: list[AIMessage], system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        """Yields incremental text chunks. Default: single chunk from chat()."""
        yield await self.chat(messages, system_prompt)


class GroqProvider(BaseAIProvider):
    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.groq_api_key
        self.model = settings.groq_model or "llama-3.3-70b-versatile"

    def _msgs(self, messages: list[AIMessage], system_prompt: str | None) -> list[dict]:
        msgs: list[dict] = []
        if system_prompt:
            msgs.append({"role": "system", "content": system_prompt})
        for m in messages:
            msgs.append({"role": m.role, "content": m.content})
        return msgs

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            resp = await client.chat.completions.create(
                model=self.model,
                messages=self._msgs(messages, system_prompt),
                temperature=0.4,
                max_tokens=2048,
            )
            return resp.choices[0].message.content or ""
        except Exception as e:
            logger.warning("Groq API error: %s", e)
            raise

    async def chat_stream(
        self, messages: list[AIMessage], system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GROQ_API_KEY not configured")
        try:
            from groq import AsyncGroq

            client = AsyncGroq(api_key=self.api_key)
            stream = await client.chat.completions.create(
                model=self.model,
                messages=self._msgs(messages, system_prompt),
                temperature=0.4,
                max_tokens=2048,
                stream=True,
            )
            async for chunk in stream:
                if getattr(chunk, "choices", None):
                    delta = chunk.choices[0].delta.content
                    if delta:
                        yield delta
        except Exception as e:
            logger.warning("Groq stream error: %s", e)
            raise


class GeminiProvider(BaseAIProvider):
    ROLE_MAP = {"assistant": "model", "system": "user"}

    def __init__(self) -> None:
        settings = get_settings()
        self.api_key = settings.gemini_api_key
        self.model = settings.gemini_model or "gemini-2.0-flash"

    @staticmethod
    def _contents(messages: list[AIMessage]) -> list:
        from google.genai import types

        return [
            types.Content(
                role=GeminiProvider.ROLE_MAP.get(m.role, m.role),
                parts=[types.Part.from_text(text=m.content)],
            )
            for m in messages
        ]

    async def chat(self, messages: list[AIMessage], system_prompt: str | None = None) -> str:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                temperature=0.4,
            )
            resp = client.models.generate_content(
                model=self.model,
                contents=self._contents(messages),
                config=config,
            )
            return resp.text or ""
        except Exception as e:
            logger.warning("Gemini API error: %s", e)
            raise

    async def chat_stream(
        self, messages: list[AIMessage], system_prompt: str | None = None
    ) -> AsyncIterator[str]:
        if not self.api_key:
            raise RuntimeError("GEMINI_API_KEY not configured")
        try:
            from google import genai
            from google.genai import types

            client = genai.Client(api_key=self.api_key)
            config = types.GenerateContentConfig(
                system_instruction=system_prompt,
                max_output_tokens=2048,
                temperature=0.4,
            )
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
        except Exception as e:
            logger.warning("Gemini stream error: %s", e)
            raise


AIProvider = BaseAIProvider

SYSTEM_PROMPT_DASHBOARD = (
    "You are Insightful AI, a senior business-intelligence analyst for a retail dashboard. "
    "You are measured on accuracy, brevity and actionable advice.\n\n"
    "Rules:\n"
    "1. Answer ONLY from the LIVE BUSINESS DATA block below. Never invent, estimate, or "
    "approximate figures. If the user asks about a metric that is not in the data block, "
    "say clearly that it is not available in the current data and point them to the "
    "relevant dashboard panel.\n"
    "2. Format replies as clean Markdown and always follow this exact structure:\n"
    "   ### A short heading that summarises the answer\n"
    "   One or two sentences with the headline figure, then:\n"
    "   - **Label:** value for each key metric (bold the label, put a colon after it)\n"
    "   Use a markdown table when comparing several items. Keep the whole reply "
    "   scannable and under ~200 words unless the user asks for depth.\n"
    "3. State the exact figures from the data block (e.g. रू 40,76,608, +17.6%) and the "
    "period they cover (last 30 days). Distinguish the current period from the previous "
    "period when comparing.\n"
    "4. Always finish with '**Suggested action:**' plus one concrete, data-grounded "
    "action (name the top product or metric involved), unless the user asked something "
    "trivial.\n"
    "5. Currency is Nepali rupees — write amounts like रू 40,76,608 (Indian digit "
    "grouping: last three digits, then groups of two).\n"
    "6. Be direct and professional: no filler, no disclaimers, no emoji unless the user "
    "uses them.\n"
    "7. If the user only greets or chats casually (hi, hello, how are you, thanks), "
    "reply conversationally in 1-2 sentences and do NOT dump data.\n\n"
    "Example of the expected style:\n"
    "### Revenue — last 30 days\n"
    "- **Total revenue:** रू 40,76,608 (+17.6% vs previous period)\n"
    "- **Top product:** Basmati Rice 25kg — रू 12,35,880 (30.3% of sales)\n"
    "- **Orders:** 1,218\n\n"
    "**Suggested action:** Protect supply of Basmati Rice 25kg, your strongest "
    "growth driver at 30.3% of sales."
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