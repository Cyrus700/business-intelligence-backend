"""High-level AI orchestration: live-context prompting, tool-calling + fallback.

Flow for every question:
  1. Detect intent; conversational intents never touch the LLM.
  2. Build a role-aware system prompt + a compact live business snapshot.
  3. Run a tool-calling loop: the LLM may call read-only query tools against
     live data (live numbers, no stale snapshot), up to MAX_TOOL_TURNS.
  4. Prefer the configured LLM provider (Groq → Gemini), each guarded by a
     circuit breaker; if no provider is configured or every provider fails,
     answer with the deterministic local engine.
  5. Verify every figure in the finished reply against the evidence that
     produced it; an unsupported number buys one repair round, then the
     deterministic local engine, which cannot invent one.
"""

import logging
import re
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.ai.context import build_business_context
from app.services.ai.dates import parse_period
from app.services.ai.grounding import repair_instruction, unsupported_figures
from app.services.ai.intents import Intent, detect_intent
from app.services.ai.local_engine import local_answer
from app.services.ai.provider import (
    SYSTEM_PROMPT_DASHBOARD,
    AIMessage,
    BaseAIProvider,
    ToolResponse,
    get_ai_response,
    get_ai_stream,
    polish_reply,
    repair_mojibake,
)
from app.services.ai.tools import dispatch_tool, tool_declarations

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 16  # cap conversation history fed to the LLM
MAX_TOOL_TURNS = 4  # how many tool rounds per user question

# Casual/social exchanges never touch the LLM: the deterministic local
# engine answers them instantly and consistently, so "hello" always gets
# a friendly greeting instead of a data dump.
CONVERSATIONAL_INTENTS = frozenset(
    {Intent.GREETING, Intent.THANKS, Intent.HELP, Intent.CAPABILITIES}
)


async def _reset_transaction(db: AsyncSession) -> None:
    """Clear a failed transaction so later reads can still run.

    Postgres aborts the whole transaction on the first error, and every
    subsequent statement then fails with InFailedSQLTransactionError — turning
    one recoverable query error into a 500 for the entire request. Rolling back
    here lets the fallback path (and the caller's own commit) proceed.
    """
    try:
        await db.rollback()
    except Exception:  # pragma: no cover - the session is unusable either way
        logger.debug("rollback after failed AI query did not succeed", exc_info=True)


@dataclass
class AnswerResult:
    reply: str
    source: str  # "llm" | "local"
    tool_calls: int = 0  # how many tool rounds the LLM used


def _build_system_prompt(
    role: str, business_context: str, page: str | None = None, intent: Intent | None = None,
    *, tools_enabled: bool = True,
) -> str:
    role_guide = {
        "admin": "The user is an ADMIN with full access to all modules and can act "
        "on any recommendation.",
        "manager": "The user is a MANAGER focused on operational decisions and team priorities.",
        "analyst": "The user is an ANALYST and appreciates precise, data-heavy detail.",
    }.get(role, "The user has dashboard access.")
    page_note = f"\nThe user is currently on the {page} page." if page else ""
    intent_note = f"\nThe user's question is about: {intent}." if intent else ""
    tools_note = (
        "\n\nYou have read-only analytics tools. If the question needs live numbers, "
        "figures or comparisons beyond the snapshot above, CALL THE TOOLS instead of "
        "guessing. Always prefer tool data over the snapshot when they differ. "
        "Any question naming a date, month, year or 'today'/'yesterday'/'last week' "
        "REQUIRES a tool call with that range — the snapshot is only the last 30 days. "
        "The tools accept date_from/date_to, a single `date`, or a named `period`; "
        "get_data_coverage tells you which dates actually exist. "
        "For 'why did X change', call explain_change and revenue_bridge rather than "
        "fetching two periods and subtracting them. For 'are we on track' / 'how will "
        "this month end', call project_period_end. For 'what if', call "
        "simulate_scenario. For 'how exposed are we', call analyse_concentration. "
        "Calling two or three tools to give one grounded, explained answer is better "
        "than one tool and a guess. "
        "Format final answers as clean markdown bullet lists."
        if tools_enabled
        else ""
    )
    return (
        f"{SYSTEM_PROMPT_DASHBOARD}\n\n{role_guide}{page_note}{intent_note}"
        f"{tools_note}\n\n### CURRENT DATE & LIVE BUSINESS DATA\n{business_context}"
    )


def _recent_history(history: list[AIMessage]) -> list[AIMessage]:
    return history[-HISTORY_LIMIT:]


def _compact_history(messages: list[AIMessage], budget_chars: int = 6000) -> list[AIMessage]:
    """Drop the middle of long histories (front-loaded memory persistence)."""
    total = sum(len(m.content) for m in messages)
    if total <= budget_chars:
        return messages
    keep_head = messages[:3]
    rest = messages[3:]
    # drop oldest while over budget, but never the latest user message
    while rest and total > budget_chars and len(rest) > 1:
        dropped = rest.pop(0)
        total -= len(dropped.content)
        logger.info("compressed %d chars of history", len(dropped.content))
    return keep_head + rest


async def _healthy_providers() -> list[BaseAIProvider]:
    """Configured providers whose circuit breaker is currently closed."""
    from app.services.ai.provider import _providers

    healthy: list[BaseAIProvider] = []
    for p in _providers():
        try:
            if await p._circuit_open():  # noqa: SLF001
                continue
        except Exception:  # never let breaker checks block the answer path
            continue
        healthy.append(p)
    return healthy


async def _run_tool_loop(
    db: AsyncSession,
    user,
    messages: list[AIMessage],
    system_prompt: str,
) -> tuple[str, int, list[str]]:
    """One tool-calling session.

    Returns (final_reply_markdown, tool_turn_count, tool_results). The tool
    results come back with the reply because they are the evidence the answer
    is checked against — a figure that appears in neither them nor the live
    snapshot was invented.

    Providers are tried in configuration order; a provider whose circuit is
    open is skipped. Falls back to plain chat if the loop or every provider
    fails.
    """
    providers = await _healthy_providers()
    if not providers:
        return "", 0, []

    # Try each healthy provider in turn: a mid-loop failure on one shouldn't
    # cost the user their tool-grounded answer.
    last_error: Exception | None = None
    for provider in providers:
        try:
            reply, turns, evidence = await _tool_session(
                db, user, messages, system_prompt, provider
            )
            if reply.strip():
                return reply, turns, evidence
        except Exception as e:  # noqa: PERF203 - provider failover is the point
            last_error = e
            logger.warning("tool loop failed on %s: %s", provider.circuit_name, e)
            await _reset_transaction(db)
    if last_error:
        logger.warning("every provider failed the tool loop: %s", last_error)
    return "", 0, []


async def _tool_session(
    db: AsyncSession,
    user,
    messages: list[AIMessage],
    system_prompt: str,
    provider: BaseAIProvider,
) -> tuple[str, int, list[str]]:
    """Run the request → tools → answer cycle against one provider.

    Returns the reply, how many tool rounds it took, and every tool result
    verbatim so the caller can check the reply's figures against them.
    """
    loop_messages: list[AIMessage] = list(_compact_history(messages))
    evidence: list[str] = []
    turns = 0

    # When the question names a date the 30-day snapshot cannot answer, a model
    # that declines to call a tool has nothing to answer *from* — so require a
    # call on the first turn instead of relying on its judgement.
    question = next(
        (m.content for m in reversed(loop_messages) if m.role == "user"), ""
    )
    force_tools = parse_period(question) is not None

    for attempt in range(MAX_TOOL_TURNS):
        resp: ToolResponse = await provider.chat_with_tools(
            loop_messages,
            tool_declarations(),
            system_prompt=system_prompt,
            tool_choice="required" if (force_tools and attempt == 0) else "auto",
        )
        if not resp.has_tool_calls:
            # The model is done gathering data and has written its answer.
            return polish_reply(repair_mojibake(resp.content or "")), turns, evidence

        turns += 1
        # The assistant turn must carry the tool_calls it made, and each result
        # must reference its call id — otherwise the next request is rejected.
        loop_messages.append(
            AIMessage(role="assistant", content=resp.content or "", tool_calls=resp.tool_calls)
        )
        for call in resp.tool_calls:
            result = await dispatch_tool(db, user, call.name, call.arguments)
            logger.info("tool %s(%s) → %d chars", call.name, call.arguments, len(result))
            evidence.append(result)
            loop_messages.append(
                AIMessage(
                    role="tool", content=result, tool_call_id=call.id, name=call.name
                )
            )

    # Turn budget spent while still calling tools. Ask once more with tools
    # withheld so the user gets a composed answer — never a raw tool dump.
    final = await provider.chat(
        [
            *loop_messages,
            AIMessage(
                role="user",
                content=(
                    "Now answer my original question using only the tool results above. "
                    "Do not call any more tools."
                ),
            ),
        ],
        system_prompt=system_prompt,
    )
    return polish_reply(repair_mojibake(final or "")), turns, evidence


async def _verified(
    db: AsyncSession,
    reply: str,
    evidence: list[str],
    msgs: list[AIMessage],
    system_prompt: str,
    tool_calls: int,
) -> AnswerResult | None:
    """Accept the reply only if its figures trace back to the evidence.

    One repair round is allowed — models correct themselves reliably when the
    offending figures are named — and a reply that still cannot be grounded
    returns ``None`` so the caller drops to the deterministic engine, which has
    no way to invent a number in the first place.
    """
    bad = unsupported_figures(reply, evidence)
    if not bad:
        return AnswerResult(reply=reply, source="llm", tool_calls=tool_calls)

    logger.warning("ungrounded figures in AI reply: %s", bad[:8])
    try:
        repaired = await get_ai_response(
            [
                *msgs,
                AIMessage(role="assistant", content=reply),
                AIMessage(role="user", content=repair_instruction(bad)),
            ],
            system_prompt=system_prompt,
        )
    except Exception as e:
        logger.warning("grounding repair round failed: %s", e)
        await _reset_transaction(db)
        return None

    if not repaired.strip():
        return None
    still_bad = unsupported_figures(repaired, evidence)
    if still_bad:
        logger.warning("reply still ungrounded after repair: %s", still_bad[:8])
        return None
    return AnswerResult(reply=repaired, source="llm", tool_calls=tool_calls)


async def answer_question(
    db: AsyncSession,
    role: str,
    question: str,
    history: list[AIMessage] | None = None,
    page: str | None = None,
    user=None,
) -> AnswerResult:
    intent = detect_intent(question)
    if intent in CONVERSATIONAL_INTENTS:
        try:
            return AnswerResult(reply=await local_answer(db, question, intent), source="local")
        except Exception as e:
            logger.warning("Local conversational answer failed: %s", e)
            await _reset_transaction(db)

    business_context = await build_business_context(db)
    system_prompt = _build_system_prompt(role, business_context, page, intent)
    msgs = _recent_history(history or [])
    # The snapshot is evidence too: an answer that quotes it without calling a
    # tool is still grounded, and checking against tool results alone would
    # flag every one of those replies.
    baseline_evidence = [business_context]

    # Tool-calling LLM path (read-only live queries) is the primary route;
    # falls back to plain chat, then the deterministic local engine.
    try:
        reply, used, tool_evidence = await _run_tool_loop(db, user, msgs, system_prompt)
        if reply.strip():
            result = await _verified(
                db, reply, baseline_evidence + tool_evidence, msgs, system_prompt, used
            )
            if result is not None:
                return result
    except Exception as e:
        logger.warning("LLM tool loop failed; falling back: %s", e)
        await _reset_transaction(db)

    settings = get_settings()
    if settings.groq_api_key or settings.gemini_api_key:
        try:
            reply = await get_ai_response(msgs, system_prompt=system_prompt)
            if reply.strip():
                # No tools ran here, so the snapshot is the only thing this
                # answer could legitimately have come from.
                result = await _verified(db, reply, baseline_evidence, msgs, system_prompt, 0)
                if result is not None:
                    return result
        except Exception as e:
            logger.warning("LLM fallback to local engine: %s", e)

    await _reset_transaction(db)
    try:
        reply = await local_answer(db, question, intent)
        if reply.strip():
            return AnswerResult(reply=reply, source="local")
    except Exception as e:
        logger.error("Local engine failed: %s", e)
        await _reset_transaction(db)

    # Last resort. The snapshot was already read successfully at the top of
    # this function, so hand it over rather than an apology — a user asking
    # about revenue is better served by real KPIs than by "try again".
    return AnswerResult(reply=_snapshot_fallback(business_context), source="local")


def _snapshot_fallback(business_context: str) -> str:
    """Degrade to the live snapshot when every answer path has failed."""
    if not business_context.strip():
        return (
            "I could not read your analytics data just now. Please try again in a "
            "moment, or check the **Data** page to confirm a source is connected."
        )
    return (
        "I could not compose a full answer for that question, but here is the live "
        "picture straight from your warehouse:\n\n"
        f"{business_context}\n\n"
        "**Suggested action:** ask me about one of these figures directly — for "
        "example a specific date, product or expense category."
    )


# Intents the 30-day snapshot simply does not carry. It holds headline KPIs,
# the top five products, the top five expense categories, an inventory summary,
# a forecast total and an anomaly count — nothing by channel, by region, and no
# period-against-period comparison. Streaming these from the snapshot alone
# produces a confident answer built on data that was never in the prompt.
SNAPSHOT_BLIND_INTENTS = frozenset(
    {Intent.CHANNELS, Intent.REGIONS, Intent.COMPARE}
)

# Phrasings that ask for detail past the snapshot's depth even when the intent
# looks covered: a breakdown, a per-day series, a ranking longer than five, or
# anything reaching into stored history.
_DETAIL_RE = re.compile(
    r"\bbreak\s?down\b|\bbreakdown\b|\bby (?:category|channel|region|city|product|day|week|month)\b"
    r"|\bday[- ]by[- ]day\b|\beach day\b|\bper day\b|\btime series\b|\bhistory\b|\bhistorical\b"
    r"|\btop\s+(?:[6-9]|[1-9]\d+)\b|\bfull list\b|\ball (?:products|categories|regions|channels)\b"
    r"|\brecommend\w*\b|\bhave we seen\b|\bpast (?:insight|anomal|finding)\w*\b"
    # Diagnostic and forward-looking asks. Each of these has a tool that returns
    # a finished decomposition; answered from the snapshot they become guesses
    # dressed as analysis.
    r"|\bwhy\b|\bwhat (?:caused|drove|changed)\b|\breason\b|\bexplain\b|\bdriver\b"
    r"|\bwhat if\b|\bwhat would happen\b|\bscenario\b|\bsimulat\w*\b"
    r"|\bon track\b|\bwill we (?:hit|make|reach)\b|\bend of (?:the )?(?:month|quarter)\b"
    r"|\bconcentrat\w*\b|\bdepend\w*\s+on\b|\bexposed\b|\brisk\b",
    re.IGNORECASE,
)


def _needs_live_tools(question: str, intent: Intent) -> bool:
    """True when answering honestly requires a query the snapshot can't serve.

    Token streaming has no tool loop, so anything this returns True for is
    routed through the tool-grounded path and arrives as one block instead.
    Streaming a guess faster is not a feature.
    """
    if parse_period(question) is not None:
        return True
    if intent in SNAPSHOT_BLIND_INTENTS:
        return True
    return bool(_DETAIL_RE.search(question))


async def stream_answer(
    db: AsyncSession,
    role: str,
    question: str,
    history: list[AIMessage] | None = None,
    page: str | None = None,
    user=None,
) -> AsyncIterator[str]:
    """Stream a reply, falling back to the tool-grounded answer when needed.

    ``user`` is required for tool dispatch (tools are role-scoped) — without it
    the streaming path cannot answer anything the 30-day snapshot doesn't
    already contain.
    """
    intent = detect_intent(question)
    if intent in CONVERSATIONAL_INTENTS:
        try:
            yield await local_answer(db, question, intent)
            return
        except Exception as e:
            logger.warning("Local conversational answer failed: %s", e)
            await _reset_transaction(db)

    if _needs_live_tools(question, intent):
        result = await answer_question(db, role, question, history, page, user=user)
        yield result.reply
        return

    business_context = await build_business_context(db)
    # No tools are attached to a streaming call, so the prompt must not claim
    # otherwise: told to "CALL THE TOOLS" with none available, the model either
    # stalls or narrates a tool call it never made.
    system_prompt = _build_system_prompt(
        role, business_context, page, intent, tools_enabled=False
    )
    msgs = _recent_history(history or [])

    settings = get_settings()
    emitted = False
    if settings.groq_api_key or settings.gemini_api_key:
        streamed: list[str] = []
        try:
            async for chunk in get_ai_stream(msgs, system_prompt=system_prompt):
                emitted = True
                streamed.append(chunk)
                yield chunk
        except Exception as e:
            logger.warning("LLM streaming failed; switching to local engine: %s", e)
            await _reset_transaction(db)

        # Streamed tokens are already on screen by the time the reply is whole,
        # so a bad figure cannot be swapped out — but it can be flagged. Saying
        # which numbers failed the check is more useful than letting them stand
        # unmarked next to the ones that passed.
        if emitted:
            bad = unsupported_figures("".join(streamed), [business_context])
            if bad:
                logger.warning("ungrounded figures in streamed reply: %s", bad[:8])
                yield (
                    "\n\n> **Unverified:** "
                    + ", ".join(bad[:5])
                    + " could not be traced to your loaded data. Ask again for a "
                    "tool-checked answer."
                )

    if not emitted:
        result = await answer_question(db, role, question, history, page, user=user)
        yield result.reply


AgentResult = AnswerResult  # alias kept for import convenience