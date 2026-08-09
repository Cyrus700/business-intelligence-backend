"""High-level AI orchestration: live-context prompting, tool-calling + fallback.

Flow for every question:
  1. Detect intent; conversational intents never touch the LLM.
  2. Build a role-aware system prompt + a compact live business snapshot.
  3. Run a tool-calling loop: the LLM may call read-only query tools against
     live data (live numbers, no stale snapshot), up to MAX_TOOL_TURNS.
  4. Prefer the configured LLM provider (Groq → Gemini), each guarded by a
     circuit breaker; if no provider is configured or every provider fails,
     answer with the deterministic local engine.
"""

import logging
from collections.abc import AsyncIterator
from dataclasses import dataclass

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.config import get_settings
from app.services.ai.context import build_business_context
from app.services.ai.dates import parse_period
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


async def _run_tool_loop(
    db: AsyncSession,
    user,
    messages: list[AIMessage],
    system_prompt: str,
) -> tuple[str, int]:
    """One tool-calling session; returns (final_reply_markdown, tool_turn_count).

    Providers are tried in configuration order; a provider whose circuit is
    open is skipped. Falls back to plain chat if the loop or every provider
    fails.
    """
    from app.services.ai.provider import _providers

    providers: list[BaseAIProvider] = []
    for p in _providers():
        try:
            if await p._circuit_open():  # noqa: SLF001
                continue
        except Exception:  # never let breaker checks block the answer path
            continue
        providers.append(p)
    if not providers:
        return "", 0

    # Try each healthy provider in turn: a mid-loop failure on one shouldn't
    # cost the user their tool-grounded answer.
    last_error: Exception | None = None
    for provider in providers:
        try:
            reply, turns = await _tool_session(db, user, messages, system_prompt, provider)
            if reply.strip():
                return reply, turns
        except Exception as e:  # noqa: PERF203 - provider failover is the point
            last_error = e
            logger.warning("tool loop failed on %s: %s", provider.circuit_name, e)
    if last_error:
        logger.warning("every provider failed the tool loop: %s", last_error)
    return "", 0


async def _tool_session(
    db: AsyncSession,
    user,
    messages: list[AIMessage],
    system_prompt: str,
    provider: BaseAIProvider,
) -> tuple[str, int]:
    """Run the request → tools → answer cycle against one provider."""
    loop_messages: list[AIMessage] = list(_compact_history(messages))
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
            return polish_reply(repair_mojibake(resp.content or "")), turns

        turns += 1
        # The assistant turn must carry the tool_calls it made, and each result
        # must reference its call id — otherwise the next request is rejected.
        loop_messages.append(
            AIMessage(role="assistant", content=resp.content or "", tool_calls=resp.tool_calls)
        )
        for call in resp.tool_calls:
            result = await dispatch_tool(db, user, call.name, call.arguments)
            logger.info("tool %s(%s) → %d chars", call.name, call.arguments, len(result))
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
    return polish_reply(repair_mojibake(final or "")), turns


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

    business_context = await build_business_context(db)
    system_prompt = _build_system_prompt(role, business_context, page, intent)
    msgs = _recent_history(history or [])

    # Tool-calling LLM path (read-only live queries) is the primary route;
    # falls back to plain chat, then the deterministic local engine.
    try:
        reply, used = await _run_tool_loop(db, user, msgs, system_prompt)
        if reply.strip():
            return AnswerResult(reply=reply, source="llm", tool_calls=used)
    except Exception as e:
        logger.warning("LLM tool loop failed; falling back: %s", e)
        await _reset_transaction(db)

    settings = get_settings()
    if settings.groq_api_key or settings.gemini_api_key:
        try:
            reply = await get_ai_response(msgs, system_prompt=system_prompt)
            if reply.strip():
                return AnswerResult(reply=reply, source="llm")
        except Exception as e:
            logger.warning("LLM fallback to local engine: %s", e)

    await _reset_transaction(db)
    try:
        reply = await local_answer(db, question, intent)
        return AnswerResult(reply=reply, source="local")
    except Exception as e:  # last-resort safety so chat never 500s
        logger.error("Local engine failed: %s", e)
        await _reset_transaction(db)
        return AnswerResult(
            reply=(
                "I hit an unexpected error reading your analytics data. "
                "Please try again in a moment."
            ),
            source="local",
        )


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

    # Token streaming has no tool loop, so a question naming a date would be
    # answered from the snapshot alone — i.e. wrongly. Those go through the
    # tool-grounded path and arrive as one block instead.
    if parse_period(question) is not None:
        result = await answer_question(db, role, question, history, page, user=user)
        yield result.reply
        return

    business_context = await build_business_context(db)
    system_prompt = _build_system_prompt(role, business_context, page, intent)
    msgs = _recent_history(history or [])

    settings = get_settings()
    emitted = False
    if settings.groq_api_key or settings.gemini_api_key:
        try:
            async for chunk in get_ai_stream(msgs, system_prompt=system_prompt):
                emitted = True
                yield chunk
        except Exception as e:
            logger.warning("LLM streaming failed; switching to local engine: %s", e)
            await _reset_transaction(db)

    if not emitted:
        result = await answer_question(db, role, question, history, page, user=user)
        yield result.reply


AgentResult = AnswerResult  # alias kept for import convenience