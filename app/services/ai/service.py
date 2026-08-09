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
        "Format final answers as clean markdown bullet lists."
        if tools_enabled
        else ""
    )
    return (
        f"{SYSTEM_PROMPT_DASHBOARD}\n\n{role_guide}{page_note}{intent_note}"
        f"{tools_note}\n\n### LIVE BUSINESS DATA (last 30 days)\n{business_context}"
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
    provider = providers[0]
    loop_messages: list[AIMessage] = list(_compact_history(messages))
    turns = 0
    last_reply = ""
    try:
        for _ in range(MAX_TOOL_TURNS):
            resp: ToolResponse = await provider.chat_with_tools(
                loop_messages, tool_declarations(), system_prompt=system_prompt
            )
            if not resp.has_tool_calls:
                last_reply = resp.content or last_reply
                break
            turns += 1
            loop_messages.append(AIMessage(role="assistant", content=resp.content or ""))
            for call in resp.tool_calls:
                result = await dispatch_tool(db, user, call.name, call.arguments)
                logger.info("tool %s → %d chars", call.name, len(result))
                loop_messages.append(
                    AIMessage(role="tool", content=f"Tool '{call.name}' returned:\n{result}")
                )
                last_reply = result
        if not last_reply:
            return "", turns
        return polish_reply(repair_mojibake(last_reply)), turns
    except Exception as e:
        logger.warning("tool loop failed: %s", e)
        return "", turns


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

    settings = get_settings()
    if settings.groq_api_key or settings.gemini_api_key:
        try:
            reply = await get_ai_response(msgs, system_prompt=system_prompt)
            if reply.strip():
                return AnswerResult(reply=reply, source="llm")
        except Exception as e:
            logger.warning("LLM fallback to local engine: %s", e)

    try:
        reply = await local_answer(db, question, intent)
        return AnswerResult(reply=reply, source="local")
    except Exception as e:  # last-resort safety so chat never 500s
        logger.error("Local engine failed: %s", e)
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
) -> AsyncIterator[str]:
    intent = detect_intent(question)
    if intent in CONVERSATIONAL_INTENTS:
        try:
            yield await local_answer(db, question, intent)
            return
        except Exception as e:
            logger.warning("Local conversational answer failed: %s", e)

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

    if not emitted:
        reply = await answer_question(db, role, question, history, page)
        yield reply.reply


AgentResult = AnswerResult  # alias kept for import convenience