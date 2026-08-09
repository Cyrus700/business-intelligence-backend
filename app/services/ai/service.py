"""High-level AI orchestration: live-context prompting + resilient fallback.

Flow for every question:
  1. Detect intent and build a live business-data snapshot.
  2. Build a role-aware system prompt that embeds the snapshot.
  3. Prefer the configured LLM provider (Groq → Gemini); stream when asked.
  4. If no provider is configured or every provider fails, answer with the
     deterministic local engine so the chatbot never degrades to canned text.
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
    get_ai_response,
    get_ai_stream,
)

logger = logging.getLogger(__name__)

HISTORY_LIMIT = 16  # cap conversation history fed to the LLM

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


def _build_system_prompt(
    role: str, business_context: str, page: str | None = None, intent: Intent | None = None
) -> str:
    role_guide = {
        "admin": "The user is an ADMIN with full access to all modules and can act "
        "on any recommendation.",
        "manager": "The user is a MANAGER focused on operational decisions and team priorities.",
        "analyst": "The user is an ANALYST and appreciates precise, data-heavy detail.",
    }.get(role, "The user has dashboard access.")
    page_note = f"\nThe user is currently on the {page} page." if page else ""
    intent_note = f"\nThe user's question is about: {intent}." if intent else ""
    return (
        f"{SYSTEM_PROMPT_DASHBOARD}\n\n{role_guide}{page_note}{intent_note}\n\n"
        f"### LIVE BUSINESS DATA (last 30 days)\n{business_context}"
    )


def _recent_history(history: list[AIMessage]) -> list[AIMessage]:
    return history[-HISTORY_LIMIT:]


async def answer_question(
    db: AsyncSession,
    role: str,
    question: str,
    history: list[AIMessage] | None = None,
    page: str | None = None,
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