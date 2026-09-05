import asyncio
import json
import logging
import re
import uuid
from datetime import UTC, datetime, timedelta
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from app.api.deps import CurrentUser, DbSession, get_current_user, require_role
from app.core.clock import business_today
from app.models.ai import Conversation, Message
from app.services.ai.provider import AIMessage, polish_reply, repair_mojibake
from app.services.ai.service import answer_question, stream_answer

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(get_current_user)])


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    context: dict | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str
    source: str = "llm"


async def _load_or_create_conversation(
    db: DbSession, user: CurrentUser, conv_id: str | None, question: str
) -> tuple[uuid.UUID, list[AIMessage]]:
    """Returns (conversation id, message history incl. the new user message)."""
    now = datetime.now(UTC).replace(tzinfo=None)
    if conv_id:
        conv = await db.get(Conversation, uuid.UUID(conv_id))
        if not conv or conv.user_id != user.id:
            raise HTTPException(404, "Conversation not found")
        # Org check: conversation must belong to caller's org (prevents IDOR via stolen conv_id across orgs)
        if conv.org_id is not None and conv.org_id != user.org_id and not getattr(user, "is_super_admin", False):
            raise HTTPException(404, "Conversation not found")
        cid = conv.id
        conv.updated_at = now
    else:
        conv = Conversation(user_id=user.id, org_id=user.org_id, title=question[:80], updated_at=now)
        db.add(conv)
        await db.flush()
        cid = conv.id

    db.add(Message(conversation_id=cid, org_id=user.org_id, role="user", content=question))
    await db.flush()

    from sqlalchemy import select as sa_select

    history = await db.execute(
        sa_select(Message).where(Message.conversation_id == cid).order_by(Message.created_at, Message.id)
    )
    msgs = [AIMessage(role=m.role, content=m.content) for m in history.scalars().all()]
    return cid, msgs


@router.post("/chat", response_model=ChatResponse)
async def ai_chat(
    body: ChatRequest,
    db: DbSession,
    user: CurrentUser,
) -> ChatResponse:
    cid, msgs = await _load_or_create_conversation(db, user, body.conversation_id, body.message)
    # Commit the user's message before generating, exactly as the streaming
    # endpoint does. Answer generation runs many read queries; if one fails the
    # transaction is aborted, and without this commit the rollback needed to
    # recover would also throw away the question the user just asked.
    await db.commit()
    page = (body.context or {}).get("page") if body.context else None

    result = await answer_question(db, user.role, body.message, msgs, page=page, user=user)

    try:
        db.add(Message(conversation_id=cid, org_id=user.org_id, role="assistant", content=result.reply))
        await db.commit()
    except Exception:
        # Never fail the request over persistence: the user still gets the answer.
        logger.exception("failed to persist assistant reply")
        await db.rollback()

    return ChatResponse(conversation_id=str(cid), reply=result.reply, source=result.source)


def _sse(payload: dict) -> str:
    return f"data: {json.dumps(payload, ensure_ascii=False)}\n\n"


@router.post("/chat/stream")
async def ai_chat_stream(
    body: ChatRequest,
    db: DbSession,
    user: CurrentUser,
):
    """Server-Sent-Events streaming chat. First event carries conversation_id.

    The conversation and the user's message are committed BEFORE streaming
    starts, so an interrupted or failed stream never loses them. The assistant
    reply is persisted when the stream completes successfully.
    """
    cid, msgs = await _load_or_create_conversation(db, user, body.conversation_id, body.message)
    await db.commit()
    page = (body.context or {}).get("page") if body.context else None

    async def _save_assistant(reply: str) -> None:
        if not reply:
            return
        db.add(Message(conversation_id=cid, org_id=user.org_id, role="assistant", content=polish_reply(reply)))
        await db.commit()

    async def event_gen():
        yield _sse({"conversation_id": str(cid)})
        reply_parts: list[str] = []
        try:
            async for chunk in stream_answer(db, user.role, body.message, msgs, page=page, user=user):
                if chunk:
                    reply_parts.append(chunk)
                    yield _sse({"delta": chunk})
        except asyncio.CancelledError:
            # Client disconnected (Stop button / tab close): keep the user
            # message, drop the partial reply.
            raise
        except Exception as e:
            logger.warning("AI stream failed: %s", e)
            yield _sse({"error": "I hit an error while generating a reply. Please try again."})
            return
        await _save_assistant("".join(reply_parts))
        yield _sse({"done": True})

    return StreamingResponse(
        event_gen(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class ConversationOut(BaseModel):
    id: str
    title: str
    created_at: str
    message_count: int


@router.get("/conversations", response_model=list[ConversationOut])
async def list_conversations(
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(50, ge=1, le=200),
) -> list[ConversationOut]:
    from sqlalchemy import func as sa_func
    from sqlalchemy import select

    subq = select(Message.conversation_id, sa_func.count().label("cnt")).group_by(Message.conversation_id).subquery()
    rows = (
        await db.execute(
            select(Conversation, subq.c.cnt)
            .outerjoin(subq, Conversation.id == subq.c.conversation_id)
            .where(Conversation.user_id == user.id)
            .order_by(Conversation.updated_at.desc())
            .limit(limit)
        )
    ).all()
    return [
        ConversationOut(id=str(c.id), title=c.title, created_at=c.created_at.isoformat(), message_count=cnt or 0)
        for c, cnt in rows
    ]


class MessageOut(BaseModel):
    id: str
    role: str
    content: str
    created_at: str


@router.get("/conversations/{conv_id}/messages", response_model=list[MessageOut])
async def get_conversation_messages(
    conv_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
    limit: int = Query(200, ge=1, le=500),
    offset: int = Query(0, ge=0),
) -> list[MessageOut]:
    conv = await db.get(Conversation, conv_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")

    from sqlalchemy import select as sa_select

    # Capped history: default 200, supports pagination for large threads.
    # Order by created_at asc so UI shows chronological flow.
    rows = (
        (
            await db.execute(
                sa_select(Message)
                .where(Message.conversation_id == conv_id)
                .order_by(Message.created_at, Message.id)
                .offset(offset)
                .limit(limit)
            )
        )
        .scalars()
        .all()
    )
    return [
        MessageOut(
            id=str(m.id),
            role=m.role,
            content=repair_mojibake(m.content),
            created_at=m.created_at.isoformat(),
        )
        for m in rows
    ]


@router.delete("/conversations")
async def flush_conversations(
    db: DbSession,
    user: CurrentUser,
) -> dict[str, int]:
    """Flush all conversations for the current user (history clear)."""
    from sqlalchemy import select as sa_select

    rows = (await db.execute(sa_select(Conversation).where(Conversation.user_id == user.id))).scalars().all()
    count = len(rows)
    for c in rows:
        await db.delete(c)
    await db.commit()
    return {"deleted": count}


@router.delete("/conversations/{conv_id}")
async def delete_conversation(
    conv_id: uuid.UUID,
    db: DbSession,
    user: CurrentUser,
) -> dict[str, str]:
    """Delete one conversation (and its messages via FK cascade)."""
    conv = await db.get(Conversation, conv_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")
    await db.delete(conv)
    await db.commit()
    return {"status": "deleted", "id": str(conv_id)}


class RetentionOut(BaseModel):
    retention_days: int
    updated_at: str | None = None
    updated_by: str | None = None
    choices: list[int] = []
    is_disabled: bool = False


class RetentionIn(BaseModel):
    retention_days: int


@router.get("/retention", response_model=RetentionOut)
async def get_retention(
    db: DbSession,
    user: CurrentUser,
) -> RetentionOut:
    """Current AI history auto-flush TTL (any authenticated user may read)."""
    from app.services.ai.retention import retention_status

    s = await retention_status(db)
    return RetentionOut(**s)


@router.put("/retention", response_model=RetentionOut)
async def put_retention(
    body: RetentionIn,
    db: DbSession,
    user: CurrentUser,
) -> RetentionOut:
    """Set AI history retention — system admin only (week/month dynamic)."""

    # super-admin or admin role check — use same guard as /admin/*
    if not getattr(user, "is_super_admin", False) and user.role != "admin":
        # fallback to role check via deps simulation
        # we cannot call Depends here, so manual check:
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Only system admin may change retention")
    from app.services.ai.retention import retention_status, set_retention_days

    await set_retention_days(db, body.retention_days, updated_by=user.id)
    await db.commit()
    s = await retention_status(db)
    return RetentionOut(**s)


@router.post("/retention/flush")
async def trigger_retention_flush(
    db: DbSession,
    user: CurrentUser,
) -> dict[str, int]:
    """Manually trigger auto-flush (admin only) — useful after changing retention."""
    if not getattr(user, "is_super_admin", False) and user.role != "admin":
        from fastapi import HTTPException

        raise HTTPException(status_code=403, detail="Only system admin may trigger flush")
    from app.services.ai.retention import flush_expired_conversations, get_retention_days

    deleted = await flush_expired_conversations(db)
    await db.commit()
    days = await get_retention_days(db)
    return {"deleted": deleted, "retention_days": days}


class ProviderStatus(BaseModel):
    name: str
    model: str
    allowed: bool
    circuit_open: bool
    open_until: str | None = None
    calls: int = 0
    failures: int = 0
    consecutive_failures: int = 0
    avg_latency_ms: float = 0.0
    est_cost_usd: float = 0.0


@router.get("/providers/status", response_model=list[ProviderStatus])
async def ai_providers_status(
    db: DbSession,
    user: CurrentUser,
) -> list[ProviderStatus]:
    """Circuit-breaker + latency/cost snapshot per AI provider (ops visibility)."""
    from app.services.ai.circuit import snapshot_all

    return [ProviderStatus(**row) for row in snapshot_all()]


class AnalyzeRequest(BaseModel):
    question: str
    data_context: dict | None = None


class AnalyzeResponse(BaseModel):
    answer: str
    suggestions: list[str]
    sources: list[str] = []
    period: str | None = None
    metrics_used: list[str] = []
    disclaimer: str = "AI-generated analysis — verify before making decisions."


@router.post("/analyze", response_model=AnalyzeResponse)
async def ai_analyze(
    body: AnalyzeRequest,
    db: DbSession,
    user: CurrentUser,
) -> AnalyzeResponse:
    from app.core.clock import business_today
    from app.services.ai.service import answer_question

    today = business_today()
    period_str = f"{today - timedelta(days=30)} to {today}"

    result = await answer_question(
        db,
        user.role,
        body.question,
        history=None,
        page=(body.data_context or {}).get("page") if body.data_context else None,
        user=user,
    )
    # Follow-ups are written against the answer the user just got, so they lead
    # somewhere the data can actually go; the keyword list is only the net for
    # when the provider is down.
    suggestions = await _followup_suggestions(body.question, result.reply)

    # The tools the answer was actually built from — not the round count, which
    # is what used to be handed to dict.fromkeys() here and raised TypeError the
    # moment a tool ran.
    sources = list(dict.fromkeys(result.tools_used))
    metrics = _metrics_from_sources(sources, body.question)

    return AnalyzeResponse(
        answer=result.reply,
        suggestions=suggestions,
        sources=sources,
        period=period_str,
        metrics_used=metrics,
        disclaimer="AI-generated analysis — verify before making decisions.",
    )


class InsightOut(BaseModel):
    title: str
    body: str
    type: str
    priority: str


@router.get("/ai/insights", response_model=list[InsightOut])
async def ai_insights(
    db: DbSession,
    user: CurrentUser,
    scope: str = Query("dashboard", pattern="^(dashboard|forecast|anomalies|inventory|all)$"),
) -> list[InsightOut]:
    from app.api.deps import is_super_admin
    from app.services.analytics import queries
    from app.services.analytics.queries import Filters

    today = business_today()
    # Scope filters to caller's org unless super-admin (sees all)
    org_id = None if is_super_admin(user) else user.org_id
    filters = Filters(date_from=today - timedelta(days=29), date_to=today, org_id=org_id)
    insights: list[InsightOut] = []

    if scope in ("dashboard", "all"):
        summary = await queries.kpi_summary(db, filters)
        for card in summary:
            change_pct = card.get("change_pct") or 0
            direction = "up" if change_pct >= 0 else "down"
            val = card.get("value", 0)
            metric = card.get("metric", "")
            insights.append(
                InsightOut(
                    title=(f"{metric.replace('_', ' ').title()} {'Increased' if direction == 'up' else 'Decreased'}"),
                    body=(
                        f"{metric.replace('_', ' ').title()} is at रू {val:,.0f}, "
                        f"{'up' if direction == 'up' else 'down'} {abs(change_pct):.1f}% "
                        f"from the previous period."
                    ),
                    type="kpi",
                    priority="medium" if abs(change_pct) > 15 else "low",
                )
            )

    if scope in ("inventory", "all"):
        stock = await queries.inventory_levels(db, below_reorder_only=True, org_id=org_id)
        if stock:
            names = [s.get("product", "") or s.get("sku", "") for s in stock[:5]]
            insights.append(
                InsightOut(
                    title=f"{len(stock)} Products Below Reorder Level",
                    body=(f"{len(stock)} products need restocking: {', '.join(names)}{'…' if len(stock) > 5 else ''}."),
                    type="inventory",
                    priority="high",
                )
            )

    if scope in ("forecast", "all"):
        from sqlalchemy import select

        from app.models.ml import Forecast as ForecastModel
        from app.models.ml import MlModel

        mq = select(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
        if org_id is not None:
            mq = mq.where(MlModel.org_id == org_id)
        model = (await db.execute(mq)).scalar_one_or_none()
        if model:
            fq = (
                select(ForecastModel)
                .where(ForecastModel.model_id == model.id)
                .order_by(ForecastModel.forecast_date)
                .limit(30)
            )
            if org_id is not None:
                fq = fq.where(ForecastModel.org_id == org_id)
            rows = (await db.execute(fq)).scalars().all()
            if rows:
                total = sum(float(r.yhat) for r in rows)
                insights.append(
                    InsightOut(
                        title="30-Day Revenue Forecast",
                        body=f"Projected revenue for the next 30 days: रू {total:,.0f}. "
                        f"Based on {model.model_type} model v{model.version}.",
                        type="forecast",
                        priority="medium",
                    )
                )

    return insights


# Which warehouse metrics each tool actually reads. Derived from the tools the
# answer used, this reports what the analysis touched — the previous version
# sniffed the question text, so "how did we do?" listed nothing and a question
# merely containing the word "revenue" claimed a revenue source it never read.
_TOOL_METRICS: dict[str, tuple[str, ...]] = {
    "query_kpis": ("revenue", "expenses", "profit"),
    "query_sales": ("revenue", "products"),
    "query_expenses": ("expenses",),
    "query_timeseries": ("revenue",),
    "get_forecast": ("forecasts",),
    "project_period_end": ("forecasts", "revenue"),
    "simulate_scenario": ("forecasts", "revenue"),
    "explain_change": ("revenue", "expenses"),
    "revenue_bridge": ("revenue",),
    "analyse_concentration": ("revenue", "products"),
    "get_anomalies": ("anomalies",),
    "get_inventory": ("inventory_levels",),
    "get_recommendations": ("recommendations",),
    "get_platform_stats": ("organizations",),
    "get_business_info": ("organizations",),
    "describe_catalog": ("catalog",),
    "sample_table": ("catalog",),
    "get_data_coverage": ("catalog",),
    "search_past_insights": ("insights",),
}


def _metrics_from_sources(sources: list[str], question: str) -> list[str]:
    """Metrics the analysis genuinely read, in first-touched order."""
    metrics: list[str] = []
    for name in sources:
        metrics.extend(_TOOL_METRICS.get(name, ()))
    if metrics:
        return list(dict.fromkeys(metrics))
    # No tool ran (snapshot-only answer): fall back to naming what was asked
    # about rather than claiming nothing was consulted.
    q = question.lower()
    guessed = [
        m
        for m, words in (
            ("revenue", ("revenue", "sales")),
            ("expenses", ("expense", "cost")),
            ("forecasts", ("forecast", "predict")),
            ("anomalies", ("anomal",)),
            ("inventory_levels", ("inventory", "stock")),
        )
        if any(w in q for w in words)
    ]
    return guessed


_SUGGESTION_PROMPT = (
    "You suggest the next question a business user should ask their analytics "
    "assistant. Given their question and the answer they received, write exactly "
    "3 short follow-up questions that dig into what the answer actually says — "
    "name the specific businesses, products, categories, periods or numbers that "
    "appeared in it. Each must be answerable from sales, expense, inventory, "
    "forecast, anomaly or organisation data. Reply with only the 3 questions, one "
    "per line, no numbering, no preamble."
)


async def _followup_suggestions(question: str, answer: str) -> list[str]:
    """Follow-ups written against this answer, with a static net beneath.

    A fixed keyword list cannot suggest 'why is Sky Print still pending?' after
    a business listing — it can only offer the same three revenue questions to
    everyone. When the provider is unavailable the static list still runs, so a
    failure here costs relevance, never the response.
    """
    from app.services.ai.provider import get_ai_response

    try:
        raw = await get_ai_response(
            [AIMessage(role="user", content=f"Question: {question}\n\nAnswer given:\n{answer[:1500]}")],
            system_prompt=_SUGGESTION_PROMPT,
        )
        lines = [re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"') for line in (raw or "").splitlines()]
        picked = [line for line in lines if line.endswith("?") and 8 < len(line) <= 120][:3]
        if picked:
            return picked
        logger.info("follow-up model returned nothing usable; using static suggestions")
    except Exception:
        logger.warning("follow-up suggestion generation failed; using static list", exc_info=True)
    return _generate_suggestions(question)


def _generate_suggestions(question: str) -> list[str]:
    q = question.lower()
    if "revenue" in q:
        return [
            "How does this compare to last month?",
            "What's driving revenue changes?",
            "Show me revenue by channel",
        ]
    if "expense" in q:
        return [
            "What's the largest expense category?",
            "How do expenses trend over time?",
            "Compare expenses to revenue",
        ]
    if "forecast" in q or "predict" in q:
        return [
            "What's the confidence interval?",
            "How accurate were past forecasts?",
            "Forecast by product category",
        ]
    if "inventory" in q or "stock" in q:
        return [
            "Which products need restocking?",
            "Inventory turnover rate?",
            "Slow-moving items?",
        ]
    if "anomaly" in q:
        return ["What caused this anomaly?", "Show anomaly trends", "How to prevent anomalies?"]
    return [
        "How is revenue trending this period?",
        "What are our top expense categories?",
        "Show me inventory alerts",
        "What does the 30-day forecast look like?",
    ]


# ── Executive Briefing (Phase 9) ────────────────────────────────────


class ExecutiveBriefing(BaseModel):
    generated_at: str
    period: str
    performance: str
    changes: list[str]
    risks: list[str]
    opportunities: list[str]
    forecast: str
    recommendations: list[str]
    sources: list[str]
    disclaimer: str


@router.get(
    "/briefing",
    response_model=ExecutiveBriefing,
    dependencies=[Depends(require_role("manager"))],
)
async def executive_briefing(
    db: DbSession,
    user: CurrentUser,
) -> ExecutiveBriefing:
    """AI-generated executive briefing — performance, changes, risks, opportunities, forecast, recommendations."""
    from app.core.clock import business_today
    from app.services.ai.service import answer_question
    from app.services.analytics import queries
    from app.services.analytics.queries import Filters

    today = business_today()
    period_str = f"{today - timedelta(days=30)} to {today}"

    # Gather structured data for the briefing
    from app.api.deps import is_super_admin as _is_super

    _org = None if _is_super(user) else user.org_id
    filters = Filters(date_from=today - timedelta(days=29), date_to=today, org_id=_org)
    kpis = await queries.kpi_summary(db, filters)
    kpi_lines = [f"{c['metric'].replace('_', ' ').title()}: {c.get('value', 0):,.0f}" for c in kpis[:5]]

    # Ask AI for the briefing
    briefing_prompt = f"""Generate a concise executive briefing for the period {period_str}.

Key metrics: {", ".join(kpi_lines)}

Structure the response with these exact sections:
1. PERFORMANCE — one paragraph summary
2. CHANGES — 3-4 bullet points of key changes
3. RISKS — 3-4 bullet points of concerns
4. OPPORTUNITIES — 3-4 bullet points of positive signals
5. FORECAST — one paragraph with projection
6. RECOMMENDATIONS — 3-4 actionable next steps

Keep each section brief. Use NPR for currency. Be direct and data-driven."""

    result = await answer_question(
        db,
        user.role,
        briefing_prompt,
        history=None,
        user=user,
    )

    sources = list(dict.fromkeys(result.tool_calls)) if result.tool_calls else []

    # Parse sections from reply (simple heuristic)
    sections: dict[str, Any] = {
        "PERFORMANCE": "",
        "CHANGES": [],
        "RISKS": [],
        "OPPORTUNITIES": [],
        "FORECAST": "",
        "RECOMMENDATIONS": [],
    }
    current = None
    for line in result.reply.split("\n"):
        line = line.strip()
        if line.upper().startswith("PERFORMANCE"):
            current = "PERFORMANCE"
            line = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("CHANGES"):
            current = "CHANGES"
            continue
        elif line.upper().startswith("RISKS"):
            current = "RISKS"
            continue
        elif line.upper().startswith("OPPORTUNITIES"):
            current = "OPPORTUNITIES"
            continue
        elif line.upper().startswith("FORECAST"):
            current = "FORECAST"
            line = line.split(":", 1)[-1].strip()
        elif line.upper().startswith("RECOMMENDATIONS"):
            current = "RECOMMENDATIONS"
            continue
        elif current:
            if line.startswith("-") or line.startswith("•"):
                sections[current].append(line.lstrip("-• ").strip())
            elif current in ("PERFORMANCE", "FORECAST"):
                sections[current] += (" " + line) if sections[current] else line

    return ExecutiveBriefing(
        generated_at=business_today().isoformat() + "T00:00:00+05:45",
        period=period_str,
        performance=sections["PERFORMANCE"] or "Performance data unavailable.",
        changes=sections["CHANGES"] or ["No significant changes detected."],
        risks=sections["RISKS"] or ["No immediate risks identified."],
        opportunities=sections["OPPORTUNITIES"] or ["Review data for opportunities."],
        forecast=sections["FORECAST"] or "Forecast unavailable.",
        recommendations=sections["RECOMMENDATIONS"] or ["No specific recommendations at this time."],
        sources=sources,
        disclaimer="AI-generated executive briefing — verify all figures before decisions.",
    )
