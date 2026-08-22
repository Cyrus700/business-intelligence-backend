import asyncio
import json
import logging
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
        cid = conv.id
        conv.updated_at = now
    else:
        conv = Conversation(user_id=user.id, title=question[:80], updated_at=now)
        db.add(conv)
        await db.flush()
        cid = conv.id

    db.add(Message(conversation_id=cid, role="user", content=question))
    await db.flush()

    from sqlalchemy import select as sa_select

    history = await db.execute(
        sa_select(Message)
        .where(Message.conversation_id == cid)
        .order_by(Message.created_at, Message.id)
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
        db.add(Message(conversation_id=cid, role="assistant", content=result.reply))
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
        db.add(Message(conversation_id=cid, role="assistant", content=polish_reply(reply)))
        await db.commit()

    async def event_gen():
        yield _sse({"conversation_id": str(cid)})
        reply_parts: list[str] = []
        try:
            async for chunk in stream_answer(
                db, user.role, body.message, msgs, page=page, user=user
            ):
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

    subq = (
        select(Message.conversation_id, sa_func.count().label("cnt"))
        .group_by(Message.conversation_id)
        .subquery()
    )
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
        ConversationOut(
            id=str(c.id), title=c.title, created_at=c.created_at.isoformat(), message_count=cnt or 0
        )
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
) -> list[MessageOut]:
    conv = await db.get(Conversation, conv_id)
    if not conv or conv.user_id != user.id:
        raise HTTPException(404, "Conversation not found")

    from sqlalchemy import select as sa_select

    rows = (
        await db.execute(
            sa_select(Message)
            .where(Message.conversation_id == conv_id)
            .order_by(Message.created_at, Message.id)
        )
    ).scalars().all()
    return [
        MessageOut(
            id=str(m.id),
            role=m.role,
            content=repair_mojibake(m.content),
            created_at=m.created_at.isoformat(),
        )
        for m in rows
    ]


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
    suggestions = _generate_suggestions(body.question)

    # Collect sources used by the tool loop
    sources = list(dict.fromkeys(result.tool_calls)) if result.tool_calls else []
    metrics = []
    if "revenue" in body.question.lower():
        metrics.append("revenue")
    if "expense" in body.question.lower() or "cost" in body.question.lower():
        metrics.append("expenses")
    if "forecast" in body.question.lower():
        metrics.append("forecasts")
    if "anomal" in body.question.lower():
        metrics.append("anomalies")
    if "inventory" in body.question.lower() or "stock" in body.question.lower():
        metrics.append("inventory_levels")

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
    from app.services.analytics import queries
    from app.services.analytics.queries import Filters

    today = business_today()
    filters = Filters(date_from=today - timedelta(days=29), date_to=today)
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
                    title=(
                        f"{metric.replace('_', ' ').title()} "
                        f"{'Increased' if direction == 'up' else 'Decreased'}"
                    ),
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
        stock = await queries.inventory_levels(db, below_reorder_only=True)
        if stock:
            names = [s.get("product", "") or s.get("sku", "") for s in stock[:5]]
            insights.append(
                InsightOut(
                    title=f"{len(stock)} Products Below Reorder Level",
                    body=(
                        f"{len(stock)} products need restocking: "
                        f"{', '.join(names)}{'…' if len(stock) > 5 else ''}."
                    ),
                    type="inventory",
                    priority="high",
                )
            )

    if scope in ("forecast", "all"):
        from sqlalchemy import select

        from app.models.ml import Forecast as ForecastModel
        from app.models.ml import MlModel

        model = (
            await db.execute(
                select(MlModel).where(
                    MlModel.target == "revenue_daily", MlModel.is_active.is_(True)
                )
            )
        ).scalar_one_or_none()
        if model:
            rows = (
                (await db.execute(
                    select(ForecastModel)
                    .where(ForecastModel.model_id == model.id)
                    .order_by(ForecastModel.forecast_date)
                    .limit(30)
                ))
                .scalars()
                .all()
            )
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
    filters = Filters(date_from=today - timedelta(days=29), date_to=today)
    kpis = await queries.kpi_summary(db, filters)
    kpi_lines = [f"{c['metric'].replace('_', ' ').title()}: {c.get('value', 0):,.0f}" for c in kpis[:5]]

    # Ask AI for the briefing
    briefing_prompt = f"""Generate a concise executive briefing for the period {period_str}.

Key metrics: {', '.join(kpi_lines)}

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
