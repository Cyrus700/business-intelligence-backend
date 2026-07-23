import uuid
from datetime import date, timedelta

from fastapi import APIRouter, Depends, HTTPException, Query
from pydantic import BaseModel

from app.api.deps import CurrentUser, DbSession, get_current_user
from app.models.ai import Conversation, Message
from app.services.ai import AIMessage, get_ai_response

router = APIRouter(prefix="/ai", tags=["ai"], dependencies=[Depends(get_current_user)])


class ChatRequest(BaseModel):
    conversation_id: str | None = None
    message: str
    context: dict | None = None


class ChatResponse(BaseModel):
    conversation_id: str
    reply: str


@router.post("/chat", response_model=ChatResponse)
async def ai_chat(
    body: ChatRequest,
    db: DbSession,
    user: CurrentUser,
) -> ChatResponse:
    conv_id = uuid.UUID(body.conversation_id) if body.conversation_id else None

    if conv_id:
        conv = await db.get(Conversation, conv_id)
        if not conv or conv.user_id != user.id:
            raise HTTPException(404, "Conversation not found")
    else:
        conv = Conversation(user_id=user.id, title=body.message[:80])
        db.add(conv)
        await db.flush()
        conv_id = conv.id

    db.add(Message(conversation_id=conv_id, role="user", content=body.message))
    await db.flush()

    from sqlalchemy import select as sa_select

    history = await db.execute(
        sa_select(Message)
        .where(Message.conversation_id == conv_id)
        .order_by(Message.created_at)
    )
    msgs = [AIMessage(role=m.role, content=m.content) for m in history.scalars().all()]

    system = _build_system_prompt(user, body.context)
    reply = await get_ai_response(msgs, system_prompt=system)

    db.add(Message(conversation_id=conv_id, role="assistant", content=reply))
    await db.commit()

    return ChatResponse(conversation_id=str(conv_id), reply=reply)


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
    from sqlalchemy import func as sa_func, select

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
            .order_by(Message.created_at)
        )
    ).scalars().all()
    return [
        MessageOut(id=str(m.id), role=m.role, content=m.content, created_at=m.created_at.isoformat())
        for m in rows
    ]


class AnalyzeRequest(BaseModel):
    question: str
    data_context: dict | None = None


class AnalyzeResponse(BaseModel):
    answer: str
    suggestions: list[str] = []


@router.post("/analyze", response_model=AnalyzeResponse)
async def ai_analyze(
    body: AnalyzeRequest,
    user: CurrentUser,
) -> AnalyzeResponse:
    system = (
        "You are a data analyst AI for a business intelligence dashboard. "
        "Answer the user's question about their business data. "
        "Be specific, use numbers from the provided context, and suggest actionable insights. "
        "If you don't have enough data, suggest what KPIs to look at on the dashboard. "
        f"The user's role is: {user.role}."
    )
    ctx_str = f"\nContext data: {body.data_context}" if body.data_context else ""
    msgs = [AIMessage(role="user", content=f"{body.question}{ctx_str}")]
    answer = await get_ai_response(msgs, system_prompt=system)
    suggestions = _generate_suggestions(body.question)
    return AnalyzeResponse(answer=answer, suggestions=suggestions)


class InsightOut(BaseModel):
    title: str
    body: str
    type: str
    priority: str


@router.get("/insights", response_model=list[InsightOut])
async def ai_insights(
    db: DbSession,
    user: CurrentUser,
    scope: str = Query("dashboard", pattern="^(dashboard|forecast|anomalies|inventory|all)$"),
) -> list[InsightOut]:
    from app.services.analytics import queries
    from app.services.analytics.queries import Filters

    today = date.today()
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
                    title=f"{metric.replace('_', ' ').title()} {'Increased' if direction == 'up' else 'Decreased'}",
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
                    body=f"{len(stock)} products need restocking: {', '.join(names)}{'…' if len(stock) > 5 else ''}.",
                    type="inventory",
                    priority="high",
                )
            )

    if scope in ("forecast", "all"):
        from sqlalchemy import select
        from app.models.ml import MlModel, Forecast as ForecastModel

        model = (
            await db.execute(
                select(MlModel).where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
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


def _build_system_prompt(user, context: dict | None = None) -> str:
    ctx = context or {}
    extra = ""
    if ctx.get("page"):
        extra += f"\nThe user is currently on the {ctx['page']} page."
    if ctx.get("metrics"):
        extra += f"\nAvailable metrics: {', '.join(ctx['metrics'])}."
    return (
        f"You are Insightful AI, a BI assistant for a retail analytics dashboard. "
        f"The user ({user.role}) can view KPIs, sales data, expenses, inventory, forecasts, and alerts. "
        f"Be concise and data-driven.{extra}"
    )


def _generate_suggestions(question: str) -> list[str]:
    q = question.lower()
    if "revenue" in q:
        return ["How does this compare to last month?", "What's driving revenue changes?", "Show me revenue by channel"]
    if "expense" in q:
        return ["What's the largest expense category?", "How do expenses trend over time?", "Compare expenses to revenue"]
    if "forecast" in q or "predict" in q:
        return ["What's the confidence interval?", "How accurate were past forecasts?", "Forecast by product category"]
    if "inventory" in q or "stock" in q:
        return ["Which products need restocking?", "Inventory turnover rate?", "Slow-moving items?"]
    if "anomaly" in q:
        return ["What caused this anomaly?", "Show anomaly trends", "How to prevent anomalies?"]
    return [
        "How is revenue trending this period?",
        "What are our top expense categories?",
        "Show me inventory alerts",
        "What does the 30-day forecast look like?",
    ]
