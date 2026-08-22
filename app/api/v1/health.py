"""Business health score + platform health checks (Phase 11)."""

from datetime import timedelta
from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel
from sqlalchemy import func, select, text

from app.api.deps import DbSession, get_current_user, require_role
from app.core.clock import business_now
from app.models import EtlJob, Message, MlModel

router = APIRouter(tags=["health"])


@router.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}


@router.get("/health/db")
async def health_db(db: DbSession) -> dict[str, str]:
    await db.execute(text("SELECT 1"))
    return {"status": "ok", "database": "reachable"}


class HealthComponent(BaseModel):
    name: str
    weight: float
    score: float
    detail: str


class BusinessHealthOut(BaseModel):
    score: float
    label: str
    formula: str
    components: list[HealthComponent]


class SystemComponent(BaseModel):
    name: str
    status: str  # ok | degraded | down
    latency_ms: float | None = None
    detail: str


class SystemHealthOut(BaseModel):
    generated_at: str
    components: list[SystemComponent]
    overall: str


async def _business_health(db: DbSession) -> dict[str, Any]:
    """Transparent Business Health Score.

    Formula: weighted mean of five components, each 0-100:
      growth          30%  — 14d revenue vs prior 14d (5% growth → 75)
      profitability   25%  — revenue / expenses over last 30 days
      forecast_health 15%  — 100 − MAPE of the active forecast
      data_freshness  15%  — 0 days stale → 100, ≥7 days stale → 0
      open_anomalies  15%  — 100 − 10×open anomalies (floored at 0)
    Label: ≥75 Healthy, ≥50 Attention, else Critical.
    """
    now = business_now()
    today = now.date()
    formula = (
        "score = 0.30*growth + 0.25*profitability + 0.15*forecast_health "
        "+ 0.15*freshness + 0.15*anomalies (each 0-100)"
    )
    components: list[dict[str, Any]] = []

    def add(name: str, weight: float, score: float, detail: str) -> None:
        score = max(0.0, min(100.0, score))
        components.append(
            {"name": name, "weight": weight, "score": round(score, 1), "detail": detail}
        )

    # growth: 14d vs previous 14d revenue
    revenue = (
        await db.execute(
            text(
                "SELECT SUM(value) AS total FROM kpi_snapshots "
                "WHERE metric = 'revenue' AND snapshot_date BETWEEN :a AND :b"
            ),
            {
                "a": today - timedelta(days=14),
                "b": today - timedelta(days=1),
            },
        )
    ).scalar() or 0
    revenue_prev = (
        await db.execute(
            text(
                "SELECT SUM(value) AS total FROM kpi_snapshots "
                "WHERE metric = 'revenue' AND snapshot_date BETWEEN :a AND :b"
            ),
            {
                "a": today - timedelta(days=28),
                "b": today - timedelta(days=15),
            },
        )
    ).scalar() or 0
    if revenue_prev > 0:
        change = (float(revenue) - float(revenue_prev)) / float(revenue_prev)
    else:
        change = 0.0
    add("growth", 0.30, 50 + change * 500, f"revenue {change:+.1%} vs prior 14d")

    # profitability: revenue vs expenses, last 30 days
    expenses = (
        await db.execute(
            text(
                "SELECT COALESCE(SUM(value), 0) FROM kpi_snapshots "
                "WHERE metric = 'expense_total' AND snapshot_date >= :a"
            ),
            {"a": today - timedelta(days=30)},
        )
    ).scalar() or 0
    margin = 1.0
    if float(expenses) > 0:
        margin = (float(revenue or 0) - float(expenses)) / float(expenses)
    add(
        "profitability",
        0.25,
        50 + margin * 50,
        f"margin vs expenses {margin:+.1%} (30d)",
    )

    # forecast health: best MAPE among active/retired models
    model_mape: float | None = None
    model = (
        await db.execute(
            select(MlModel)
            .where(MlModel.target == "revenue_daily", MlModel.is_active.is_(True))
            .order_by(MlModel.version.desc())
        )
    ).scalar_one_or_none()
    if model is not None and model.metrics:
        model_mape = float((model.metrics or {}).get("mape") or 100)
    add(
        "forecast_health",
        0.15,
        100 - (model_mape if model_mape is not None else 100),
        f"active model MAPE {model_mape:.1f}%" if model_mape is not None else "no trained model",
    )

    # data freshness: days since the latest snapshot
    latest = (
        await db.execute(
            text("SELECT MAX(snapshot_date) FROM kpi_snapshots WHERE metric = 'revenue'")
        )
    ).scalar()
    if latest is None:
        freshness = 0.0
        detail = "no data yet"
    else:
        stale_days = (today - latest).days
        freshness = 100 * max(0.0, 1.0 - stale_days / 7)
        detail = f"data {stale_days}d old"
    add("data_freshness", 0.15, freshness, detail)

    # open anomalies penalty
    open_count = (
        await db.execute(
            select(func.count()).select_from(
                text("anomalies WHERE status IN ('open', 'acknowledged')")
            )
        )
    ).scalar() or 0
    add("open_anomalies", 0.15, 100 - 10 * int(open_count), f"{open_count} open anomaly(s)")

    score = sum(c["score"] * c["weight"] for c in components)
    label = "Healthy" if score >= 75 else ("Attention" if score >= 50 else "Critical")
    return {"score": round(score, 1), "label": label, "formula": formula, "components": components}


@router.get(
    "/health/business",
    response_model=BusinessHealthOut,
    dependencies=[Depends(get_current_user)],
)
async def business_health(db: DbSession) -> BusinessHealthOut:
    result = await _business_health(db)
    return BusinessHealthOut(**result)


async def _system_health(db: DbSession) -> dict[str, Any]:
    """Per-component latency/status: DB, storage, scheduler, ETL, AI, email."""
    import time

    components: list[dict[str, Any]] = []

    def add(name: str, status: str, latency_ms: float | None, detail: str) -> None:
        components.append(
            {"name": name, "status": status, "latency_ms": latency_ms, "detail": detail}
        )

    start = time.perf_counter()
    try:
        await db.execute(text("SELECT 1"))
        add("database", "ok", round((time.perf_counter() - start) * 1000, 1), "SELECT 1 ok")
    except Exception as exc:  # noqa: BLE001
        add("database", "down", None, f"{type(exc).__name__}: {exc}")

    start = time.perf_counter()
    try:
        from app.services.storage import LOCAL_ROOT

        ok = LOCAL_ROOT.is_dir()
        add("storage", "ok" if ok else "down", round((time.perf_counter() - start) * 1000, 1),
            f"root {LOCAL_ROOT}")
    except Exception as exc:  # noqa: BLE001
        add("storage", "down", None, f"{type(exc).__name__}: {exc}")

    last_etl = (
        await db.execute(
            select(EtlJob).order_by(EtlJob.started_at.desc()).limit(1)
        )
    ).scalar_one_or_none()
    if last_etl is None:
        add("etl", "degraded", None, "no ETL job ever ran")
    elif last_etl.status == "success":
        add("etl", "ok", None, f"last run {last_etl.started_at:%Y-%m-%d %H:%M}")
    else:
        add("etl", "degraded", None, f"last run {last_etl.started_at:%Y-%m-%d %H:%M} = {last_etl.status}")

    from app.services.ai.circuit import snapshot_all

    ai = snapshot_all()
    ai_statuses = {row["name"]: row for row in ai}
    if not ai_statuses:
        add("ai_providers", "degraded", None, "no provider configured")
    else:
        worst = max((v["failures"] for v in ai_statuses.values()), default=0)
        any_open = any(v["circuit_open"] for v in ai_statuses.values())
        add(
            "ai_providers",
            "down" if any_open else ("ok" if worst == 0 else "degraded"),
            round(max(v["avg_latency_ms"] for v in ai_statuses.values()), 1),
            f"{len(ai_statuses)} provider(s), {worst} failure(s)",
        )

    overall = "ok" if all(c["status"] == "ok" for c in components) else (
        "degraded" if all(c["status"] in ("ok", "degraded") for c in components) else "down"
    )
    return {
        "generated_at": business_now().isoformat(),
        "components": components,
        "overall": overall,
    }


@router.get(
    "/health/system",
    response_model=SystemHealthOut,
    dependencies=[Depends(require_role("admin"))],
)
async def system_health(db: DbSession) -> SystemHealthOut:
    result = await _system_health(db)
    return SystemHealthOut(**result)


class UsageRow(BaseModel):
    provider: str
    calls: int
    failures: int
    circuit_open: bool
    avg_latency_ms: float
    est_cost_usd: float

    @classmethod
    def from_snapshot(cls, snap: dict) -> "UsageRow":
        return cls(
            provider=snap.get("name") or snap.get("provider", "unknown"),
            calls=snap.get("calls", 0),
            failures=snap.get("failures", 0),
            circuit_open=snap.get("circuit_open", False),
            avg_latency_ms=snap.get("avg_latency_ms", 0.0),
            est_cost_usd=snap.get("est_cost_usd", 0.0),
        )


class AiUsageOut(BaseModel):
    generated_at: str
    providers: list[UsageRow]
    requests_14d: int
    assistant_messages_14d: int
    active_users_14d: int
    total_est_cost_usd: float


@router.get(
    "/ai/usage",
    response_model=AiUsageOut,
    dependencies=[Depends(require_role("admin"))],
)
async def ai_usage(db: DbSession) -> AiUsageOut:
    """AI usage/cost monitoring (admin) — circuit snapshots + message volume."""
    from app.services.ai.circuit import snapshot_all

    providers = snapshot_all()
    # ai_messages.created_at is tz-naive; business_now() is aware
    cutoff = business_now().replace(tzinfo=None) - timedelta(days=14)
    requests = (
        await db.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.role == "user", Message.created_at >= cutoff)
        )
    ).scalar() or 0
    assistant = (
        await db.execute(
            select(func.count())
            .select_from(Message)
            .where(Message.role == "assistant", Message.created_at >= cutoff)
        )
    ).scalar() or 0
    users = (
        await db.execute(
            text(
                "SELECT COUNT(DISTINCT c.user_id) FROM ai_messages m "
                "JOIN ai_conversations c ON c.id = m.conversation_id "
                "WHERE m.role = 'user' AND m.created_at >= :cutoff"
            ),
            {"cutoff": cutoff},
        )
    ).scalar() or 0
    total_cost = sum(float(p["est_cost_usd"]) for p in providers)
    return AiUsageOut(
        generated_at=business_now().isoformat(),
        providers=[UsageRow.from_snapshot(p) for p in providers],
        requests_14d=int(requests),
        assistant_messages_14d=int(assistant),
        active_users_14d=int(users),
        total_est_cost_usd=round(total_cost, 4),
    )