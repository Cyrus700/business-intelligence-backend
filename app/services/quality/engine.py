"""Data Quality Framework — measurable quality audit over the warehouse.

Six dimensions are computed per run (app.models.quality.DQ_DIMENSIONS):

* **Completeness** — % of rows whose required fields are present.
* **Validity** — % of rows obeying domain rules (positive quantities, sane
  money, allowed categories, in-range dates).
* **Consistency** — relational integrity (orphan foreign keys) + internal
  arithmetic consistency of stored values + forecast interval sanity.
* **Uniqueness** — natural-key / row-hash duplicates.
* **Timeliness** — how recent the freshest ingested data is vs the business
  clock (score decays with staleness per domain).
* **Accuracy** — rows whose stored total matches a recomputation from
  component fields (total_amount vs qty × price − discount).

Each run persists an overall score, per-dimension scores, a breakdown keyed
by table/domain/source, and individual issues (open → acknowledged → resolved).

Score formula (documented, transparent):
    overall = Σ weightᵢ × dimensionᵢ, weights in app.models.quality.DQ_WEIGHTS.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from datetime import date
from decimal import Decimal

from sqlalchemy import func, select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now, business_today
from app.models import (
    DataSource,
    Expense,
    Forecast,
    InventoryLevel,
    MlModel,
    Product,
    SalesTransaction,
)
from app.models.quality import DQ_WEIGHTS, DataQualityIssue, DataQualityRun

logger = logging.getLogger(__name__)

# Tables audited and the domain each maps to (drives source/domain breakdown).
TABLE_DOMAINS: dict[str, str] = {
    "sales_transactions": "sales",
    "expenses": "finance",
    "inventory_levels": "inventory",
    "products": "sales",
}

VALID_EXPENSE_CATEGORIES = ("rent", "salaries", "utilities", "marketing", "logistics", "other")

# Staleness tolerance (days) before the timeliness score decays per domain.
TIMELINESS_TOLERANCE_DAYS = {"sales": 2, "finance": 3, "inventory": 1}

_MAX_SAMPLE_ROWS = 5
_MAX_ISSUES_PER_TYPE = 20


@dataclass
class _TableResult:
    rows_checked: int
    issues: list[DataQualityIssue]
    dimension_hits: dict[str, int] = field(default_factory=dict)  # passing rows per dim
    dimension_total: dict[str, int] = field(default_factory=dict)  # evaluated rows per dim


def _pct(hits: int, total: int) -> float:
    return round(100.0 * hits / total, 2) if total else 100.0


def _severity_for(issue_type: str, row_count: int) -> str:
    if row_count >= 100 or issue_type in ("orphan_fk", "duplicate_row"):
        return "critical"
    if row_count >= 10:
        return "warning"
    return "info"


def _iso_sample(rows: list[tuple]) -> list[dict]:
    out = []
    for row in rows[: _MAX_SAMPLE_ROWS]:
        out.append({str(i): str(v) for i, v in enumerate(row)})
    return out


def _issue(
    table: str,
    dim: str,
    type_: str,
    severity: str,
    scope_key: str,
    scope_label: str,
    description: str,
    row_count: int,
    sample: dict | None = None,
) -> DataQualityIssue:
    return DataQualityIssue(
        table_name=table,
        dimension=dim,
        issue_type=type_,
        severity=severity,
        scope_key=scope_key,
        scope_label=scope_label,
        description=description,
        row_count=row_count,
        sample=sample,
    )


async def _count(db: AsyncSession, stmt) -> int:
    return (await db.execute(stmt)).scalar_one()


async def _audit_sales(db: AsyncSession) -> _TableResult:
    issues: list[DataQualityIssue] = []
    today = business_today()

    total = await _count(db, select(func.count()).select_from(SalesTransaction))

    complete = await _count(
        db,
        select(func.count())
        .select_from(SalesTransaction)
        .where(
            SalesTransaction.txn_date.is_not(None),
            SalesTransaction.quantity.is_not(None),
            SalesTransaction.unit_price.is_not(None),
            SalesTransaction.total_amount.is_not(None),
            SalesTransaction.product_id.is_not(None),
        ),
    )
    bad = total - complete
    if bad:
        issues.append(
            _issue(
                "sales_transactions", "completeness", "null_required",
                _severity_for("null_required", int(bad)), "domain:sales", "Sales domain",
                f"{int(bad)} of {int(total)} sales rows missing a required field "
                "(txn_date, quantity, unit_price, total_amount or product).",
                int(bad),
            )
        )

    valid = await _count(
        db,
        select(func.count())
        .select_from(SalesTransaction)
        .where(
            SalesTransaction.quantity > 0,
            SalesTransaction.unit_price > 0,
            SalesTransaction.total_amount >= 0,
            SalesTransaction.discount >= 0,
            SalesTransaction.txn_date <= today,
        ),
    )
    invalid = total - valid
    if invalid:
        issues.append(
            _issue(
                "sales_transactions", "validity", "negative_value",
                _severity_for("negative_value", int(invalid)), "domain:sales", "Sales domain",
                f"{int(invalid)} of {int(total)} sales rows violate validity rules "
                "(non-positive quantity/price, negative discount, future date).",
                int(invalid),
            )
        )

    acc_total = await _count(
        db,
        select(func.count())
        .select_from(SalesTransaction)
        .where(
            SalesTransaction.total_amount
            == SalesTransaction.quantity * SalesTransaction.unit_price
            - SalesTransaction.discount
        ),
    )
    acc_bad = total - acc_total
    if acc_bad:
        issues.append(
            _issue(
                "sales_transactions", "accuracy", "expected_total_mismatch",
                _severity_for("expected_total_mismatch", int(acc_bad)),
                "domain:sales", "Sales domain",
                f"{int(acc_bad)} sales rows where total_amount ≠ "
                "quantity × unit_price − discount.",
                int(acc_bad),
            )
        )

    dup_rows = (
        await db.execute(
            select(
                SalesTransaction.txn_date,
                SalesTransaction.product_id,
                SalesTransaction.quantity,
                SalesTransaction.total_amount,
                func.count().label("n"),
            )
            .select_from(SalesTransaction)
            .group_by(
                SalesTransaction.txn_date,
                SalesTransaction.product_id,
                SalesTransaction.quantity,
                SalesTransaction.total_amount,
            )
            .having(func.count() > 1)
        )
    ).all()
    dup_count = sum(int(r.n) - 1 for r in dup_rows)
    if dup_count:
        issues.append(
            _issue(
                "sales_transactions", "uniqueness", "duplicate_row",
                _severity_for("duplicate_row", int(dup_count)), "domain:sales", "Sales domain",
                f"{int(dup_count)} duplicate sales rows (same date, product, quantity, total).",
                int(dup_count),
                _iso_sample([tuple(r) for r in dup_rows[:_MAX_SAMPLE_ROWS]]),
            )
        )

    orphan = await _count(
        db,
        select(func.count())
        .select_from(SalesTransaction)
        .join(Product, SalesTransaction.product_id == Product.id, isouter=True)
        .where(SalesTransaction.product_id.is_not(None), Product.id.is_(None)),
    )
    if orphan:
        issues.append(
            _issue(
                "sales_transactions", "consistency", "orphan_fk",
                "critical", "domain:sales", "Sales domain",
                f"{int(orphan)} sales rows reference a product that no longer exists "
                "(orphaned product_id).",
                int(orphan),
            )
        )

    return _TableResult(
        rows_checked=total or 0,
        issues=issues,
        dimension_hits={
            "completeness": complete or 0,
            "validity": valid or 0,
            "accuracy": acc_total or 0,
            "uniqueness": (total - dup_count) or 0,
            "consistency": (total - orphan) or 0,
        },
        dimension_total={
            "completeness": total or 0,
            "validity": total or 0,
            "accuracy": total or 0,
            "uniqueness": total or 0,
            "consistency": total or 0,
        },
    )


async def _audit_expenses(db: AsyncSession) -> _TableResult:
    issues: list[DataQualityIssue] = []
    today = business_today()

    total = await _count(db, select(func.count()).select_from(Expense))

    complete = await _count(
        db,
        select(func.count())
        .select_from(Expense)
        .where(
            Expense.expense_date.is_not(None),
            Expense.amount.is_not(None),
            Expense.category.is_not(None),
        ),
    )
    bad = total - complete
    if bad:
        issues.append(
            _issue(
                "expenses", "completeness", "null_required",
                _severity_for("null_required", int(bad)), "domain:finance", "Finance domain",
                f"{int(bad)} of {int(total)} expense rows missing date, amount or category.",
                int(bad),
            )
        )

    valid = await _count(
        db,
        select(func.count())
        .select_from(Expense)
        .where(
            Expense.amount >= 0,
            Expense.expense_date <= today,
            Expense.category.in_(VALID_EXPENSE_CATEGORIES),
        ),
    )
    invalid = total - valid
    if invalid:
        issues.append(
            _issue(
                "expenses", "validity", "invalid_category",
                _severity_for("invalid_category", int(invalid)), "domain:finance", "Finance domain",
                f"{int(invalid)} expense rows with negative amounts or categories outside "
                f"{VALID_EXPENSE_CATEGORIES}.",
                int(invalid),
            )
        )

    dup_rows = (
        await db.execute(
            select(
                Expense.expense_date,
                Expense.category,
                Expense.amount,
                func.count().label("n"),
            )
            .select_from(Expense)
            .group_by(Expense.expense_date, Expense.category, Expense.amount)
            .having(func.count() > 1)
        )
    ).all()
    dup_count = sum(int(r.n) - 1 for r in dup_rows)
    if dup_count:
        issues.append(
            _issue(
                "expenses", "uniqueness", "duplicate_row",
                _severity_for("duplicate_row", int(dup_count)), "domain:finance", "Finance domain",
                f"{int(dup_count)} duplicate expense rows (same date, category, amount).",
                int(dup_count),
                _iso_sample([tuple(r) for r in dup_rows[:_MAX_SAMPLE_ROWS]]),
            )
        )

    return _TableResult(
        rows_checked=total or 0,
        issues=issues,
        dimension_hits={
            "completeness": complete or 0,
            "validity": valid or 0,
            "uniqueness": (total - dup_count) or 0,
        },
        dimension_total={
            "completeness": total or 0,
            "validity": total or 0,
            "uniqueness": total or 0,
        },
    )


async def _audit_inventory(db: AsyncSession) -> _TableResult:
    issues: list[DataQualityIssue] = []
    today = business_today()

    total = await _count(db, select(func.count()).select_from(InventoryLevel))

    complete = await _count(
        db,
        select(func.count())
        .select_from(InventoryLevel)
        .where(
            InventoryLevel.snapshot_date.is_not(None),
            InventoryLevel.quantity_on_hand.is_not(None),
            InventoryLevel.product_id.is_not(None),
        ),
    )
    bad = total - complete
    if bad:
        issues.append(
            _issue(
                "inventory_levels", "completeness", "null_required",
                _severity_for("null_required", int(bad)),
                "domain:inventory", "Inventory domain",
                f"{int(bad)} of {int(total)} inventory rows missing snapshot date, "
                "quantity or product.",
                int(bad),
            )
        )

    valid = await _count(
        db,
        select(func.count())
        .select_from(InventoryLevel)
        .where(
            InventoryLevel.quantity_on_hand >= 0,
            InventoryLevel.snapshot_date <= today,
        ),
    )
    invalid = total - valid
    if invalid:
        issues.append(
            _issue(
                "inventory_levels", "validity", "negative_value",
                _severity_for("negative_value", int(invalid)),
                "domain:inventory", "Inventory domain",
                f"{int(invalid)} inventory rows with negative quantity or future snapshot date.",
                int(invalid),
            )
        )

    orphan = await _count(
        db,
        select(func.count())
        .select_from(InventoryLevel)
        .join(Product, InventoryLevel.product_id == Product.id, isouter=True)
        .where(InventoryLevel.product_id.is_not(None), Product.id.is_(None)),
    )
    if orphan:
        issues.append(
            _issue(
                "inventory_levels", "consistency", "orphan_fk",
                "critical", "domain:inventory", "Inventory domain",
                f"{int(orphan)} inventory rows reference a missing product.",
                int(orphan),
            )
        )

    return _TableResult(
        rows_checked=total or 0,
        issues=issues,
        dimension_hits={
            "completeness": complete or 0,
            "validity": valid or 0,
            "consistency": (total - orphan) or 0,
        },
        dimension_total={
            "completeness": total or 0,
            "validity": total or 0,
            "consistency": total or 0,
        },
    )


async def _audit_products(db: AsyncSession) -> _TableResult:
    issues: list[DataQualityIssue] = []

    total = await _count(db, select(func.count()).select_from(Product))

    complete = await _count(
        db,
        select(func.count()).select_from(Product).where(
            Product.sku.is_not(None), Product.name.is_not(None)
        ),
    )
    bad = total - complete
    if bad:
        issues.append(
            _issue(
                "products", "completeness", "null_required",
                _severity_for("null_required", int(bad)),
                "table:products", "Product dimension",
                f"{int(bad)} products missing SKU or name.",
                int(bad),
            )
        )

    dup_rows = (
        await db.execute(
            select(Product.sku, func.count().label("n"))
            .select_from(Product)
            .group_by(Product.sku)
            .having(func.count() > 1)
        )
    ).all()
    dup_count = sum(int(r.n) - 1 for r in dup_rows)
    if dup_count:
        issues.append(
            _issue(
                "products", "uniqueness", "duplicate_row",
                _severity_for("duplicate_row", int(dup_count)),
                "table:products", "Product dimension",
                f"{int(dup_count)} duplicate product SKUs.",
                int(dup_count),
                _iso_sample([tuple(r) for r in dup_rows[:_MAX_SAMPLE_ROWS]]),
            )
        )

    return _TableResult(
        rows_checked=total or 0,
        issues=issues,
        dimension_hits={
            "completeness": complete or 0,
            "uniqueness": (total - dup_count) or 0,
        },
        dimension_total={"completeness": total or 0, "uniqueness": total or 0},
    )


async def _compute_timeliness(
    db: AsyncSession, today: date
) -> tuple[float, list[DataQualityIssue]]:
    """Timeliness per domain: how long since the freshest row was ingested.

    Score 100 when within tolerance, decaying linearly to 0 at 4× tolerance.
    """
    issues: list[DataQualityIssue] = []
    scores: list[float] = []
    domains = {"sales": SalesTransaction, "finance": Expense, "inventory": InventoryLevel}
    for domain, model in domains.items():
        latest = (
            await db.execute(select(func.max(model.ingested_at)).select_from(model))
        ).scalar_one()
        tolerance = TIMELINESS_TOLERANCE_DAYS[domain]
        if latest is None:
            scores.append(100.0)  # empty domain: nothing stale
            continue
        stale_days = max(0, (today - latest.date()).days)
        score = max(0.0, 100.0 * (1 - max(0, stale_days - tolerance) / (4 * tolerance)))
        scores.append(score)
        if stale_days > tolerance:
            issues.append(
                _issue(
                    domain, "timeliness", "stale_ingestion",
                    "warning" if stale_days <= 14 else "critical",
                    f"domain:{domain}", f"{domain.capitalize()} domain",
                    f"No {domain} rows ingested for {stale_days} day(s) "
                    f"(tolerance: {tolerance} day(s)).",
                    1,
                )
            )
    return round(sum(scores) / len(scores), 2) if scores else 100.0, issues


async def _verify_forecast_consistency(db: AsyncSession) -> list[DataQualityIssue]:
    """Consistency indicator: active forecast rows whose interval is inverted."""
    rows = (
        await db.execute(
            select(Forecast.id)
            .join(MlModel, Forecast.model_id == MlModel.id)
            .where(
                MlModel.is_active.is_(True),
                Forecast.yhat_upper.is_not(None),
                Forecast.yhat_lower.is_not(None),
                Forecast.yhat_upper < Forecast.yhat_lower,
            )
        )
    ).all()
    if not rows:
        return []
    return [
        _issue(
            "forecasts", "consistency", "invalid_date",
            "warning", "domain:sales", "Sales domain",
            f"{len(rows)} active forecast rows have upper bound below lower bound "
            "(inverted confidence interval).",
            len(rows),
        )
    ]


async def _source_quality(
    db: AsyncSession,
) -> dict:
    """Per-source pipeline health: latest ETL job outcome per data source."""
    from app.models import EtlJob

    sources = (await db.execute(select(DataSource))).scalars().all()
    out: dict[str, dict] = {}
    for source in sources:
        job = (
            await db.execute(
                select(EtlJob)
                .where(EtlJob.data_source_id == source.id)
                .order_by(EtlJob.started_at.desc())
                .limit(1)
            )
        ).scalars().first()
        out[source.name] = {
            "kind": source.kind,
            "status": source.status,
            "last_job": job.status if job else None,
            "last_run_at": job.started_at.isoformat() if job else None,
            "rows_loaded": job.rows_loaded if job else 0,
            "rows_rejected": job.rows_rejected if job else 0,
        }
    return out


async def run_quality_audit(
    db: AsyncSession, triggered_by: str = "schedule", org_id=None
) -> DataQualityRun | None:
    """Run the full quality audit and persist a DataQualityRun + issues."""
    start = time.perf_counter()
    today = business_today()

    audits = {
        "sales_transactions": _audit_sales,
        "expenses": _audit_expenses,
        "inventory_levels": _audit_inventory,
        "products": _audit_products,
    }
    results: dict[str, _TableResult] = {}
    all_issues: list[DataQualityIssue] = []
    rows_checked = 0

    for table, audit in audits.items():
        try:
            results[table] = await audit(db)
            rows_checked += results[table].rows_checked
            all_issues.extend(results[table].issues)
        except Exception:
            logger.exception("DQ audit of %s failed", table)

    timeliness, timeliness_issues = await _compute_timeliness(db, today)
    all_issues.extend(timeliness_issues)
    all_issues.extend(await _verify_forecast_consistency(db))

    dim_scores = _dimension_scores(results, timeliness)
    overall = round(sum(DQ_WEIGHTS[d] * dim_scores[d] for d in DQ_WEIGHTS), 2)

    breakdown = {
        "by_table": _table_breakdown(results, timeliness),
        "by_domain": _domain_breakdown(results),
        "by_source": await _source_quality(db),
        "summary": {
            "dimensions": dim_scores,
            "timeliness_score": timeliness,
            "rows_checked": rows_checked,
        },
    }

    run = DataQualityRun(
        run_date=today,
        score=Decimal(str(overall)),
        dimensions=dim_scores,
        breakdown=breakdown,
        rows_checked=rows_checked,
        issues_found=len(all_issues),
        triggered_by=triggered_by,
        duration_ms=int((time.perf_counter() - start) * 1000),
        status="succeeded",
        org_id=org_id,
    )
    db.add(run)
    await db.flush()

    all_issues.sort(key=lambda i: (i.row_count, i.issue_type), reverse=True)
    for issue in all_issues[:_MAX_ISSUES_PER_TYPE]:
        issue.run_id = run.id
        issue.org_id = org_id
        db.add(issue)

    try:
        await db.commit()
    except Exception:
        logger.exception("DQ run commit failed")
        await db.rollback()
        return None

    logger.info(
        "data quality run: score=%s rows=%d issues=%d (%s)",
        overall, rows_checked, len(all_issues[:_MAX_ISSUES_PER_TYPE]), triggered_by,
    )
    return run


def _dimension_scores(results: dict[str, _TableResult], timeliness: float) -> dict[str, float]:
    agg: dict[str, list[int]] = {}
    for res in results.values():
        for dim, hits in res.dimension_hits.items():
            bucket = agg.setdefault(dim, [0, 0])
            bucket[0] += hits
            bucket[1] += res.dimension_total[dim]

    out: dict[str, float] = {}
    for dim, (hits, total) in agg.items():
        out[dim] = _pct(hits, total)
    out["timeliness"] = timeliness
    return out


def _table_breakdown(results: dict[str, _TableResult], timeliness: float) -> dict:
    out: dict[str, dict] = {}
    for table, res in results.items():
        table_scores = {
            dim: _pct(res.dimension_hits[dim], res.dimension_total[dim])
            for dim in res.dimension_hits
        }
        if table in TABLE_DOMAINS:
            table_scores["timeliness"] = timeliness
        out[table] = {
            "rows_checked": res.rows_checked,
            "issues": len(res.issues),
            "scores": table_scores,
        }
    return out


def _domain_breakdown(results: dict[str, _TableResult]) -> dict:
    domains: dict[str, dict] = {}
    for table, res in results.items():
        domain = TABLE_DOMAINS.get(table, "other")
        entry = domains.setdefault(
            domain, {"rows_checked": 0, "tables": [], "issues": 0}
        )
        entry["rows_checked"] += res.rows_checked
        entry["issues"] += len(res.issues)
        entry["tables"].append(table)
    return domains


async def acknowledge_issue(
    db: AsyncSession, issue_id, status: str, user_id
) -> DataQualityIssue | None:
    status = status if status in ("acknowledged", "resolved") else "acknowledged"
    issue = await db.get(DataQualityIssue, issue_id)
    if issue is None:
        return None
    issue.status = status
    issue.resolved_by = user_id
    issue.resolved_at = business_now().replace(tzinfo=None)
    await db.commit()
    return issue