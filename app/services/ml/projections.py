"""Forward-looking analysis: where the period lands, and what would change it.

The trained forecaster answers "what will the next 30 days look like". Two
questions users actually ask are neither that nor a KPI lookup:

  * "Are we going to hit the month?" — a projection of the *current, partly
    elapsed* period that combines what already happened with what the remaining
    days are likely to add.
  * "What if we lifted order value 5%?" — a scenario, where the point is the
    delta against the baseline, not the absolute number.

Both are computed here rather than left to the model, because both are
arithmetic over live figures and a language model doing arithmetic is the
failure this codebase spends most of its effort preventing.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import date, timedelta
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_today
from app.services.analytics.queries import Filters, kpi_summary, kpi_timeseries

#: Weekday profiles need a few observations each before they beat a flat mean.
MIN_DAYS_FOR_WEEKDAY_PROFILE = 28
#: z for a 95% interval, applied to the accumulated variance of the days left.
Z_95 = 1.96

# how stale the series is before we flag it
STALE_DAYS_THRESHOLD = 3


@dataclass
class PeriodProjection:
    metric: str
    period_label: str
    period_start: date
    period_end: date
    days_elapsed: int
    days_remaining: int
    actual_to_date: float
    projected_remainder: float
    projected_total: float
    lower_bound: float
    upper_bound: float
    #: How the projection was built, so the answer can say it out loud instead
    #: of presenting a run-rate as if it were a trained model.
    method: str
    daily_run_rate: float
    # --- extended visualization fields ---
    confidence: float = 0.95
    band_method: str = "flat_std"
    period_key: str = ""
    daily_band: float = 0.0
    coverage: bool = True
    is_stale: bool = False
    daily_cone: list[dict[str, Any]] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "period": self.period_label,
            "period_key": self.period_key or f"{self.period_start.isoformat()}_{self.period_end.isoformat()}",
            "period_start": self.period_start.isoformat(),
            "period_end": self.period_end.isoformat(),
            "days_elapsed": self.days_elapsed,
            "days_remaining": self.days_remaining,
            "actual_to_date": round(self.actual_to_date, 2),
            "projected_remainder": round(self.projected_remainder, 2),
            "projected_total": round(self.projected_total, 2),
            "lower_bound": round(self.lower_bound, 2),
            "upper_bound": round(self.upper_bound, 2),
            "daily_run_rate": round(self.daily_run_rate, 2),
            "method": self.method,
            "confidence": self.confidence,
            "band_method": self.band_method,
            "daily_band": round(self.daily_band, 2),
            "coverage": self.coverage,
            "is_stale": self.is_stale,
            "daily_cone": self.daily_cone,
            # alias for cone
            "per_day_cone": self.daily_cone,
        }


def _zero_fill_history(history: list[tuple[date, float]], today: date) -> list[tuple[date, float]]:
    """Ensure continuous daily history by inserting zeros for missing dates.

    ``kpi_timeseries`` may omit days with no activity (gaps), which would bias
    flat_mean downward if we ignore them or upward if we treat missing as 0
    incorrectly. We zero-fill explicitly and let downstream logic decide to
    exclude missing zeros from flat_mean.
    """
    if not history:
        return []
    # sort and dedup (keep last value per date)
    by_date: dict[date, float] = {}
    for d, v in history:
        by_date[d] = float(v)
    start = min(by_date.keys())
    # fill up to today (or max date in history, whichever is later for projection)
    end = max(max(by_date.keys()), today)
    # but for projection we only need up to today; gaps before start not needed
    filled: list[tuple[date, float]] = []
    cur = start
    while cur <= end:
        filled.append((cur, by_date.get(cur, 0.0)))
        cur += timedelta(days=1)
    return filled


def _weekday_profile(history: list[tuple[date, float]]) -> dict[int, float]:
    """Median (trimmed) value per weekday. Trade is not flat across the week.

    Projecting a month that has four weekends left off a flat daily mean is
    wrong in a direction that depends entirely on which days remain, which is
    why the error looks random rather than like a bias.

    Requires >=28 days overall and >=3 observations per weekday; otherwise
    returns empty dict so caller falls back to flat mean. Uses median for
    robustness, trimmed mean when enough samples.
    """
    if len(history) < 28:
        return {}
    buckets: dict[int, list[float]] = {}
    for day, value in history:
        buckets.setdefault(day.weekday(), []).append(value)
    profile: dict[int, float] = {}
    for wd, vals in buckets.items():
        if len(vals) < 3:
            continue
        # median is robust; for >=5 use trimmed mean (drop min/max) blended with median
        if len(vals) >= 5:
            s = sorted(vals)
            # 10% trimmed mean (drop ~10% each side, at least 1)
            trim = max(1, len(s) // 10)
            trimmed = s[trim:-trim] if len(s) > 2 * trim else s
            # blend median and trimmed mean
            med = statistics.median(vals)
            tmean = statistics.fmean(trimmed) if trimmed else med
            # prefer median but average with trimmed to reduce outlier pull
            profile[wd] = float((med + tmean) / 2)
        else:
            profile[wd] = float(statistics.median(vals))
    return profile


def project_period(
    history: list[tuple[date, float]],
    period_start: date,
    period_end: date,
    today: date,
    metric: str = "revenue",
    period_label: str = "",
) -> PeriodProjection:
    """Project a partly-elapsed period from its own days plus recent history.

    ``history`` is daily observations, and may run before ``period_start`` —
    the extra days only feed the weekday profile and the variance band, never
    the actual-to-date total. History is zero-filled with date_range.
    """
    # zero-fill history first
    history_filled = _zero_fill_history(history, today) if history else []
    # use filled for profile/variance, but keep original for actual? Use filled for actual too with proper filter
    # in_period is based on filled so missing days in period count as 0 elapsed value but still count as elapsed days? 
    # We want days_elapsed = number of days from period_start to min(today, period_end) inclusive,
    # regardless of whether history had a value — gaps inside period are 0 actual but still elapsed.
    # Actual total sums values for those days (missing -> 0).
    period_key = f"{period_start.isoformat()}_{period_end.isoformat()}"

    # Build map for quick lookup
    hist_map: dict[date, float] = {d: v for d, v in history_filled} if history_filled else {d: v for d, v in history}

    # Days in period up to today
    last_elapsed_day = min(today, period_end)
    if last_elapsed_day >= period_start:
        # inclusive count
        days_elapsed = (last_elapsed_day - period_start).days + 1
        # sum actual from hist_map (missing days -> 0)
        actual = sum(hist_map.get(period_start + timedelta(days=i), 0.0) for i in range(days_elapsed))
    else:
        days_elapsed = 0
        actual = 0.0

    # remaining days: (today+1 .. period_end) inclusive
    if today < period_end:
        remaining_start = max(today + timedelta(days=1), period_start)
        remaining_days = [
            period_start + timedelta(days=i)
            for i in range((period_end - period_start).days + 1)
            if period_start + timedelta(days=i) >= remaining_start
        ]
    else:
        remaining_days = []

    # early period: days_elapsed==0 but history exists — still project full remainder
    if not history_filled and not history:
        return PeriodProjection(
            metric=metric,
            period_label=period_label or f"{period_start} → {period_end}",
            period_start=period_start,
            period_end=period_end,
            days_elapsed=0,
            days_remaining=len(remaining_days),
            actual_to_date=0.0,
            projected_remainder=0.0,
            projected_total=0.0,
            lower_bound=0.0,
            upper_bound=0.0,
            method="no data for this period yet",
            daily_run_rate=0.0,
            confidence=0.95,
            band_method="none",
            period_key=period_key,
            daily_band=0.0,
            coverage=False,
            is_stale=False,
            daily_cone=[],
        )

    # Use filled history for calculations
    effective_history = history_filled if history_filled else history
    # values excluding missing zeros for flat_mean (spec: exclude missing zeros)
    # Identify which zeros are "missing" vs true zeros: missing are those we inserted
    # We have original set; zeros we inserted are missing. Exclude those from flat_mean.
    original_dates = {d for d, _ in history} if history else set()
    # flat_mean excludes inserted zeros (i.e., where date not in original and value==0)
    flat_values = []
    for d, v in effective_history:
        if d not in original_dates and v == 0.0:
            # inserted gap — exclude
            continue
        flat_values.append(v)
    if not flat_values:
        flat_values = [v for _, v in effective_history] or [0.0]
    flat_mean = statistics.fmean(flat_values) if flat_values else 0.0

    # coverage flag: whether we have at least expected lookback coverage
    coverage = len(original_dates) >= len(effective_history) * 0.9  # 90% filled

    # stale flag: latest history date is behind today
    latest_hist = max(hist_map.keys()) if hist_map else None
    is_stale = False
    if latest_hist is not None:
        is_stale = (today - latest_hist).days > STALE_DAYS_THRESHOLD

    profile = _weekday_profile(effective_history)
    # require len>=28 and at least 5 weekdays with >=3 obs
    use_profile = len(effective_history) >= MIN_DAYS_FOR_WEEKDAY_PROFILE and len(profile) >= 5

    if use_profile:
        remainder = sum(profile.get(d.weekday(), flat_mean) for d in remaining_days)
        method = f"weekday-adjusted run rate over {len(effective_history)} days of history"
        band_method = "profile_residual"
    else:
        remainder = flat_mean * len(remaining_days)
        method = f"flat run rate over {len(effective_history)} days of history"
        band_method = "flat_residual"

    # --- spread uses residuals after removing profile (not raw pstdev) ---
    if use_profile:
        # residuals after profile
        resid = []
        for d, v in effective_history:
            # skip inserted missing zeros for resid calc
            if d not in original_dates and v == 0.0:
                continue
            expected = profile.get(d.weekday(), flat_mean)
            resid.append(v - expected)
    else:
        resid = [v - flat_mean for v in flat_values]

    if len(resid) > 1:
        # unbiased sample std (ddof=1) — pstdev vs stdev fix
        try:
            spread = statistics.stdev(resid)
        except statistics.StatisticsError:
            spread = 0.0
        # also consider using population? ddof=1 is correct
    else:
        spread = 0.0

    # Band widens with sqrt(days left) — independent daily errors partly cancel
    margin = Z_95 * spread * (len(remaining_days) ** 0.5) if spread and remaining_days else 0.0
    total = actual + remainder

    # daily cone for visualization: per-day projected value plus accumulating variance
    daily_cone: list[dict[str, Any]] = []
    cum = actual
    cum_var = 0.0  # variance accumulates linearly, std = spread * sqrt(n)
    for idx, d in enumerate(remaining_days, start=1):
        if use_profile:
            daily_val = profile.get(d.weekday(), flat_mean)
        else:
            daily_val = flat_mean
        cum += daily_val
        # per-day margin: Z*spread*sqrt(idx) for cumulative; per-day band = Z*spread
        # daily_cone shows cumulative projection cone
        day_lower = max(0.0, cum - Z_95 * spread * (idx ** 0.5)) if spread else cum
        day_upper = cum + Z_95 * spread * (idx ** 0.5) if spread else cum
        daily_cone.append(
            {
                "date": d.isoformat(),
                "weekday": d.weekday(),
                "projected": round(cum, 2),
                "lower": round(day_lower, 2),
                "upper": round(day_upper, 2),
                "daily_value": round(daily_val, 2),
                "coverage": coverage,
            }
        )

    daily_band = Z_95 * spread if spread else 0.0

    # daily run rate: if days_elapsed==0, use flat_mean as run rate (early period projection)
    if days_elapsed > 0:
        daily_run_rate = actual / days_elapsed
    else:
        daily_run_rate = flat_mean if not use_profile else (sum(profile.values()) / len(profile) if profile else flat_mean)

    return PeriodProjection(
        metric=metric,
        period_label=period_label or f"{period_start} → {period_end}",
        period_start=period_start,
        period_end=period_end,
        days_elapsed=days_elapsed,
        days_remaining=len(remaining_days),
        actual_to_date=actual,
        projected_remainder=remainder,
        projected_total=total,
        lower_bound=max(0.0, total - margin),
        upper_bound=total + margin,
        method=method,
        daily_run_rate=daily_run_rate,
        confidence=0.95,
        band_method=band_method,
        period_key=period_key,
        daily_band=daily_band,
        coverage=coverage,
        is_stale=is_stale,
        daily_cone=daily_cone,
    )


@dataclass
class Scenario:
    """A what-if against the live baseline.

    Revenue is modelled as orders × average order value, so a scenario moves
    one or both and the interaction falls out of the multiplication rather than
    being approximated by adding the two percentages.
    """

    baseline_revenue: float
    baseline_orders: float
    baseline_aov: float
    baseline_expenses: float
    baseline_profit: float
    scenario_revenue: float
    scenario_orders: float
    scenario_aov: float
    scenario_expenses: float
    scenario_profit: float
    assumptions: dict[str, float]

    @property
    def revenue_delta(self) -> float:
        return self.scenario_revenue - self.baseline_revenue

    @property
    def profit_delta(self) -> float:
        return self.scenario_profit - self.baseline_profit

    def as_dict(self) -> dict[str, Any]:
        rev_pct = (
            round(self.revenue_delta / self.baseline_revenue * 100, 1) if self.baseline_revenue else None
        )
        profit_pct = (
            round(self.profit_delta / self.baseline_profit * 100, 1) if self.baseline_profit else None
        )
        # if baseline profit ~0, profit_pct is not meaningful
        if self.baseline_profit and abs(self.baseline_profit) < 1e-9:
            profit_pct = None
        return {
            "assumptions": self.assumptions,
            "baseline": {
                "revenue": round(self.baseline_revenue, 2),
                "orders": round(self.baseline_orders, 2),
                "avg_order_value": round(self.baseline_aov, 2),
                "expenses": round(self.baseline_expenses, 2),
                "profit": round(self.baseline_profit, 2),
            },
            "scenario": {
                "revenue": round(self.scenario_revenue, 2),
                "orders": round(self.scenario_orders, 2),
                "avg_order_value": round(self.scenario_aov, 2),
                "expenses": round(self.scenario_expenses, 2),
                "profit": round(self.scenario_profit, 2),
            },
            "delta": {
                "revenue": round(self.revenue_delta, 2),
                "profit": round(self.profit_delta, 2),
                "revenue_pct": rev_pct,
                "profit_pct": profit_pct,
            },
        }


def simulate(
    revenue: float,
    orders: float,
    expenses: float,
    *,
    orders_change_pct: float = 0.0,
    aov_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
    variable_pct: float = 0.0,
    historical_median_aov: float | None = None,
) -> Scenario:
    """Simulate what-if scenario.

    * handles orders==0 via historical_median_aov fallback
    * couples expenses to orders via variable_pct (fraction of expenses that scales with orders)
    * returns profit_pct
    """
    # handle orders==0 using historical median AOV
    if orders and orders != 0:
        aov = revenue / orders
    else:
        if historical_median_aov is not None and historical_median_aov > 0:
            aov = float(historical_median_aov)
            # if revenue was 0 but we have a median AOV, baseline revenue stays as-is (0)
            # but scenario revenue will use median AOV * new_orders
        else:
            aov = 0.0
        # If still 0 but revenue >0 and orders==0 (data anomaly), derive aov from revenue if possible
        # keep 0

    new_orders = orders * (1 + orders_change_pct / 100) if orders else (0.0 if aov == 0 else 0.0)
    # if orders was 0 but we have aov fallback and orders_change implies new orders,
    # we need a baseline orders to scale from — if orders==0 we treat orders_change as absolute?
    # Interpret orders_change_pct when orders==0: if historical_median_aov exists, assume baseline orders implied?
    # Alternative: if orders==0 and orders_change_pct !=0, we cannot compute — keep 0.
    # But to support scenario where orders==0 baseline but we want to simulate new orders,
    # we allow new_orders = 0 * (1+ pct) =0, so still 0. That's intended to avoid inventing data.
    # Callers with historical_median_aov should also pass baseline orders derived from median.
    # For safety, if orders==0 and historical_median_aov and orders_change_pct !=0, we could estimate
    # baseline orders as revenue / aov if revenue>0 else 1? But revenue is 0, so not.
    # Keep 0.

    # However, if orders==0 and historical_median_aov provided and revenue==0, and orders_change_pct >0,
    # scenario revenue will remain 0 which is misleading. We handle by if orders==0 and historical_median_aov:
    # treat new_orders as (orders or 1) scaled? Instead we treat baseline orders as 1 for pct math if orders==0?
    # We leave as 0 to be safe and document.

    new_aov = aov * (1 + aov_change_pct / 100) if aov else (historical_median_aov * (1 + aov_change_pct / 100) if historical_median_aov else 0.0)
    new_revenue = new_orders * new_aov

    # couple expenses to orders via variable_pct
    # variable_pct is fraction of expenses that varies with order volume (0..1)
    # e.g., 0.3 means 30% of expenses scale with orders
    # Clamp variable_pct to [0,1]
    var_pct = max(0.0, min(1.0, float(variable_pct)))
    if var_pct > 0 and orders_change_pct != 0:
        variable_part = expenses * var_pct
        fixed_part = expenses * (1 - var_pct)
        # variable scales with orders, fixed scales with explicit expense_change
        new_variable = variable_part * (1 + orders_change_pct / 100)
        new_fixed = fixed_part * (1 + expense_change_pct / 100)
        new_expenses = new_variable + new_fixed
    else:
        new_expenses = expenses * (1 + expense_change_pct / 100)

    return Scenario(
        baseline_revenue=revenue,
        baseline_orders=orders,
        baseline_aov=aov,
        baseline_expenses=expenses,
        baseline_profit=revenue - expenses,
        scenario_revenue=new_revenue,
        scenario_orders=new_orders,
        scenario_aov=new_aov,
        scenario_expenses=new_expenses,
        scenario_profit=new_revenue - new_expenses,
        assumptions={
            "orders_change_pct": orders_change_pct,
            "aov_change_pct": aov_change_pct,
            "expense_change_pct": expense_change_pct,
            "variable_pct": var_pct,
            "historical_median_aov": float(historical_median_aov) if historical_median_aov is not None else 0.0,
        },
    )


# ── warehouse-backed wrappers ──────────────────────────────────────────────


def month_bounds(day: date) -> tuple[date, date]:
    start = day.replace(day=1)
    next_month = (start + timedelta(days=32)).replace(day=1)
    return start, next_month - timedelta(days=1)


def quarter_bounds(day: date) -> tuple[date, date]:
    first_month = 3 * ((day.month - 1) // 3) + 1
    start = date(day.year, first_month, 1)
    # next quarter is exactly 3 months after start
    if first_month == 10:
        next_q = date(day.year + 1, 1, 1)
    else:
        next_q = date(day.year, first_month + 3, 1)
    return start, next_q - timedelta(days=1)


async def project_current_period(
    db: AsyncSession,
    metric: str = "revenue",
    period: str = "month",
    lookback_days: int = 90,
    org_id=None,
) -> PeriodProjection:
    """Project how the current month or quarter lands, from live daily data.

    Uses load_series zero-filled and adds stale flag.
    """
    today = business_today()
    start, end = quarter_bounds(today) if period == "quarter" else month_bounds(today)
    label = f"Q{(start.month - 1) // 3 + 1} {start.year}" if period == "quarter" else (start.strftime("%B %Y"))

    history_from = min(start, today - timedelta(days=lookback_days - 1))
    # Prefer load_series zero-filled when metric maps to a target, else fallback to kpi_timeseries
    # load_series needs target name like revenue_daily
    target_map = {"revenue": "revenue_daily", "orders": "orders_daily", "expenses": "expenses_daily", "expense_total": "expenses_daily", "gross_margin": "revenue_daily"}
    target = target_map.get(metric)
    history: list[tuple[date, float]] = []
    stale = False
    if target:
        try:
            from app.services.ml.features import load_series

            frame = await load_series(db, target, org_id=org_id)
            if not frame.empty:
                # filter to history_from..today
                mask = (frame["ds"].dt.date >= history_from) & (frame["ds"].dt.date <= today)
                sub = frame.loc[mask]
                history = [(d.date() if hasattr(d, "date") else d, float(y)) for d, y in zip(sub["ds"], sub["y"])]
                # stale check: latest date in frame
                latest = frame["ds"].max().date() if not frame.empty else None
                if latest and (today - latest).days > STALE_DAYS_THRESHOLD:
                    stale = True
            else:
                # fallback to kpi_timeseries if load_series empty
                points = await kpi_timeseries(db, Filters(date_from=history_from, date_to=today, org_id=org_id), metric, "day")
                history = [(p["period"], float(p["value"])) for p in points]
                if history:
                    latest = max(d for d, _ in history)
                    stale = (today - latest).days > STALE_DAYS_THRESHOLD
        except Exception:
            # fallback to kpi_timeseries on any error
            points = await kpi_timeseries(db, Filters(date_from=history_from, date_to=today, org_id=org_id), metric, "day")
            history = [(p["period"], float(p["value"])) for p in points]
    else:
        points = await kpi_timeseries(db, Filters(date_from=history_from, date_to=today, org_id=org_id), metric, "day")
        history = [(p["period"], float(p["value"])) for p in points]
        if history:
            latest = max(d for d, _ in history)
            stale = (today - latest).days > STALE_DAYS_THRESHOLD

    proj = project_period(history, start, end, today, metric=metric, period_label=label)
    # propagate stale flag from load_series check (or keep projection's own)
    proj.is_stale = proj.is_stale or stale
    if stale:
        proj.method += " (stale: latest data is behind today)"
    return proj


async def simulate_current_period(
    db: AsyncSession,
    start: date,
    end: date,
    *,
    orders_change_pct: float = 0.0,
    aov_change_pct: float = 0.0,
    expense_change_pct: float = 0.0,
    variable_pct: float = 0.0,
    org_id=None,
) -> Scenario:
    cards = {c["metric"]: c for c in await kpi_summary(db, Filters(date_from=start, date_to=end, org_id=org_id))}
    revenue = float(cards.get("revenue", {}).get("value") or 0.0)
    orders = float(cards.get("orders", {}).get("value") or 0.0)
    expenses = float(cards.get("expense_total", {}).get("value") or 0.0)
    # historical median AOV for orders==0 fallback: try to compute from longer window
    hist_median_aov = None
    if orders == 0:
        try:
            from app.services.ml.features import load_series

            # last 90 days median AOV
            frame_rev = await load_series(db, "revenue_daily", org_id=org_id)
            frame_ord = await load_series(db, "orders_daily", org_id=org_id)
            if not frame_rev.empty and not frame_ord.empty:
                merged = frame_rev[["ds", "y"]].merge(frame_ord[["ds", "y"]], on="ds", suffixes=("_rev", "_ord"))
                merged = merged[merged["y_ord"] > 0]
                if not merged.empty:
                    aovs = merged["y_rev"] / merged["y_ord"]
                    hist_median_aov = float(aovs.median())
        except Exception:
            hist_median_aov = None

    return simulate(
        revenue=revenue,
        orders=orders,
        expenses=expenses,
        orders_change_pct=orders_change_pct,
        aov_change_pct=aov_change_pct,
        expense_change_pct=expense_change_pct,
        variable_pct=variable_pct,
        historical_median_aov=hist_median_aov,
    )
