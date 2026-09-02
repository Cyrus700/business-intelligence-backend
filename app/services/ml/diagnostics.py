"""Diagnostic analysis: not what the numbers are, but why they moved.

A KPI card says revenue fell 12%. The next question is always "because of
what?", and answering it by hand means pulling two periods, aligning them,
diffing every member and ranking the result — exactly the arithmetic a language
model gets wrong. So the server does it and returns the finished decomposition.

Three lenses:
  * ``decompose`` — attributes a period-over-period movement to the products,
    categories, channels or regions that caused it, separating growth from
    decline and flagging members that appeared or disappeared entirely.
  * ``price_volume_bridge`` — splits a revenue movement into "we sold more" and
    "we sold at a higher value", which point at completely different actions.
  * ``concentration`` — how much of the business rests on how few members, the
    risk that a good headline number hides.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date
from typing import Any

from sqlalchemy.ext.asyncio import AsyncSession

from app.services.analytics.queries import Filters, sales_by_dimension

#: Members contributing less than this share of the total movement are rolled
#: into an "other" line: a decomposition listing forty products explains
#: nothing.
MATERIAL_SHARE_PCT = 2.0
DEFAULT_TOP_N = 5


@dataclass(frozen=True)
class Contribution:
    key: str
    current: float
    previous: float
    delta: float
    #: Signed share of the total movement. A member that fell while the total
    #: rose gets a negative contribution — it worked against the change.
    contribution_pct: float
    status: str  # grew | declined | new | lost | flat

    def as_dict(self) -> dict[str, Any]:
        return {
            "key": self.key,
            "current": round(self.current, 2),
            "previous": round(self.previous, 2),
            "delta": round(self.delta, 2),
            "contribution_pct": round(self.contribution_pct, 1),
            "status": self.status,
        }


@dataclass
class ChangeBreakdown:
    dimension: str
    total_current: float
    total_previous: float
    total_delta: float
    change_pct: float | None
    drivers: list[Contribution] = field(default_factory=list)   # pushed it up
    drags: list[Contribution] = field(default_factory=list)     # held it down
    new_members: list[str] = field(default_factory=list)
    lost_members: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "total_current": round(self.total_current, 2),
            "total_previous": round(self.total_previous, 2),
            "total_delta": round(self.total_delta, 2),
            "change_pct": self.change_pct,
            "drivers": [c.as_dict() for c in self.drivers],
            "drags": [c.as_dict() for c in self.drags],
            "new_members": self.new_members,
            "lost_members": self.lost_members,
        }


def _status(current: float, previous: float) -> str:
    if previous == 0 and current > 0:
        return "new"
    if current == 0 and previous > 0:
        return "lost"
    if current > previous:
        return "grew"
    if current < previous:
        return "declined"
    return "flat"


def decompose(
    current: dict[str, float],
    previous: dict[str, float],
    dimension: str = "product",
    top_n: int = DEFAULT_TOP_N,
) -> ChangeBreakdown:
    """Attribute the movement between two periods to individual members.

    ``contribution_pct`` is measured against the *gross* movement (the sum of
    all absolute deltas), not the net. Netting first is what produces the
    nonsense figure of "product A caused 400% of the change" whenever gains and
    losses roughly cancel — and near-cancellation is the normal case.
    """
    keys = set(current) | set(previous)
    total_current = sum(current.values())
    total_previous = sum(previous.values())
    total_delta = total_current - total_previous

    contributions: list[Contribution] = []
    gross = sum(abs(current.get(k, 0.0) - previous.get(k, 0.0)) for k in keys) or 1.0
    for key in keys:
        curr = current.get(key, 0.0)
        prev = previous.get(key, 0.0)
        delta = curr - prev
        if delta == 0:
            continue
        contributions.append(
            Contribution(
                key=key,
                current=curr,
                previous=prev,
                delta=delta,
                contribution_pct=delta / gross * 100,
                status=_status(curr, prev),
            )
        )

    contributions.sort(key=lambda c: c.delta, reverse=True)
    material = [c for c in contributions if abs(c.contribution_pct) >= MATERIAL_SHARE_PCT]
    drivers = [c for c in material if c.delta > 0][:top_n]
    drags = [c for c in reversed(material) if c.delta < 0][:top_n]

    return ChangeBreakdown(
        dimension=dimension,
        total_current=total_current,
        total_previous=total_previous,
        total_delta=total_delta,
        change_pct=(
            round(total_delta / total_previous * 100, 1) if total_previous else None
        ),
        drivers=drivers,
        drags=drags,
        new_members=sorted(c.key for c in contributions if c.status == "new"),
        lost_members=sorted(c.key for c in contributions if c.status == "lost"),
    )


@dataclass
class PriceVolumeBridge:
    """Why revenue moved: more orders, bigger orders, or both.

    Revenue = orders × average order value, so the movement splits exactly
    three ways and the terms sum back to the total. The interaction term is
    reported rather than folded into the others, because silently attributing
    it makes the two headline effects wrong by however large it is.
    """

    revenue_current: float
    revenue_previous: float
    revenue_delta: float
    orders_current: float
    orders_previous: float
    aov_current: float
    aov_previous: float
    volume_effect: float       # Δorders at last period's average order value
    value_effect: float        # ΔAOV on last period's order count
    interaction_effect: float  # the part only explained by both moving
    verdict: str

    def as_dict(self) -> dict[str, Any]:
        return {
            "revenue_current": round(self.revenue_current, 2),
            "revenue_previous": round(self.revenue_previous, 2),
            "revenue_delta": round(self.revenue_delta, 2),
            "orders_current": round(self.orders_current, 2),
            "orders_previous": round(self.orders_previous, 2),
            "aov_current": round(self.aov_current, 2),
            "aov_previous": round(self.aov_previous, 2),
            "volume_effect": round(self.volume_effect, 2),
            "value_effect": round(self.value_effect, 2),
            "interaction_effect": round(self.interaction_effect, 2),
            "verdict": self.verdict,
        }


def price_volume_bridge(
    orders_current: float,
    orders_previous: float,
    revenue_current: float,
    revenue_previous: float,
) -> PriceVolumeBridge:
    aov_current = revenue_current / orders_current if orders_current else 0.0
    aov_previous = revenue_previous / orders_previous if orders_previous else 0.0
    d_orders = orders_current - orders_previous
    d_aov = aov_current - aov_previous

    volume = d_orders * aov_previous
    value = d_aov * orders_previous
    interaction = d_orders * d_aov

    # The "nothing moved" case has to be tested first: with both effects at
    # zero, `abs(0) >= abs(0) * 2` holds and an empty period would be reported
    # as driven by volume.
    if volume == 0 and value == 0:
        verdict = "no material change"
    elif abs(volume) >= abs(value) * 2:
        verdict = "driven by order volume"
    elif abs(value) >= abs(volume) * 2:
        verdict = "driven by order value"
    else:
        verdict = "volume and order value moved together"

    return PriceVolumeBridge(
        revenue_current=revenue_current,
        revenue_previous=revenue_previous,
        revenue_delta=revenue_current - revenue_previous,
        orders_current=orders_current,
        orders_previous=orders_previous,
        aov_current=aov_current,
        aov_previous=aov_previous,
        volume_effect=volume,
        value_effect=value,
        interaction_effect=interaction,
        verdict=verdict,
    )


@dataclass
class Concentration:
    """How exposed the business is to losing a handful of members."""

    dimension: str
    members: int
    top1_share_pct: float
    top3_share_pct: float
    #: Herfindahl-Hirschman index over revenue shares, 0–1. Above 0.25 is the
    #: conventional "highly concentrated" line.
    hhi: float
    risk: str
    leaders: list[str]

    def as_dict(self) -> dict[str, Any]:
        return {
            "dimension": self.dimension,
            "members": self.members,
            "top1_share_pct": round(self.top1_share_pct, 1),
            "top3_share_pct": round(self.top3_share_pct, 1),
            "hhi": round(self.hhi, 3),
            "risk": self.risk,
            "leaders": self.leaders,
        }


def concentration(values: dict[str, float], dimension: str = "product") -> Concentration:
    ranked = sorted(
        ((k, v) for k, v in values.items() if v > 0), key=lambda kv: kv[1], reverse=True
    )
    total = sum(v for _, v in ranked)
    if not ranked or total <= 0:
        return Concentration(dimension, 0, 0.0, 0.0, 0.0, "no data", [])

    shares = [v / total for _, v in ranked]
    hhi = sum(s * s for s in shares)
    top1 = shares[0] * 100
    top3 = sum(shares[:3]) * 100
    if hhi >= 0.25 or top1 >= 40:
        risk = "high — a single loss would move the headline number"
    elif hhi >= 0.15 or top3 >= 60:
        risk = "moderate — revenue leans on a few members"
    else:
        risk = "low — revenue is well spread"
    return Concentration(
        dimension=dimension,
        members=len(ranked),
        top1_share_pct=top1,
        top3_share_pct=top3,
        hhi=hhi,
        risk=risk,
        leaders=[k for k, _ in ranked[:3]],
    )


# ── warehouse-backed wrappers ──────────────────────────────────────────────

async def _revenue_by(db: AsyncSession, dimension: str, start: date, end: date, org_id=None) -> dict[str, float]:
    rows = await sales_by_dimension(db, Filters(date_from=start, date_to=end, org_id=org_id), dimension)
    return {r["key"]: float(r["revenue"]) for r in rows}


async def explain_change(
    db: AsyncSession,
    dimension: str,
    current: tuple[date, date],
    previous: tuple[date, date],
    top_n: int = DEFAULT_TOP_N,
    org_id=None,
) -> ChangeBreakdown:
    """Decomposition of the movement between two explicit windows."""
    return decompose(
        await _revenue_by(db, dimension, *current, org_id=org_id),
        await _revenue_by(db, dimension, *previous, org_id=org_id),
        dimension=dimension,
        top_n=top_n,
    )


async def analyse_concentration(
    db: AsyncSession, dimension: str, start: date, end: date, org_id=None
) -> Concentration:
    return concentration(await _revenue_by(db, dimension, start, end, org_id=org_id), dimension=dimension)
