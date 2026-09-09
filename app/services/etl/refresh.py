"""Refresh the derived layer immediately after new data lands.

Loading rows rebuilds the KPI snapshots and stops there. Everything else the
dashboard and the assistant read from — anomaly alerts, generated insights, the
assistant's retrieval index — was only ever rebuilt by the nightly cron. So an
upload at 10am produced a dashboard with fresh KPIs sitting next to anomalies,
insights and AI answers describing yesterday's business, for the rest of the
day. Users read that as "my upload didn't work".

This closes the gap: the moment an ingest commits, the derived layer is brought
forward over the window that actually changed. Retraining is deliberately not
included — it takes tens of seconds and belongs to the weekly job and the drift
check, which now see the new rows anyway.
"""

import logging
from dataclasses import dataclass
from datetime import date

from sqlalchemy import text
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.clock import business_now

logger = logging.getLogger(__name__)

#: Never rescan more than this after one load. A backfill covering two years
#: would otherwise turn a single upload into a multi-minute anomaly sweep.
MAX_RESCAN_DAYS = 120


@dataclass
class RefreshResult:
    anomalies_found: int = 0
    insights_created: int = 0
    alerts_evaluated: int = 0
    recommendations_created: int = 0
    retriever_reset: bool = False
    #: Populated with the stage name when a stage failed; the ingest itself is
    #: never failed by a refresh problem, so this is how it stays visible.
    errors: list[str] | None = None

    def as_log(self) -> dict:
        return {
            "anomalies_found": self.anomalies_found,
            "insights_created": self.insights_created,
            "alerts_evaluated": self.alerts_evaluated,
            "recommendations_created": self.recommendations_created,
            "retriever_reset": self.retriever_reset,
            "errors": self.errors or [],
        }


def _lookback_days(start: date, end: date) -> int:
    """How far back the rescan has to reach to cover the loaded window."""
    earliest = min(start, end)
    span = (business_now().date() - earliest).days + 1
    return max(1, min(span, MAX_RESCAN_DAYS))


async def refresh_derived(db: AsyncSession, start: date, end: date, org_id=None) -> RefreshResult:
    """Bring anomalies, alerts, insights, recommendations and the AI index
    forward over [start, end].

    Every stage is independently guarded. A load that put rows in the warehouse
    has already succeeded; failing it here would roll back real data because a
    downstream convenience did not work.
    """
    result = RefreshResult()
    errors: list[str] = []

    lookback = _lookback_days(start, end)

    try:
        from app.services.ml.anomaly import scan_all

        result.anomalies_found = await scan_all(db, lookback_days=lookback, org_id=org_id)
        await db.commit()
    except Exception:
        logger.exception("post-load anomaly scan failed")
        errors.append("anomaly_scan")
        await _rollback(db)

    try:
        from app.services.alerts.engine import evaluate_alerts

        result.alerts_evaluated = await evaluate_alerts(db, org_id=org_id)
        await db.commit()
    except Exception:
        logger.exception("post-load alert evaluation failed")
        errors.append("alerts")
        await _rollback(db)

    try:
        from app.services.insights.engine import generate_insights

        result.insights_created = await generate_insights(db, org_id=org_id)
        await db.commit()
    except Exception:
        logger.exception("post-load insight generation failed")
        errors.append("insights")
        await _rollback(db)

    try:
        from app.services.ml.recommendations import persist_recommendations

        recs = await persist_recommendations(db, org_id=org_id)
        result.recommendations_created = recs.get("new", 0)
        await db.commit()
    except Exception:
        logger.exception("post-load recommendation generation failed")
        errors.append("recommendations")
        await _rollback(db)

    # Record watermark so the frontend can show "last refreshed" — per-org if org_id given
    try:
        if org_id is not None:
            # Try per-org watermark (requires org_id column added by 7303a1912965)
            try:
                await db.execute(
                    text(
                        """
                        INSERT INTO data_watermarks (org_id, last_refresh_at, last_source, last_trigger, affected_range_start, affected_range_end)
                        VALUES (:org_id, :now, 'etl', 'auto', :start, :end)
                        ON CONFLICT (org_id) DO UPDATE SET
                            last_refresh_at = EXCLUDED.last_refresh_at,
                            last_source = EXCLUDED.last_source,
                            last_trigger = EXCLUDED.last_trigger,
                            affected_range_start = EXCLUDED.affected_range_start,
                            affected_range_end = EXCLUDED.affected_range_end
                        """
                    ),
                    {"org_id": str(org_id), "now": business_now(), "start": start, "end": end},
                )
                await db.commit()
            except Exception as e:
                msg = str(e).lower()
                if "org_id" in msg or "does not exist" in msg or "no unique" in msg or "constraint" in msg or "column" in msg:
                    await _rollback(db)
                    # Fallback to legacy global watermark if per-org column not yet migrated
                    await db.execute(
                        text(
                            """
                            INSERT INTO data_watermarks (id, last_refresh_at, last_source, last_trigger, affected_range_start, affected_range_end)
                            VALUES (1, :now, 'etl', 'auto', :start, :end)
                            ON CONFLICT (id) DO UPDATE SET
                                last_refresh_at = EXCLUDED.last_refresh_at,
                                last_source = EXCLUDED.last_source,
                                last_trigger = EXCLUDED.last_trigger,
                                affected_range_start = EXCLUDED.affected_range_start,
                                affected_range_end = EXCLUDED.affected_range_end
                            """
                        ),
                        {"now": business_now(), "start": start, "end": end},
                    )
                    await db.commit()
                else:
                    raise
        else:
            await db.execute(
                text(
                    """
                    INSERT INTO data_watermarks (id, last_refresh_at, last_source, last_trigger, affected_range_start, affected_range_end)
                    VALUES (1, :now, 'etl', 'auto', :start, :end)
                    ON CONFLICT (id) DO UPDATE SET
                        last_refresh_at = EXCLUDED.last_refresh_at,
                        last_source = EXCLUDED.last_source,
                        last_trigger = EXCLUDED.last_trigger,
                        affected_range_start = EXCLUDED.affected_range_start,
                        affected_range_end = EXCLUDED.affected_range_end
                    """
                ),
                {"now": business_now(), "start": start, "end": end},
            )
            await db.commit()
    except Exception:
        logger.exception("could not record data watermark")
        errors.append("watermark")
        await _rollback(db)

    # Cheap and last: the assistant's retrieval index is an in-memory snapshot
    # with a five-minute TTL, so without this the chat keeps answering "have we
    # seen this before?" from the pre-upload world for another five minutes.
    try:
        from app.services.ai.retrieval import reset_retriever

        reset_retriever()
        result.retriever_reset = True
    except Exception:
        logger.exception("could not reset the AI retrieval index")
        errors.append("retriever")

    result.errors = errors or None
    logger.info(
        "post-load refresh over %s → %s (lookback %dd): %s",
        start,
        end,
        lookback,
        result.as_log(),
    )
    return result


async def _rollback(db: AsyncSession) -> None:
    try:
        await db.rollback()
    except Exception:  # pragma: no cover - session already unusable
        logger.debug("rollback during post-load refresh failed", exc_info=True)
