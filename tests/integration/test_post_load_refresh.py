"""An upload must move the whole dashboard forward, not just the KPI cards.

Loading rows rebuilt the KPI snapshots and stopped. Anomalies, insights and the
assistant's retrieval index waited for the nightly cron, so for the rest of the
day the dashboard showed fresh KPIs beside stale everything-else and the chat
answered from the pre-upload world. To a user that reads as "my upload didn't
work".
"""

import io

from sqlalchemy import select

from app.core.database import get_session_factory
from app.models import EtlJob
from app.services.ai import retrieval
from tests.conftest import auth
from tests.integration.test_etl_flow import CSV


def _upload(token: str, content: str = CSV, name: str = "sales.csv", domain: str = "sales"):
    return {
        "url": "/api/v1/uploads",
        "headers": auth(token),
        "files": {"file": (name, io.BytesIO(content.encode()), "text/csv")},
        "data": {"domain": domain},
    }


async def _latest_job() -> EtlJob:
    async with get_session_factory()() as db:
        return (
            (await db.execute(select(EtlJob).order_by(EtlJob.started_at.desc()).limit(1)))
            .scalars()
            .first()
        )


async def test_upload_refreshes_the_derived_layer(client, manager_token):
    _, token = manager_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code in (200, 201), resp.text

    job = await _latest_job()
    refresh = (job.log or {}).get("post_load_refresh")
    assert refresh is not None, "the ingest did not run a post-load refresh"
    assert refresh["retriever_reset"] is True
    assert refresh["errors"] == []


async def test_upload_invalidates_the_assistant_index(client, manager_token):
    """The chat's index is an in-memory snapshot; new rows must not wait on its TTL."""
    _, token = manager_token
    retrieval.get_retriever()  # force an instance into existence
    assert retrieval._retriever is not None

    kwargs = _upload(token)
    await client.post(kwargs.pop("url"), **kwargs)

    assert retrieval._retriever is None, "stale AI index survived the upload"


async def test_a_failed_refresh_never_fails_the_ingest(client, manager_token, monkeypatch):
    """The rows are already committed; a downstream hiccup must not undo them."""
    async def exploding_scan(*args, **kwargs):
        raise RuntimeError("anomaly scanner is down")

    import app.services.ml.anomaly as anomaly_mod

    monkeypatch.setattr(anomaly_mod, "scan_all", exploding_scan)

    _, token = manager_token
    kwargs = _upload(token)
    resp = await client.post(kwargs.pop("url"), **kwargs)
    assert resp.status_code == 201, resp.text
    body = resp.json()
    assert body["status"] == "loaded"
    assert body["error_report"]["loaded"] == 3

    job = await _latest_job()
    # The failure is recorded rather than swallowed, so a silently degraded
    # refresh is still visible in the job log.
    assert "anomaly_scan" in (job.log or {}).get("post_load_refresh", {}).get("errors", [])
