"""Phase 13 observability: request correlation id on responses, logs and audit."""

from app.core.request_context import current_request_id, set_request_id


async def test_request_id_echoed_in_response(client, user_token):
    _, token = user_token
    resp = await client.get(
        "/api/v1/health",
        headers={"Authorization": f"Bearer {token}", "X-Request-ID": "trace-me-123"},
    )
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") == "trace-me-123"


async def test_request_id_generated_when_absent(client, user_token):
    _, token = user_token
    resp = await client.get("/api/v1/health", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200
    assert resp.headers.get("X-Request-ID") is not None


async def test_mutating_request_audited_with_request_id(client, manager_token, admin_token):
    """The audit row for a mutation carries the caller's request id."""
    _, token = manager_token
    resp = await client.get("/api/v1/anomalies", headers={"Authorization": f"Bearer {token}"})
    assert resp.status_code == 200, resp.text
    anomaly_id = resp.json()[0]["id"] if resp.json() else None

    if anomaly_id is None:
        from app.core.database import get_session_factory
        from app.models import Anomaly

        async with get_session_factory()() as db:
            anomaly = Anomaly(metric="revenue", observed_value=1, expected_value=1)
            db.add(anomaly)
            await db.commit()
            anomaly_id = anomaly.id

    resp = await client.patch(
        f"/api/v1/anomalies/{anomaly_id}",
        json={"status": "acknowledged"},
        headers={"Authorization": f"Bearer {token}", "X-Request-ID": "audit-trace-77"},
    )
    assert resp.status_code == 200, resp.text

    _, admin = admin_token
    audit_resp = await client.get(
        "/api/v1/audit-logs?limit=50", headers={"Authorization": f"Bearer {admin}"}
    )
    assert audit_resp.status_code == 200, audit_resp.text
    rows = audit_resp.json()
    assert isinstance(rows, list) and rows
    match = next(
        (r for r in rows if r.get("detail", {}).get("request_id") == "audit-trace-77"),
        None,
    )
    assert match is not None, "audit row should carry the request id"
    assert match["action"].startswith("PATCH /api/v1/anomalies")


def test_contextvar_scope():
    assert current_request_id() is None
    set_request_id("ctx-1")
    assert current_request_id() == "ctx-1"
    set_request_id("ctx-2")
    assert current_request_id() == "ctx-2"
