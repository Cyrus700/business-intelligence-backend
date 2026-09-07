from sqlalchemy import select

from app.core.database import get_session_factory
from app.main import app
from app.models import AuditLog
from tests.conftest import auth


def teardown_function():
    app.dependency_overrides.clear()


async def test_analyst_cannot_list_users(client, user_token):
    _, token = user_token
    resp = await client.get("/api/v1/users", headers=auth(token))
    assert resp.status_code == 403


async def test_admin_lists_users(client, admin_token):
    profile, token = admin_token
    resp = await client.get("/api/v1/users", headers=auth(token))
    assert resp.status_code == 200
    emails = [u["email"] for u in resp.json()["items"]]
    assert profile.email in emails


async def test_admin_creates_user(client, admin_token):
    _, token = admin_token
    resp = await client.post(
        "/api/v1/users",
        headers=auth(token),
        json={
            "email": "new.manager@example.com",
            "password": "s3cure-Pass!",
            "role": "manager",
            "full_name": "New Manager",
        },
    )
    assert resp.status_code == 201, resp.text
    assert resp.json()["role"] == "manager"
    assert resp.json()["email"] == "new.manager@example.com"


async def test_duplicate_email_conflict(client, admin_token):
    profile, token = admin_token
    resp = await client.post(
        "/api/v1/users",
        headers=auth(token),
        json={"email": profile.email, "password": "s3cure-Pass!", "role": "analyst"},
    )
    assert resp.status_code == 409


async def test_role_change_bumps_token_version(client, admin_token, user_token):
    target, _ = user_token
    _, token = admin_token
    # Capture token_version before
    from app.models import Profile

    async with get_session_factory()() as session:
        before = await session.get(Profile, target.id)
        before_ver = before.token_version
    resp = await client.patch(f"/api/v1/users/{target.id}", headers=auth(token), json={"role": "manager"})
    assert resp.status_code == 200
    assert resp.json()["role"] == "manager"
    async with get_session_factory()() as session:
        after = await session.get(Profile, target.id)
        assert after.token_version == before_ver + 1


async def test_cannot_deactivate_self(client, admin_token):
    profile, token = admin_token
    resp = await client.patch(f"/api/v1/users/{profile.id}", headers=auth(token), json={"is_active": False})
    assert resp.status_code == 400
    assert "cannot deactivate your own" in resp.json()["detail"].lower()


async def test_mutating_request_writes_audit_log(client, admin_token, user_token):
    target, _ = user_token
    _, token = admin_token
    await client.patch(f"/api/v1/users/{target.id}", headers=auth(token), json={"department": "finance"})
    async with get_session_factory()() as session:
        logs = (await session.execute(select(AuditLog))).scalars().all()
    assert any(f"PATCH /api/v1/users/{target.id}" == log.action for log in logs)


async def test_analyst_cannot_read_audit_logs(client, user_token):
    _, token = user_token
    resp = await client.get("/api/v1/audit-logs", headers=auth(token))
    assert resp.status_code == 403


async def test_admin_reads_audit_logs(client, admin_token):
    _, token = admin_token
    resp = await client.get("/api/v1/audit-logs", headers=auth(token))
    assert resp.status_code == 200
