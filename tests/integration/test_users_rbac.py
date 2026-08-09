import uuid

from sqlalchemy import select

from app.core.database import get_session_factory
from app.main import app
from app.models import AuditLog
from app.services.supabase_admin import get_supabase_admin
from tests.conftest import auth


class FakeSupabaseAdmin:
    def __init__(self):
        self.created: list[dict] = []
        self.role_updates: list[tuple[uuid.UUID, str]] = []

    async def create_user(self, email, password, role, full_name):
        user_id = uuid.uuid4()
        self.created.append({"id": user_id, "email": email, "role": role})
        return user_id

    async def set_role(self, user_id, role):
        self.role_updates.append((user_id, role))


def override_admin_api() -> FakeSupabaseAdmin:
    fake = FakeSupabaseAdmin()
    app.dependency_overrides[get_supabase_admin] = lambda: fake
    return fake


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
    fake = override_admin_api()
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
    assert fake.created[0]["email"] == "new.manager@example.com"


async def test_duplicate_email_conflict(client, admin_token):
    profile, token = admin_token
    override_admin_api()
    resp = await client.post(
        "/api/v1/users",
        headers=auth(token),
        json={"email": profile.email, "password": "s3cure-Pass!", "role": "analyst"},
    )
    assert resp.status_code == 409


async def test_role_change_syncs_jwt_metadata(client, admin_token, user_token):
    target, _ = user_token
    _, token = admin_token
    fake = override_admin_api()
    resp = await client.patch(
        f"/api/v1/users/{target.id}", headers=auth(token), json={"role": "manager"}
    )
    assert resp.status_code == 200
    assert resp.json()["role"] == "manager"
    assert fake.role_updates == [(target.id, "manager")]


async def test_mutating_request_writes_audit_log(client, admin_token, user_token):
    target, _ = user_token
    _, token = admin_token
    override_admin_api()
    await client.patch(
        f"/api/v1/users/{target.id}", headers=auth(token), json={"department": "finance"}
    )
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
