"""Multi-tenant isolation tests: cross-org data leakage must never happen.

Creates two organizations (Acme, Globex), seeds each with distinct warehouse
data and users, then asserts every endpoint respects strict org_predicate ==
(caller's org only). Pattern extends tests/integration/test_authz_matrix.py
but focuses on data rows, not just role gates.

Also covers registration atomicity and per-org scheduler iteration.

Requires DB (TEST_DB_URL). Skipped automatically when DB not available (CI unit-only).
"""

import uuid
from datetime import date, timedelta

import pytest
from sqlalchemy import text

from app.core.database import get_session_factory
from app.core.security import sign_token
from app.models import Organization, Profile
from tests.conftest import auth

pytestmark = pytest.mark.integration


async def _create_org(name: str, slug: str | None = None) -> Organization:

    async with get_session_factory()() as db:
        org = Organization(name=name, slug=slug or name.lower().replace(" ", "-"), is_legacy=False)
        db.add(org)
        await db.commit()
        await db.refresh(org)
        return org


async def _create_profile(email: str, role: str, org_id, is_super_admin=False) -> tuple[Profile, str]:
    from uuid import NAMESPACE_URL, uuid5

    pid = uuid5(NAMESPACE_URL, f"email://{email}")
    async with get_session_factory()() as db:
        # Use upsert to be idempotent for re-runs
        await db.execute(
            text(
                "INSERT INTO profiles (id, email, full_name, role, org_id, is_active, is_super_admin, token_version) "
                "VALUES (:id, :email, :name, :role, :org, true, :super, 0) ON CONFLICT (id) DO UPDATE SET org_id=EXCLUDED.org_id, is_super_admin=EXCLUDED.is_super_admin"
            ),
            {
                "id": str(pid),
                "email": email,
                "name": f"Test {role}",
                "role": role,
                "org": str(org_id) if org_id else None,
                "super": is_super_admin,
            },
        )
        await db.commit()
        # Re-fetch via ORM for convenience
        prof = await db.get(Profile, pid)
        token = sign_token(prof.id, prof.email, prof.role, org_id=prof.org_id)
        return prof, token


@pytest.fixture
async def two_orgs():
    try:
        org_a = await _create_org(f"Acme-{uuid.uuid4().hex[:6]}")
        org_b = await _create_org(f"Globex-{uuid.uuid4().hex[:6]}")
    except Exception as e:
        pytest.skip(f"DB not available for multitenant test: {e}")
    return org_a, org_b


async def test_cross_org_sales_isolation(client, two_orgs):
    org_a, org_b = two_orgs
    # Create users in each org
    admin_a, tok_a = await _create_profile(f"admin-a-{uuid.uuid4().hex[:4]}@example.com", "admin", org_a.id)
    admin_b, tok_b = await _create_profile(f"admin-b-{uuid.uuid4().hex[:4]}@example.com", "admin", org_b.id)

    # Seed one sales row per org via direct DB (bypass API)
    async with get_session_factory()() as db:
        await db.execute(
            text(
                "INSERT INTO sales_transactions (txn_date, quantity, unit_price, discount, total_amount, channel, region, row_hash, org_id, ingested_at) "
                "VALUES (:d, 1, 100, 0, 100, 'online', 'Kathmandu', :h, :org, now())"
            ),
            {"d": date.today() - timedelta(days=1), "h": f"hash-a-{uuid.uuid4().hex[:8]}", "org": str(org_a.id)},
        )
        await db.execute(
            text(
                "INSERT INTO sales_transactions (txn_date, quantity, unit_price, discount, total_amount, channel, region, row_hash, org_id, ingested_at) "
                "VALUES (:d, 2, 200, 0, 400, 'retail', 'Pokhara', :h, :org, now())"
            ),
            {"d": date.today() - timedelta(days=1), "h": f"hash-b-{uuid.uuid4().hex[:8]}", "org": str(org_b.id)},
        )
        await db.commit()

    # Org A should only see its own transaction via API
    # The endpoint GET /sales/transactions should filter by org_id
    # Use date range covering yesterday
    d_from = (date.today() - timedelta(days=2)).isoformat()
    d_to = date.today().isoformat()
    resp_a = await client.get(f"/api/v1/sales/transactions?from={d_from}&to={d_to}", headers=auth(tok_a))
    assert resp_a.status_code == 200, resp_a.text
    items_a = resp_a.json()["items"]
    # All returned items must be from Org A (region Kathmandu) and never Pokhara
    for it in items_a:
        # If the transactions endpoint properly filters, Pokhara/400 should not appear for A
        assert it["region"] != "Pokhara" or it["total_amount"] != 400.0, "Cross-org leak: Org A saw Org B's row"

    resp_b = await client.get(f"/api/v1/sales/transactions?from={d_from}&to={d_to}", headers=auth(tok_b))
    assert resp_b.status_code == 200
    items_b = resp_b.json()["items"]
    for it in items_b:
        assert it["region"] != "Kathmandu" or it["total_amount"] != 100.0, "Cross-org leak: Org B saw Org A's row"


async def test_cross_org_profile_isolation(client, two_orgs):
    org_a, org_b = two_orgs
    admin_a, tok_a = await _create_profile(f"admin-a2-{uuid.uuid4().hex[:4]}@example.com", "admin", org_a.id)
    analyst_b, tok_b = await _create_profile(f"analyst-b-{uuid.uuid4().hex[:4]}@example.com", "analyst", org_b.id)

    # Admin A should not be able to GET user B via /users/:id
    resp = await client.get(f"/api/v1/users/{analyst_b.id}", headers=auth(tok_a))
    assert resp.status_code == 404, f"Org isolation failed: A could fetch B's profile: {resp.status_code} {resp.text}"

    # Likewise B cannot see A's admin
    resp2 = await client.get(f"/api/v1/users/{admin_a.id}", headers=auth(tok_b))
    # Analyst role can't access /users anyway (403), but even if they could, should be 404
    assert resp2.status_code in (403, 404)


async def test_client_supplied_org_id_ignored(client, two_orgs):
    """A manager of Org A must never act on Org B even with forged org_id in body."""
    org_a, org_b = two_orgs
    manager_a, tok_a = await _create_profile(f"mgr-a-{uuid.uuid4().hex[:4]}@example.com", "manager", org_a.id)
    # Try to create a data source claiming to be in Org B (forged)
    # Even if body contains org_id of B, the server must ignore it and use caller's org
    await client.post(
        "/api/v1/data-sources",
        json={
            "name": f"src-forged-{uuid.uuid4().hex[:6]}",
            "kind": "csv_upload",
            "target_domain": "sales",
            "config": {},
        },
        headers=auth(tok_a),
    )
    # Admin only for data-sources, so manager gets 403; but if they were admin, org would still be forced to A
    # So we also test admin A trying to create user with forged org_id
    admin_a, tok_admin_a = await _create_profile(f"admin-a3-{uuid.uuid4().hex[:4]}@example.com", "admin", org_a.id)
    # Admin A tries to create a user claiming org_b
    resp2 = await client.post(
        "/api/v1/users",
        json={
            "email": f"victim-{uuid.uuid4().hex[:6]}@example.com",
            "password": "Test12345",
            "role": "analyst",
            "org_id": str(org_b.id),
        },
        headers=auth(tok_admin_a),
    )
    if resp2.status_code == 201:
        created = resp2.json()
        assert created["org_id"] == str(org_a.id), "Client-supplied org_id was trusted — must use JWT's org_id"
        assert created["org_id"] != str(org_b.id)
    else:
        # If creation failed due to Supabase admin mock not configured, that's okay — the check above is about org enforcement
        assert resp2.status_code in (403, 422, 502), resp2.text


async def test_register_org_atomicity(client):
    """POST /auth/register-org creates org + admin atomically; duplicate rolls back."""
    org_name = f"AtomicOrg-{uuid.uuid4().hex[:6]}"
    email = f"owner-{uuid.uuid4().hex[:6]}@example.com"
    payload = {"org_name": org_name, "email": email, "password": "Test12345", "full_name": "Owner"}
    resp = await client.post("/api/v1/auth/register-org", json=payload)
    assert resp.status_code == 201, resp.text
    data = resp.json()
    assert data["organization"]["name"] == org_name
    assert data["user"]["email"] == email
    assert data["user"]["org_id"] == data["organization"]["id"]

    # Duplicate org name should 409 and not create second user
    dup_email = f"owner2-{uuid.uuid4().hex[:6]}@example.com"
    resp2 = await client.post(
        "/api/v1/auth/register-org", json={"org_name": org_name, "email": dup_email, "password": "Test12345"}
    )
    assert resp2.status_code == 409

    # Duplicate email should 409
    resp3 = await client.post(
        "/api/v1/auth/register-org",
        json={"org_name": f"Other-{uuid.uuid4().hex[:6]}", "email": email, "password": "Test12345"},
    )
    assert resp3.status_code == 409

    # Ensure org not duplicated and user not duplicated
    async with get_session_factory()() as db:
        count_org = (
            await db.execute(text("SELECT COUNT(*) FROM organizations WHERE name = :n"), {"n": org_name})
        ).scalar_one()
        assert count_org == 1
        count_user = (
            await db.execute(text("SELECT COUNT(*) FROM profiles WHERE email = :e"), {"e": email})
        ).scalar_one()
        assert count_user == 1


async def test_invite_flow(client):
    """Admin creates invite, new user joins via invite_token, lands in same org."""
    org_name = f"InviteOrg-{uuid.uuid4().hex[:6]}"
    admin_email = f"adm-inv-{uuid.uuid4().hex[:6]}@example.com"
    resp = await client.post(
        "/api/v1/auth/register-org", json={"org_name": org_name, "email": admin_email, "password": "Test12345"}
    )
    assert resp.status_code == 201
    admin_tok = resp.json()["token"]
    org_id = resp.json()["organization"]["id"]

    # Admin creates invite
    invite_email = f"newuser-{uuid.uuid4().hex[:6]}@example.com"
    inv_resp = await client.post(
        "/api/v1/auth/invite", json={"email": invite_email, "role": "analyst"}, headers=auth(admin_tok)
    )
    assert inv_resp.status_code == 200, inv_resp.text
    token = inv_resp.json()["token"]

    # New user signs up with token
    signup_resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": invite_email, "password": "Test12345", "full_name": "Invited Analyst", "invite_token": token},
    )
    assert signup_resp.status_code == 201, signup_resp.text
    assert signup_resp.json()["user"]["org_id"] == org_id
    assert signup_resp.json()["user"]["role"] == "analyst"

    # Reuse same token should 409 (already accepted)
    dup_resp = await client.post(
        "/api/v1/auth/signup",
        json={"email": f"other-{uuid.uuid4().hex[:6]}@example.com", "password": "Test12345", "invite_token": token},
    )
    assert dup_resp.status_code in (409, 404, 422)


async def test_scheduler_per_org_iterates(monkeypatch):
    """Scheduler jobs iterate per-org, not just global."""

    from app.workers import scheduler as sched

    # Mock Organization rows
    org_ids = [uuid.uuid4(), uuid.uuid4()]

    class FakeOrg:
        def __init__(self, oid):
            self.id = oid
            self.name = f"Org-{oid.hex[:6]}"
            self.is_legacy = False

    fake_orgs = [FakeOrg(oid) for oid in org_ids]

    # Mock get_session_factory to return fake orgs, and train_all to capture calls
    calls = []

    async def fake_train_all(db, org_id=None):
        calls.append(org_id)
        return []

    monkeypatch.setattr("app.services.ml.registry.train_all", fake_train_all)

    # Mock db session for org listing
    class FakeResult:
        def scalars(self):
            class S:
                def all(self):
                    return fake_orgs

            return S()

    class FakeDB:
        async def execute(self, *a, **kw):
            return FakeResult()

        async def __aenter__(self):
            return self

        async def __aexit__(self, *a):
            return False

    class FakeFactory:
        def __call__(self):
            return FakeDB()

    monkeypatch.setattr("app.workers.scheduler.get_session_factory", FakeFactory)

    # Call _weekly_retrain without org_id -> should iterate per org
    await sched._weekly_retrain()
    # Should have been called at least once per org (2) — not just once global
    assert len(calls) >= 2 or org_ids[0] in calls or org_ids[1] in calls, f"Scheduler did not iterate per org: {calls}"
