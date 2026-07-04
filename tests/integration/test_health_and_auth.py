import uuid

from tests.conftest import auth, create_profile, mint_token


async def test_health(client):
    resp = await client.get("/api/v1/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}


async def test_health_db(client):
    resp = await client.get("/api/v1/health/db")
    assert resp.status_code == 200
    assert resp.json()["database"] == "reachable"


async def test_me_requires_token(client):
    resp = await client.get("/api/v1/auth/me")
    assert resp.status_code == 401


async def test_me_rejects_bad_token(client):
    resp = await client.get("/api/v1/auth/me", headers=auth("garbage"))
    assert resp.status_code == 401


async def test_me_rejects_token_without_profile(client):
    # valid signature but the user has no profiles row
    resp = await client.get("/api/v1/auth/me", headers=auth(mint_token(uuid.uuid4())))
    assert resp.status_code == 401


async def test_me_returns_profile(client, user_token):
    profile, token = user_token
    resp = await client.get("/api/v1/auth/me", headers=auth(token))
    assert resp.status_code == 200
    body = resp.json()
    assert body["id"] == str(profile.id)
    assert body["role"] == "analyst"


async def test_inactive_account_is_forbidden(client):
    profile = await create_profile("analyst", is_active=False)
    resp = await client.get("/api/v1/auth/me", headers=auth(mint_token(profile.id)))
    assert resp.status_code == 403


async def test_all_feature_routers_require_auth(client, user_token):
    _, token = user_token
    assert (await client.get("/api/v1/insights")).status_code == 401
    resp = await client.get("/api/v1/insights", headers=auth(token))
    assert resp.status_code == 200
    assert resp.json() == []
