"""Integration tests for the Power BI-grade analytics endpoints.

Require a live test DB (see conftest). When the DB is unavailable they are
auto-skipped by the ``not integration`` marker, exactly like the rest of the
suite, so the local run stays green without infrastructure.
"""

import pytest

pytestmark = pytest.mark.integration


async def test_decomposition_tree(client, user_token):
    _, token = user_token
    r = await client.get(
        "/api/v1/advanced/decomposition-tree?metric=revenue", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    body = r.json()
    assert "root" in body and "hierarchy" in body


async def test_waterfall(client, user_token):
    _, token = user_token
    r = await client.get(
        "/api/v1/advanced/waterfall?metric=revenue&dimension=category", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    assert "steps" in r.json()


async def test_heatmap(client, user_token):
    _, token = user_token
    r = await client.get("/api/v1/advanced/heatmap", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    body = r.json()
    assert "matrix" in body and "rows" in body


async def test_scatter(client, user_token):
    _, token = user_token
    r = await client.get("/api/v1/advanced/scatter?dimension=product", headers={"Authorization": f"Bearer {token}"})
    assert r.status_code == 200
    assert "points" in r.json()


async def test_funnel_radar_small_multiples(client, user_token):
    _, token = user_token
    h = {"Authorization": f"Bearer {token}"}
    assert (await client.get("/api/v1/advanced/funnel", headers=h)).status_code == 200
    assert (await client.get("/api/v1/advanced/radar?dimension=region", headers=h)).status_code == 200
    assert (await client.get("/api/v1/advanced/small-multiples", headers=h)).status_code == 200


async def test_key_influencers(client, user_token):
    _, token = user_token
    r = await client.get(
        "/api/v1/advanced/key-influencers?target=revenue", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    assert "leading_dimension" in r.json()


async def test_segmentation(client, user_token):
    _, token = user_token
    r = await client.get(
        "/api/v1/advanced/segmentation?dimension=product&n_clusters=4", headers={"Authorization": f"Bearer {token}"}
    )
    assert r.status_code == 200
    assert "entities" in r.json()


async def test_forecast_scenarios_and_comparison(client, user_token):
    _, token = user_token
    h = {"Authorization": f"Bearer {token}"}
    r = await client.get("/api/v1/advanced/forecast-scenarios?metric=revenue&horizon=21", headers=h)
    assert r.status_code == 200
    body = r.json()
    # Either a valid simulation or an explicit "insufficient history" notice.
    assert "dates" in body or "error" in body
    assert (await client.get("/api/v1/advanced/model-comparison?metric=revenue", headers=h)).status_code == 200
