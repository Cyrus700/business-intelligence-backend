"""Authorization matrix: every v1 endpoint × {anon, analyst, manager, admin}.

MIN_ROLE below is the security contract for the whole API. The test asserts:
- anon → 401 (except public endpoints)
- roles below the minimum → 403
- roles at/above the minimum → anything but 401/403 (404/422 prove the caller
  passed authorization and failed later, which is the point of this test)

It also fails if an endpoint exists that isn't listed here, so any new route
must declare its access level. On success the matrix is exported to
docs/completions/assets/phase-6/authz-matrix.md.
"""

import os
import uuid
from pathlib import Path

import pytest

from app.main import app
from tests.conftest import auth, create_profile, mint_token

ROLE_RANK = {"analyst": 1, "manager": 2, "admin": 3}

# (method, path) -> minimum role; None = public
MIN_ROLE: dict[tuple[str, str], str | None] = {
    ("GET", "/api/v1/health"): None,
    ("GET", "/api/v1/health/db"): None,
    ("GET", "/api/v1/auth/me"): "analyst",
    # user management: admin only
    ("GET", "/api/v1/users"): "admin",
    ("POST", "/api/v1/users"): "admin",
    ("PATCH", "/api/v1/users/{user_id}"): "admin",
    ("GET", "/api/v1/audit-logs"): "admin",
    # data integration: admin only
    ("GET", "/api/v1/data-sources"): "admin",
    ("POST", "/api/v1/data-sources"): "admin",
    ("PATCH", "/api/v1/data-sources/{source_id}"): "admin",
    ("GET", "/api/v1/etl/jobs"): "admin",
    ("GET", "/api/v1/etl/jobs/{job_id}"): "admin",
    ("POST", "/api/v1/etl/run/{source_id}"): "admin",
    ("POST", "/api/v1/uploads"): "analyst",
    ("GET", "/api/v1/uploads/{upload_id}"): "analyst",
    # analytics: any role; P&L is manager+
    ("GET", "/api/v1/kpis/summary"): "analyst",
    ("GET", "/api/v1/kpis/timeseries"): "analyst",
    ("GET", "/api/v1/sales/transactions"): "analyst",
    ("GET", "/api/v1/sales/by-product"): "analyst",
    ("GET", "/api/v1/sales/by-category"): "analyst",
    ("GET", "/api/v1/sales/by-channel"): "analyst",
    ("GET", "/api/v1/sales/by-region"): "analyst",
    ("GET", "/api/v1/finance/expenses-by-category"): "analyst",
    ("GET", "/api/v1/finance/pnl"): "manager",
    ("GET", "/api/v1/inventory/levels"): "analyst",
    # ML: reads for all; retrain admin; anomaly triage manager+
    ("GET", "/api/v1/forecasts"): "analyst",
    ("GET", "/api/v1/forecasts/accuracy"): "analyst",
    ("POST", "/api/v1/forecasts/retrain"): "admin",
    ("GET", "/api/v1/anomalies"): "analyst",
    ("PATCH", "/api/v1/anomalies/{anomaly_id}"): "manager",
    ("GET", "/api/v1/trends"): "analyst",
    # decision support
    ("GET", "/api/v1/insights"): "analyst",
    ("POST", "/api/v1/insights/generate"): "admin",
    ("PATCH", "/api/v1/insights/{insight_id}/pin"): "manager",
    ("GET", "/api/v1/alert-rules"): "manager",
    ("POST", "/api/v1/alert-rules"): "manager",
    ("PATCH", "/api/v1/alert-rules/{rule_id}"): "manager",
    ("DELETE", "/api/v1/alert-rules/{rule_id}"): "manager",
    ("POST", "/api/v1/alert-rules/evaluate"): "manager",
    ("GET", "/api/v1/notifications"): "analyst",
    ("PATCH", "/api/v1/notifications/{notification_id}/read"): "analyst",
    ("GET", "/api/v1/reports"): "analyst",
    ("POST", "/api/v1/reports/generate"): "manager",
    ("GET", "/api/v1/reports/{report_id}/download"): "analyst",
}

MATRIX_PATH = Path(__file__).parents[2].parent / ("docs/completions/assets/phase-6/authz-matrix.md")


def fill_params(path: str) -> str:
    out = path
    while "{" in out:
        start, end = out.index("{"), out.index("}")
        out = out[:start] + str(uuid.uuid4()) + out[end + 1 :]
    return out


def all_spec_endpoints() -> set[tuple[str, str]]:
    spec = app.openapi()
    return {
        (method.upper(), path)
        for path, ops in spec["paths"].items()
        for method in ops
        if method.upper() in ("GET", "POST", "PATCH", "DELETE", "PUT")
    }


def test_matrix_covers_every_endpoint():
    missing = all_spec_endpoints() - set(MIN_ROLE)
    extra = set(MIN_ROLE) - all_spec_endpoints()
    assert not missing, f"endpoints missing from MIN_ROLE: {sorted(missing)}"
    assert not extra, f"MIN_ROLE lists endpoints that do not exist: {sorted(extra)}"


@pytest.mark.parametrize("caller", ["anon", "analyst", "manager", "admin"])
async def test_authorization_matrix(client, caller):
    headers = {}
    if caller != "anon":
        profile = await create_profile(caller)
        headers = auth(mint_token(profile.id, caller))

    results: list[tuple[str, str, int, str]] = []
    for (method, path), min_role in sorted(MIN_ROLE.items(), key=lambda kv: kv[0][1]):
        resp = await client.request(method, fill_params(path), headers=headers)
        if min_role is None:
            expected = "allow"
        elif caller == "anon":
            expected = "401"
        elif ROLE_RANK[caller] < ROLE_RANK[min_role]:
            expected = "403"
        else:
            expected = "allow"

        if expected == "401":
            assert resp.status_code == 401, f"{caller} {method} {path}: {resp.status_code}"
        elif expected == "403":
            assert resp.status_code == 403, f"{caller} {method} {path}: {resp.status_code}"
        else:
            assert resp.status_code not in (401, 403), (
                f"{caller} {method} {path}: {resp.status_code}"
            )
        results.append((method, path, resp.status_code, min_role or "public"))

    _RESULTS[caller] = {(m, p): code for m, p, code, _ in results}
    if len(_RESULTS) == 4 and os.environ.get("EXPORT_AUTHZ_MATRIX"):
        _export()


_RESULTS: dict[str, dict[tuple[str, str], int]] = {}


def _export() -> None:
    MATRIX_PATH.parent.mkdir(parents=True, exist_ok=True)
    callers = ["anon", "analyst", "manager", "admin"]
    with MATRIX_PATH.open("w") as f:
        f.write(
            "# Authorization matrix\n\n"
            "Generated by `tests/integration/test_authz_matrix.py` "
            "(run with `EXPORT_AUTHZ_MATRIX=1`). Cells are observed HTTP status codes: "
            "401 = unauthenticated, 403 = role denied; any other code means "
            "authorization passed (404/422 = failed later on data, which is expected "
            "with placeholder IDs).\n\n"
            "| Method | Endpoint | Min role | anon | analyst | manager | admin |\n"
            "|---|---|---|---|---|---|---|\n"
        )
        for (method, path), min_role in sorted(MIN_ROLE.items(), key=lambda kv: kv[0][1]):
            cells = " | ".join(str(_RESULTS[c][(method, path)]) for c in callers)
            f.write(f"| {method} | `{path}` | {min_role or 'public'} | {cells} |\n")
