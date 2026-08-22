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
    ("GET", "/api/v1/health/business"): "analyst",
    ("GET", "/api/v1/health/system"): "admin",
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
    # ETL execution is manager+ (etl:manage in the RBAC defaults); defining the
    # sources themselves stays admin-only (data-sources:manage).
    ("GET", "/api/v1/etl/jobs"): "manager",
    ("GET", "/api/v1/etl/jobs/{job_id}"): "manager",
    ("POST", "/api/v1/etl/run/{source_id}"): "manager",
    ("POST", "/api/v1/uploads"): "manager",
    ("GET", "/api/v1/uploads"): "manager",
    ("GET", "/api/v1/uploads/{upload_id}"): "manager",
    # analytics: any role; P&L is manager+
    ("GET", "/api/v1/kpis/summary"): "analyst",
    ("GET", "/api/v1/kpis/timeseries"): "analyst",
    ("GET", "/api/v1/kpis/definitions"): "analyst",
    ("PATCH", "/api/v1/kpis/definitions/{metric}"): "admin",
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
    ("GET", "/api/v1/models"): "analyst",
    ("POST", "/api/v1/models/{model_id}/retire"): "admin",
    ("GET", "/api/v1/backtest"): "analyst",
    ("GET", "/api/v1/anomalies"): "analyst",
    ("PATCH", "/api/v1/anomalies/{anomaly_id}"): "manager",
    ("GET", "/api/v1/trends"): "analyst",
    # diagnostic analytics — read for all authenticated roles
    ("GET", "/api/v1/diagnostics/change"): "analyst",
    ("GET", "/api/v1/cache/stats"): "analyst",
    ("POST", "/api/v1/cache/clear"): "admin",
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
    # report schedules: self-service, manager+
    ("GET", "/api/v1/report-schedules"): "manager",
    ("POST", "/api/v1/report-schedules"): "manager",
    ("PATCH", "/api/v1/report-schedules/{schedule_id}"): "manager",
    ("DELETE", "/api/v1/report-schedules/{schedule_id}"): "manager",
    # data quality: read for all authenticated, audit run manager+,
    # issue triage available to analysts (quality:resolve default)
    ("GET", "/api/v1/data-quality/overview"): "analyst",
    ("GET", "/api/v1/data-quality/issues"): "analyst",
    ("POST", "/api/v1/data-quality/run"): "manager",
    ("PATCH", "/api/v1/data-quality/issues/{issue_id}"): "analyst",
    ("GET", "/api/v1/data-quality/quality/history"): "analyst",
    # public marketing + auth surfaces
    ("GET", "/api/v1/landing"): None,
    ("GET", "/api/v1/landing/live"): None,
    ("POST", "/api/v1/auth/login"): None,
    ("POST", "/api/v1/auth/signup"): None,
    ("POST", "/api/v1/auth/forgot-password"): None,
    ("POST", "/api/v1/auth/reset-password"): None,
    ("GET", "/api/v1/auth/google/login"): None,
    ("GET", "/api/v1/auth/google/callback"): None,
    # self-service profile
    ("PATCH", "/api/v1/auth/me"): "analyst",
    ("GET", "/api/v1/auth/me/preferences"): "analyst",
    ("PATCH", "/api/v1/auth/me/preferences"): "analyst",
    # assistant: any authenticated role
    ("POST", "/api/v1/ai/chat"): "analyst",
    ("POST", "/api/v1/ai/chat/stream"): "analyst",
    ("POST", "/api/v1/ai/analyze"): "analyst",
    ("GET", "/api/v1/ai/conversations"): "analyst",
    ("GET", "/api/v1/ai/conversations/{conv_id}/messages"): "analyst",
    ("GET", "/api/v1/ai/ai/insights"): "analyst",
    ("GET", "/api/v1/ai/providers/status"): "analyst",
    ("GET", "/api/v1/ai/usage"): "admin",
    ("GET", "/api/v1/ai/briefing"): "manager",
    ("GET", "/api/v1/auth/me/permissions"): "analyst",
    # recommendations
    ("GET", "/api/v1/recommendations"): "analyst",
    ("POST", "/api/v1/recommendations/generate"): "manager",
    ("GET", "/api/v1/recommendations/history"): "analyst",
    ("POST", "/api/v1/recommendations/{insight_id}/decide"): "manager",
    # data freshness — read-only, any authenticated role
    ("GET", "/api/v1/data-coverage"): "analyst",
    ("GET", "/api/v1/watermark"): "analyst",
    ("GET", "/api/v1/data-quality/quality/history"): "analyst",
    # admin endpoints
    ("GET", "/api/v1/admin/scheduler/status"): "admin",
    ("POST", "/api/v1/admin/scheduler/trigger/{job_id}"): "admin",
    ("POST", "/api/v1/admin/scheduler/pause/{job_id}"): "admin",
    ("POST", "/api/v1/admin/scheduler/resume/{job_id}"): "admin",
    ("GET", "/api/v1/admin/storage"): "admin",
    ("GET", "/api/v1/admin/security"): "admin",
    ("GET", "/api/v1/admin/stats"): "admin",
    # admin-only reads
    ("GET", "/api/v1/users/{user_id}"): "admin",
    ("GET", "/api/v1/audit-logs/role-changes"): "admin",
    # RBAC catalog: everyone reads the matrix (the UI renders it),
    # only roles:manage (admin by default) edits it
    ("GET", "/api/v1/rbac/matrix"): "analyst",
    ("GET", "/api/v1/rbac/me"): "analyst",
    ("GET", "/api/v1/rbac/rbac/me"): "analyst",
    ("GET", "/api/v1/rbac/roles"): "analyst",
    ("GET", "/api/v1/rbac/permissions"): "analyst",
    ("GET", "/api/v1/rbac/audit"): "admin",
    ("PATCH", "/api/v1/rbac/matrix"): "admin",
    ("PUT", "/api/v1/rbac/roles/{name}/permissions"): "admin",
    ("POST", "/api/v1/rbac/roles"): "admin",
    ("PATCH", "/api/v1/rbac/roles/{name}"): "admin",
    ("DELETE", "/api/v1/rbac/roles/{name}"): "admin",
    ("POST", "/api/v1/rbac/permissions"): "admin",
    ("PATCH", "/api/v1/rbac/permissions/{key}"): "admin",
    ("DELETE", "/api/v1/rbac/permissions/{key}"): "admin",
    ("POST", "/api/v1/rbac/reset"): "admin",
    ("POST", "/api/v1/rbac/sync-catalog"): "admin",
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
