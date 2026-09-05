"""Canonical default RBAC catalog.

This is the *seed* — not the runtime source of truth. Once the ``roles`` /
``permissions`` / ``role_permissions`` tables are populated an admin owns the
policy and can diverge from these defaults freely. They are kept here so that

* the Alembic migration can seed a fresh deployment,
* :mod:`app.services.rbac` can bootstrap a database whose RBAC tables are empty
  (fresh dev DB, truncated test DB),
* "Reset to defaults" in the admin UI has something to reset *to*.
"""

from typing import TypedDict


class RoleSeed(TypedDict):
    name: str
    label: str
    description: str
    rank: int
    color: str


class PermissionSeed(TypedDict):
    key: str
    label: str
    description: str
    group_label: str
    sort_order: int


DEFAULT_ROLES: list[RoleSeed] = [
    {
        "name": "analyst",
        "label": "Analyst",
        "description": ("View data, dashboards, forecasts and insights. Read-only access to most features."),
        "rank": 1,
        "color": "green",
    },
    {
        "name": "manager",
        "label": "Manager",
        "description": (
            "Everything an analyst can do, plus upload data, run ETL, manage "
            "anomalies, alert rules, reports and view P&L."
        ),
        "rank": 2,
        "color": "blue",
    },
    {
        "name": "admin",
        "label": "Admin",
        "description": ("Full platform control — manage users, roles, data sources, audit logs and ML models."),
        "rank": 3,
        "color": "purple",
    },
]

# Roles that must always exist: the app's auth layer and the seeded fixtures
# assume them, and deleting the top role would lock every admin out.
SYSTEM_ROLES = frozenset({"analyst", "manager", "admin"})

# Permissions that may never be revoked from the highest-ranked role, otherwise
# an admin could lock themselves out of the very screen that grants access back.
LOCKED_ADMIN_PERMISSIONS = frozenset({"users:manage", "roles:manage"})


def _perms(group: str, rows: list[tuple[str, str, str]]) -> list[PermissionSeed]:
    return [
        {
            "key": key,
            "label": label,
            "description": description,
            "group_label": group,
            "sort_order": i,
        }
        for i, (key, label, description) in enumerate(rows)
    ]


DEFAULT_PERMISSIONS: list[PermissionSeed] = [
    *_perms(
        "Dashboards & KPIs",
        [
            ("dashboard:view", "View dashboard", "Access the main overview dashboard"),
            ("kpis:view", "View KPIs", "See KPI cards and summary metrics"),
            ("timeseries:view", "View timeseries", "View trend charts over time"),
            ("compare:view", "Compare periods", "Compare 2+ months or years side-by-side with insights and AI suggestions"),
        ],
    ),
    *_perms(
        "Sales & Finance",
        [
            (
                "sales:view",
                "View sales data",
                "See sales transactions, product and channel breakdowns",
            ),
            ("expenses:view", "View expenses", "See expense breakdowns by category"),
            ("pnl:view", "View P&L", "See profit & loss statements"),
            ("inventory:view", "View inventory", "See inventory levels and low-stock alerts"),
        ],
    ),
    *_perms(
        "ML & Analytics",
        [
            ("forecasts:view", "View forecasts", "See revenue and demand forecasts"),
            ("forecasts:retrain", "Retrain models", "Trigger ML model retraining"),
            ("anomalies:view", "View anomalies", "See detected anomaly alerts"),
            ("anomalies:update", "Dismiss anomalies", "Acknowledge or dismiss anomaly alerts"),
            ("trends:view", "View trends", "See trend analysis"),
        ],
    ),
    *_perms(
        "Insights & Reports",
        [
            ("insights:view", "View insights", "See AI-generated business insights"),
            ("insights:pin", "Pin insights", "Mark insights as important"),
            ("insights:generate", "Generate insights", "Trigger AI insight generation"),
            ("reports:view", "View reports", "See generated reports"),
            ("reports:download", "Download reports", "Download reports as files"),
            ("reports:generate", "Generate reports", "Create and schedule reports"),
        ],
    ),
    *_perms(
        "Operations",
        [
            (
                "alert-rules:manage",
                "Manage alert rules",
                "Create, edit and delete alert rules",
            ),
            ("notifications:view", "View notifications", "See system notifications"),
            ("notifications:read", "Read notifications", "Mark notifications as read"),
        ],
    ),
    *_perms(
        "Data Integration",
        [
            ("uploads:create", "Upload data", "Upload CSV and Excel data files"),
            ("etl:manage", "Manage ETL", "Run and monitor ETL pipelines"),
            ("data-sources:manage", "Manage data sources", "Add and configure data sources"),
            ("quality:view", "View data quality", "See data quality score, history and issues"),
            (
                "quality:run",
                "Run quality audits",
                "Trigger a manual data-quality audit run",
            ),
            (
                "quality:resolve",
                "Resolve quality issues",
                "Acknowledge or resolve data-quality issues",
            ),
        ],
    ),
    *_perms(
        "Administration",
        [
            ("users:manage", "Manage users", "Create, edit and deactivate users"),
            (
                "roles:manage",
                "Manage roles & permissions",
                "Edit the role/permission matrix and define custom roles",
            ),
            ("audit-logs:view", "View audit logs", "See system audit trail"),
        ],
    ),
]

DEFAULT_GRANTS: dict[str, list[str]] = {
    "analyst": [
        "dashboard:view",
        "kpis:view",
        "timeseries:view",
        "compare:view",
        "sales:view",
        "expenses:view",
        "inventory:view",
        "forecasts:view",
        "anomalies:view",
        "trends:view",
        "insights:view",
        "notifications:view",
        "notifications:read",
        "reports:view",
        "reports:download",
        "quality:view",
        "quality:resolve",
    ],
    "manager": [
        "dashboard:view",
        "kpis:view",
        "timeseries:view",
        "compare:view",
        "sales:view",
        "expenses:view",
        "pnl:view",
        "inventory:view",
        "forecasts:view",
        "anomalies:view",
        "anomalies:update",
        "trends:view",
        "insights:view",
        "insights:pin",
        "alert-rules:manage",
        "notifications:view",
        "notifications:read",
        "reports:view",
        "reports:download",
        "reports:generate",
        "uploads:create",
        "etl:manage",
        "quality:view",
        "quality:run",
        "quality:resolve",
    ],
    "admin": [p["key"] for p in DEFAULT_PERMISSIONS],
}
