"""Public landing-page content endpoints (no auth required)."""

from fastapi import APIRouter
from fastapi.responses import JSONResponse

from app.api.deps import DbSession
from app.services.analytics.landing_live import CACHE_TTL_SECONDS, build_live_metrics

router = APIRouter(prefix="/landing", tags=["landing"])


LANDING_DATA = {
    "hero": {
        "eyebrow": "AI-driven cloud business intelligence",
        "title": ["One dashboard for", "every business decision."],
        "subtitle": "Insightful brings your spreadsheets, databases and APIs into a single cloud dashboard, then uses AI to forecast trends, flag anomalies and explain what to do next.",
        "primaryCta": "Get started",
        "secondaryCta": "See how it works",
    },
    "stats": [
        {"value": 63, "suffix": "", "label": "professionals surveyed"},
        {"value": 90.5, "suffix": "%", "label": "want an AI BI dashboard"},
        {"value": 50, "prefix": "−", "suffix": "%", "label": "time spent on reports"},
        {"value": 100, "suffix": "%", "label": "asked for automated reports"},
    ],
    "features": [
        {
            "icon": "chart",
            "title": "Real-time dashboards",
            "body": "Live KPIs that refresh as your business moves — no more waiting days for a static report.",
        },
        {
            "icon": "trend",
            "title": "Predictive analytics",
            "body": "Forecast demand, revenue and performance with ML models built on your own history.",
        },
        {
            "icon": "alert",
            "title": "Anomaly detection",
            "body": "Isolation-Forest powered alerts surface unusual activity before it becomes a problem.",
        },
        {
            "icon": "spark",
            "title": "Automated insights",
            "body": "Plain-language summaries and recommendations turn raw model output into clear next steps.",
        },
        {
            "icon": "pipe",
            "title": "Multi-source ETL",
            "body": "Connect CSV, Excel, PostgreSQL and REST APIs into one trusted single source of truth.",
        },
        {
            "icon": "lock",
            "title": "RBAC & security",
            "body": "Role-based views, JWT auth and row-level security keep sensitive data with the right people.",
        },
    ],
    "steps": [
        {
            "no": "01",
            "title": "Connect your data",
            "body": "Point Insightful at your spreadsheets, databases and APIs. Our ETL pipelines clean, standardise and centralise everything automatically.",
        },
        {
            "no": "02",
            "title": "AI analyses it",
            "body": "Predictive models, trend detection and anomaly engines run continuously over your unified warehouse — no data scientist required.",
        },
        {
            "no": "03",
            "title": "You decide faster",
            "body": "Insights arrive as visual KPIs, forecasts and written recommendations, delivered to role-based dashboards on any device.",
        },
    ],
    "pricing": [
        {
            "name": "Starter",
            "price": "Free",
            "period": "",
            "blurb": "For trying it out and small teams getting started.",
            "features": ["1 data source", "Core dashboards", "7-day history", "Community support"],
            "cta": "Start free",
            "featured": False,
        },
        {
            "name": "Growth",
            "price": "$29",
            "period": "/mo",
            "blurb": "For SMEs that run on data, day to day.",
            "features": [
                "Unlimited data sources",
                "Predictive analytics + anomalies",
                "Automated insights & alerts",
                "Full RBAC",
                "Email support",
            ],
            "cta": "Start 14-day trial",
            "featured": True,
        },
        {
            "name": "Enterprise",
            "price": "Custom",
            "period": "",
            "blurb": "For organisations with scale and compliance needs.",
            "features": ["Dedicated cloud", "SSO & audit logs", "Custom models", "SLA & onboarding"],
            "cta": "Talk to us",
            "featured": False,
        },
    ],
    "faqs": [
        {
            "q": "How is this affordable for a small business?",
            "a": "Insightful is cloud-native and pay-as-you-go — there's no upfront hardware or licensing. You only pay for what you use, and there's a free tier to start.",
        },
        {
            "q": "Is my data secure?",
            "a": "Yes. We use TLS in transit, AES-256 at rest, JWT authentication and database row-level security so only authorised roles see sensitive data.",
        },
        {
            "q": "Does it work on low bandwidth?",
            "a": "Dashboards are optimised to stay fast and legible on modest connections, and the interface is fully responsive for mobile and on-the-go access.",
        },
        {
            "q": "What data sources can I connect?",
            "a": "CSV and Excel files, relational databases like PostgreSQL, and external REST APIs — all unified through our ETL pipelines.",
        },
        {
            "q": "Do I need a data scientist?",
            "a": "No. The AI runs automatically and explains its findings in plain language, designed for non-technical business users.",
        },
    ],
}


# Marketing copy is static, so let the CDN/browser hold it for an hour.
_CONTENT_CACHE = "public, max-age=3600, stale-while-revalidate=86400"
_LIVE_CACHE = f"public, max-age={CACHE_TTL_SECONDS}, stale-while-revalidate={CACHE_TTL_SECONDS * 5}"


@router.get("")
async def get_landing_data():
    return JSONResponse(LANDING_DATA, headers={"Cache-Control": _CONTENT_CACHE})


@router.get("/live")
async def get_landing_live(db: DbSession):
    """Platform figures behind the landing page.

    Public and unauthenticated, so it returns plumbing counters only — see the
    module docstring in services/analytics/landing_live.py before adding fields.
    """
    return JSONResponse(await build_live_metrics(db), headers={"Cache-Control": _LIVE_CACHE})
