"""Intent classification for the BI assistant (keyword-based, fast, offline)."""

import re
from enum import StrEnum


class Intent(StrEnum):
    REVENUE = "revenue"
    EXPENSES = "expenses"
    PROFIT = "profit"
    FORECAST = "forecast"
    INVENTORY = "inventory"
    ANOMALIES = "anomalies"
    PRODUCTS = "products"
    CHANNELS = "channels"
    REGIONS = "regions"
    COMPARE = "compare"
    BUSINESS = "business"
    PLATFORM = "platform"
    USERS = "users"
    CATALOG = "catalog"
    UPDATE = "update"
    HELP = "help"
    GREETING = "greeting"
    THANKS = "thanks"
    CAPABILITIES = "capabilities"
    UNKNOWN = "unknown"


_KEYWORDS: dict[Intent, tuple[str, ...]] = {
    # Platform counting must be checked BEFORE generic business identity — otherwise
    # "how many businesses" hits the BUSINESS phrase and never reaches PLATFORM.
    Intent.PLATFORM: (
        "how many business",
        "how many businesses",
        "how many organization",
        "how many organisations",
        "number of business",
        "number of businesses",
        "number of organization",
        "total business",
        "total businesses",
        "total organization",
        "total organisations",
        "count business",
        "count organization",
        "businesses registered",
        "business registered",
        "organizations registered",
        "registered business",
        "registered organization",
        "business count",
        "organization count",
        "workspaces registered",
        "tenants registered",
        "businesses are there",
        "business are there",
    ),
    Intent.USERS: (
        "how many users",
        "number of users",
        "total users",
        "count users",
        "users registered",
        "user count",
        "how many members",
        "team size",
    ),
    Intent.CATALOG: (
        "what tables",
        "what data",
        "available data",
        "data catalog",
        "schema",
        "columns",
        "what fields",
        "what datasets",
        "data sources available",
        "database tables",
        "show tables",
        "list tables",
        "what kind of data",
        "what information do you have",
    ),
    Intent.UPDATE: (
        "whats the update",
        "what is the update",
        "what is update",
        "give update",
        "daily update",
        "weekly update",
        "monthly update",
        "status update",
        "current status",
        "overall status",
        "how are we doing",
        "how is business",
        "business update",
        "summary",
        "overview",
        "brief me",
        "briefing",
        "report update",
        "what happened",
        "what changed",
        "recent changes",
        "latest update",
        "today update",
        "whats new",
        "what is new",
        "any update",
        "give me update",
        "update please",
    ),
    Intent.BUSINESS: (
        "business name",
        "company name",
        "organization name",
        "organisation name",
        "org name",
        "workspace name",
        "what business",
        "which business",
        "my business",
        "this business",
        "business is this",
        "who am i",
        "what is my",
        "account name",
        "tenant name",
    ),
    # Specific intents first: "revenue forecast" must match FORECAST, not REVENUE.
    Intent.FORECAST: (
        "forecast",
        "predict",
        "projection",
        "future",
        "trend next",
        "outlook",
        "will revenue",
    ),
    Intent.ANOMALIES: ("anomal", "alert", "outlier", "unusual", "abnormal", "spike", "suspicious"),
    Intent.INVENTORY: ("inventory", "stock", "restock", "reorder", "warehouse", "sku", "supply"),
    Intent.EXPENSES: ("expense", "spend", "cost", "overhead", "outgoing", "billing"),
    Intent.PROFIT: ("profit", "margin", "net income", "bottom line", "pnl", "p&l", "earnings"),
    Intent.CHANNELS: ("channel", "online", "offline", "retail", "wholesale", "ecommerce"),
    Intent.REGIONS: ("region", "city", "province", "location", "district", "geography"),
    Intent.PRODUCTS: ("product", "best seller", "top seller", "item", "category sales"),
    Intent.COMPARE: ("compare", "versus", "vs", "vs.", "difference between", "which is higher"),
    Intent.REVENUE: (
        "revenue",
        "sales",
        "income",
        "earning",
        "turnover",
        "selling",
        "trend",
        "performance",
    ),
    Intent.HELP: (
        "help",
        "what can you do",
        "how do i use",
        "how does this work",
        "capabilities",
        "commands",
    ),
    Intent.GREETING: (
        "hello",
        "hi ",
        "hey",
        "namaste",
        "good morning",
        "good afternoon",
        "good evening",
    ),
    Intent.THANKS: ("thank", "thanks", "thx", "appreciate"),
}

# Catches greeting variants the keyword list misses: "hii", "how are you",
# "how r u", "whats up", "hi there!" etc. Only consulted when no data
# keyword matched, so "hi, what's our revenue?" still routes to REVENUE.
_GREETING_RE = re.compile(
    r"^(?:hi+|hii+|hello+|hey+|yo|namaste|namaskar|"
    r"good\s+(?:morning|afternoon|evening)|"
    r"how\s+(?:are|r|re)\s+(?:you|u)|how's?\s+it\s+going|"
    r"what'?s\s+up|wassup|sup)\b",
    re.IGNORECASE,
)


def detect_intent(question: str) -> Intent:
    q = question.lower()
    for intent, keywords in _KEYWORDS.items():
        if any(k in q for k in keywords):
            return intent
    if _GREETING_RE.search(q):
        return Intent.GREETING
    return Intent.UNKNOWN
