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
        # list / show variants — must route to PLATFORM with detail=true
        "list business",
        "list businesses",
        "list the businesses",
        "list the business",
        "list organization",
        "list organizations",
        "list all businesses",
        "list all business",
        "show business",
        "show businesses",
        "show all businesses",
        "show organizations",
        "display businesses",
        "get businesses",
        "fetch businesses",
        # status-specific (must also hit PLATFORM so filter can be extracted later)
        "how many approved",
        "how many pending",
        "how many rejected",
        "how many legacy",
        "how many personal",
        "number of approved",
        "number of pending",
        "number of rejected",
        "count approved",
        "count pending",
        "count rejected",
        "total approved",
        "total pending",
        "total rejected",
        "approved businesses",
        "pending businesses",
        "rejected businesses",
        "approved business",
        "pending business",
        "rejected business",
        "approved organization",
        "pending organization",
        "rejected organization",
        "list approved",
        "list pending",
        "list rejected",
        "show approved",
        "show pending",
        "show rejected",
        # common typos — keep detection precise even with misspelling
        "aprrved",
        "apprved",
        "approveed",
        "rejeect",
        "rejeected",
        "rejefct",
        "rejefect",
        "pendng",
        "pendding",
        "bussiness",
        "bussines",
        "busines",
        "orgnization",
        "regisgtered",
        "regisgterd",
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


def _is_platform_typo(q: str) -> bool:
    """Catch PLATFORM with typos / varied phrasing that exact keywords miss.

    Handles: how many / list / show / total / count + business/organization
    even when misspelled (aprrved, rejeect, bussiness, regisgtered).
    """
    # any business-like root + action
    has_business = bool(re.search(r"business|bussin|organization|orgniza|workspace|tenant", q))
    has_action = any(
        p in q for p in ("how many", "how much", "count", "total", "number of", "are there", "registered", "regisg")
    )
    has_list = any(p in q for p in ("list", "show", "display", "get ", "fetch"))
    if (has_action or has_list) and has_business:
        return True
    if has_business and re.search(r"approv|apprv|aprrv|pend|rej|bussin|orgniza|regisg", q):
        return True
    if any(p in q for p in ("how many", "count", "total", "number of")) and re.search(
        r"approv|aprrv|pend|rejec|rejfe|bussin|orgniza|regisg", q
    ):
        return True
    # bare status words with typo roots also count as platform
    if re.search(r"\b(approv\w*|aprrv\w*|pend\w*|rej\w*|bussin\w*|regisg\w*)\b", q):
        # limit to platform context: must have platform noun nearby or action word
        if has_business or has_action or has_list:
            return True
    # fallback: any platform root + status root
    if re.search(r"business|bussin|organization|orgniza|workspace|tenant", q) and re.search(
        r"approv|aprrv|pend|rej", q
    ):
        return True
    return False


def detect_intent(question: str) -> Intent:
    q = question.lower()
    for intent, keywords in _KEYWORDS.items():
        if any(k in q for k in keywords):
            return intent
    if _is_platform_typo(q):
        return Intent.PLATFORM
    if _GREETING_RE.search(q):
        return Intent.GREETING
    return Intent.UNKNOWN
