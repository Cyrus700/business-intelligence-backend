"""Per-provider circuit breaker + latency/cost tracking for LLM failover.

The existing Groq→Gemini failover retries _every_ request. This module adds a
breaker on top: after FAILURE_THRESHOLD consecutive failures a provider is
skipped for a cooldown window (no retry-into-failure), while per-provider
latency and token-cost estimates are recorded so ops can see which provider is
healthy and at what cost (surfaced by GET /api/v1/ai/providers/status).
"""

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime, timedelta

logger = logging.getLogger(__name__)

FAILURE_THRESHOLD = 3  # consecutive failures before the breaker opens
COOLDOWN_SECONDS = 300  # how long a tripped provider is skipped (5 min)

# Approximate list prices per 1M tokens (inputs, outputs) — used only for
# cost *estimates* in the ops view; keep in sync with the configured models.
PRICING_PER_1M = {
    "llama-3.3-70b": (0.66, 0.79),
    "llama-3.1-8b": (0.05, 0.08),
    "gemini-2.0-flash": (0.10, 0.40),
    "gemini-2.5-flash": (0.30, 2.50),
}
DEFAULT_PRICE = (0.30, 0.60)  # unknown models


def _estimate_tokens(text: str) -> int:
    """Rough token estimate (chars/4); close enough for cost bookkeeping."""
    return max(1, len(text) // 4)


def estimate_cost_usd(model: str, input_text: str, output_text: str) -> float:
    in_price, out_price = next(
        (p for prefix, p in PRICING_PER_1M.items() if model.startswith(prefix)), DEFAULT_PRICE
    )
    tokens_in = _estimate_tokens(input_text)
    tokens_out = _estimate_tokens(output_text)
    return (tokens_in * in_price + tokens_out * out_price) / 1e6


@dataclass
class CircuitState:
    name: str
    model: str
    failures: int = 0
    consecutive_failures: int = 0
    circuit_open_until: datetime | None = None
    calls: int = 0
    total_latency_ms: int = 0
    total_cost_usd: float = 0.0
    latency_buckets: list[float] = field(default_factory=list)

    @property
    def avg_latency_ms(self) -> float:
        return round(self.total_latency_ms / self.calls, 1) if self.calls else 0.0

    @property
    def is_open(self) -> bool:
        if self.circuit_open_until is None:
            return False
        return datetime.now(UTC) < self.circuit_open_until

    def record_success(self, latency_ms: int | None, cost_usd: float) -> None:
        self.calls += 1
        self.consecutive_failures = 0
        self.circuit_open_until = None
        if latency_ms is not None:
            self.total_latency_ms += latency_ms
            self.latency_buckets.append(latency_ms)
            if len(self.latency_buckets) > 500:
                self.latency_buckets = self.latency_buckets[-500:]

    def record_failure(self) -> None:
        self.calls += 1
        self.consecutive_failures += 1
        self.failures += 1
        if self.consecutive_failures >= FAILURE_THRESHOLD:
            self.circuit_open_until = datetime.now(UTC) + timedelta(seconds=COOLDOWN_SECONDS)
            logger.warning(
                "circuit breaker opened for provider %s (%d consecutive failures)",
                self.name,
                self.consecutive_failures,
            )


_registry: dict[str, CircuitState] = {}


def get_circuit(name: str, model: str) -> CircuitState:
    """Shared per-name state so every request sees the same breaker."""
    existing = _registry.get(name)
    if existing is None:
        existing = CircuitState(name=name, model=model)
        _registry[name] = existing
    return existing


def reset_circuits() -> None:
    _registry.clear()


def snapshot_all() -> list[dict]:
    return [
        {
            "name": s.name,
            "model": s.model,
            "allowed": not s.is_open,
            "circuit_open": s.is_open,
            "open_until": s.circuit_open_until.isoformat() if s.circuit_open_until else None,
            "calls": s.calls,
            "failures": s.failures,
            "consecutive_failures": s.consecutive_failures,
            "avg_latency_ms": s.avg_latency_ms,
            "est_cost_usd": round(s.total_cost_usd, 6),
        }
        for s in _registry.values()
    ]