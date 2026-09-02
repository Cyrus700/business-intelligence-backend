"""Query cache layer with TTL and hit/miss metrics."""

import asyncio
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from functools import wraps
from typing import Any, TypeVar

T = TypeVar("T")


@dataclass
class CacheEntry:
    value: Any
    expires_at: float


@dataclass
class QueryCache:
    _store: dict[str, CacheEntry] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        import json

        key_data = {"fn": func_name, "args": args, "kwargs": kwargs}
        return f"query:{func_name}:{hash(json.dumps(key_data, sort_keys=True, default=str))}"

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        async with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=time.time() + ttl_seconds)

    async def clear(self) -> None:
        async with self._lock:
            self._store.clear()

    async def stats(self) -> dict[str, int]:
        return {"hits": self._hits, "misses": self._misses, "size": len(self._store)}

    def cache_key(self, *args, **kwargs) -> str:
        return self._make_key("query", args, kwargs)


# Global cache instance
_query_cache: QueryCache | None = None


def get_query_cache() -> QueryCache:
    global _query_cache
    if _query_cache is None:
        _query_cache = QueryCache()
    return _query_cache


def cached_query(ttl_seconds: int = 30):
    """Decorator to cache query results with TTL."""

    def decorator(func: Callable[..., T]) -> Callable[..., T]:
        @wraps(func)
        async def wrapper(*args, **kwargs) -> T:
            cache = get_query_cache()
            key = cache.cache_key(func.__name__, args, kwargs)

            cached = await cache.get(key)
            if cached is not None:
                return cached

            result = await func(*args, **kwargs)
            await cache.set(key, result, ttl_seconds)
            return result

        return wrapper

    return decorator


async def get_cache_stats() -> dict[str, int]:
    """Get cache hit/miss statistics."""
    cache = get_query_cache()
    return await cache.stats()


async def clear_query_cache() -> None:
    """Clear all cached queries."""
    cache = get_query_cache()
    await cache.clear()
