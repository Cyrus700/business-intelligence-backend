"""Query cache layer with TTL and hit/miss metrics."""

import asyncio
import hashlib
import json
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
    _key_json: dict[str, str] = field(default_factory=dict)
    _hits: int = 0
    _misses: int = 0
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock)

    def _make_key(self, func_name: str, args: tuple, kwargs: dict) -> str:
        def _stable(v: Any) -> Any:
            if v is None:
                return None
            # Profile / user object — only tenant and role matter for cache
            t = type(v).__name__
            if t == "Profile":
                try:
                    return {"org_id": str(getattr(v, "org_id", None)), "role": getattr(v, "role", None)}
                except Exception:
                    return str(getattr(v, "org_id", None))
            # Filters dataclass → stable dict of its fields
            if hasattr(v, "__dataclass_fields__"):
                try:
                    d = {k: _stable(getattr(v, k)) for k in sorted(v.__dataclass_fields__.keys())}  # type: ignore[attr-defined]
                    return d
                except Exception:
                    return str(v)
            if isinstance(v, (list, tuple)):
                return [_stable(x) for x in v]
            if isinstance(v, dict):
                return {str(k): _stable(val) for k, val in sorted(v.items())}
            try:
                json.dumps(v)
                return v
            except TypeError:
                return str(v)

        # First arg is always the DB session — drop it; it differs per request
        # and would make every cache lookup miss, causing the 27-query thundering
        # herd that triggered 429/500 on the dashboard.
        stable_args: list[Any] = []
        for idx, a in enumerate(args):
            if idx == 0 and "Session" in type(a).__name__:
                continue
            stable_args.append(_stable(a))
        stable_kwargs = {k: _stable(v) for k, v in kwargs.items()}
        key_data = {"fn": func_name, "args": stable_args, "kwargs": stable_kwargs}
        json_str = json.dumps(key_data, sort_keys=True, default=str)
        digest = hashlib.md5(json_str.encode()).hexdigest()
        key = f"query:{func_name}:{digest}"
        # store mapping for per-org invalidation (org_id substring search)
        try:
            self._key_json[key] = json_str
        except Exception:
            pass
        return key

    async def get(self, key: str) -> Any | None:
        async with self._lock:
            entry = self._store.get(key)
            if entry is None:
                self._misses += 1
                return None
            if time.time() > entry.expires_at:
                del self._store[key]
                self._key_json.pop(key, None)
                self._misses += 1
                return None
            self._hits += 1
            return entry.value

    async def set(self, key: str, value: Any, ttl_seconds: int) -> None:
        async with self._lock:
            self._store[key] = CacheEntry(value=value, expires_at=time.time() + ttl_seconds)

    async def clear(self, org_id: Any | None = None) -> None:
        async with self._lock:
            if org_id is None:
                self._store.clear()
                self._key_json.clear()
                return
            org_str = str(org_id)
            # Find keys where the original json payload contains the org_id
            to_delete = [k for k, js in list(self._key_json.items()) if org_str in js]
            # Fallback: legacy keys may contain org_str directly (pre-md5 keys)
            for k in list(self._store.keys()):
                if org_str in k and k not in to_delete:
                    to_delete.append(k)
            if not to_delete:
                # No trackable keys for this org — nothing to do (avoid global clear which would affect other tenants)
                return
            for k in to_delete:
                self._store.pop(k, None)
                self._key_json.pop(k, None)

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


async def clear_query_cache(org_id: Any | None = None) -> None:
    """Clear cached queries — per-org if org_id given, else global."""
    cache = get_query_cache()
    await cache.clear(org_id=org_id)
