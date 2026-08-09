"""Runtime resolution of the admin-editable RBAC policy.

Every authorization decision goes through :func:`get_policy`, which returns an
immutable in-memory snapshot of the ``roles`` / ``permissions`` /
``role_permissions`` tables. The snapshot is cached process-wide for
``CACHE_TTL_SECONDS`` and busted synchronously by :func:`invalidate` whenever an
admin edits the matrix, so a permission change takes effect on the next request
without a redeploy — while a hot path never pays for three extra queries.

If the RBAC tables are empty (fresh database, truncated test database) the
policy falls back to :mod:`app.core.rbac_defaults`, and :func:`ensure_seeded`
writes those defaults back idempotently.
"""

from __future__ import annotations

import asyncio
import time
from dataclasses import dataclass, field

from sqlalchemy import select
from sqlalchemy.dialects.postgresql import insert as pg_insert
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.rbac_defaults import (
    DEFAULT_GRANTS,
    DEFAULT_PERMISSIONS,
    DEFAULT_ROLES,
    LOCKED_ADMIN_PERMISSIONS,
)
from app.models.rbac import Permission, Role, RolePermission

CACHE_TTL_SECONDS = 30.0


@dataclass(frozen=True)
class RoleInfo:
    name: str
    label: str
    description: str | None
    rank: int
    color: str
    is_system: bool
    is_active: bool


@dataclass(frozen=True)
class Policy:
    """Immutable snapshot of the effective access-control policy."""

    roles: dict[str, RoleInfo] = field(default_factory=dict)
    grants: dict[str, frozenset[str]] = field(default_factory=dict)
    permission_keys: frozenset[str] = frozenset()

    def rank(self, role: str | None) -> int:
        info = self.roles.get(role or "")
        return info.rank if info else 0

    def permissions_for(self, role: str | None) -> frozenset[str]:
        if not role:
            return frozenset()
        return self.grants.get(role, frozenset())

    @property
    def top_role(self) -> str | None:
        """Highest-ranked active role — the one that can never be locked out."""
        active = [r for r in self.roles.values() if r.is_active]
        if not active:
            return None
        return max(active, key=lambda r: r.rank).name


def _fallback_policy() -> Policy:
    roles = {
        r["name"]: RoleInfo(
            name=r["name"],
            label=r["label"],
            description=r["description"],
            rank=r["rank"],
            color=r["color"],
            is_system=True,
            is_active=True,
        )
        for r in DEFAULT_ROLES
    }
    return Policy(
        roles=roles,
        grants={name: frozenset(keys) for name, keys in DEFAULT_GRANTS.items()},
        permission_keys=frozenset(p["key"] for p in DEFAULT_PERMISSIONS),
    )


FALLBACK_POLICY = _fallback_policy()

_cache: Policy | None = None
_cached_at: float = 0.0
_lock = asyncio.Lock()


def invalidate() -> None:
    """Drop the cached snapshot — call after any write to the RBAC tables."""
    global _cache, _cached_at
    _cache = None
    _cached_at = 0.0


async def _load(db: AsyncSession) -> Policy:
    roles = (await db.execute(select(Role))).scalars().all()
    if not roles:
        return FALLBACK_POLICY

    permissions = (await db.execute(select(Permission))).scalars().all()
    by_id = {p.id: p.key for p in permissions}
    links = (await db.execute(select(RolePermission))).scalars().all()

    grants: dict[str, set[str]] = {r.name: set() for r in roles}
    role_names = {r.id: r.name for r in roles}
    for link in links:
        name = role_names.get(link.role_id)
        key = by_id.get(link.permission_id)
        if name and key:
            grants[name].add(key)

    return Policy(
        roles={
            r.name: RoleInfo(
                name=r.name,
                label=r.label,
                description=r.description,
                rank=r.rank,
                color=r.color,
                is_system=r.is_system,
                is_active=r.is_active,
            )
            for r in roles
        },
        grants={name: frozenset(keys) for name, keys in grants.items()},
        permission_keys=frozenset(by_id.values()),
    )


async def get_policy(db: AsyncSession, *, fresh: bool = False) -> Policy:
    global _cache, _cached_at
    now = time.monotonic()
    if not fresh and _cache is not None and now - _cached_at < CACHE_TTL_SECONDS:
        return _cache
    async with _lock:
        # Another coroutine may have refreshed while we waited for the lock.
        if not fresh and _cache is not None and time.monotonic() - _cached_at < CACHE_TTL_SECONDS:
            return _cache
        policy = await _load(db)
        _cache = policy
        _cached_at = time.monotonic()
        return policy


async def ensure_seeded(db: AsyncSession) -> Policy:
    """Idempotently write the default catalog; safe to call concurrently.

    Existing rows are never overwritten — an admin who removed a default grant
    does not get it silently restored on the next boot. Only genuinely missing
    roles/permissions are inserted, and grants are seeded only for roles that
    had no grants at all (i.e. brand-new roles).
    """
    await db.execute(
        pg_insert(Role)
        .values(
            [
                {
                    "name": r["name"],
                    "label": r["label"],
                    "description": r["description"],
                    "rank": r["rank"],
                    "color": r["color"],
                    "is_system": True,
                    "is_active": True,
                }
                for r in DEFAULT_ROLES
            ]
        )
        .on_conflict_do_nothing(index_elements=["name"])
    )
    await db.execute(
        pg_insert(Permission)
        .values(
            [
                {
                    "key": p["key"],
                    "label": p["label"],
                    "description": p["description"],
                    "group_label": p["group_label"],
                    "sort_order": p["sort_order"],
                    "is_system": True,
                }
                for p in DEFAULT_PERMISSIONS
            ]
        )
        .on_conflict_do_nothing(index_elements=["key"])
    )
    await db.flush()

    role_ids = dict(
        (await db.execute(select(Role.name, Role.id))).all()  # type: ignore[arg-type]
    )
    perm_ids = dict(
        (await db.execute(select(Permission.key, Permission.id))).all()  # type: ignore[arg-type]
    )
    ungranted = {
        name
        for name in role_ids
        if not (
            await db.execute(
                select(RolePermission.id).where(RolePermission.role_id == role_ids[name]).limit(1)
            )
        ).first()
    }

    rows = [
        {"role_id": role_ids[role], "permission_id": perm_ids[key]}
        for role, keys in DEFAULT_GRANTS.items()
        if role in ungranted and role in role_ids
        for key in keys
        if key in perm_ids
    ]
    if rows:
        await db.execute(
            pg_insert(RolePermission)
            .values(rows)
            .on_conflict_do_nothing(index_elements=["role_id", "permission_id"])
        )
    await db.commit()
    invalidate()
    return await get_policy(db, fresh=True)


def locked_permissions_for(policy: Policy, role_name: str) -> frozenset[str]:
    """Permissions that cannot be revoked from ``role_name``.

    Only the top-ranked role is protected: stripping ``users:manage`` or
    ``roles:manage`` from it would leave nobody able to grant them back.
    """
    if policy.top_role == role_name:
        return LOCKED_ADMIN_PERMISSIONS & policy.permission_keys
    return frozenset()
