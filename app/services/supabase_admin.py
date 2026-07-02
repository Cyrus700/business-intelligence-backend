"""Thin wrapper around Supabase's GoTrue admin API (service-role key).

Kept minimal on purpose: only what user management needs. Tests override this
via FastAPI dependency injection, so no Supabase project is needed to run them.
"""

from typing import Any
from uuid import UUID

import httpx

from app.core.config import get_settings


class SupabaseAdminError(Exception):
    pass


class SupabaseAdmin:
    def __init__(self) -> None:
        settings = get_settings()
        if not settings.supabase_url or not settings.supabase_service_key:
            raise SupabaseAdminError("Supabase admin credentials are not configured")
        self._base = f"{settings.supabase_url}/auth/v1"
        self._headers = {
            "apikey": settings.supabase_service_key,
            "Authorization": f"Bearer {settings.supabase_service_key}",
        }

    async def create_user(
        self, email: str, password: str, role: str, full_name: str | None
    ) -> UUID:
        payload: dict[str, Any] = {
            "email": email,
            "password": password,
            "email_confirm": True,
            "app_metadata": {"role": role},
            "user_metadata": {"full_name": full_name},
        }
        async with httpx.AsyncClient() as client:
            resp = await client.post(
                f"{self._base}/admin/users", json=payload, headers=self._headers
            )
        if resp.status_code >= 400:
            raise SupabaseAdminError(f"Supabase user creation failed: {resp.text}")
        return UUID(resp.json()["id"])

    async def set_role(self, user_id: UUID, role: str) -> None:
        async with httpx.AsyncClient() as client:
            resp = await client.put(
                f"{self._base}/admin/users/{user_id}",
                json={"app_metadata": {"role": role}},
                headers=self._headers,
            )
        if resp.status_code >= 400:
            raise SupabaseAdminError(f"Supabase role update failed: {resp.text}")


def get_supabase_admin() -> SupabaseAdmin:
    return SupabaseAdmin()
