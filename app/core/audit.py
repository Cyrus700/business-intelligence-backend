"""Audit trail middleware: records every authenticated mutating request.

Runs after the response is produced; failures to write the audit row are logged,
never surfaced to the client (availability over audit completeness for the MVP —
documented trade-off in docs/02-architecture.md).
"""

import logging

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

from app.core.database import get_session_factory
from app.core.request_context import current_request_id
from app.models import AuditLog

logger = logging.getLogger(__name__)

AUDITED_METHODS = {"POST", "PATCH", "PUT", "DELETE"}


class AuditMiddleware(BaseHTTPMiddleware):
    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        response = await call_next(request)

        user = getattr(request.state, "user", None)
        if request.method in AUDITED_METHODS and user is not None and response.status_code < 500:
            try:
                async with get_session_factory()() as session:
                    session.add(
                        AuditLog(
                            user_id=user.id,
                            action=f"{request.method} {request.url.path}",
                            entity=request.url.path.strip("/").split("/")[-1] or None,
                            detail={
                                "status_code": response.status_code,
                                "request_id": current_request_id(),
                            },
                            ip_address=request.client.host if request.client else None,
                        )
                    )
                    await session.commit()
            except Exception:
                logger.exception("failed to write audit log")
        return response
