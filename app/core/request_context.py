"""Request correlation: a request_id on every log line and API response.

Phase 13 observability. The id is generated here (or taken from the caller's
X-Request-ID), stored in a contextvar so any logger call anywhere in the
request path — API handler, ETL run, ML inference, AI tool, DB commit — emits
the same id, and echoed back in the X-Request-ID response header so a frontend
or an operator can trace a failing call end to end.
"""

import uuid
from contextvars import ContextVar
from typing import Any

from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint
from starlette.requests import Request
from starlette.responses import Response

_request_id: ContextVar[str | None] = ContextVar("request_id", default=None)

REQUEST_ID_HEADER = "X-Request-ID"


def current_request_id() -> str | None:
    return _request_id.get()


def set_request_id(request_id: str) -> None:
    _request_id.set(request_id)


class RequestContextMiddleware(BaseHTTPMiddleware):
    """Assigns/echoes a request id and exposes it on the request state."""

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        incoming = request.headers.get(REQUEST_ID_HEADER)
        request_id = incoming.strip()[:64] if incoming and incoming.strip() else uuid.uuid4().hex[:16]
        token = _request_id.set(request_id)
        request.state.request_id = request_id
        try:
            response = await call_next(request)
        finally:
            _request_id.reset(token)
        response.headers.setdefault(REQUEST_ID_HEADER, request_id)
        return response


def request_extra() -> dict[str, Any]:
    rid = current_request_id()
    return {"request_id": rid} if rid else {}
