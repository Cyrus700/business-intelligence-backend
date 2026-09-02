import json
import logging
import sys
from datetime import UTC, datetime

from app.core.request_context import current_request_id

SENSITIVE_KEYS = {"password", "passwd", "secret", "token", "authorization", "api_key", "apikey", "dsn", "smtp_password"}


def _redact(obj: object) -> object:
    if isinstance(obj, dict):
        return {k: ("***" if any(s in k.lower() for s in SENSITIVE_KEYS) else _redact(v)) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_redact(v) for v in obj]  # type: ignore[return-value]
    if isinstance(obj, str) and len(obj) > 200 and "postgresql" in obj.lower():
        return "***DSN***"
    return obj


class JsonFormatter(logging.Formatter):
    def format(self, record: logging.LogRecord) -> str:
        msg = record.getMessage()
        # Redact obvious secret patterns from the message itself
        lowered = msg.lower()
        if any(k in lowered for k in ("password", "secret", "dsn")) and "postgresql" in lowered:
            msg = "***redacted***"
        payload: dict[str, object] = {
            "ts": datetime.now(UTC).isoformat(timespec="milliseconds"),
            "level": record.levelname,
            "logger": record.name,
            "message": msg,
        }
        if record.exc_info:
            # In prod, strip exc text that might contain DSNs or tokens
            exc_text = self.formatException(record.exc_info)
            if "postgresql" in exc_text.lower() or "password" in exc_text.lower():
                exc_text = "***redacted exception***"
            payload["exc"] = exc_text
        extra = getattr(record, "extra_fields", None)
        if isinstance(extra, dict):
            redacted = _redact(extra)
            if isinstance(redacted, dict):
                payload.update(redacted)
        request_id = current_request_id()
        if request_id:
            payload["request_id"] = request_id
        return json.dumps(payload, default=str)


def setup_logging(level: int = logging.INFO) -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.handlers = [handler]
    root.setLevel(level)
    # uvicorn's own access log stays readable; app logs are JSON
    logging.getLogger("uvicorn.access").handlers = [handler]
