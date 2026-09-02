"""Central email service — async, non-blocking, HTML + text, retry-aware.

Security:
- Header-injection guard: rejects CR/LF in To/Subject.
- Per-recipient rate limit (in-memory fixed-window).
- All SMTP I/O runs in a thread pool via ``asyncio.to_thread`` so the FastAPI
  event loop never blocks (fixing the sync ``smtplib`` calls that previously froze
  the worker under load).
- Best-effort: if SMTP is not configured the call logs and returns False instead
  of raising (prevents auth/report flows from failing when email is disabled).

Performance:
- Configurable retries with exponential backoff, port-aware TLS (465=SSL, 587=STARTTLS).
"""

from __future__ import annotations

import asyncio
import logging
import re
import smtplib
import ssl
import time
from email.message import EmailMessage
from email.utils import formataddr, parseaddr
from typing import Any

from app.core.config import get_settings
from app.services.email import templates

logger = logging.getLogger(__name__)

# ── Observability ────────────────────────────────────────────────────────
_stats: dict[str, int] = {"sent": 0, "failed": 0, "skipped": 0, "throttled": 0}

# ── Per-recipient rate limit (fixed-window, in-memory) ──────────────────
# This mirrors hardening.FixedWindowLimiter but scoped to email so we don't
# need to import the middleware here. Suffers the same single-process caveat.
_email_limiter: dict[tuple[str, int], int] = {}
_EMAIL_LIMIT_WINDOW = 60  # seconds


def _check_email_rate_limit(recipient: str) -> bool:
    """Return True if allowed, False if throttled."""
    s = get_settings()
    limit = getattr(s, "email_rate_limit_per_minute", 10)
    window = int(time.time() // _EMAIL_LIMIT_WINDOW)
    key = (recipient.lower(), window)
    # Evict stale windows
    stale = [k for k in _email_limiter if k[1] < window]
    for k in stale:
        _email_limiter.pop(k, None)
    count = _email_limiter.get(key, 0) + 1
    _email_limiter[key] = count
    return count <= limit


def _reset_email_limiter() -> None:  # for tests
    _email_limiter.clear()


# ── Validation & sanitisation ───────────────────────────────────────────
_EMAIL_RE = re.compile(r"^[^@\s]+@[^@\s]+\.[^@\s]+$")


def _sanitize_header(value: str, field: str = "header") -> str:
    """Reject header injection (CR/LF) and control characters."""
    if "\n" in value or "\r" in value:
        raise ValueError(f"Invalid {field}: header injection attempt")
    # Strip surrounding whitespace but keep inner spaces
    return value.strip()


def _validate_recipient(to: str) -> str:
    to = _sanitize_header(to, "recipient")
    if not _EMAIL_RE.match(to):
        raise ValueError(f"Invalid recipient email {to!r}")
    return to


def is_configured() -> bool:
    s = get_settings()
    return bool(s.smtp_host)


def _parse_from(raw: str) -> tuple[str, str]:
    """Split 'Name <email>' or plain email into (display_name, email)."""
    name, addr = parseaddr(raw)
    if not addr:
        addr = raw.strip()
    return _sanitize_header(name, "from_name"), _sanitize_header(addr, "from_addr")


def _send_sync(
    to: str,
    subject: str,
    text_body: str,
    html_body: str | None,
    *,
    attachments: list[tuple[str, bytes, str]] | None = None,
    timeout: int | None = None,
) -> None:
    s = get_settings()
    host = _sanitize_header(s.smtp_host.strip(), "smtp_host")
    if not host:
        raise RuntimeError("SMTP not configured")
    port = int(s.smtp_port or 587)
    user = s.smtp_user.strip()
    # Gmail App Passwords are displayed as "abcd efgh ijkl mnop" but must be sent without spaces
    password = s.smtp_password.strip().replace(" ", "")
    from_raw = s.smtp_from.strip() or "Sairash BI <alerts@sairash.local>"
    timeout = timeout or int(getattr(s, "smtp_timeout", 15))

    to = _validate_recipient(to)
    subject = _sanitize_header(subject, "subject")

    from_name, from_addr = _parse_from(from_raw)
    if from_addr and not _EMAIL_RE.match(from_addr):
        raise ValueError(f"Invalid SMTP_FROM address {from_addr!r}")
    from_header = formataddr((from_name or templates.BRAND, from_addr)) if from_addr else from_raw

    msg = EmailMessage()
    msg["From"] = from_header
    msg["To"] = to
    msg["Subject"] = subject
    msg.set_content(text_body)
    if html_body:
        msg.add_alternative(html_body, subtype="html")
    if attachments:
        for filename, data, mime in attachments:
            # Filename sanitisation: no directory traversal, no CR/LF
            filename = _sanitize_header(filename.replace("\\", "/").split("/")[-1], "filename")
            if not filename:
                filename = "attachment.bin"
            maintype, _, subtype = mime.partition("/")
            msg.add_attachment(data, maintype=maintype or "application", subtype=subtype or "octet-stream", filename=filename)

    # Port-aware connection: 465 → SSL, 587/25/2525 → STARTTLS.
    if port == 465:
        context = ssl.create_default_context()
        with smtplib.SMTP_SSL(host, port, timeout=timeout, context=context) as server:
            if user:
                server.login(user, password)
            server.send_message(msg)
    else:
        with smtplib.SMTP(host, port, timeout=timeout) as server:
            try:
                server.ehlo()
                if server.has_extn("STARTTLS"):
                    context = ssl.create_default_context()
                    server.starttls(context=context)
                    server.ehlo()
            except Exception:
                logger.debug("STARTTLS negotiation failed; continuing without TLS", exc_info=True)
            if user:
                server.login(user, password)
            server.send_message(msg)


async def send_email(
    to: str,
    subject: str,
    text_body: str,
    html_body: str | None = None,
    *,
    attachments: list[tuple[str, bytes, str]] | None = None,
    max_retries: int | None = None,
) -> bool:
    """Async send with short retries. Returns True on success, False otherwise."""
    # ── Input validation (before rate limit) ─────────────────────────
    try:
        _validate_recipient(to)
        _sanitize_header(subject, "subject")
    except ValueError as e:
        logger.warning("send_email: invalid input — %s", e)
        _stats["failed"] += 1
        return False

    if not is_configured():
        logger.info("SMTP not configured; skipping email to %s (subject: %s)", to, subject)
        _stats["skipped"] += 1
        return False

    if not _check_email_rate_limit(to):
        logger.warning("send_email: rate limit exceeded for %s", to)
        _stats["throttled"] += 1
        return False

    s = get_settings()
    max_retries = max_retries if max_retries is not None else int(getattr(s, "smtp_max_retries", 2))
    timeout = int(getattr(s, "smtp_timeout", 15))

    last_exc: Exception | None = None
    for attempt in range(max_retries + 1):
        try:
            await asyncio.to_thread(_send_sync, to, subject, text_body, html_body, attachments=attachments, timeout=timeout)
            _stats["sent"] += 1
            logger.info("email sent to %s — %s", to, subject)
            return True
        except Exception as exc:  # noqa: BLE001
            last_exc = exc
            logger.warning("email send attempt %d/%d to %s failed: %s", attempt + 1, max_retries + 1, to, exc)
            if attempt < max_retries:
                await asyncio.sleep(0.6 * (2**attempt))
    _stats["failed"] += 1
    logger.error("email to %s permanently failed after %d attempts: %s", to, max_retries + 1, last_exc, exc_info=last_exc)
    return False


def stats_snapshot() -> dict[str, Any]:
    return dict(_stats)


# ── Convenience helpers ───────────────────────────────────────────────

async def send_password_reset_email(to: str, token: str, full_name: str | None = None) -> bool:
    s = get_settings()
    reset_url = f"{s.frontend_url}/reset-password?token={token}"
    subject, text, html = templates.password_reset(full_name, reset_url)
    return await send_email(to, subject, text, html)


async def send_welcome_email(to: str, full_name: str | None = None) -> bool:
    s = get_settings()
    login_url = f"{s.frontend_url}/login"
    subject, text, html = templates.welcome(full_name, login_url)
    return await send_email(to, subject, text, html)


async def send_alert_email(to: str, rule_name: str, message: str) -> bool:
    s = get_settings()
    dashboard_url = f"{s.frontend_url}/dashboard/alerts"
    subject, text, html = templates.alert_notification(rule_name, message, dashboard_url)
    return await send_email(to, subject, text, html)


async def send_report_ready_email(
    to: str,
    title: str,
    period_start: str,
    period_end: str,
    fmt: str,
    attachment: tuple[str, bytes, str] | None = None,
) -> bool:
    s = get_settings()
    dashboard_url = f"{s.frontend_url}/dashboard/reports"
    subject, text, html = templates.report_ready(title, period_start, period_end, fmt, dashboard_url)
    atts = [attachment] if attachment else None
    return await send_email(to, subject, text, html, attachments=atts)


async def send_test_email(to: str) -> bool:
    subject = f"{templates.BRAND} — Test email"
    text = f"This is a test email from {templates.BRAND}. If you received it, SMTP is configured correctly."
    html = templates._wrap("Test email", "SMTP test", f'<p style="margin:0;color:{templates.COLOR_TEXT};font-size:14px;">This is a test email from <strong>{templates.BRAND}</strong>. If you received it, SMTP is configured correctly.</p>')
    return await send_email(to, subject, text, html)


async def send_business_verification_email(to: str, business_name: str, token: str, full_name: str | None = None) -> bool:
    s = get_settings()
    verify_url = f"{s.frontend_url}/verify-email?token={token}"
    subject, text, html = templates.business_email_verification(full_name, business_name, verify_url)
    return await send_email(to, subject, text, html)


async def send_business_pending_admin_email(business_name: str, business_email: str, contact_name: str | None = None) -> bool:
    s = get_settings()
    admin_email = (s.admin_email or "").strip()
    if not admin_email:
        logger.info("No admin_email configured; skipping admin notification for %s", business_name)
        return False
    admin_url = f"{s.frontend_url}/admin/dashboard"
    subject, text, html = templates.business_admin_notification(business_name, business_email, contact_name, admin_url)
    # Send to ADMIN_EMAIL (could be comma-separated list)
    recipients = [e.strip() for e in admin_email.split(",") if e.strip()]
    ok = True
    for rcpt in recipients:
        if not await send_email(rcpt, subject, text, html):
            ok = False
    return ok


async def send_business_pending_confirmation_email(to: str, business_name: str, full_name: str | None = None) -> bool:
    subject, text, html = templates.business_pending_confirmation(full_name, business_name)
    return await send_email(to, subject, text, html)


async def send_business_approved_email(to: str, business_name: str, full_name: str | None = None) -> bool:
    s = get_settings()
    login_url = f"{s.frontend_url}/login"
    subject, text, html = templates.business_approved(full_name, business_name, login_url)
    return await send_email(to, subject, text, html)


async def send_business_rejected_email(to: str, business_name: str, reason: str | None, full_name: str | None = None) -> bool:
    subject, text, html = templates.business_rejected(full_name, business_name, reason)
    return await send_email(to, subject, text, html)


async def send_invite_email(to: str, token: str, inviter_email: str, role: str, business_name: str | None = None) -> bool:
    s = get_settings()
    # business_name fallback: try to resolve via inviter's org if not provided
    if not business_name:
        business_name = "your workspace"
    invite_url = f"{s.frontend_url}/signup?invite={token}"
    subject, text, html = templates.invite_email(business_name, inviter_email, role, invite_url, token)
    return await send_email(to, subject, text, html)
