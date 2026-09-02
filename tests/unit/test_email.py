"""Email service — templates + async SMTP logic."""

from unittest.mock import MagicMock, patch

import pytest

from app.services.email import templates
from app.services.email.service import (
    is_configured,
    send_alert_email,
    send_email,
    send_password_reset_email,
    send_report_ready_email,
    send_welcome_email,
    stats_snapshot,
)


def test_password_reset_template():
    subj, text, html = templates.password_reset("Ashok", "https://app.example.com/reset-password?token=abc123")
    assert "Reset" in subj
    assert "Ashok" in text
    assert "abc123" in html
    assert "30" in text or "30" in html  # reset window is 30 min now


def test_welcome_template():
    subj, text, html = templates.welcome("Jane", "https://app.example.com/login")
    assert "Welcome" in subj
    assert "Jane" in text
    assert "login" in html.lower()


def test_alert_template():
    subj, text, html = templates.alert_notification(
        "Revenue anomaly", "revenue changed +60%", "https://app.example.com/dashboard/alerts"
    )
    assert "Revenue anomaly" in subj
    assert "revenue changed" in text
    assert "dashboard" in html


def test_report_ready_template():
    subj, text, html = templates.report_ready(
        "Monthly — Jan 2026", "2026-01-01", "2026-01-31", "pdf", "https://app.example.com/dashboard/reports"
    )
    assert "Monthly" in subj
    assert "2026-01-01" in text
    assert "PDF" in html or "pdf" in html.lower()


@pytest.mark.anyio
async def test_send_email_skipped_when_no_smtp(monkeypatch):
    # explicit: a developer .env with SMTP configured must not change the result
    monkeypatch.setenv("SMTP_HOST", "")
    ok = await send_email("user@example.com", "Subject", "body", "<b>html</b>")
    assert ok is False
    snap = stats_snapshot()
    assert snap["skipped"] >= 1


@pytest.mark.anyio
async def test_send_email_rejects_invalid_recipient():
    ok = await send_email("not-an-email", "Subject", "body")
    assert ok is False


@pytest.mark.anyio
async def test_send_email_with_mock_smtp(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    monkeypatch.setenv("SMTP_PORT", "587")
    monkeypatch.setenv("SMTP_FROM", "Test <test@example.com>")
    from app.core.config import get_settings

    get_settings.cache_clear()
    assert is_configured() is True

    with patch("app.services.email.service.smtplib.SMTP") as mock_smtp:
        inst = MagicMock()
        mock_smtp.return_value.__enter__.return_value = inst
        inst.has_extn.return_value = True
        ok = await send_email("to@example.com", "Hello", "plain", "<b>html</b>")
        assert ok is True
        assert inst.send_message.called

    # also test SSL path
    monkeypatch.setenv("SMTP_PORT", "465")
    get_settings.cache_clear()
    with patch("app.services.email.service.smtplib.SMTP_SSL") as mock_ssl:
        inst2 = MagicMock()
        mock_ssl.return_value.__enter__.return_value = inst2
        ok2 = await send_email("to2@example.com", "Hi", "body")
        assert ok2 is True
        assert inst2.send_message.called

    monkeypatch.delenv("SMTP_HOST", raising=False)
    monkeypatch.delenv("SMTP_PORT", raising=False)
    monkeypatch.delenv("SMTP_FROM", raising=False)
    get_settings.cache_clear()


@pytest.mark.anyio
async def test_helpers_delegate_to_send_email(monkeypatch):
    monkeypatch.setenv("SMTP_HOST", "smtp.example.com")
    from app.core.config import get_settings

    get_settings.cache_clear()
    with patch("app.services.email.service.smtplib.SMTP") as mock_smtp:
        inst = MagicMock()
        mock_smtp.return_value.__enter__.return_value = inst
        inst.has_extn.return_value = False
        ok = await send_welcome_email("new@example.com", "New User")
        assert ok is True
        ok = await send_password_reset_email("user@example.com", "tok123", "User")
        assert ok is True
        ok = await send_alert_email("mgr@example.com", "Anomaly", "something happened")
        assert ok is True
        ok = await send_report_ready_email("mgr@example.com", "Report", "2026-01-01", "2026-01-31", "pdf")
        assert ok is True
    monkeypatch.delenv("SMTP_HOST", raising=False)
    get_settings.cache_clear()


def test_header_injection_rejected():
    import pytest

    from app.services.email.service import _sanitize_header

    with pytest.raises(ValueError):
        _sanitize_header("bad\nInjection", "subject")
    with pytest.raises(ValueError):
        _sanitize_header("bad\rInjection", "subject")


def test_email_rate_limit(monkeypatch):
    from app.services.email.service import _check_email_rate_limit, _reset_email_limiter

    monkeypatch.setenv("EMAIL_RATE_LIMIT_PER_MINUTE", "2")
    from app.core.config import get_settings

    get_settings.cache_clear()
    _reset_email_limiter()
    assert _check_email_rate_limit("a@example.com") is True
    assert _check_email_rate_limit("a@example.com") is True
    assert _check_email_rate_limit("a@example.com") is False  # throttled on 3rd
    _reset_email_limiter()
    monkeypatch.delenv("EMAIL_RATE_LIMIT_PER_MINUTE", raising=False)
    get_settings.cache_clear()
