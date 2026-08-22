"""Phase 12 security: upload hardening + auth audit events."""

import uuid

import bcrypt

from app.core.database import get_session_factory
from app.models import Profile
from app.services.etl.extractors import extract_tabular, sanitize_upload_filename
from tests.conftest import auth


def test_sanitize_rejects_path_traversal():
    assert sanitize_upload_filename("../../../etc/passwd.csv") == "passwd.csv"
    assert sanitize_upload_filename("..\\..\\win.ini.csv") == "win.ini.csv"
    assert sanitize_upload_filename("report.xlsx") == "report.xlsx"


def test_sanitize_rejects_unknown_extensions_and_hidden_files():
    for bad in ("evil.sh", "secret.csv.exe", ".csv", "noext", "a.bat"):
        try:
            sanitize_upload_filename(bad)
            raise AssertionError(f"{bad} should be rejected")
        except ValueError:
            pass


def test_magic_bytes_sniffing_rejects_disguised_files():
    good_csv = b"txn_date,amount\n2026-01-01,100\n"
    extract_tabular(good_csv, "data.csv")
    try:
        extract_tabular(b"PK\x03\x04 fake zip content", "data.csv")
        raise AssertionError("CSV containing zip magic should be rejected")
    except ValueError:
        pass
    try:
        extract_tabular(b"<html><body>hello</body></html>", "data.csv")
        raise AssertionError("CSV containing HTML should be rejected")
    except ValueError:
        pass
    try:
        extract_tabular(b"plain text pretending to be xlsx", "data.xlsx")
        raise AssertionError("xlsx without zip magic should be rejected")
    except ValueError:
        pass


async def test_login_success_and_failure_are_audited(client, admin_token):
    """Successful and failed logins both leave audit events."""
    async with get_session_factory()() as db:
        profile = Profile(
            id=uuid.uuid4(),
            email="audit@example.com",
            password_hash=bcrypt.hashpw(b"correct-horse-battery", bcrypt.gensalt()).decode(),
            full_name="Audit Test",
            role="analyst",
            is_active=True,
        )
        db.add(profile)
        await db.commit()

    bad = await client.post(
        "/api/v1/auth/login",
        json={"email": "audit@example.com", "password": "wrong-password"},
    )
    assert bad.status_code == 401

    ok = await client.post(
        "/api/v1/auth/login",
        json={"email": "audit@example.com", "password": "correct-horse-battery"},
    )
    assert ok.status_code == 200, ok.text

    _, token = admin_token
    resp = await client.get("/api/v1/audit-logs?page_size=100", headers=auth(token))
    assert resp.status_code == 200, resp.text
    actions = [r["action"] for r in resp.json()]
    assert "auth.login_failed" in actions
    assert "auth.login" in actions
    failed = next(r for r in resp.json() if r["action"] == "auth.login_failed")
    assert failed["detail"].get("email") == "audit@example.com"