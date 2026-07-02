import uuid

import pytest

from app.core.security import AuthError, verify_token
from tests.conftest import mint_token

USER_ID = uuid.uuid4()


def test_valid_token_roundtrip():
    claims = verify_token(mint_token(USER_ID, "manager"))
    assert claims.user_id == USER_ID
    assert claims.role == "manager"
    assert claims.email == "manager@example.com"


def test_expired_token_rejected():
    token = mint_token(USER_ID, expires_in=-3600)
    with pytest.raises(AuthError, match="expired"):
        verify_token(token)


def test_wrong_signature_rejected():
    token = mint_token(USER_ID, secret="attacker-secret")
    with pytest.raises(AuthError, match="Invalid token"):
        verify_token(token)


def test_wrong_audience_rejected():
    token = mint_token(USER_ID, audience="anon")
    with pytest.raises(AuthError, match="Invalid token"):
        verify_token(token)


def test_garbage_token_rejected():
    with pytest.raises(AuthError):
        verify_token("not.a.jwt")


def test_missing_role_claim_is_none():
    from datetime import UTC, datetime, timedelta

    import jwt as pyjwt

    now = datetime.now(UTC)
    token = pyjwt.encode(
        {"sub": str(USER_ID), "aud": "authenticated", "exp": now + timedelta(hours=1)},
        "test-jwt-secret-0123456789abcdef-0123456789abcdef",
        algorithm="HS256",
    )
    claims = verify_token(token)
    assert claims.role is None
