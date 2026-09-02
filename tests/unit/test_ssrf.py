"""Phase 12 security: SSRF guard on REST sources (unit + API)."""

import pytest

from app.services.etl.ssrf import validate_public_http_url

GOOD_URLS = [
    "https://example.com/data.json",
    "https://api.example.com/v1/records?page=2",
    "http://analytics.example.net/sales",
]

BAD_URLS = [
    "http://127.0.0.1:8000/internal",
    "http://10.0.0.1/secret",
    "https://192.168.1.10/",
    "http://169.254.169.254/latest/meta-data/",
    "http://0.0.0.0/",
    "https://[::1]/",
    "http://localhost:5432/db",
    "http://db.internal:5432/",
    "http://postgres.local/",
    "ftp://example.com/file",
    "https://user:pass@example.com/",
    "http://example.com:8080/api",
    "http://",
]


def _public_dns(*args, **kwargs):
    import socket

    return [
        (
            socket.AddressFamily.AF_INET,
            socket.SocketKind.SOCK_STREAM,
            6,
            "",
            ("93.184.216.34", 443),
        )
    ]


@pytest.mark.parametrize("url", GOOD_URLS)
def test_public_urls_allowed(url: str, monkeypatch) -> None:
    import socket

    monkeypatch.setattr(socket, "getaddrinfo", _public_dns)
    assert validate_public_http_url(url) == url


@pytest.mark.parametrize("url", BAD_URLS)
def test_internal_urls_blocked(url: str) -> None:
    with pytest.raises(ValueError):
        validate_public_http_url(url)


def test_private_dns_resolution_blocked(monkeypatch) -> None:
    """A hostname resolving to a private address is refused."""
    import socket

    fake = [
        (socket.AddressFamily.AF_INET, socket.SocketKind.SOCK_STREAM, 6, "", ("127.0.0.1", 443)),
        (socket.AddressFamily.AF_INET, socket.SocketKind.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: fake)
    with pytest.raises(ValueError, match="private address"):
        validate_public_http_url("https://hostname-that-resolves-privately.test/data")


def test_public_dns_resolution_allowed(monkeypatch) -> None:
    import socket

    fake = [
        (socket.AddressFamily.AF_INET, socket.SocketKind.SOCK_STREAM, 6, "", ("93.184.216.34", 443)),
    ]
    monkeypatch.setattr(socket, "getaddrinfo", lambda *a, **k: fake)
    assert validate_public_http_url("https://public.example.com/data") == ("https://public.example.com/data")


async def test_rest_api_source_with_private_url_rejected(client, admin_token):
    from tests.conftest import auth

    try:
        _, token = admin_token
    except Exception as e:  # DB not available
        pytest.skip(f"DB not available: {e}")
    try:
        resp = await client.post(
            "/api/v1/data-sources",
            json={
                "name": "Internal scraper",
                "kind": "rest_api",
                "target_domain": "sales",
                "config": {"url": "http://127.0.0.1/private/orders.json"},
            },
            headers=auth(token),
        )
    except OSError as e:
        pytest.skip(f"DB not available: {e}")
    assert resp.status_code == 422
    assert "SSRF" in resp.text


async def test_rest_api_source_with_public_url_created(client, admin_token, monkeypatch):
    import socket

    from tests.conftest import auth

    try:
        _, token = admin_token
    except Exception as e:
        pytest.skip(f"DB not available: {e}")
    monkeypatch.setattr(
        socket,
        "getaddrinfo",
        lambda *a, **k: [(socket.AddressFamily.AF_INET, socket.SocketKind.SOCK_STREAM, 6, "", ("93.184.216.34", 443))],
    )
    try:
        resp = await client.post(
            "/api/v1/data-sources",
            json={
                "name": "Public feed",
                "kind": "rest_api",
                "target_domain": "sales",
                "config": {"url": "https://data.example.com/orders.json"},
            },
            headers=auth(token),
        )
    except OSError as e:
        pytest.skip(f"DB not available: {e}")
    assert resp.status_code == 201, resp.text
    assert resp.json()["config"]["url"] == "https://data.example.com/orders.json"
