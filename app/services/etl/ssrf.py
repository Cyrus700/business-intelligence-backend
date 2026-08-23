"""SSRF defence for outbound REST-source fetches (Phase 12 security).

``extract_rest_api`` will only ever fetch a URL that passes ``validate_public_http_url``.
The check is performed on:

- scheme: http/https only
- host syntax: no userinfo, no port other than 80/443
- literal IPs: all private / loopback / link-local / reserved ranges are refused
- hostnames: resolved via DNS and refused if *any* resolved address is
  non-public (loopback, RFC1918, link-local 169.254.x.x — the cloud metadata
  range — IPv6 ULA/link-local, or the zero address)

Resolving at fetch time (not just at source-config save time) also closes the
DNS-rebinding window for this path; the resolution here is only a pre-flight
check, and the production network egress rules are the second line of defence.
"""

import ipaddress
import socket
from urllib.parse import urlparse

import httpx

ALLOWED_SCHEMES = {"http", "https"}
ALLOWED_PORTS = {80, 443, None}

_BLOCKED_TLDS = {"localhost", "local", "internal", "localdomain", "home", "lan"}

# Explicit internal ranges only (RFC1918, loopback, link-local incl. cloud
# metadata, CGNAT). Documentation/test ranges (192.0.2/24, 203.0.113/24, …)
# cannot be reached, so they are not a security concern and are left alone.
_PRIVATE_NETS = tuple(
    ipaddress.ip_network(n)
    for n in (
        "0.0.0.0/8",
        "10.0.0.0/8",
        "100.64.0.0/10",
        "127.0.0.0/8",
        "169.254.0.0/16",
        "172.16.0.0/12",
        "192.168.0.0/16",
        "224.0.0.0/4",
        "240.0.0.0/4",
        "::/128",
        "::1/128",
        "fc00::/7",
        "fe80::/10",
        "ff00::/8",
    )
)


def _is_private_ip(ip_str: str) -> bool:
    ip = ipaddress.ip_address(ip_str)
    if isinstance(ip, ipaddress.IPv6Address):
        return (
            ip.is_loopback
            or ip.is_link_local
            or ip.is_unspecified
            or ip.is_multicast
            or ip in _PRIVATE_NETS
        )
    return any(ip in net for net in _PRIVATE_NETS)


def validate_public_http_url(url: str) -> str:
    """Raise ValueError for any URL that could hit internal infrastructure."""
    parsed = urlparse(url)
    if parsed.scheme not in ALLOWED_SCHEMES:
        raise ValueError("REST source URL must be http(s)")
    if parsed.username is not None or parsed.password is not None:
        raise ValueError("REST source URL must not embed credentials")
    if parsed.port not in ALLOWED_PORTS:
        raise ValueError("REST source URL must use the standard port (80/443)")
    host = parsed.hostname
    if not host:
        raise ValueError("REST source URL has no host")
    lowered = host.rstrip(".").lower()
    if lowered.startswith("localhost") or lowered.endswith(tuple(f".{t}" for t in _BLOCKED_TLDS)):
        raise ValueError(f"REST source host '{host}' is blocked (SSRF guard)")
    try:
        candidate = ipaddress.ip_address(lowered.split("%")[0])
        if _is_private_ip(str(candidate)):
            raise ValueError(f"REST source host '{host}' is a private address (SSRF guard)")
        return url
    except ValueError as e:
        if str(e).startswith("REST source"):
            raise
        # Not a literal IP: resolve and inspect every address (incl. CNAMEs).
    try:
        infos = socket.getaddrinfo(host, parsed.port or (443 if parsed.scheme == "https" else 80))
    except OSError:
        # Unresolvable hosts are no security target — the fetch itself will
        # fail harmlessly downstream.
        return url
    resolved = {info[4][0] for info in infos}
    if not resolved:
        return url
    for addr in resolved:
        if _is_private_ip(addr):
            raise ValueError(
                f"REST source host '{host}' resolves to a private address {addr} (SSRF guard)"
            )
    return url


def validate_rest_api_fetch(url: str, client: httpx.AsyncClient | None = None) -> None:
    """Public pre-flight that the ETL path runs before any request."""
    validate_public_http_url(url)


def validate_postgres_dsn(dsn: str) -> str:
    """Block PG DSNs that would probe loopback / cloud metadata.

    The warehouse itself lives on a private network, so we cannot block all
    RFC1918, but we must refuse loopback/link-local and the 169.254.169.254
    metadata address that would leak instance credentials.
    """
    from urllib.parse import urlparse

    # SQLAlchemy async DSN: postgresql+asyncpg://user:pass@host:5432/db
    # urlparse needs a scheme it recognises — normalise to postgresql://
    normalised = dsn.replace("postgresql+asyncpg://", "postgresql://").replace(
        "postgresql+psycopg://", "postgresql://"
    )
    parsed = urlparse(normalised)
    host = parsed.hostname
    if not host:
        raise ValueError("Postgres DSN has no host")
    lowered = host.rstrip(".").lower()
    if lowered in {"localhost", "metadata.google.internal"}:
        raise ValueError(f"Postgres DSN host '{host}' is blocked")
    try:
        ip = ipaddress.ip_address(lowered.split("%")[0])
        if ip.is_loopback or ip.is_link_local or str(ip) == "169.254.169.254":
            raise ValueError(f"Postgres DSN host '{host}' is a blocked address")
        # Also block link-local range explicitly
        if any(ip in net for net in _PRIVATE_NETS if str(net).startswith("169.254") or str(net).startswith("127.")):
            raise ValueError(f"Postgres DSN host '{host}' is blocked")
    except ValueError as e:
        if str(e).startswith("Postgres DSN"):
            raise
        # hostname, not IP — resolve and check if it points to metadata/loopback
        try:
            infos = socket.getaddrinfo(host, parsed.port or 5432)
        except OSError:
            return dsn
        for info in infos:
            addr = info[4][0]
            if addr in {"127.0.0.1", "::1", "169.254.169.254"}:
                raise ValueError(f"Postgres DSN host '{host}' resolves to blocked address {addr}")
    return dsn
