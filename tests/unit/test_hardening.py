from app.core.hardening import SECURITY_HEADERS, FixedWindowLimiter


def test_limiter_allows_up_to_limit_then_blocks():
    limiter = FixedWindowLimiter(3)
    now = 1_000_000.0
    assert limiter.hit("k", now) == (True, 2)
    assert limiter.hit("k", now + 1) == (True, 1)
    assert limiter.hit("k", now + 2) == (True, 0)
    allowed, remaining = limiter.hit("k", now + 3)
    assert allowed is False and remaining == 0
    # other callers are unaffected
    assert limiter.hit("other", now + 3)[0] is True


def test_limiter_resets_on_new_window():
    limiter = FixedWindowLimiter(1)
    assert limiter.hit("k", 60.0)[0] is True
    assert limiter.hit("k", 61.0)[0] is False
    assert limiter.hit("k", 120.0)[0] is True  # next minute window


async def test_security_headers_present(client):
    resp = await client.get("/api/v1/health")
    for header in SECURITY_HEADERS:
        assert header in resp.headers, header
    assert "X-RateLimit-Remaining" in resp.headers
