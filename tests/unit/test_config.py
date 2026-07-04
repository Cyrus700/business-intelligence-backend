from app.core.config import Settings


def test_cors_origins_parsing():
    s = Settings(frontend_origins="http://localhost:3000, https://app.example.com ,")
    assert s.cors_origins == ["http://localhost:3000", "https://app.example.com"]


def test_defaults_are_dev_safe(monkeypatch):
    monkeypatch.delenv("ENV", raising=False)
    s = Settings(_env_file=None)
    assert s.env == "dev"
    assert s.jwt_audience == "authenticated"
