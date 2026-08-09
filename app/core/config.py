from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = ""

    supabase_url: str = ""
    supabase_jwt_secret: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    # Supabase signs user tokens with aud=authenticated
    jwt_audience: str = "authenticated"

    aws_region: str = "ap-south-1"
    s3_bucket: str = "bi-fyp-dev-uploads"

    frontend_origins: str = "http://localhost:3000"

    # Google OAuth
    google_client_id: str = ""
    google_client_secret: str = ""
    google_redirect_uri: str = "http://localhost:8000/api/v1/auth/google/callback"

    # Seed / default admin email for bootstrap and user management
    admin_email: str = "bhattaraiashok101@gmail.com"
    admin_password: str = "Admin@123456"

    # AI / LLM providers
    groq_api_key: str = ""
    groq_model: str = "llama-3.3-70b-versatile"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.0-flash"

    # optional SMTP for the email alert channel (in-app channel needs nothing)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "alerts@bi-dashboard.local"

    # requests per minute per caller (token, else IP); strict paths use 20/min
    rate_limit_per_minute: int = 240

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]

    @property
    def frontend_url(self) -> str:
        return self.cors_origins[0] if self.cors_origins else "http://localhost:3000"


@lru_cache
def get_settings() -> Settings:
    return Settings()
