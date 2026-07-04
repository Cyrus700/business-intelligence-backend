from functools import lru_cache

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    env: str = "dev"
    database_url: str = "postgresql+asyncpg://postgres:postgres@localhost:54329/bi_dev"

    supabase_url: str = ""
    supabase_jwt_secret: str = ""
    supabase_service_key: str = ""
    supabase_anon_key: str = ""
    # Supabase signs user tokens with aud=authenticated
    jwt_audience: str = "authenticated"

    aws_region: str = "ap-south-1"
    s3_bucket: str = "bi-fyp-dev-uploads"

    frontend_origins: str = "http://localhost:3000"

    # optional SMTP for the email alert channel (in-app channel needs nothing)
    smtp_host: str = ""
    smtp_port: int = 587
    smtp_user: str = ""
    smtp_password: str = ""
    smtp_from: str = "alerts@bi-dashboard.local"

    @property
    def cors_origins(self) -> list[str]:
        return [o.strip() for o in self.frontend_origins.split(",") if o.strip()]


@lru_cache
def get_settings() -> Settings:
    return Settings()
