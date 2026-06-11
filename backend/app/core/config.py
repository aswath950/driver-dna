from __future__ import annotations

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
        case_sensitive=True,
    )

    ENV: str = "local"
    LOG_LEVEL: str = "INFO"

    DATABASE_URL: str = "postgresql+asyncpg://dna:dna@localhost:5432/driver_dna"
    DATABASE_URL_SYNC: str = "postgresql+psycopg://dna:dna@localhost:5432/driver_dna"

    OPENAI_API_KEY: str = ""
    OPENF1_BASE_URL: str = "https://api.openf1.org/v1"

    CORS_ORIGINS: str = "http://localhost:3000"
    COOKIE_SECRET: str = "dev-secret-change-me"
    RATE_LIMIT_PER_MIN: int = 60

    API_TITLE: str = "Driver DNA API"
    API_VERSION: str = "1.0.0"

    @property
    def cors_origins_list(self) -> list[str]:
        return [o.strip() for o in self.CORS_ORIGINS.split(",") if o.strip()]


settings = Settings()
