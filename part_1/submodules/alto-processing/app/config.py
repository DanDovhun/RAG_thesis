from typing import Optional

from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    app_name: str = "Alto Processing Web"
    environment: str = "development"
    auth_token: Optional[str] = None
    auth_cookie_name: str = "alto_auth"
    auth_cookie_max_age_seconds: int = 60 * 60 * 24 * 30  # 30 dní
    auth_cookie_secure: bool = False

    class Config:
        env_prefix = "ALTO_WEB_"


def get_settings() -> Settings:
    return Settings()
