"""Common dependency providers for FastAPI routes."""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from fastapi.templating import Jinja2Templates

from .config import Settings, get_settings
from .services.alto import AltoProcessingService


def get_templates() -> Jinja2Templates:
    template_dir = Path(__file__).resolve().parent / "templates"
    return Jinja2Templates(directory=str(template_dir))


@lru_cache
def get_alto_service() -> AltoProcessingService:
    return AltoProcessingService()


def get_app_settings() -> Settings:
    return get_settings()
