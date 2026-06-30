"""Simple token-based access gate for the whole application."""

from __future__ import annotations

import hashlib
from fastapi import Request
from fastapi.responses import JSONResponse, RedirectResponse, Response
from starlette.middleware.base import BaseHTTPMiddleware, RequestResponseEndpoint

from ..config import Settings


def _hash_token(token: str) -> str:
    return hashlib.sha256(token.encode("utf-8")).hexdigest()


class TokenAuthMiddleware(BaseHTTPMiddleware):
    """Middleware that enforces a shared secret token before allowing access."""

    def __init__(self, app, settings: Settings):
        super().__init__(app)
        self.settings = settings
        self.token_hash = _hash_token(settings.auth_token) if settings.auth_token else None
        self.allowed_paths = {"/auth", "/auth/", "/healthz", "/favicon.ico"}
        self.allowed_prefixes = ("/static",)

    def _is_public_path(self, path: str) -> bool:
        return path in self.allowed_paths or any(path.startswith(prefix) for prefix in self.allowed_prefixes)

    def _has_valid_cookie(self, request: Request) -> bool:
        if not self.token_hash:
            return True
        cookie_value = request.cookies.get(self.settings.auth_cookie_name)
        return bool(cookie_value) and cookie_value == self.token_hash

    def _has_valid_header(self, request: Request) -> bool:
        """Allow Authorization: Bearer <token> as an alternative to the cookie."""
        if not self.token_hash:
            return True
        auth_header = request.headers.get("authorization", "")
        if not auth_header.lower().startswith("bearer "):
            return False
        submitted = auth_header.split(" ", 1)[1].strip()
        if not submitted:
            return False
        submitted_hash = _hash_token(submitted)
        return submitted_hash == self.token_hash or submitted == self.settings.auth_token

    def _wants_html(self, request: Request) -> bool:
        accept_header = request.headers.get("accept", "").lower()
        return "text/html" in accept_header or "*/*" in accept_header

    async def dispatch(self, request: Request, call_next: RequestResponseEndpoint) -> Response:
        if not self.token_hash:
            return await call_next(request)

        path = request.url.path
        if self._is_public_path(path) or self._has_valid_cookie(request) or self._has_valid_header(request):
            return await call_next(request)

        if self._wants_html(request):
            return RedirectResponse(url="/auth", status_code=303)
        return JSONResponse(status_code=401, content={"error": "unauthorized"})
