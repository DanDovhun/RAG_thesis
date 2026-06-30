from __future__ import annotations

from fastapi import APIRouter, Depends, Form, Request, Response
from fastapi.responses import HTMLResponse, RedirectResponse

from ..dependencies import get_app_settings, get_templates
from ..core.comparison_legacy import MODEL_REGISTRY_JSON, DEFAULT_AGENT_PROMPT_TEXT, DEFAULT_MODEL
from ..middleware.auth import _hash_token

router = APIRouter()


@router.get("/auth", response_class=HTMLResponse)
async def auth_form(
    request: Request,
    templates=Depends(get_templates),
    settings=Depends(get_app_settings),
) -> Response:
    if not settings.auth_token:
        return RedirectResponse(url="/", status_code=303)

    return templates.TemplateResponse(
        "auth.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "error": request.query_params.get("error", ""),
        },
    )


@router.post("/auth", response_class=HTMLResponse)
async def auth_submit(
    request: Request,
    token: str = Form(""),
    templates=Depends(get_templates),
    settings=Depends(get_app_settings),
) -> Response:
    if not settings.auth_token:
        return RedirectResponse(url="/", status_code=303)

    submitted = token.strip()
    if submitted != settings.auth_token:
        return templates.TemplateResponse(
            "auth.html",
            {
                "request": request,
                "app_name": settings.app_name,
                "error": "Neplatný přístupový kód.",
            },
            status_code=401,
        )

    response = RedirectResponse(url="/", status_code=303)
    response.set_cookie(
        settings.auth_cookie_name,
        _hash_token(settings.auth_token),
        max_age=settings.auth_cookie_max_age_seconds,
        httponly=True,
        secure=settings.auth_cookie_secure,
        samesite="lax",
    )
    return response


@router.get("/", response_class=HTMLResponse)
async def show_form(
    request: Request,
    templates=Depends(get_templates),
    settings=Depends(get_app_settings),
) -> HTMLResponse:
    return templates.TemplateResponse(
        "compare.html",
        {
            "request": request,
            "app_name": settings.app_name,
            "model_registry_json": MODEL_REGISTRY_JSON,
            "default_agent_model": DEFAULT_MODEL,
            "default_agent_prompt": DEFAULT_AGENT_PROMPT_TEXT,
        },
    )
