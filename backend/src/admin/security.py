"""Admin security helpers."""

import hmac
import secrets

from fastapi import HTTPException, status
from starlette.requests import Request


CSRF_FORM_FIELD = "csrf_token"
CSRF_SESSION_KEY = "admin_csrf_token"


def get_csrf_token(request: Request) -> str:
    """Return the per-session CSRF token, creating it when needed."""
    token = request.session.get(CSRF_SESSION_KEY)
    if not token:
        token = secrets.token_urlsafe(32)
        request.session[CSRF_SESSION_KEY] = token
    return token


def validate_csrf_token(request: Request, submitted_token: str | None) -> None:
    """Reject requests with a missing or mismatched CSRF token."""
    session_token = request.session.get(CSRF_SESSION_KEY)
    if not session_token or not submitted_token:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )
    if not hmac.compare_digest(session_token, submitted_token):
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Invalid CSRF token",
        )
