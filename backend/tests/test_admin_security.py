"""Tests for focused admin security hardening."""

from fastapi import HTTPException
from starlette.requests import Request
from starlette.middleware.sessions import SessionMiddleware

from src.admin import _safe_redirect_target
from src.admin.security import (
    get_csrf_token,
    validate_csrf_token,
)
from src.core.config import settings
from src.main import app


def _session_middleware_options() -> dict[str, object]:
    for middleware in app.user_middleware:
        if middleware.cls is SessionMiddleware:
            return middleware.kwargs

    msg = "SessionMiddleware is not configured"
    raise AssertionError(msg)


def test_safe_redirect_target_rejects_external_urls() -> None:
    """Only local absolute paths should be accepted as login redirects."""
    fallback_url = "https://admin.point-source.org/"

    assert _safe_redirect_target("https://evil.example/steal", fallback_url) == (
        fallback_url
    )
    assert _safe_redirect_target("//evil.example/steal", fallback_url) == fallback_url
    assert _safe_redirect_target("javascript:alert(1)", fallback_url) == fallback_url
    assert _safe_redirect_target("admin/users", fallback_url) == fallback_url


def test_safe_redirect_target_allows_local_paths() -> None:
    """Known-good in-app redirects should remain intact."""
    fallback_url = "https://admin.point-source.org/"

    assert (
        _safe_redirect_target("/users?page=2", fallback_url) == "/users?page=2"
    )


def test_session_middleware_is_hardened() -> None:
    """Admin sessions should use strict cookie settings."""
    options = _session_middleware_options()

    assert options["same_site"] == "strict"
    assert options["https_only"] is (settings.environment != "development")
    assert options["max_age"] == 60 * 60 * 8
    assert options["secret_key"] == settings.admin_session_secret_key.get_secret_value()


def _request_with_session(session: dict[str, str] | None = None) -> Request:
    scope = {
        "type": "http",
        "http_version": "1.1",
        "method": "GET",
        "scheme": "https",
        "path": "/admin",
        "raw_path": b"/admin",
        "query_string": b"",
        "headers": [],
        "client": ("testclient", 123),
        "server": ("testserver", 443),
        "session": session or {},
    }
    return Request(scope)


def test_get_csrf_token_reuses_session_value() -> None:
    """CSRF tokens should be stable for a session."""
    request = _request_with_session()

    first_token = get_csrf_token(request)
    second_token = get_csrf_token(request)

    assert first_token == second_token
    assert request.session["admin_csrf_token"] == first_token


def test_validate_csrf_token_rejects_invalid_values() -> None:
    """Missing or mismatched CSRF tokens must be rejected."""
    request = _request_with_session({"admin_csrf_token": "expected"})

    try:
        validate_csrf_token(request, "wrong")
    except HTTPException as exc:
        assert exc.status_code == 403
    else:
        raise AssertionError("Expected CSRF validation to fail")
