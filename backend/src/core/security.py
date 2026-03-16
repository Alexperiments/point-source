"""Security utilities for password hashing and verification using bcrypt."""

from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi import HTTPException, Request, Response, status
from fastapi.security import OAuth2PasswordBearer
from fastapi.security.utils import get_authorization_scheme_param
from jose import jwt

from src.core.config import settings


AUTH_COOKIE_NAME = (
    "__Host-point-source-auth"
    if settings.environment != "development"
    else "point-source-auth"
)
AUTH_COOKIE_MAX_AGE_SECONDS = settings.access_token_expire_minutes * 60

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/v1/auth/token",
    scheme_name="Bearer",
    auto_error=False,
)


def hash_password(password: str) -> str:
    """Hash a password using bcrypt.

    Args:
        password: The plain text password to hash.

    Returns:
        The hashed password as a string.

    """
    salt = bcrypt.gensalt()
    hashed = bcrypt.hashpw(password.encode("utf-8"), salt)
    return hashed.decode("utf-8")


def verify_password(plain_password: str, hashed_password: str) -> bool:
    """Verify a password against a bcrypt hash.

    Args:
        plain_password: The plain text password to verify.
        hashed_password: The hashed password to compare against.

    Returns:
        True if the password matches, False otherwise.

    """
    return bcrypt.checkpw(
        plain_password.encode("utf-8"),
        hashed_password.encode("utf-8"),
    )


def create_access_token(
    data: dict,
    expires_delta: timedelta | None = None,
) -> str:
    """Create a JWT access token with issuer and audience claims.

    Args:
        data: The data to encode in the token
            (e.g., {"sub": "user@example.com"}).
        expires_delta: Optional expiration time delta.
            If None, uses default from settings.

    Returns:
        The encoded JWT token string.

    """
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.now(UTC) + expires_delta
    else:
        expire = datetime.now(UTC) + timedelta(minutes=15)
    to_encode.update(
        {
            "exp": expire,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
    )
    return jwt.encode(
        to_encode,
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )


def set_auth_cookie(response: Response, token: str) -> None:
    """Set the browser auth cookie for the current session."""
    response.set_cookie(
        key=AUTH_COOKIE_NAME,
        value=token,
        httponly=True,
        secure=settings.environment != "development",
        samesite="lax",
        path="/",
        max_age=AUTH_COOKIE_MAX_AGE_SECONDS,
    )


def clear_auth_cookie(response: Response) -> None:
    """Clear the browser auth cookie."""
    response.delete_cookie(
        key=AUTH_COOKIE_NAME,
        path="/",
        secure=settings.environment != "development",
        httponly=True,
        samesite="lax",
    )


def get_request_access_token(request: Request) -> str:
    """Resolve an auth token from the cookie first, then bearer auth."""
    cookie_token = request.cookies.get(AUTH_COOKIE_NAME)
    if cookie_token:
        return cookie_token

    authorization = request.headers.get("Authorization")
    scheme, token = get_authorization_scheme_param(authorization)
    if authorization and scheme.lower() == "bearer" and token:
        return token

    raise HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
