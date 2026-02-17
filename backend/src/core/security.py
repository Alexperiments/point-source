"""Security utilities for password hashing and verification using bcrypt."""

from datetime import UTC, datetime, timedelta

import bcrypt
from fastapi.security import OAuth2PasswordBearer
from jose import jwt

from src.core.config import settings


oauth2_scheme = OAuth2PasswordBearer(tokenUrl="/token", scheme_name="Bearer")


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
