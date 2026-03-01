"""Auth service."""

import uuid
from datetime import timedelta

from jose import JWTError, jwt
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.security import create_access_token, oauth2_scheme
from src.models.user import User
from src.schemas.user import Token, UserCreate, UserLogin
from src.services.user_service import (
    UserService,
    UserServiceError,
)


class AuthServiceError(Exception):
    """Exception for auth service errors."""


class InvalidCredentialsError(AuthServiceError):
    """Exception for invalid credentials errors."""


class TokenValidationError(AuthServiceError):
    """Exception for token validation errors."""


class AuthService:
    """Service for authentication operations."""

    def __init__(
        self,
        session: AsyncSession,
        redis: Redis | None = None,
    ) -> None:
        """Initialize the auth service.

        Args:
            session: The database session to use.
            redis: Optional Redis client used for token revocation/versioning.

        """
        self.session = session
        self.redis = redis
        self.user_service = UserService(session)

    @staticmethod
    def _token_version_key(user_id: uuid.UUID) -> str:
        return f"auth:token_version:{user_id}"

    async def _get_token_version(self, user_id: uuid.UUID) -> int:
        if self.redis is None:
            return 1

        key = self._token_version_key(user_id)
        raw_version = await self.redis.get(key)
        if raw_version is None:
            await self.redis.set(key, "1")
            return 1
        try:
            return int(raw_version)
        except (TypeError, ValueError):
            await self.redis.set(key, "1")
            return 1

    @staticmethod
    def _decode_payload(token: str) -> dict:
        try:
            payload = jwt.decode(
                token,
                settings.jwt_secret_key.get_secret_value(),
                algorithms=[settings.jwt_algorithm],
                issuer=settings.jwt_issuer,
                audience=settings.jwt_audience,
                options={
                    "verify_aud": True,
                    "leeway": settings.clock_skew_leeway_seconds,
                },
            )
        except JWTError as e:
            error_msg = str(e)
            if "issuer" in error_msg.lower():
                raise TokenValidationError(
                    f"Token issuer validation failed: {e!s}",
                ) from e
            if "audience" in error_msg.lower() or "aud" in error_msg.lower():
                raise TokenValidationError(
                    f"Token audience validation failed: {e!s}",
                ) from e
            raise TokenValidationError(f"Invalid token: {e!s}") from e
        if payload.get("sub") is None:
            raise TokenValidationError("Token missing subject")
        if payload.get("aud") != settings.jwt_audience:
            raise TokenValidationError(
                "Token audience validation failed: "
                "audience claim missing or invalid",
            )
        return payload

    async def _resolve_user_from_subject(self, token_subject: object) -> User | None:
        try:
            user_id = uuid.UUID(str(token_subject))
        except (TypeError, ValueError):
            # Backward-compatible fallback for legacy tokens using email in `sub`.
            return await self.user_service.get_user_by_email(str(token_subject))
        return await self.user_service.get_user_by_id(user_id)

    @staticmethod
    def _decode_token_version(payload: dict) -> int:
        token_version_raw = payload.get("tv", 1)
        try:
            return int(token_version_raw)
        except (TypeError, ValueError):
            raise TokenValidationError("Token version claim is invalid") from None

    async def register_user(self, user_data: UserCreate) -> User:
        """Register a new user.

        Args:
            user_data: The user data to create.

        Returns:
            The created user.

        Raises:
            UserAlreadyExistsError: If the email already exists.
            FailedToCreateUserError: If user creation fails.

        """
        try:
            return await self.user_service.create_user(user_data)
        except UserServiceError as e:
            raise e from e

    async def authenticate_user(self, user_data: UserLogin) -> User:
        """Authenticate a user with email and password.

        Args:
            user_data: The user login credentials.

        Returns:
            The authenticated user.

        Raises:
            InvalidCredentialsError: If credentials are invalid.

        """
        user = await self.user_service.verify_user_password(
            user_data.email,
            user_data.password.get_secret_value(),
        )
        if not user or user.is_deleted:
            raise InvalidCredentialsError("Incorrect email or password")
        return user

    def create_token(
        self,
        user: User,
        expires_delta: timedelta | None = None,
        *,
        token_version: int = 1,
    ) -> Token:
        """Create an access token for a user.

        Args:
            user: The user to create a token for.
            expires_delta: Optional expiration time delta.
                If None, uses default from settings.
            token_version: Token version included in the JWT for revocation checks.

        Returns:
            A Token object with access_token and token_type.

        """
        if expires_delta is None:
            expires_delta = timedelta(
                minutes=settings.access_token_expire_minutes,
            )

        access_token = create_access_token(
            data={
                "sub": str(user.id),
                "tv": token_version,
            },
            expires_delta=expires_delta,
        )
        return Token(
            access_token=access_token,
            token_type=oauth2_scheme.scheme_name,
        )

    async def create_token_for_user(
        self,
        user: User,
        expires_delta: timedelta | None = None,
    ) -> Token:
        """Create an access token using the current persisted token version."""
        token_version = await self._get_token_version(user.id)
        return self.create_token(
            user,
            expires_delta=expires_delta,
            token_version=token_version,
        )

    async def login(self, user_data: UserLogin) -> Token:
        """Login a user and return an access token.

        Args:
            user_data: The user login credentials.

        Returns:
            A Token object with access_token and token_type.

        Raises:
            InvalidCredentialsError: If credentials are invalid.

        """
        user = await self.authenticate_user(user_data)
        return await self.create_token_for_user(user)

    async def revoke_user_tokens(self, user: User) -> int:
        """Invalidate existing tokens for a user by bumping token version."""
        if self.redis is None:
            return 1

        key = self._token_version_key(user.id)
        await self._get_token_version(user.id)
        return int(await self.redis.incr(key))

    async def get_user_from_token(self, token: str) -> User:
        """Get a user from a JWT token with issuer and audience validation.

        Args:
            token: The JWT token string.

        Returns:
            The user associated with the token.

        Raises:
            TokenValidationError: If the token is invalid,
                has wrong issuer/audience, or user not found.

        """
        payload = self._decode_payload(token)
        token_subject = payload.get("sub")
        user = await self._resolve_user_from_subject(token_subject)
        if user is None or user.is_deleted:
            raise TokenValidationError("User not found")

        token_version = self._decode_token_version(payload)
        current_version = await self._get_token_version(user.id)
        if token_version != current_version:
            raise TokenValidationError("Token has been revoked")

        return user

    async def validate_token(self, token: str) -> tuple[bool, User | None]:
        """Validate a JWT token and return the user if valid.

        Args:
            token: The JWT token string.

        Returns:
            A tuple of (is_valid, user). user is None if token is invalid.

        """
        try:
            return True, await self.get_user_from_token(token)
        except TokenValidationError:
            return False, None
