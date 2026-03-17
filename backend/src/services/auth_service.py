"""Auth service."""

import hashlib
import secrets
import uuid
from datetime import UTC, datetime, timedelta
from html import escape
from urllib.parse import urlencode

from jose import JWTError, jwt
from loguru import logger
from redis.asyncio import Redis
from sqlalchemy import select, update
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.security import create_access_token, hash_password, oauth2_scheme
from src.models.email_action_token import EmailActionToken
from src.models.user import User
from src.schemas.user import Token, UserCreate, UserLogin
from src.services.email_service import (
    BaseEmailService,
    EmailDeliveryError,
    EmailMessage,
)
from src.services.user_service import UserService


EMAIL_VERIFICATION_PURPOSE = "verify_email"
PASSWORD_RESET_PURPOSE = "reset_password"  # noqa: S105


class AuthServiceError(Exception):
    """Exception for auth service errors."""


class InvalidCredentialsError(AuthServiceError):
    """Exception for invalid credentials errors."""


class EmailNotVerifiedError(AuthServiceError):
    """Raised when a user has not yet verified their email address."""


class TokenValidationError(AuthServiceError):
    """Exception for token validation errors."""


class EmailActionTokenError(AuthServiceError):
    """Raised when an email action token is invalid or expired."""


class AuthService:
    """Service for authentication operations."""

    def __init__(
        self,
        session: AsyncSession,
        redis: Redis | None = None,
        email_service: BaseEmailService | None = None,
    ) -> None:
        """Initialize the auth service."""
        self.session = session
        self.redis = redis
        self.email_service = email_service
        self.user_service = UserService(session)

    @staticmethod
    def _token_version_key(user_id: uuid.UUID) -> str:
        return f"auth:token_version:{user_id}"

    @staticmethod
    def _hash_action_token(token: str) -> str:
        return hashlib.sha256(token.encode("utf-8")).hexdigest()

    @staticmethod
    def _format_expiration_window(
        *,
        hours: int | None = None,
        minutes: int | None = None,
    ) -> str:
        if hours is not None:
            return f"{hours} hour{'s' if hours != 1 else ''}"
        if minutes is not None:
            return f"{minutes} minute{'s' if minutes != 1 else ''}"
        return "a limited time"

    @staticmethod
    def _coerce_utc_datetime(value: datetime | None) -> datetime | None:
        if value is None or value.tzinfo is not None:
            return value
        return value.replace(tzinfo=UTC)

    def _build_frontend_link(self, path: str, token: str) -> str:
        base_url = str(settings.frontend_base_url).rstrip("/")
        query = urlencode({"token": token})
        return f"{base_url}{path}?{query}"

    @staticmethod
    def _build_auth_email_text(
        *,
        greeting_name: str,
        intro: str,
        action_label: str,
        action_link: str,
        expiration: str,
        ignore_message: str,
    ) -> str:
        return (
            f"Hi {greeting_name},\n\n"
            f"{intro}\n\n"
            f"{action_label}: {action_link}\n\n"
            f"This link expires in {expiration}.\n\n"
            "If the button does not work, copy and paste the full URL into your "
            "browser.\n\n"
            f"{ignore_message}\n"
        )

    @staticmethod
    def _build_auth_email_html(
        *,
        preheader: str,
        eyebrow: str,
        title: str,
        greeting_name: str,
        intro: str,
        action_label: str,
        action_link: str,
        expiration: str,
        ignore_message: str,
    ) -> str:
        escaped_preheader = escape(preheader)
        escaped_eyebrow = escape(eyebrow.upper())
        escaped_title = escape(title)
        escaped_name = escape(greeting_name)
        escaped_intro = escape(intro)
        escaped_label = escape(action_label)
        escaped_link = escape(action_link, quote=True)
        escaped_expiration = escape(expiration)
        escaped_ignore_message = escape(ignore_message)

        return f"""
<!DOCTYPE html>
<html lang="en">
  <body style="margin:0;padding:0;background-color:#f4f1ed;color:#171717;font-family:-apple-system,BlinkMacSystemFont,'Segoe UI',sans-serif;">
    <div style="display:none;max-height:0;overflow:hidden;opacity:0;mso-hide:all;">
      {escaped_preheader}
    </div>
    <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="background:linear-gradient(180deg,#f4f1ed 0%,#ece7e0 100%);padding:32px 16px;">
      <tr>
        <td align="center">
          <table role="presentation" cellpadding="0" cellspacing="0" border="0" width="100%" style="max-width:640px;">
            <tr>
              <td style="padding:0 0 16px 4px;font-size:12px;letter-spacing:0.18em;text-transform:uppercase;color:#58534d;font-weight:700;">
                Point-source
              </td>
            </tr>
            <tr>
              <td style="background-color:#fbfaf8;border:1px solid #d8d1c7;border-radius:28px;padding:40px 36px;box-shadow:0 18px 40px rgba(23,23,23,0.08);">
                <div style="font-size:11px;letter-spacing:0.2em;text-transform:uppercase;color:#4f46e5;font-weight:700;margin-bottom:18px;">
                  {escaped_eyebrow}
                </div>
                <h1 style="margin:0 0 16px;font-size:32px;line-height:1.15;color:#171717;font-family:Georgia,'Times New Roman',serif;font-weight:600;">
                  {escaped_title}
                </h1>
                <p style="margin:0 0 12px;font-size:16px;line-height:1.7;color:#2f2a24;">
                  Hi {escaped_name},
                </p>
                <p style="margin:0 0 28px;font-size:16px;line-height:1.7;color:#2f2a24;">
                  {escaped_intro}
                </p>
                <table role="presentation" cellpadding="0" cellspacing="0" border="0" style="margin:0 0 28px;">
                  <tr>
                    <td align="center" bgcolor="#4f46e5" style="border-radius:999px;">
                      <a href="{escaped_link}" style="display:inline-block;padding:14px 24px;font-size:15px;font-weight:700;line-height:1;text-decoration:none;color:#f5f7ff;">
                        {escaped_label}
                      </a>
                    </td>
                  </tr>
                </table>
                <div style="margin:0 0 24px;padding:16px 18px;border-radius:20px;background-color:#f1ede7;border:1px solid #ddd4ca;">
                  <p style="margin:0 0 10px;font-size:13px;line-height:1.6;color:#5e564d;font-weight:700;">
                    Link expires in {escaped_expiration}
                  </p>
                  <p style="margin:0;font-size:13px;line-height:1.7;color:#5e564d;word-break:break-all;">
                    If the button does not work, copy and paste this URL into your browser:<br />
                    <a href="{escaped_link}" style="color:#4f46e5;text-decoration:underline;">{escaped_link}</a>
                  </p>
                </div>
                <p style="margin:0;font-size:14px;line-height:1.7;color:#6b645d;">
                  {escaped_ignore_message}
                </p>
              </td>
            </tr>
            <tr>
              <td style="padding:18px 8px 0;font-size:12px;line-height:1.7;color:#6f685f;text-align:center;">
                Transactional security email from Point-source. This mailbox is not monitored.
              </td>
            </tr>
          </table>
        </td>
      </tr>
    </table>
  </body>
</html>
""".strip()

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
                "Token audience validation failed: audience claim missing or invalid",
            )
        return payload

    async def _resolve_user_from_subject(self, token_subject: object) -> User | None:
        try:
            user_id = uuid.UUID(str(token_subject))
        except (TypeError, ValueError):
            return await self.user_service.get_user_by_email(str(token_subject))
        return await self.user_service.get_user_by_id(user_id)

    @staticmethod
    def _decode_token_version(payload: dict) -> int:
        token_version_raw = payload.get("tv", 1)
        try:
            return int(token_version_raw)
        except (TypeError, ValueError):
            raise TokenValidationError("Token version claim is invalid") from None

    async def _invalidate_email_action_tokens(
        self,
        user_id: uuid.UUID,
        purpose: str,
    ) -> None:
        now = datetime.now(UTC)
        await self.session.execute(
            update(EmailActionToken)
            .where(
                EmailActionToken.user_id == user_id,
                EmailActionToken.purpose == purpose,
                EmailActionToken.consumed_at.is_(None),
            )
            .values(consumed_at=now),
        )

    async def _create_email_action_token(
        self,
        user: User,
        purpose: str,
        *,
        email: str,
        expires_delta: timedelta,
    ) -> str:
        await self._invalidate_email_action_tokens(user.id, purpose)

        raw_token = secrets.token_urlsafe(32)
        token = EmailActionToken(
            user_id=user.id,
            purpose=purpose,
            token_hash=self._hash_action_token(raw_token),
            email=email,
            expires_at=datetime.now(UTC) + expires_delta,
        )
        self.session.add(token)
        await self.session.flush()
        return raw_token

    async def _get_valid_email_action_token(
        self,
        raw_token: str,
        purpose: str,
    ) -> tuple[EmailActionToken, User]:
        token_hash = self._hash_action_token(raw_token)
        result = await self.session.execute(
            select(EmailActionToken).where(
                EmailActionToken.token_hash == token_hash,
                EmailActionToken.purpose == purpose,
            ),
        )
        token = result.scalar_one_or_none()
        now = datetime.now(UTC)
        expires_at = (
            self._coerce_utc_datetime(token.expires_at) if token is not None else None
        )
        if (
            token is None
            or token.consumed_at is not None
            or expires_at is None
            or expires_at < now
        ):
            raise EmailActionTokenError("Token is invalid or has expired.")

        user = await self.user_service.get_user_by_id(token.user_id)
        if user is None or user.is_deleted or user.email != token.email:
            raise EmailActionTokenError("Token is invalid or has expired.")

        return token, user

    def _require_email_service(self) -> BaseEmailService:
        if self.email_service is None:
            msg = "Email delivery is not configured."
            raise EmailDeliveryError(msg)
        return self.email_service

    @staticmethod
    def _verification_email_message(user: User, verification_link: str) -> EmailMessage:
        expiration = AuthService._format_expiration_window(
            hours=settings.email_verification_token_expire_hours,
        )
        intro = (
            "Confirm your email address to finish creating your Point-source account."
        )
        ignore_message = (
            "If you did not create a Point-source account, you can ignore this email."
        )
        text_body = AuthService._build_auth_email_text(
            greeting_name=user.name,
            intro=intro,
            action_label="Verify your email",
            action_link=verification_link,
            expiration=expiration,
            ignore_message=ignore_message,
        )
        html_body = AuthService._build_auth_email_html(
            preheader="Confirm your email to activate your Point-source account.",
            eyebrow="Email verification",
            title="Verify your email",
            greeting_name=user.name,
            intro=intro,
            action_label="Verify email",
            action_link=verification_link,
            expiration=expiration,
            ignore_message=ignore_message,
        )
        return EmailMessage(
            to_email=user.email,
            subject="Verify your Point-source email",
            text_body=text_body,
            html_body=html_body,
            tag="verify-email",
        )

    @staticmethod
    def _password_reset_email_message(user: User, reset_link: str) -> EmailMessage:
        expiration = AuthService._format_expiration_window(
            minutes=settings.password_reset_token_expire_minutes,
        )
        intro = "We received a request to reset your Point-source password."
        ignore_message = (
            "If you did not request a password reset, you can ignore this email."
        )
        text_body = AuthService._build_auth_email_text(
            greeting_name=user.name,
            intro=intro,
            action_label="Reset your password",
            action_link=reset_link,
            expiration=expiration,
            ignore_message=ignore_message,
        )
        html_body = AuthService._build_auth_email_html(
            preheader="Reset your Point-source password securely.",
            eyebrow="Password reset",
            title="Reset your password",
            greeting_name=user.name,
            intro=intro,
            action_label="Reset password",
            action_link=reset_link,
            expiration=expiration,
            ignore_message=ignore_message,
        )
        return EmailMessage(
            to_email=user.email,
            subject="Reset your Point-source password",
            text_body=text_body,
            html_body=html_body,
            tag="password-reset",
        )

    async def _send_verification_email(self, user: User) -> None:
        raw_token = await self._create_email_action_token(
            user,
            EMAIL_VERIFICATION_PURPOSE,
            email=user.email,
            expires_delta=timedelta(
                hours=settings.email_verification_token_expire_hours,
            ),
        )
        verification_link = self._build_frontend_link(
            "/auth/verify-email",
            raw_token,
        )
        message = self._verification_email_message(user, verification_link)
        await self._require_email_service().send(message)

    async def _send_password_reset_email(self, user: User) -> None:
        raw_token = await self._create_email_action_token(
            user,
            PASSWORD_RESET_PURPOSE,
            email=user.email,
            expires_delta=timedelta(
                minutes=settings.password_reset_token_expire_minutes,
            ),
        )
        reset_link = self._build_frontend_link("/auth/reset-password", raw_token)
        message = self._password_reset_email_message(user, reset_link)
        await self._require_email_service().send(message)

    async def register_user(self, user_data: UserCreate) -> User:
        """Register a new user and send a verification email."""
        user = await self.user_service.create_user(user_data)
        await self._send_verification_email(user)
        return user

    async def resend_verification_email(self, email: str) -> None:
        """Resend the email verification link when the account is unverified."""
        user = await self.user_service.get_user_by_email(email)
        if user is None or user.is_deleted or user.email_verified:
            return

        try:
            await self._send_verification_email(user)
        except EmailDeliveryError:
            logger.exception(
                "Failed to deliver verification email for user {}",
                user.id,
            )

    async def verify_email(self, raw_token: str) -> User:
        """Consume a verification token and mark the user email as verified."""
        token_hash = self._hash_action_token(raw_token)
        result = await self.session.execute(
            select(EmailActionToken).where(
                EmailActionToken.token_hash == token_hash,
                EmailActionToken.purpose == EMAIL_VERIFICATION_PURPOSE,
            ),
        )
        token = result.scalar_one_or_none()
        if token is None:
            raise EmailActionTokenError("Token is invalid or has expired.")

        user = await self.user_service.get_user_by_id(token.user_id)
        now = datetime.now(UTC)
        expires_at = self._coerce_utc_datetime(token.expires_at)
        consumed_at = self._coerce_utc_datetime(token.consumed_at)
        verified_at = (
            self._coerce_utc_datetime(user.email_verified_at)
            if user is not None
            else None
        )
        if (
            user is not None
            and not user.is_deleted
            and user.email_verified
            and user.email == token.email
            and consumed_at is not None
            and verified_at is not None
            and consumed_at == verified_at
            and expires_at is not None
            and expires_at >= now
        ):
            return user

        token, user = await self._get_valid_email_action_token(
            raw_token,
            EMAIL_VERIFICATION_PURPOSE,
        )
        now = datetime.now(UTC)
        token.consumed_at = now
        user.email_verified = True
        user.email_verified_at = now
        await self._invalidate_email_action_tokens(user.id, EMAIL_VERIFICATION_PURPOSE)
        await self.session.flush()
        return user

    async def request_password_reset(self, email: str) -> None:
        """Send a password reset link without revealing whether an account exists."""
        user = await self.user_service.get_user_by_email(email)
        if user is None or user.is_deleted:
            return

        try:
            await self._send_password_reset_email(user)
        except EmailDeliveryError:
            logger.exception(
                "Failed to deliver password reset email for user {}",
                user.id,
            )

    async def reset_password(self, raw_token: str, new_password: str) -> User:
        """Consume a password reset token and rotate the stored password."""
        token, user = await self._get_valid_email_action_token(
            raw_token,
            PASSWORD_RESET_PURPOSE,
        )
        now = datetime.now(UTC)
        token.consumed_at = now
        user.hashed_password = hash_password(new_password)
        if not user.email_verified:
            user.email_verified = True
            user.email_verified_at = now
        await self._invalidate_email_action_tokens(user.id, PASSWORD_RESET_PURPOSE)
        await self._invalidate_email_action_tokens(user.id, EMAIL_VERIFICATION_PURPOSE)
        await self.session.flush()
        await self.revoke_user_tokens(user)
        return user

    async def authenticate_user(self, user_data: UserLogin) -> User:
        """Authenticate a user with email and password."""
        user = await self.user_service.verify_user_password(
            user_data.email,
            user_data.password.get_secret_value(),
        )
        if not user or user.is_deleted:
            raise InvalidCredentialsError("Incorrect email or password")
        if not user.email_verified:
            raise EmailNotVerifiedError(
                "Email address not verified. Check your inbox or request another "
                "verification email.",
            )
        return user

    def create_token(
        self,
        user: User,
        expires_delta: timedelta | None = None,
        *,
        token_version: int = 1,
    ) -> Token:
        """Create an access token for a user."""
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
        """Login a user and return an access token."""
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
        """Get a user from a JWT token with issuer and audience validation."""
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
        """Validate a JWT token and return the user if valid."""
        try:
            return True, await self.get_user_from_token(token)
        except TokenValidationError:
            return False, None
