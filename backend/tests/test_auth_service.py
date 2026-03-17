"""Tests for auth service."""

from datetime import UTC, datetime, timedelta

import pytest
from jose import jwt
from pydantic import SecretStr
from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.config import settings
from src.core.security import hash_password
from src.models.email_action_token import EmailActionToken
from src.models.user import User
from src.schemas.user import UserCreate, UserLogin
from src.services.auth_service import (
    AuthService,
    EmailActionTokenError,
    EmailNotVerifiedError,
    InvalidCredentialsError,
    TokenValidationError,
)
from src.services.user_service import UserAlreadyExistsError


class _FakeRedis:
    def __init__(self) -> None:
        self._values: dict[str, str] = {}

    async def get(self, key: str) -> str | None:
        return self._values.get(key)

    async def set(self, key: str, value: str) -> bool:
        self._values[key] = value
        return True

    async def incr(self, key: str) -> int:
        next_value = int(self._values.get(key, "0")) + 1
        self._values[key] = str(next_value)
        return next_value


@pytest.mark.asyncio
async def test_register_user_success(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Test successful user registration."""
    auth_service = AuthService(db_session, email_service=fake_email_service)

    user_data = UserCreate(
        name="Test User",
        email="test@example.com",
        password=SecretStr("SecurePass123"),
    )

    user = await auth_service.register_user(user_data)

    assert user.id is not None
    assert user.name == "Test User"
    assert user.email == "test@example.com"
    assert user.is_deleted is False
    assert user.email_verified is False
    assert len(fake_email_service.messages) == 1


@pytest.mark.asyncio
async def test_verification_email_template_escapes_user_name(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Verification emails should escape user-controlled content in HTML."""
    auth_service = AuthService(db_session, email_service=fake_email_service)

    await auth_service.register_user(
        UserCreate(
            name="<Editor & Co.>",
            email="escape@example.com",
            password=SecretStr("SecurePass123"),
        )
    )

    message = fake_email_service.messages[0]

    assert message.subject == "Verify your Point-source email"
    assert message.html_body is not None
    assert "Verify email" in message.html_body
    assert "&lt;Editor &amp; Co.&gt;" in message.html_body
    assert "<Editor & Co.>" not in message.html_body
    assert "copy and paste this URL into your browser" in message.html_body
    assert "token=" in message.text_body


@pytest.mark.asyncio
async def test_password_reset_email_template_includes_fallback_link(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Password reset emails should include both CTA copy and the raw fallback URL."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    user = await auth_service.register_user(
        UserCreate(
            name="Reset User",
            email="reset-template@example.com",
            password=SecretStr("SecurePass123"),
        )
    )
    fake_email_service.messages.clear()

    await auth_service.request_password_reset(user.email)

    message = fake_email_service.messages[0]

    assert message.subject == "Reset your Point-source password"
    assert message.html_body is not None
    assert "Reset password" in message.html_body
    assert "copy and paste this URL into your browser" in message.html_body
    assert "Transactional security email from Point-source" in message.html_body
    assert "token=" in message.text_body


@pytest.mark.asyncio
async def test_register_user_duplicate_email(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Test registration with duplicate email raises error."""
    auth_service = AuthService(db_session, email_service=fake_email_service)

    user_data = UserCreate(
        name="Test User",
        email="test@example.com",
        password=SecretStr("SecurePass123"),
    )

    # Create first user
    await auth_service.register_user(user_data)

    # Try to register again with same email
    with pytest.raises(UserAlreadyExistsError):
        await auth_service.register_user(user_data)


@pytest.mark.asyncio
async def test_authenticate_user_success(db_session: AsyncSession) -> None:
    """Test successful user authentication."""
    auth_service = AuthService(db_session)

    # Create a user first
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Authenticate
    user_data = UserLogin(
        email="test@example.com",
        password=SecretStr(password),
    )
    authenticated_user = await auth_service.authenticate_user(user_data)

    assert authenticated_user.id == user.id
    assert authenticated_user.email == user.email


@pytest.mark.asyncio
async def test_authenticate_user_invalid_email(
    db_session: AsyncSession,
) -> None:
    """Test authentication with non-existent email fails."""
    auth_service = AuthService(db_session)

    user_data = UserLogin(
        email="nonexistent@example.com",
        password=SecretStr("SecurePass123"),
    )

    with pytest.raises(InvalidCredentialsError):
        await auth_service.authenticate_user(user_data)


@pytest.mark.asyncio
async def test_authenticate_user_invalid_password(
    db_session: AsyncSession,
) -> None:
    """Test authentication with incorrect password fails."""
    auth_service = AuthService(db_session)

    # Create a user first
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()

    # Try to authenticate with wrong password
    user_data = UserLogin(
        email="test@example.com",
        password=SecretStr("WrongPassword123"),
    )

    with pytest.raises(InvalidCredentialsError):
        await auth_service.authenticate_user(user_data)


@pytest.mark.asyncio
async def test_authenticate_user_deleted_user_fails(
    db_session: AsyncSession,
) -> None:
    """Deleted users should not authenticate."""
    auth_service = AuthService(db_session)
    user = User(
        name="Deleted User",
        email="deleted@example.com",
        hashed_password=hash_password("SecurePass123"),
        is_deleted=True,
    )
    db_session.add(user)
    await db_session.commit()

    with pytest.raises(InvalidCredentialsError):
        await auth_service.authenticate_user(
            UserLogin(
                email="deleted@example.com",
                password=SecretStr("SecurePass123"),
            ),
        )


@pytest.mark.asyncio
async def test_authenticate_user_unverified_user_fails(
    db_session: AsyncSession,
) -> None:
    """Unverified users must not receive login sessions."""
    auth_service = AuthService(db_session)
    user = User(
        name="Pending User",
        email="pending@example.com",
        hashed_password=hash_password("SecurePass123"),
        email_verified=False,
    )
    db_session.add(user)
    await db_session.commit()

    with pytest.raises(EmailNotVerifiedError):
        await auth_service.authenticate_user(
            UserLogin(
                email="pending@example.com",
                password=SecretStr("SecurePass123"),
            ),
        )


@pytest.mark.asyncio
async def test_create_token_success(db_session: AsyncSession) -> None:
    """Test successful token creation."""
    auth_service = AuthService(db_session)

    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    token = auth_service.create_token(user)

    assert token.access_token is not None
    assert token.token_type == "Bearer"
    assert len(token.access_token) > 0

    # Verify token can be decoded
    payload = jwt.decode(
        token.access_token,
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    assert payload["sub"] == str(user.id)
    assert "exp" in payload


@pytest.mark.asyncio
async def test_create_token_with_custom_expires_delta(
    db_session: AsyncSession,
) -> None:
    """Test token creation with custom expiration time."""
    auth_service = AuthService(db_session)

    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    custom_delta = timedelta(minutes=60)
    token = auth_service.create_token(user, expires_delta=custom_delta)

    # Verify token can be decoded
    payload = jwt.decode(
        token.access_token,
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    assert payload["sub"] == str(user.id)

    # Check expiration is approximately 60 minutes from now
    exp_timestamp = payload["exp"]

    expected_exp = datetime.now(UTC) + custom_delta
    actual_exp = datetime.fromtimestamp(exp_timestamp, tz=UTC)
    # Allow 5 second tolerance
    assert abs((actual_exp - expected_exp).total_seconds()) < 5


@pytest.mark.asyncio
async def test_login_success(db_session: AsyncSession) -> None:
    """Test successful login."""
    auth_service = AuthService(db_session)

    # Create a user first
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Login
    user_data = UserLogin(
        email="test@example.com",
        password=SecretStr(password),
    )
    token = await auth_service.login(user_data)

    assert token.access_token is not None
    assert token.token_type == "Bearer"

    # Verify token contains correct email
    payload = jwt.decode(
        token.access_token,
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    assert payload["sub"] == str(user.id)


@pytest.mark.asyncio
async def test_login_invalid_credentials(db_session: AsyncSession) -> None:
    """Test login with invalid credentials fails."""
    auth_service = AuthService(db_session)

    user_data = UserLogin(
        email="nonexistent@example.com",
        password=SecretStr("SecurePass123"),
    )

    with pytest.raises(InvalidCredentialsError):
        await auth_service.login(user_data)


@pytest.mark.asyncio
async def test_verify_email_marks_user_as_verified(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Verification should mark the account as active for login."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    user = await auth_service.register_user(
        UserCreate(
            name="Verify User",
            email="verify@example.com",
            password=SecretStr("SecurePass123"),
        )
    )

    message = fake_email_service.messages[0]
    raw_token = message.text_body.split("token=")[1].split()[0]

    verified_user = await auth_service.verify_email(raw_token)

    assert verified_user.id == user.id
    assert verified_user.email_verified is True
    assert verified_user.email_verified_at is not None


@pytest.mark.asyncio
async def test_verify_email_is_idempotent_for_already_verified_user(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Submitting the same verify link twice should still be safe."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    await auth_service.register_user(
        UserCreate(
            name="Verify User",
            email="verify-twice@example.com",
            password=SecretStr("SecurePass123"),
        )
    )

    message = fake_email_service.messages[0]
    raw_token = message.text_body.split("token=")[1].split()[0]

    first_user = await auth_service.verify_email(raw_token)
    second_user = await auth_service.verify_email(raw_token)

    assert first_user.id == second_user.id
    assert second_user.email_verified is True


@pytest.mark.asyncio
async def test_verify_email_rejects_invalidated_older_token_after_resend(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """A resent verification link must invalidate previously issued links."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    user = await auth_service.register_user(
        UserCreate(
            name="Verify User",
            email="verify-resend@example.com",
            password=SecretStr("SecurePass123"),
        )
    )

    first_token = fake_email_service.messages[0].text_body.split("token=")[1].split()[0]

    await auth_service.resend_verification_email(user.email)

    second_token = fake_email_service.messages[-1].text_body.split("token=")[1].split()[0]
    await auth_service.verify_email(second_token)

    with pytest.raises(EmailActionTokenError):
        await auth_service.verify_email(first_token)


@pytest.mark.asyncio
async def test_verify_email_rejects_expired_reuse_after_success(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """An already-consumed verify token should not remain reusable after expiry."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    await auth_service.register_user(
        UserCreate(
            name="Verify User",
            email="verify-expired@example.com",
            password=SecretStr("SecurePass123"),
        )
    )

    raw_token = fake_email_service.messages[0].text_body.split("token=")[1].split()[0]
    verified_user = await auth_service.verify_email(raw_token)

    verified_user.email_verified_at = datetime.now(UTC) - timedelta(days=2)

    token_hash = auth_service._hash_action_token(raw_token)  # noqa: SLF001
    result = await db_session.execute(
        select(EmailActionToken).where(
            EmailActionToken.token_hash == token_hash,
            EmailActionToken.purpose == "verify_email",
        )
    )
    token = result.scalar_one()
    token.expires_at = datetime.now(UTC) - timedelta(minutes=1)
    token.consumed_at = verified_user.email_verified_at
    await db_session.flush()

    with pytest.raises(EmailActionTokenError):
        await auth_service.verify_email(raw_token)


@pytest.mark.asyncio
async def test_reset_password_marks_token_as_one_time_use(
    db_session: AsyncSession,
    fake_email_service,
) -> None:
    """Password reset tokens should be invalid after first successful use."""
    auth_service = AuthService(db_session, email_service=fake_email_service)
    user = await auth_service.register_user(
        UserCreate(
            name="Reset User",
            email="reset@example.com",
            password=SecretStr("SecurePass123"),
        )
    )
    await auth_service.request_password_reset(user.email)

    reset_message = fake_email_service.messages[-1]
    raw_token = reset_message.text_body.split("token=")[1].split()[0]

    await auth_service.reset_password(raw_token, "NewSecurePass456")

    with pytest.raises(EmailActionTokenError):
        await auth_service.reset_password(raw_token, "AnotherPass789")


@pytest.mark.asyncio
async def test_get_user_from_token_success(db_session: AsyncSession) -> None:
    """Test getting user from valid token."""
    auth_service = AuthService(db_session)

    # Create a user
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Create a token
    token = auth_service.create_token(user)

    # Get user from token
    retrieved_user = await auth_service.get_user_from_token(token.access_token)

    assert retrieved_user.id == user.id
    assert retrieved_user.email == user.email
    assert retrieved_user.name == user.name


@pytest.mark.asyncio
async def test_get_user_from_token_invalid_token(
    db_session: AsyncSession,
) -> None:
    """Test getting user from invalid token fails."""
    auth_service = AuthService(db_session)

    invalid_token = "invalid.token.here"

    with pytest.raises(TokenValidationError):
        await auth_service.get_user_from_token(invalid_token)


@pytest.mark.asyncio
async def test_get_user_from_token_missing_subject(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token without subject fails."""
    auth_service = AuthService(db_session)

    # Create a token without 'sub' field
    token_without_sub = jwt.encode(
        {
            "exp": 9999999999,  # Far future expiration
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_without_sub)

    assert "Token missing subject" in str(exc_info.value)


@pytest.mark.asyncio
async def test_get_user_from_token_user_not_found(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token when user doesn't exist."""
    auth_service = AuthService(db_session)

    # Create a token for a non-existent user
    token_for_nonexistent = jwt.encode(
        {
            "sub": "nonexistent@example.com",
            "exp": 9999999999,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_for_nonexistent)

    assert "not found" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_get_user_from_token_expired_token(
    db_session: AsyncSession,
) -> None:
    """Test getting user from expired token fails."""
    auth_service = AuthService(db_session)

    # Create an expired token
    expired_token = jwt.encode(
        {
            "sub": "test@example.com",
            # Expired 1 hour ago
            "exp": datetime.now(UTC) - timedelta(hours=1),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError):
        await auth_service.get_user_from_token(expired_token)


@pytest.mark.asyncio
async def test_validate_token_success(db_session: AsyncSession) -> None:
    """Test token validation with valid token."""
    auth_service = AuthService(db_session)

    # Create a user
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Create a token
    token = auth_service.create_token(user)

    # Validate token
    is_valid, retrieved_user = await auth_service.validate_token(
        token.access_token
    )

    assert is_valid is True
    assert retrieved_user is not None
    assert retrieved_user.id == user.id
    assert retrieved_user.email == user.email


@pytest.mark.asyncio
async def test_validate_token_invalid_token(db_session: AsyncSession) -> None:
    """Test token validation with invalid token."""
    auth_service = AuthService(db_session)

    invalid_token = "invalid.token.here"

    is_valid, user = await auth_service.validate_token(invalid_token)

    assert is_valid is False
    assert user is None


@pytest.mark.asyncio
async def test_validate_token_expired_token(db_session: AsyncSession) -> None:
    """Test token validation with expired token."""
    auth_service = AuthService(db_session)

    # Create an expired token
    expired_token = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": datetime.now(UTC) - timedelta(hours=1),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    is_valid, user = await auth_service.validate_token(expired_token)

    assert is_valid is False
    assert user is None


@pytest.mark.asyncio
async def test_validate_token_user_not_found(db_session: AsyncSession) -> None:
    """Test token validation when user doesn't exist."""
    auth_service = AuthService(db_session)

    # Create a token for non-existent user
    token_for_nonexistent = jwt.encode(
        {
            "sub": "nonexistent@example.com",
            "exp": 9999999999,
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    is_valid, user = await auth_service.validate_token(token_for_nonexistent)

    assert is_valid is False
    assert user is None


@pytest.mark.asyncio
async def test_get_user_from_token_wrong_issuer(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token with wrong issuer fails."""
    auth_service = AuthService(db_session)

    # Create a token with wrong issuer
    token_with_wrong_issuer = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": 9999999999,
            "iss": "wrong-issuer",
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_with_wrong_issuer)

    assert "issuer" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_get_user_from_token_wrong_audience(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token with wrong audience fails."""
    auth_service = AuthService(db_session)

    # Create a token with wrong audience
    token_with_wrong_audience = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": 9999999999,
            "iss": settings.jwt_issuer,
            "aud": "wrong-audience",
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_with_wrong_audience)

    assert "audience" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_get_user_from_token_missing_issuer(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token without issuer fails."""
    auth_service = AuthService(db_session)

    # Create a token without issuer
    token_without_issuer = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": 9999999999,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_without_issuer)

    assert "issuer" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_get_user_from_token_missing_audience(
    db_session: AsyncSession,
) -> None:
    """Test getting user from token without audience fails."""
    auth_service = AuthService(db_session)

    # Create a token without audience
    token_without_audience = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": 9999999999,
            "iss": settings.jwt_issuer,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    with pytest.raises(TokenValidationError) as exc_info:
        await auth_service.get_user_from_token(token_without_audience)

    assert "audience" in str(exc_info.value).lower()


@pytest.mark.asyncio
async def test_create_token_includes_issuer_audience(
    db_session: AsyncSession,
) -> None:
    """Test that created tokens include issuer and audience claims."""
    auth_service = AuthService(db_session)

    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    token = auth_service.create_token(user)

    # Verify token can be decoded and contains issuer/audience
    payload = jwt.decode(
        token.access_token,
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithms=[settings.jwt_algorithm],
        issuer=settings.jwt_issuer,
        audience=settings.jwt_audience,
    )
    assert payload["sub"] == str(user.id)
    assert payload["iss"] == settings.jwt_issuer
    assert payload["aud"] == settings.jwt_audience
    assert "exp" in payload


@pytest.mark.asyncio
async def test_get_user_from_token_expired_within_leeway(
    db_session: AsyncSession,
) -> None:
    """Test token expired within clock skew leeway is still accepted."""
    auth_service = AuthService(db_session)

    # Create a user
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Create token expired 60s ago (within default 120s leeway)
    expired_token = jwt.encode(
        {
            "sub": str(user.id),
            "exp": datetime.now(UTC) - timedelta(seconds=60),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    # Should succeed because it's within leeway
    retrieved_user = await auth_service.get_user_from_token(expired_token)
    assert retrieved_user.id == user.id
    assert retrieved_user.email == user.email


@pytest.mark.asyncio
async def test_get_user_from_token_expired_beyond_leeway(
    db_session: AsyncSession,
) -> None:
    """Test that token expired beyond clock skew leeway is rejected."""
    auth_service = AuthService(db_session)

    # Create token expired 5 min ago (beyond default 120s leeway)
    expired_token = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": datetime.now(UTC) - timedelta(minutes=5),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    # Should fail because it's beyond leeway
    with pytest.raises(TokenValidationError):
        await auth_service.get_user_from_token(expired_token)


@pytest.mark.asyncio
async def test_get_user_from_token_nbf_within_leeway(
    db_session: AsyncSession,
) -> None:
    """Test token with nbf claim slightly in future within leeway accepted."""
    auth_service = AuthService(db_session)

    # Create a user
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Create token with nbf 60s in future (within default 120s leeway)
    future_token = jwt.encode(
        {
            "sub": str(user.id),
            "exp": datetime.now(UTC) + timedelta(hours=1),
            "nbf": datetime.now(UTC) + timedelta(seconds=60),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    # Should succeed because nbf is within leeway
    retrieved_user = await auth_service.get_user_from_token(future_token)
    assert retrieved_user.id == user.id
    assert retrieved_user.email == user.email


@pytest.mark.asyncio
async def test_get_user_from_token_nbf_beyond_leeway(
    db_session: AsyncSession,
) -> None:
    """Test token with nbf claim far in future beyond leeway is rejected."""
    auth_service = AuthService(db_session)

    # Create token with nbf 5 min in future (beyond default 120s leeway)
    future_token = jwt.encode(
        {
            "sub": "test@example.com",
            "exp": datetime.now(UTC) + timedelta(hours=1),
            "nbf": datetime.now(UTC) + timedelta(minutes=5),
            "iss": settings.jwt_issuer,
            "aud": settings.jwt_audience,
        },
        # pylint: disable=no-member
        settings.jwt_secret_key.get_secret_value(),
        algorithm=settings.jwt_algorithm,
    )

    # Should fail because nbf is beyond leeway
    with pytest.raises(TokenValidationError):
        await auth_service.get_user_from_token(future_token)


@pytest.mark.asyncio
async def test_revoked_token_is_rejected(db_session: AsyncSession) -> None:
    """Token version bump should invalidate previously issued tokens."""
    redis = _FakeRedis()
    auth_service = AuthService(db_session, redis=redis)  # type: ignore[arg-type]
    user = User(
        name="Revoked User",
        email="revoked@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    token = await auth_service.create_token_for_user(user)
    await auth_service.revoke_user_tokens(user)

    with pytest.raises(TokenValidationError, match="revoked"):
        await auth_service.get_user_from_token(token.access_token)
