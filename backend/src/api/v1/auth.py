"""Auth API."""

from collections.abc import AsyncIterator
from typing import Annotated, Any

from fastapi import APIRouter, Depends, HTTPException, Response, status
from pydantic import SecretStr, ValidationError
from redis.asyncio import Redis
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database.base import get_async_session
from src.core.database.redis import get_redis_pool
from src.core.security import hash_password, oauth2_scheme, verify_password
from src.models.user import User
from src.schemas.user import (
    ProfileUpdateRequest,
    ProfileUpdateResponse,
    Token,
    TokenValidationResponse,
    UserCreate,
    UserLogin,
    UserResponse,
    UserUpdate,
)
from src.services.auth_service import (
    AuthService,
    InvalidCredentialsError,
    TokenValidationError,
)
from src.services.user_service import UserAlreadyExistsError, UserService


router = APIRouter()


async def get_redis() -> AsyncIterator[Redis]:
    """Get Redis client dependency for auth/token operations."""
    redis = await get_redis_pool()
    try:
        yield redis
    finally:
        await redis.aclose()


async def get_current_user(
    token: str = Depends(oauth2_scheme),
    db: AsyncSession = Depends(get_async_session),
    redis: Redis = Depends(get_redis),
) -> User:
    """Get the current user."""
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        auth_service = AuthService(db, redis)
        return await auth_service.get_user_from_token(token)
    except TokenValidationError as e:
        raise credentials_exception from e


@router.post("/register", response_model=UserResponse)
async def register(
    user_data: UserCreate,
    db: Annotated[AsyncSession, Depends(get_async_session)],
) -> User:
    """Register a new user."""
    try:
        auth_service = AuthService(db)
        return await auth_service.register_user(user_data)
    except UserAlreadyExistsError as e:
        raise HTTPException(status_code=400, detail=str(e)) from e


@router.post("/token", response_model=Token)
async def login(
    user_data: UserLogin,
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> Token:
    """Login a user."""
    try:
        auth_service = AuthService(db, redis)
        return await auth_service.login(user_data)
    except InvalidCredentialsError as e:
        raise HTTPException(
            status_code=401,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        ) from e


@router.get("/validate-token", response_model=TokenValidationResponse)
async def validate_token(
    current_user: Annotated[User, Depends(get_current_user)],
) -> TokenValidationResponse:
    """Validate token."""
    return TokenValidationResponse(valid=True, user_id=str(current_user.id))


@router.post("/logout", status_code=status.HTTP_204_NO_CONTENT)
async def logout(
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> Response:
    """Invalidate currently active access tokens for the user."""
    auth_service = AuthService(db, redis)
    await auth_service.revoke_user_tokens(current_user)
    return Response(status_code=status.HTTP_204_NO_CONTENT)


@router.get("/users/me", response_model=UserResponse)
async def get_current_user_info(
    current_user: Annotated[User, Depends(get_current_user)],
) -> User:
    """Get current user information."""
    return current_user


@router.patch("/users/me", response_model=ProfileUpdateResponse)
async def update_current_user_info(
    profile_data: ProfileUpdateRequest,
    current_user: Annotated[User, Depends(get_current_user)],
    db: Annotated[AsyncSession, Depends(get_async_session)],
    redis: Annotated[Redis, Depends(get_redis)],
) -> ProfileUpdateResponse:
    """Update the current user's profile and optional password."""
    name = (profile_data.name or "").strip()
    email = (profile_data.email or "").strip()
    current_password = profile_data.current_password or ""
    new_password = profile_data.new_password or ""
    confirm_password = profile_data.confirm_password or ""

    errors: list[str] = []
    update_data = _build_profile_update_data(name, email, current_user)
    user, updated_profile = await _apply_profile_updates(
        db,
        current_user,
        update_data,
        errors,
    )
    updated_password = await _apply_password_change(
        db,
        user,
        current_password,
        new_password,
        confirm_password,
        errors,
    )

    if errors:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=errors)

    updated = updated_profile or updated_password
    access_token = None
    token_type = None

    if updated:
        auth_service = AuthService(db, redis)
        token = await auth_service.create_token_for_user(user)
        access_token = token.access_token
        token_type = token.token_type

    await db.refresh(user)

    return ProfileUpdateResponse(
        user=user,
        access_token=access_token,
        token_type=token_type,
    )


def _build_profile_update_data(
    name: str,
    email: str,
    user: User,
) -> dict[str, Any]:
    update_data: dict[str, Any] = {}
    if name and name != user.name:
        update_data["name"] = name
    if email and email != user.email:
        update_data["email"] = email
    return update_data


async def _apply_profile_updates(
    db: AsyncSession,
    user: User,
    update_data: dict[str, Any],
    errors: list[str],
) -> tuple[User, bool]:
    if not update_data:
        return user, False

    try:
        user_service = UserService(db)
        user = await user_service.update_user(user.id, UserUpdate(**update_data))
    except UserAlreadyExistsError:
        errors.append("That email is already in use.")
    except ValidationError as e:
        errors.append(e.errors()[0].get("msg", "Invalid input"))
    else:
        return user, True

    return user, False


async def _apply_password_change(
    db: AsyncSession,
    user: User,
    current_password: str,
    new_password: str,
    confirm_password: str,
    errors: list[str],
) -> bool:
    if not _password_change_requested(
        current_password,
        new_password,
        confirm_password,
    ):
        return False

    if not current_password:
        errors.append("Current password is required to change your password.")
        return False

    if new_password != confirm_password:
        errors.append("New passwords do not match.")
        return False

    try:
        UserUpdate(password=SecretStr(new_password))
    except ValidationError as e:
        errors.append(e.errors()[0].get("msg", "Invalid password"))
        return False

    if not verify_password(current_password, user.hashed_password):
        errors.append("Current password is incorrect.")
        return False

    user.hashed_password = hash_password(new_password)
    await db.flush()
    return True


def _password_change_requested(
    current_password: str,
    new_password: str,
    confirm_password: str,
) -> bool:
    return any([current_password, new_password, confirm_password])
