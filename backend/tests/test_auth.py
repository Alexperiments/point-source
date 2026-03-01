"""Tests for authentication endpoints."""

import pytest
from httpx import AsyncClient
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.security import hash_password
from src.models.user import User


@pytest.mark.asyncio
async def test_register_success(client: AsyncClient, db_session: AsyncSession) -> None:
    """Test successful user registration."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "SecurePass123",
    }

    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["name"] == "Test User"
    assert "id" in data
    assert "hashed_password" not in data
    assert "password" not in data


@pytest.mark.asyncio
async def test_register_duplicate_email(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test registration with duplicate email fails."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "SecurePass123",
    }

    # Create first user
    await client.post("/v1/auth/register", json=user_data)

    # Try to register again with same email
    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 400
    assert "already exists" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_register_invalid_password(client: AsyncClient) -> None:
    """Test registration with invalid password fails."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "short",  # Too short
    }

    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_register_invalid_email(client: AsyncClient) -> None:
    """Test registration with invalid email fails."""
    user_data = {
        "name": "Test User",
        "email": "not-an-email",
        "password": "SecurePass123",
    }

    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_login_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test successful user login."""
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
    login_data = {
        "email": "test@example.com",
        "password": password,
    }

    response = await client.post("/v1/auth/token", json=login_data)

    assert response.status_code == 200
    data = response.json()
    assert "access_token" in data
    assert data["token_type"] == "Bearer"
    assert isinstance(data["access_token"], str)
    assert len(data["access_token"]) > 0


@pytest.mark.asyncio
async def test_login_invalid_email(client: AsyncClient) -> None:
    """Test login with non-existent email fails."""
    login_data = {
        "email": "nonexistent@example.com",
        "password": "SecurePass123",
    }

    response = await client.post("/v1/auth/token", json=login_data)

    assert response.status_code == 401
    assert "incorrect email or password" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_login_invalid_password(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test login with incorrect password fails."""
    # Create a user first
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Try to login with wrong password
    login_data = {
        "email": "test@example.com",
        "password": "WrongPassword123",
    }

    response = await client.post("/v1/auth/token", json=login_data)

    assert response.status_code == 401
    assert "incorrect email or password" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_login_deleted_user_fails(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Soft-deleted users should not be able to obtain access tokens."""
    password = "SecurePass123"
    user = User(
        name="Deleted User",
        email="deleted@example.com",
        hashed_password=hash_password(password),
        is_deleted=True,
    )
    db_session.add(user)
    await db_session.commit()

    response = await client.post(
        "/v1/auth/token",
        json={"email": user.email, "password": password},
    )

    assert response.status_code == 401
    assert "incorrect email or password" in response.json()["detail"].lower()


@pytest.mark.asyncio
async def test_validate_token_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test token validation with valid token."""
    # Create a user and get a token
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Login to get token
    login_data = {
        "email": "test@example.com",
        "password": password,
    }
    login_response = await client.post("/v1/auth/token", json=login_data)
    token = login_response.json()["access_token"]

    # Validate token
    response = await client.get(
        "/v1/auth/validate-token",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["valid"] is True
    assert data["user_id"] == str(user.id)


@pytest.mark.asyncio
async def test_validate_token_missing_header(client: AsyncClient) -> None:
    """Test token validation without authorization header fails."""
    response = await client.get("/v1/auth/validate-token")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_validate_token_invalid_token(client: AsyncClient) -> None:
    """Test token validation with invalid token fails."""
    response = await client.get(
        "/v1/auth/validate-token",
        headers={"Authorization": "Bearer invalid_token_here"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test getting current user info with valid token."""
    # Create a user and get a token
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
        is_superuser=False,
    )
    db_session.add(user)
    await db_session.commit()
    await db_session.refresh(user)

    # Login to get token
    login_data = {
        "email": "test@example.com",
        "password": password,
    }
    login_response = await client.post("/v1/auth/token", json=login_data)
    token = login_response.json()["access_token"]

    # Get current user info
    response = await client.get(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"},
    )

    assert response.status_code == 200
    data = response.json()
    assert data["email"] == "test@example.com"
    assert data["name"] == "Test User"
    assert data["id"] == str(user.id)
    assert data["is_superuser"] is False
    assert "hashed_password" not in data
    assert "password" not in data


@pytest.mark.asyncio
async def test_get_current_user_missing_token(client: AsyncClient) -> None:
    """Test getting current user without token fails."""
    response = await client.get("/v1/auth/users/me")

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_logout_revokes_access_token(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Logout should invalidate existing bearer tokens."""
    password = "SecurePass123"
    user = User(
        name="Token User",
        email="token-user@example.com",
        hashed_password=hash_password(password),
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": user.email, "password": password},
    )
    token = login_response.json()["access_token"]
    headers = {"Authorization": f"Bearer {token}"}

    me_before = await client.get("/v1/auth/users/me", headers=headers)
    assert me_before.status_code == 200

    logout_response = await client.post("/v1/auth/logout", headers=headers)
    assert logout_response.status_code == 204

    me_after = await client.get("/v1/auth/users/me", headers=headers)
    assert me_after.status_code == 401


@pytest.mark.asyncio
async def test_get_current_user_invalid_token(client: AsyncClient) -> None:
    """Test getting current user with invalid token fails."""
    response = await client.get(
        "/v1/auth/users/me",
        headers={"Authorization": "Bearer invalid_token_here"},
    )

    assert response.status_code == 401


@pytest.mark.asyncio
async def test_register_empty_name(client: AsyncClient) -> None:
    """Test registration with empty name fails."""
    user_data = {
        "name": "",
        "email": "test@example.com",
        "password": "SecurePass123",
    }

    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_register_password_no_digit(client: AsyncClient) -> None:
    """Test registration with password without digit fails."""
    user_data = {
        "name": "Test User",
        "email": "test@example.com",
        "password": "NoDigitPassword",
    }

    response = await client.post("/v1/auth/register", json=user_data)

    assert response.status_code == 422  # Validation error


@pytest.mark.asyncio
async def test_login_missing_fields(client: AsyncClient) -> None:
    """Test login with missing fields fails."""
    # Missing password
    response = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com"},
    )
    assert response.status_code == 422

    # Missing email
    response = await client.post(
        "/v1/auth/token",
        json={"password": "SecurePass123"},
    )
    assert response.status_code == 422


@pytest.mark.asyncio
async def test_update_current_user_profile_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test updating current user profile fields returns updated user + token."""
    password = "SecurePass123"
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(password),
        is_superuser=False,
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com", "password": password},
    )
    token = login_response.json()["access_token"]

    response = await client.patch(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Updated User",
            "email": "updated@example.com",
        },
    )

    assert response.status_code == 200
    data = response.json()
    assert data["user"]["name"] == "Updated User"
    assert data["user"]["email"] == "updated@example.com"
    assert isinstance(data["access_token"], str)
    assert data["token_type"] == "Bearer"

    me_response = await client.get(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {data['access_token']}"},
    )
    assert me_response.status_code == 200
    assert me_response.json()["email"] == "updated@example.com"


@pytest.mark.asyncio
async def test_update_current_user_password_success(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test updating current user password rotates credentials."""
    current_password = "SecurePass123"
    new_password = "NewSecurePass456"

    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password(current_password),
        is_superuser=False,
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com", "password": current_password},
    )
    token = login_response.json()["access_token"]

    response = await client.patch(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Test User",
            "email": "test@example.com",
            "current_password": current_password,
            "new_password": new_password,
            "confirm_password": new_password,
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload["access_token"], str)

    old_login = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com", "password": current_password},
    )
    assert old_login.status_code == 401

    new_login = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com", "password": new_password},
    )
    assert new_login.status_code == 200


@pytest.mark.asyncio
async def test_update_current_user_password_missing_current_fails(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test password update requires current password."""
    user = User(
        name="Test User",
        email="test@example.com",
        hashed_password=hash_password("SecurePass123"),
        is_superuser=False,
    )
    db_session.add(user)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": "test@example.com", "password": "SecurePass123"},
    )
    token = login_response.json()["access_token"]

    response = await client.patch(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Test User",
            "email": "test@example.com",
            "new_password": "NewSecurePass456",
            "confirm_password": "NewSecurePass456",
        },
    )

    assert response.status_code == 400
    detail = response.json().get("detail", [])
    assert isinstance(detail, list)
    assert "Current password is required to change your password." in detail


@pytest.mark.asyncio
async def test_update_current_user_duplicate_email_fails(
    client: AsyncClient,
    db_session: AsyncSession,
) -> None:
    """Test profile update fails when target email already exists."""
    user = User(
        name="Primary User",
        email="primary@example.com",
        hashed_password=hash_password("SecurePass123"),
        is_superuser=False,
    )
    other = User(
        name="Other User",
        email="other@example.com",
        hashed_password=hash_password("SecurePass123"),
        is_superuser=False,
    )
    db_session.add(user)
    db_session.add(other)
    await db_session.commit()

    login_response = await client.post(
        "/v1/auth/token",
        json={"email": "primary@example.com", "password": "SecurePass123"},
    )
    token = login_response.json()["access_token"]

    response = await client.patch(
        "/v1/auth/users/me",
        headers={"Authorization": f"Bearer {token}"},
        json={
            "name": "Primary User",
            "email": "other@example.com",
        },
    )

    assert response.status_code == 400
    detail = response.json().get("detail", [])
    assert isinstance(detail, list)
    assert "That email is already in use." in detail
