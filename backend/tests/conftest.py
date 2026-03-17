"""Pytest configuration and fixtures."""

from collections.abc import AsyncGenerator

import pytest_asyncio
from httpx import ASGITransport, AsyncClient
from sqlalchemy.ext.asyncio import (
    AsyncSession,
    async_sessionmaker,
    create_async_engine,
)
from sqlalchemy.pool import StaticPool

from src.api.v1.auth import get_email_service, get_redis as get_auth_redis
from src.core.database.base import Base, get_async_session
from src.main import app
from src.services.email_service import EmailMessage


# Use in-memory SQLite for testing
TEST_DATABASE_URL = "sqlite+aiosqlite:///:memory:"


class _FakeRedis:
    """Simple in-memory Redis stub for auth token versioning tests."""

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

    async def aclose(self) -> None:
        return None


class _FakeEmailService:
    """Collect outbound emails for assertions in tests."""

    def __init__(self) -> None:
        self.messages: list[EmailMessage] = []

    async def send(self, message: EmailMessage) -> None:
        self.messages.append(message)


@pytest_asyncio.fixture(scope="session")
async def test_engine() -> AsyncGenerator:
    """Create a test engine and dispose it at session end."""
    engine = create_async_engine(
        TEST_DATABASE_URL,
        echo=False,
        future=True,
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
        execution_options={"schema_translate_map": {"processed": None}},
    )
    yield engine
    await engine.dispose()


@pytest_asyncio.fixture(scope="function")
async def db_session(
    test_engine,
) -> AsyncGenerator[AsyncSession, None]:
    """Create a test database session."""
    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.create_all)

    session_factory = async_sessionmaker(
        bind=test_engine,
        class_=AsyncSession,
        autoflush=False,
        expire_on_commit=False,
    )

    async with session_factory() as session:
        yield session
        await session.rollback()

    async with test_engine.begin() as conn:
        await conn.run_sync(Base.metadata.drop_all)


@pytest_asyncio.fixture(scope="function")
async def fake_email_service() -> _FakeEmailService:
    """Create an in-memory email sink for auth-related tests."""
    return _FakeEmailService()


@pytest_asyncio.fixture(scope="function")
async def client(
    db_session: AsyncSession,
    fake_email_service: _FakeEmailService,
) -> AsyncGenerator[AsyncClient, None]:
    """Create a test client with database session override."""

    async def override_get_session() -> AsyncGenerator[AsyncSession, None]:
        yield db_session

    redis = _FakeRedis()

    async def override_auth_redis() -> AsyncGenerator[_FakeRedis, None]:
        yield redis
        await redis.aclose()

    async def override_email_service() -> AsyncGenerator[_FakeEmailService, None]:
        yield fake_email_service

    app.dependency_overrides[get_async_session] = override_get_session
    app.dependency_overrides[get_auth_redis] = override_auth_redis
    app.dependency_overrides[get_email_service] = override_email_service

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac

    app.dependency_overrides.clear()


@pytest_asyncio.fixture(scope="function")
async def client_no_db() -> AsyncGenerator[AsyncClient, None]:
    """Create a test client without database session override.

    Use this for endpoints that don't require database access.
    """
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://test",
    ) as ac:
        yield ac
