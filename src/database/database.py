"""Database utilities."""

import abc
from collections.abc import AsyncGenerator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import DeclarativeMeta, declarative_base
from sqlalchemy.orm import DeclarativeBase

from src.core.config import settings


class Base(DeclarativeBase):
    """Base class for all models."""


class DeclarativeMetaBase(DeclarativeMeta, abc.ABCMeta):
    """DeclarativeBase class that works as base for abstract classes."""


class AbstractBase(declarative_base(metaclass=DeclarativeMetaBase)):
    """Abstract base class for all models."""

    __abstract__ = True


engine = create_async_engine(
    settings.database_url.unicode_string(),
    future=True,
    echo=settings.debug,
    pool_pre_ping=True,
)

async_session_factory = async_sessionmaker(
    bind=engine,
    class_=AsyncSession,
    autoflush=False,
    expire_on_commit=False,
)


async def get_async_session() -> AsyncGenerator[AsyncSession]:
    """Get a database session."""
    async with async_session_factory() as session:
        try:
            yield session
            await session.commit()
        except SQLAlchemyError:
            raise
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()
