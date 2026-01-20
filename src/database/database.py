"""Database utilities."""

import abc
from collections.abc import AsyncGenerator

from sqlalchemy.exc import SQLAlchemyError
from sqlalchemy.ext.asyncio import async_sessionmaker, create_async_engine
from sqlalchemy.ext.asyncio.session import AsyncSession
from sqlalchemy.ext.declarative import DeclarativeMeta
from sqlalchemy.orm import declarative_base

from src.core.config import settings


class DeclarativeMetaBase(DeclarativeMeta, abc.ABC):
    """Metaclass that combines SQLAlchemy declarative behavior with ABC."""


# Create ONE metadata object
Base = declarative_base()

# Second base shares the SAME metadata
AbstractBase = declarative_base(
    metadata=Base.metadata,
    metaclass=DeclarativeMetaBase,
)


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
