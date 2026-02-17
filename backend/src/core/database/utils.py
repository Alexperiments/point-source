"""Database primitive operations."""

from collections.abc import Sequence

from sqlalchemy import literal
from sqlalchemy.dialects.postgresql import insert
from sqlalchemy.ext.asyncio.session import AsyncSession

from src.core.database.base import DeclarativeMetaBase


async def insert_many(
    session: AsyncSession,
    table: type[DeclarativeMetaBase],
    values: Sequence[dict],
) -> int:
    """Insert many in the table referenced by the provided SQLAlchemy table object.

    Return the number of inserted rows. Caller manages the transaction.
    """
    result = await session.execute(
        insert(table).on_conflict_do_nothing().returning(literal(1)),
        values,
    )
    return sum(1 for _ in result)
