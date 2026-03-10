"""Script to run document ingestion from HuggingFace using DocumentIngestionService."""

import asyncio

import logfire
from sqlalchemy import select

from src.core.database.base import async_session
from src.core.telemetry import configure_logfire
from src.models.node import DocumentNode
from src.services.chunking_service import MarkdownChunker


configure_logfire()


async def main() -> None:
    """Run the document insertion."""
    with logfire.span("Document chunking job"):
        chunking_service = MarkdownChunker()

        async with async_session() as session:
            while True:
                result = await session.execute(
                    select(DocumentNode)
                    .where(
                        DocumentNode.text.is_not(None),
                        DocumentNode.text != "",
                        ~DocumentNode.children.any(),
                    )
                    .order_by(
                        DocumentNode.id,
                    )
                    .with_for_update(skip_locked=True)
                    .limit(1024),
                )
                rows = result.scalars().all()
                if not rows:
                    break

                with logfire.span("Chunking batch"):
                    chunking_result = chunking_service.chunk(
                        documents=rows,
                    )

                with logfire.span("Inserting batch in database"):
                    session.add_all(chunking_result)
                    await session.commit()


if __name__ == "__main__":
    asyncio.run(main())
