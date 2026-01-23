"""Script to run document ingestion from HuggingFace using DocumentIngestionService."""

import asyncio

import logfire
from datasets import load_dataset

from src.core.database.base import async_session_factory
from src.services.doc_ingestion_service import DocumentIngestionService


logfire.configure()


async def main() -> None:
    """Run the document insertion."""
    dataset_name = "marin-community/ar5iv-no-problem-markdown"
    category = "astro-ph"
    dataset = (
        load_dataset(
            dataset_name,
            split="train",
            streaming=True,
        )
        .filter(lambda row: "astro-ph" in row["id"])
        .remove_columns(["source", "format"])
        .rename_column("id", "source_id")
    )

    async with async_session_factory() as session:
        service = DocumentIngestionService(
            session=session,
            batch_size=1000,
            metadata={"source": dataset_name, "category": category},
        )

        await service.ingest(rows=dataset)


if __name__ == "__main__":
    asyncio.run(main())
