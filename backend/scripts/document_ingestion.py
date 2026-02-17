"""Script to run document ingestion from HuggingFace using DocumentIngestionService."""

import asyncio

import logfire
from datasets import load_dataset

from src.core.database.base import async_session
from src.services.document_service import DocumentService


logfire.configure()


async def main() -> None:
    """Run the document insertion."""
    dataset_name = "marin-community/ar5iv-no-problem-markdown"
    category = "astro-ph"

    with logfire.span(
        "document_ingestion_job",
        dataset_name=dataset_name,
        category=category,
    ):
        dataset = (
            load_dataset(
                dataset_name,
                split="train",
                streaming=False,
            )
            .filter(lambda row: (category in row["id"]) & (len(row["text"]) >= 1000))
            .remove_columns(["source", "format"])
            .rename_column("id", "source_id")
            .take(1000)
        )

        async with async_session() as session:
            service = DocumentService(
                session=session,
            )

            await service.ingest_document(
                rows=dataset,
                batch_size=1000,
                perform_chunking=True,
                metadata={"source": dataset_name, "category": category},
            )
            await session.commit()

        logfire.info(
            "Document ingestion job finished",
            dataset_name=dataset_name,
            category=category,
        )


if __name__ == "__main__":
    asyncio.run(main())
