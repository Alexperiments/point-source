"""Document ingestion service."""

import logfire
from datasets import load_dataset
from sqlalchemy.exc import IntegrityError
from sqlalchemy.ext.asyncio import AsyncSession

from src.db_models.node import DocumentNode


papers_processed_metric = logfire.metric_counter("papers_processed")
documents_ingested_metric = logfire.metric_counter("documents_ingested")


class DocumentIngestionServiceError(Exception):
    """Exception for document ingestion service errors."""


class DocumentAlreadyExistsError(DocumentIngestionServiceError):
    """Exception for document already exists errors."""


class FailedToCreateDocumentError(DocumentIngestionServiceError):
    """Exception for failed to create document errors."""


class DocumentIngestionService:
    """Service for document ingestion operations."""

    def __init__(self, session: AsyncSession) -> None:
        """Initialize the document ingestion service.

        Args:
            session: The database session to use.

        """
        self.session = session

    async def ingest_ar5iv_papers(
        self,
        batch_size: int = 1000,
        category: str | None = None,
    ) -> int:
        """Ingest ar5iv papers from the dataset, filtering by category in id.

        Args:
            batch_size: Number of documents to process in each batch.
            category: Arxiv category to filter the request, e.g. 'astro-ph'.

        Returns:
            The number of documents ingested.

        """
        with logfire.span(
            "ingest_ar5iv_papers",
            batch_size=batch_size,
            category=category,
        ):
            logfire.log(
                "info",
                "Starting ingestion of astro-ph papers from ar5iv dataset",
            )
            # Load dataset in streaming mode
            dataset = load_dataset(
                "marin-community/ar5iv-no-problem-markdown",
                split="train",
                streaming=True,
            )
            logfire.log("info", "Dataset loaded in streaming mode")
            if category is not None:
                dataset = dataset.filter(lambda row: category in row["id"])
                logfire.log("info", f"Applied filter for category: {category}")

            batch = []
            count = 0
            processed = 0
            logfire.log(
                "info",
                f"Starting to process papers with batch_size={batch_size}",
            )

            for paper in dataset:
                ar5iv_id = paper.get("id")
                if not ar5iv_id:
                    continue

                processed += 1
                if processed % 10 == 0:
                    logfire.log("info", f"Processed {processed} papers")

                document = DocumentNode(
                    provider_id=ar5iv_id,
                    text=paper.get("markdown", ""),
                    node_metadata={
                        "source": "ar5iv",
                        "category": "astro-ph",
                    },
                )
                batch.append(document)

                if len(batch) >= batch_size:
                    logfire.log("info", f"Inserting batch of {len(batch)} documents")
                    inserted = await self._insert_batch(batch)
                    count += inserted
                    batch = []
                    logfire.log(
                        "info",
                        f"Inserted batch of {inserted} documents, total: {count}",
                    )

            if batch:
                logfire.log("info", f"Inserting final batch of {len(batch)} documents")
                inserted = await self._insert_batch(batch)
                count += inserted
                logfire.log(
                    "info",
                    f"Inserted final batch of {inserted} documents, total: {count}",
                )

            logfire.log(
                "info",
                f"Ingestion completed successfully, total papers processed: {processed}, total documents: {count}",
            )
            papers_processed_metric.add(processed)
            documents_ingested_metric.add(count)
            return count

    async def _insert_batch(self, documents: list[DocumentNode]) -> int:
        """Insert a batch of documents into the database, skipping duplicates.

        Args:
            documents: List of DocumentNode instances to insert.

        Returns:
            Number of documents successfully inserted.

        Raises:
            FailedToCreateDocumentError: If insertion fails for reasons other than duplicates.

        """
        with logfire.span("_insert_batch", batch_size=len(documents)):
            inserted_count = 0
            for doc in documents:
                self.session.add(doc)
                try:
                    await self.session.flush()
                    inserted_count += 1
                except IntegrityError:
                    await self.session.rollback()
                    logfire.log(
                        "warning",
                        f"Skipped duplicate document with provider_id: {doc.provider_id}",
                    )
                except Exception as e:
                    await self.session.rollback()
                    msg = f"Failed to insert document {doc.provider_id}: {e!s}"
                    logfire.exception(msg)
                    raise FailedToCreateDocumentError(msg) from e

            await self.session.commit()
            return inserted_count
