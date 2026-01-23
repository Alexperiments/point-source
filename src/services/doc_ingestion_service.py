"""Document ingestion service."""

import itertools
from collections.abc import AsyncIterable, Iterable, Sequence

import logfire
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from src.core.database.utils import insert_many
from src.models.node import DocumentNode
from src.schemas.node import DocumentNodeCreate


DocumentNodeBatchAdapter = TypeAdapter(list[DocumentNodeCreate])


validated_documents_metric = logfire.metric_counter(
    "papers_processed",
    unit="1",
    description="Number of valide papers fetched.",
)
ingested_documents_metric = logfire.metric_counter(
    "documents_ingested",
    unit="1",
    description="Number of papers inserted in the database.",
)


class DocumentIngestionService:
    """Service for document ingestion operations."""

    def __init__(
        self,
        session: AsyncSession,
        batch_size: int = 1000,
        metadata: dict | None = None,
    ) -> None:
        """Initialize the document ingestion service."""
        self.session = session
        self.batch_size = batch_size
        self.base_metadata = metadata

    async def ingest(self, rows: Iterable[dict] | AsyncIterable[dict]) -> None:
        """Ingest all the documents from an iterable, after pydantic validation following DocumentNode model."""
        with logfire.span("documents_ingestion"):
            if isinstance(rows, Iterable) and not isinstance(rows, AsyncIterable):
                iterator = iter(rows)
                for batch in itertools.batched(iterable=iterator, n=self.batch_size):
                    await self._ingest_batch(batch)
            else:
                batch: list[dict] = []
                async for row in rows:
                    batch.append(row)
                    if len(batch) >= self.batch_size:
                        await self._ingest_batch(batch)
                        batch = []
                if batch:
                    await self._ingest_batch(batch)

    async def _ingest_batch(self, batch: Sequence[dict]) -> None:
        """Ingest a batch of documents."""
        validated_models = DocumentNodeBatchAdapter.validate_python(
            batch,
            context={"base_metadata": self.base_metadata},
        )

        validated_documents_metric.add(len(validated_models))

        validated_inputs = [model.model_dump() for model in validated_models]

        inserted_rows_count = await insert_many(
            self.session,
            DocumentNode,
            validated_inputs,
        )

        ingested_documents_metric.add(inserted_rows_count)
