"""Document ingestion service."""

from collections.abc import AsyncIterable, Iterable, Sequence

import logfire
from aioitertools import itertools as async_itertools
from pydantic import TypeAdapter
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import logging
from src.core.database.utils import insert_many
from src.models.node import DocumentNode
from src.schemas.node import DocumentNodeCreate
from src.services.chunking_service import MarkdownChunker


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


class DocumentService:
    """Service for document operations."""

    def __init__(
        self,
        session: AsyncSession,
        *,
        chunker: MarkdownChunker | None = None,
    ) -> None:
        """Initialize the document ingestion service."""
        self.session = session
        self.chunker = chunker or MarkdownChunker()

    @logfire.instrument(
        "document_servie.ingest_document",
        extract_args=["batch_size", "metadata"],
    )
    async def ingest_document(
        self,
        rows: Iterable[dict] | AsyncIterable[dict],
        batch_size: int = 1000,
        metadata: dict | None = None,
    ) -> None:
        """Ingest all the documents from an iterable, after pydantic validation following DocumentNode model."""
        async for batch in async_itertools.batched(rows, batch_size):
            validated_batch = await self._validate_batch_with_metadata(
                batch=batch,
                metadata=metadata,
            )
            await self._ingest_batch(batch=validated_batch)

        logfire.info("documents_ingestion completed")

    @logging.auto_instrument()
    async def _validate_batch_with_metadata(
        self,
        batch: Sequence[dict],
        metadata: dict | None = None,
    ) -> list[dict]:
        validated_models = DocumentNodeBatchAdapter.validate_python(
            batch,
            context={"base_metadata": metadata},
        )

        validated_documents_metric.add(len(validated_models))

        return [model.model_dump() for model in validated_models]

    @logging.auto_instrument()
    async def _ingest_batch(
        self,
        batch: Sequence[dict],
    ) -> None:
        """Ingest a batch of documents."""
        inserted_rows_count = await insert_many(
            self.session,
            DocumentNode,
            batch,
        )

        ingested_documents_metric.add(inserted_rows_count)

    @logfire.instrument(
        "document_service.chunk_documents",
        extract_args=["batch_size", "metadata"],
    )
    async def chunk_documents(
        self,
        documents: Sequence[DocumentNode],
        batch_size: int = 1000,
        metadata: dict | None = None,
    ) -> None:
        """Chunk the documents pointed by the provided UUIDs."""
        if self.chunker is None:
            raise ValueError(
                "Chunker not initialized, construct a DocumentService instance defining a chunking service instance.",
            )
        async for batch in async_itertools.batched(documents, batch_size):
            self.chunker.chunk(batch, metadata=metadata)
            with logfire.span("Commit chunks into database"):
                await self.session.commit()
