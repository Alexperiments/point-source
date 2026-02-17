"""Document ingestion service."""

from collections.abc import AsyncIterable, Iterable, Sequence

import logfire
from aioitertools import itertools as async_itertools
from sqlalchemy.ext.asyncio import AsyncSession

from src.core import logging
from src.models.node import DocumentNode
from src.services.chunking_service import MarkdownChunker


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
        "document_service.ingest_document",
        extract_args=["batch_size", "metadata", "perform_chunking"],
    )
    async def ingest_document(
        self,
        rows: Iterable[dict] | AsyncIterable[dict],
        batch_size: int = 1000,
        *,
        perform_chunking: bool = False,
        metadata: dict | None = None,
    ) -> None:
        """Ingest all the documents from an iterable of row dicts."""
        async for batch in async_itertools.batched(rows, batch_size):
            with logfire.span(f"Ingest {len(batch)} documents."):
                documents = await self._build_documents_from_dicts(
                    dict_documents_list=batch,
                    metadata=metadata,
                )
                if perform_chunking:
                    await self.chunk_documents(
                        documents,
                        batch_size=batch_size,
                        metadata=metadata,
                    )
                self.session.add_all(documents)
                await self.session.flush()

                ingested_documents_metric.add(len(batch))

        logfire.info("documents_ingestion completed")

    @logging.auto_instrument()
    async def _build_documents_from_dicts(
        self,
        dict_documents_list: Sequence[dict],
        metadata: dict | None = None,
    ) -> list[DocumentNode]:
        base_metadata = dict(metadata) if metadata else None
        documents: list[DocumentNode] = []

        for row in dict_documents_list:
            document = DocumentNode(
                text=row["text"],
                source_id=row["source_id"],
            )
            if "id" in row and row["id"] is not None:
                document.id = row["id"]
            if "embedding" in row:
                document.embedding = row["embedding"]

            node_metadata = row.get("node_metadata")
            if base_metadata:
                if node_metadata is None:
                    node_metadata = dict(base_metadata)
                else:
                    node_metadata = {
                        **base_metadata,
                        **node_metadata,
                    }
            if node_metadata is not None:
                document.node_metadata = node_metadata

            documents.append(document)

        return documents

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
