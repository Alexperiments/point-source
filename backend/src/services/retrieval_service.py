"""Retrieval service."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

from sqlalchemy import bindparam, func, select

from src.core.rag_config import EMBEDDING_SETTINGS, RETRIEVAL_SETTINGS
from src.models.node import DocumentNode, TextNode
from src.schemas.retrieval import RetrievalFilters, RetrievedChunk
from src.services.embedding_service import (
    MLXQwen3EmbeddingService,
    get_embedding_service,
)
from src.services.reranking_service import RerankingService


if TYPE_CHECKING:
    from uuid import UUID

    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select

_QUERY_WHITESPACE_RE = re.compile(r"\s+")


class RetrievalService:
    """Retrieve relevant chunks for a query."""

    def __init__(
        self,
        session: AsyncSession,
        reranker: RerankingService | None = None,
    ) -> None:
        """Initialize the retrieval service."""
        self.session = session
        self._embedder: MLXQwen3EmbeddingService | None = None
        self._reranker = reranker or RerankingService()

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        use_prev_next: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        embedding = await self._embed_query_async(normalized_query)
        text_chunks = await self._text_candidates(normalized_query, filters)
        vector_chunks = await self._vector_candidates(embedding, filters)

        merged = self._rrf_merge(text_chunks, vector_chunks)
        reranked = await asyncio.to_thread(
            self._reranker.rerank,
            normalized_query,
            merged,
        )
        results = reranked[: RETRIEVAL_SETTINGS.top_n]

        include_neighbors = (
            use_prev_next
            if use_prev_next is not None
            else RETRIEVAL_SETTINGS.use_prev_next
        )
        if include_neighbors and results:
            await self._expand_with_neighbors(results)

        return results

    @staticmethod
    def _normalize_query(query: str) -> str:
        return _QUERY_WHITESPACE_RE.sub(" ", query).strip()

    def _get_embedder(self) -> MLXQwen3EmbeddingService:
        if self._embedder is None:
            self._embedder = get_embedding_service()
        return self._embedder

    def _embed_query(self, query: str) -> list[float]:
        embedder = self._get_embedder()
        embedding = embedder.encode_query(
            [query],
            batch_size=EMBEDDING_SETTINGS.query_batch_size,
        )
        return embedding[0].tolist()

    async def _embed_query_async(self, query: str) -> list[float]:
        return await asyncio.to_thread(self._embed_query, query)

    async def _text_candidates(
        self,
        query: str,
        filters: RetrievalFilters | None,
    ) -> list[RetrievedChunk]:
        ts_query = func.websearch_to_tsquery("simple", bindparam("query"))
        ts_vector = func.to_tsvector("simple", TextNode.text)
        ts_rank = func.ts_rank_cd(ts_vector, ts_query)

        stmt = (
            select(
                TextNode.id,
                TextNode.document_id,
                DocumentNode.source_id,
                TextNode.text,
                TextNode.node_metadata,
            )
            .join(DocumentNode, TextNode.document_id == DocumentNode.id)
            .where(TextNode.text != "")
            .where(ts_vector.op("@@")(ts_query))
            .order_by(ts_rank.desc())
            .limit(RETRIEVAL_SETTINGS.text_top_k)
        )
        stmt = self._apply_filters(stmt, filters)
        rows = await self.session.execute(stmt.params(query=query))

        chunks: list[RetrievedChunk] = []
        for chunk_id, document_id, source_id, text, metadata in rows.tuples():
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_id=source_id,
                    path=self._metadata_path(metadata),
                    text=text,
                    citations=[source_id],
                ),
            )
        return chunks

    async def _vector_candidates(
        self,
        embedding: list[float],
        filters: RetrievalFilters | None,
    ) -> list[RetrievedChunk]:
        distance = TextNode.embedding.op("<=>")(bindparam("embedding"))

        stmt = (
            select(
                TextNode.id,
                TextNode.document_id,
                DocumentNode.source_id,
                TextNode.text,
                TextNode.node_metadata,
            )
            .join(DocumentNode, TextNode.document_id == DocumentNode.id)
            .where(TextNode.text != "", TextNode.embedding.is_not(None))
            .order_by(distance)
            .limit(RETRIEVAL_SETTINGS.vector_top_k)
        )
        stmt = self._apply_filters(stmt, filters)
        rows = await self.session.execute(stmt.params(embedding=embedding))

        chunks: list[RetrievedChunk] = []
        for chunk_id, document_id, source_id, text, metadata in rows.tuples():
            chunks.append(
                RetrievedChunk(
                    chunk_id=chunk_id,
                    document_id=document_id,
                    source_id=source_id,
                    path=self._metadata_path(metadata),
                    text=text,
                    citations=[source_id],
                ),
            )
        return chunks

    @staticmethod
    def _metadata_path(node_metadata: dict[str, object] | None) -> str | None:
        if not node_metadata:
            return None
        return str(node_metadata.get("path") or "") or None

    @staticmethod
    def _apply_filters(
        stmt: Select[tuple[object, ...]],
        filters: RetrievalFilters | None,
    ) -> Select[tuple[object, ...]]:
        if filters is None:
            return stmt
        if filters.document_id is not None:
            stmt = stmt.where(TextNode.document_id == filters.document_id)
        if filters.source_id is not None:
            stmt = stmt.where(DocumentNode.source_id == filters.source_id)
        # TODO: Category filtering disabled for now because the model can  # noqa: FIX002, TD002, TD003
        # emit nonexistent categories, which zeroes out results.
        return stmt

    def _rrf_merge(
        self,
        text_chunks: list[RetrievedChunk],
        vector_chunks: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        candidates: dict[UUID, RetrievedChunk] = {}

        for rank, chunk in enumerate(text_chunks, start=1):
            candidate = candidates.get(chunk.chunk_id)
            if candidate is None:
                candidate = chunk.model_copy(deep=True)
                candidates[chunk.chunk_id] = candidate
            candidate.text_rank = rank

        for rank, chunk in enumerate(vector_chunks, start=1):
            candidate = candidates.get(chunk.chunk_id)
            if candidate is None:
                candidate = chunk.model_copy(deep=True)
                candidates[chunk.chunk_id] = candidate
            candidate.vector_rank = rank

        for candidate in candidates.values():
            candidate.rrf_score = self._rrf_score(
                text_rank=candidate.text_rank,
                vector_rank=candidate.vector_rank,
            )

        return sorted(
            candidates.values(),
            key=lambda item: item.rrf_score,
            reverse=True,
        )

    @staticmethod
    def _rrf_score(
        *,
        text_rank: int | None,
        vector_rank: int | None,
    ) -> float:
        score = 0.0
        if text_rank is not None:
            score += RETRIEVAL_SETTINGS.text_weight / (
                RETRIEVAL_SETTINGS.rrf_k + text_rank
            )
        if vector_rank is not None:
            score += RETRIEVAL_SETTINGS.semantic_weight / (
                RETRIEVAL_SETTINGS.rrf_k + vector_rank
            )
        return score

    async def _expand_with_neighbors(self, chunks: list[RetrievedChunk]) -> None:
        chunk_ids = [chunk.chunk_id for chunk in chunks]
        if not chunk_ids:
            return

        base = await self._load_base_chunks(chunk_ids)
        prev_texts = await self._load_prev_texts(base)
        next_texts = await self._load_next_texts(chunk_ids)

        for chunk in chunks:
            base_row = base.get(chunk.chunk_id)
            if base_row is None:
                continue
            chunk.text = self._merge_neighbor_texts(
                current_text=base_row[0] or chunk.text,
                prev_text=prev_texts.get(base_row[1]),
                next_text=next_texts.get(chunk.chunk_id),
            )

    async def _load_base_chunks(
        self,
        chunk_ids: list[UUID],
    ) -> dict[UUID, tuple[str | None, UUID | None]]:
        stmt = select(
            TextNode.id,
            TextNode.text,
            TextNode.prev_id,
        ).where(TextNode.id.in_(chunk_ids))
        rows = await self.session.execute(stmt)
        return {chunk_id: (text, prev_id) for chunk_id, text, prev_id in rows.tuples()}

    async def _load_prev_texts(
        self,
        base: dict[UUID, tuple[str | None, UUID | None]],
    ) -> dict[UUID, str]:
        prev_ids = [prev_id for _, prev_id in base.values() if prev_id is not None]
        if not prev_ids:
            return {}

        stmt = select(
            TextNode.id,
            TextNode.text,
        ).where(TextNode.id.in_(prev_ids))
        rows = await self.session.execute(stmt)
        return {chunk_id: text for chunk_id, text in rows.tuples() if text is not None}

    async def _load_next_texts(self, chunk_ids: list[UUID]) -> dict[UUID, str]:
        stmt = select(
            TextNode.prev_id,
            TextNode.text,
        ).where(TextNode.prev_id.in_(chunk_ids))
        rows = await self.session.execute(stmt)

        next_texts: dict[UUID, str] = {}
        for prev_id, text in rows.tuples():
            if prev_id is None or text is None:
                continue
            next_texts.setdefault(prev_id, text)
        return next_texts

    @staticmethod
    def _merge_neighbor_texts(
        current_text: str,
        *,
        prev_text: str | None,
        next_text: str | None,
    ) -> str:
        parts: list[str] = []
        if prev_text:
            parts.append(prev_text)
        parts.append(current_text)
        if next_text:
            parts.append(next_text)

        merged = "\n\n".join(parts)
        if len(merged) > RETRIEVAL_SETTINGS.max_merged_chars:
            return merged[: RETRIEVAL_SETTINGS.max_merged_chars]
        return merged
