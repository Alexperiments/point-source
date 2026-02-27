"""Retrieval service."""

from __future__ import annotations

import asyncio
import re
from typing import TYPE_CHECKING

import logfire
from sqlalchemy import text as sa_text

from src.core.rag_config import RERANKER_SETTINGS, RETRIEVAL_SETTINGS
from src.schemas.reranking import Passage
from src.schemas.retrieval import RetrievedChunk
from src.services.embedding_service import (
    EmbeddingService,
)
from src.services.reranking_service import RerankingService, RerankingServiceError


if TYPE_CHECKING:
    from sqlalchemy.ext.asyncio import AsyncSession

    from src.schemas.reranking import RerankingModelResult

_QUERY_WHITESPACE_RE = re.compile(r"\s+")
_HYBRID_RETRIEVE_SQL = sa_text(
    """
    SELECT
        r.chunk_id,
        r.document_id,
        d.doi_url,
        d.authors,
        d.title,
        d.journal_ref,
        r.chunk_text,
        r.node_metadata,
        r.text_rank,
        r.vector_rank,
        r.rrf_score
    FROM processed.hybrid_retrieve(
        :query,
        CAST(:embedding AS halfvec(1024)),
        :text_k,
        :vec_k,
        :fused_k,
        :rrf_k,
        :text_weight,
        :vector_weight
    ) AS r
    JOIN processed.documents AS d
      ON d.id = r.document_id
    """,
)


class RetrievalService:
    """Retrieve relevant chunks for a query."""

    def __init__(
        self,
        session: AsyncSession,
        reranker: RerankingService | None = None,
        embedder: EmbeddingService | None = None,
    ) -> None:
        """Initialize the retrieval service."""
        self.session = session
        self._embedder = embedder or EmbeddingService()
        self._reranker = reranker or RerankingService()

    async def retrieve(
        self,
        query: str,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        normalized_query = self._normalize_query(query)
        if not normalized_query:
            return []

        embedding_result = await self._embedder.embed_query(normalized_query)
        query_embedding = list(embedding_result.embeddings[0])

        merged = await self._hybrid_candidates(normalized_query, query_embedding)
        if merged and RERANKER_SETTINGS.enabled:
            try:
                reranked = await asyncio.to_thread(
                    self._reranker.rerank,
                    normalized_query,
                    [
                        Passage(id=chunk.chunk_id, passage=chunk.text)
                        for chunk in merged
                    ],
                )
                merged = self._apply_reranking(merged, reranked)
            except RerankingServiceError as e:
                logfire.exception(f"Reranking failed: {e!s}")

        return merged[: RERANKER_SETTINGS.top_k]

    @staticmethod
    def _normalize_query(query: str) -> str:
        return _QUERY_WHITESPACE_RE.sub(" ", query).strip()

    async def _hybrid_candidates(
        self,
        query: str,
        embedding: list[float],
    ) -> list[RetrievedChunk]:
        fused_k = RETRIEVAL_SETTINGS.text_top_k + RETRIEVAL_SETTINGS.vector_top_k
        rows = await self.session.execute(
            _HYBRID_RETRIEVE_SQL,
            {
                "query": query,
                "embedding": self._vector_literal(embedding),
                "text_k": RETRIEVAL_SETTINGS.text_top_k,
                "vec_k": RETRIEVAL_SETTINGS.vector_top_k,
                "fused_k": fused_k,
                "rrf_k": RETRIEVAL_SETTINGS.rrf_k,
                "text_weight": RETRIEVAL_SETTINGS.text_weight,
                "vector_weight": RETRIEVAL_SETTINGS.semantic_weight,
            },
        )

        return [
            RetrievedChunk(
                chunk_id=chunk_id,
                document_id=document_id,
                doi_url=doi_url,
                authors=authors,
                title=title,
                journal_ref=journal_ref,
                path=node_metadata.get("path") if node_metadata else None,
                text=chunk_text,
                text_rank=text_rank,
                vector_rank=vector_rank,
                rrf_score=rrf_score,
            )
            for (
                chunk_id,
                document_id,
                doi_url,
                authors,
                title,
                journal_ref,
                chunk_text,
                node_metadata,
                text_rank,
                vector_rank,
                rrf_score,
            ) in rows.tuples()
        ]

    @staticmethod
    def _vector_literal(embedding: list[float]) -> str:
        return f"[{','.join(format(value, '.12g') for value in embedding)}]"

    @staticmethod
    def _apply_reranking(
        candidates: list[RetrievedChunk],
        reranked: list[RerankingModelResult],
    ) -> list[RetrievedChunk]:
        if not reranked:
            return candidates

        ordered_candidates: list[RetrievedChunk] = []
        candidates_by_id = {
            candidate.chunk_id: candidate.model_copy(deep=True)
            for candidate in candidates
        }

        for result in reranked:
            chunk_id = result.passage.id
            candidate = candidates_by_id.pop(chunk_id, None)
            if candidate is None:
                continue

            candidate.relevance_score = result.relevance_score
            ordered_candidates.append(candidate)

        ordered_candidates.extend(candidates_by_id.values())

        return ordered_candidates
