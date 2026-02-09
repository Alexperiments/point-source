"""Retrieval service."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import logfire
from sqlalchemy import Float, bindparam, func, select
from sqlalchemy.dialects.postgresql import JSONB
from sqlalchemy.orm import selectinload
from sqlalchemy.sql import cast

from src.core.rag_config import (
    EMBEDDING_SETTINGS,
    RERANKER_SETTINGS,
    RETRIEVAL_SETTINGS,
)
from src.models.node import DocumentNode, TextNode
from src.schemas.retrieval import RetrievalFilters, RetrievedChunk
from src.services.embedding_service import get_embedding_service


if TYPE_CHECKING:
    from redis.asyncio import Redis
    from sqlalchemy.ext.asyncio import AsyncSession
    from sqlalchemy.sql import Select


@dataclass
class _Candidate:
    chunk_id: Any
    document_id: Any
    source_id: str
    text: str
    node_metadata: dict[str, Any] | None
    text_rank: int | None = None
    vector_rank: int | None = None
    rrf_score: float = 0.0

    @property
    def path(self) -> str | None:
        if not self.node_metadata:
            return None
        return str(self.node_metadata.get("path") or "") or None


class RetrievalService:
    """Service for retrieving relevant chunks."""

    def __init__(self, session: AsyncSession, redis: Redis) -> None:
        """Initialize the retrieval service."""
        self.session = session
        self.redis = redis
        self._embedder = get_embedding_service()

    async def retrieve(
        self,
        query: str,
        *,
        filters: RetrievalFilters | None = None,
        use_prev_next: bool | None = None,
    ) -> list[RetrievedChunk]:
        """Retrieve relevant chunks for a query."""
        normalized = self._normalize_query(query)
        if not normalized:
            return []

        include_neighbors = (
            use_prev_next
            if use_prev_next is not None
            else RETRIEVAL_SETTINGS.use_prev_next
        )

        cache_key = self._cache_key(
            normalized_query=normalized,
            filters=filters,
            include_neighbors=include_neighbors,
        )

        cached = await self._get_cached(cache_key)
        if cached is not None:
            return cached

        with logfire.span("retrieval.embed_query"):
            embedding = self._embed_query(normalized)

        with logfire.span("retrieval.text_search"):
            text_rows = await self._text_candidates(normalized, filters)

        with logfire.span("retrieval.vector_search"):
            vector_rows = await self._vector_candidates(embedding, filters)

        with logfire.span("retrieval.rrf_merge"):
            merged = self._rrf_merge(text_rows, vector_rows)

        with logfire.span("retrieval.rerank"):
            reranked = self._mock_rerank(normalized, merged)

        reranked = reranked[: RETRIEVAL_SETTINGS.top_n]

        results = [
            RetrievedChunk(
                chunk_id=c.chunk_id,
                document_id=c.document_id,
                source_id=c.source_id,
                path=c.path,
                text=c.text,
                text_rank=c.text_rank,
                vector_rank=c.vector_rank,
                rrf_score=c.rrf_score,
                citations=[c.source_id],
            )
            for c in reranked
        ]

        if include_neighbors and results:
            with logfire.span("retrieval.expand_context"):
                await self._expand_with_neighbors(results)

        await self._set_cached(cache_key, results)
        return results

    @staticmethod
    def _normalize_query(query: str) -> str:
        return re.compile(r"\s+").sub(" ", query).strip()

    def _cache_key(
        self,
        *,
        normalized_query: str,
        filters: RetrievalFilters | None,
        include_neighbors: bool,
    ) -> str:
        payload = {
            "query": normalized_query,
            "filters": (filters.model_dump(exclude_none=True) if filters else None),
            "include_neighbors": include_neighbors,
            "retrieval": {
                "top_n": RETRIEVAL_SETTINGS.top_n,
                "text_top_k": RETRIEVAL_SETTINGS.text_top_k,
                "vector_top_k": RETRIEVAL_SETTINGS.vector_top_k,
                "rrf_k": RETRIEVAL_SETTINGS.rrf_k,
                "semantic_weight": RETRIEVAL_SETTINGS.semantic_weight,
                "text_weight": RETRIEVAL_SETTINGS.text_weight,
            },
            "embedding": {"model_name": EMBEDDING_SETTINGS.model_name},
            "reranker": {
                "enabled": RERANKER_SETTINGS.enabled,
                "model_name": RERANKER_SETTINGS.model_name,
                "top_k": RERANKER_SETTINGS.top_k,
            },
        }
        raw = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(raw).hexdigest()
        return f"{RETRIEVAL_SETTINGS.cache_prefix}{digest}"

    async def _get_cached(self, cache_key: str) -> list[RetrievedChunk] | None:
        if self.redis is None:
            return None
        cached = await self.redis.get(cache_key)
        if not cached:
            return None
        if isinstance(cached, bytes):
            cached = cached.decode("utf-8")
        try:
            payload = json.loads(cached)
        except json.JSONDecodeError:
            return None
        return [RetrievedChunk.model_validate(item) for item in payload]

    async def _set_cached(self, cache_key: str, results: list[RetrievedChunk]) -> None:
        if self.redis is None:
            return
        payload = json.dumps([r.to_cache_dict() for r in results], default=str)
        await self.redis.set(
            cache_key,
            payload,
            ex=RETRIEVAL_SETTINGS.cache_ttl_seconds,
        )

    def _embed_query(self, query: str) -> list[float]:
        embedding = self._embedder.encode_query(
            [query],
            batch_size=EMBEDDING_SETTINGS.query_batch_size,
        )
        return embedding[0].tolist()

    async def _text_candidates(
        self,
        query: str,
        filters: RetrievalFilters | None,
    ) -> list[Any]:
        ts_query = func.websearch_to_tsquery("simple", bindparam("query"))
        ts_vector = func.to_tsvector("simple", TextNode.text)
        ts_rank = func.ts_rank_cd(ts_vector, ts_query)

        stmt = (
            select(
                TextNode.id.label("chunk_id"),
                TextNode.document_id.label("document_id"),
                DocumentNode.source_id.label("source_id"),
                TextNode.text.label("text"),
                TextNode.node_metadata.label("node_metadata"),
                ts_rank.label("text_rank"),
            )
            .join(DocumentNode, TextNode.document_id == DocumentNode.id)
            .where(TextNode.text != "")
            .where(ts_vector.op("@@")(ts_query))
            .order_by(ts_rank.desc())
            .limit(RETRIEVAL_SETTINGS.text_top_k)
        )

        stmt = self._apply_filters(stmt, filters)
        result = await self.session.execute(stmt.params(query=query))
        return list(result.all())

    async def _vector_candidates(
        self,
        embedding: list[float],
        filters: RetrievalFilters | None,
    ) -> list[Any]:
        distance = TextNode.embedding.op("<=>")(bindparam("embedding"))

        stmt = (
            select(
                TextNode.id.label("chunk_id"),
                TextNode.document_id.label("document_id"),
                DocumentNode.source_id.label("source_id"),
                TextNode.text.label("text"),
                TextNode.node_metadata.label("node_metadata"),
                func.cast(distance, Float).label("distance"),
            )
            .join(DocumentNode, TextNode.document_id == DocumentNode.id)
            .where(TextNode.text != "", TextNode.embedding.is_not(None))
            .order_by(distance)
            .limit(RETRIEVAL_SETTINGS.vector_top_k)
        )

        stmt = self._apply_filters(stmt, filters)
        result = await self.session.execute(stmt.params(embedding=embedding))
        return list(result.all())

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
        if filters.category:
            stmt = stmt.where(
                cast(TextNode.node_metadata, JSONB).contains(
                    {"category": filters.category},
                ),
            )
        return stmt

    @staticmethod
    def _rrf_merge(text_rows: list[Any], vector_rows: list[Any]) -> list[_Candidate]:
        candidates: dict[Any, _Candidate] = {}

        for rank, row in enumerate(text_rows, start=1):
            candidate = candidates.get(row.chunk_id)
            if candidate is None:
                candidate = _Candidate(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    source_id=row.source_id,
                    text=row.text,
                    node_metadata=row.node_metadata,
                )
                candidates[row.chunk_id] = candidate
            candidate.text_rank = rank

        for rank, row in enumerate(vector_rows, start=1):
            candidate = candidates.get(row.chunk_id)
            if candidate is None:
                candidate = _Candidate(
                    chunk_id=row.chunk_id,
                    document_id=row.document_id,
                    source_id=row.source_id,
                    text=row.text,
                    node_metadata=row.node_metadata,
                )
                candidates[row.chunk_id] = candidate
            candidate.vector_rank = rank

        for candidate in candidates.values():
            score = 0.0
            if candidate.text_rank is not None:
                score += RETRIEVAL_SETTINGS.text_weight / (
                    RETRIEVAL_SETTINGS.rrf_k + candidate.text_rank
                )
            if candidate.vector_rank is not None:
                score += RETRIEVAL_SETTINGS.semantic_weight / (
                    RETRIEVAL_SETTINGS.rrf_k + candidate.vector_rank
                )
            candidate.rrf_score = score

        return sorted(
            candidates.values(),
            key=lambda c: c.rrf_score,
            reverse=True,
        )

    @staticmethod
    def _mock_rerank(query: str, candidates: list[_Candidate]) -> list[_Candidate]:
        del query
        if not RERANKER_SETTINGS.enabled:
            return candidates
        return candidates[: RERANKER_SETTINGS.top_k]

    async def _expand_with_neighbors(self, results: list[RetrievedChunk]) -> None:
        ids = [r.chunk_id for r in results]
        if not ids:
            return

        stmt = (
            select(TextNode)
            .options(
                selectinload(TextNode.prev_node),
                selectinload(TextNode.next_node),
            )
            .where(TextNode.id.in_(ids))
        )
        rows = await self.session.execute(stmt)
        nodes = {node.id: node for node in rows.scalars().all()}

        for item in results:
            node = nodes.get(item.chunk_id)
            if node is None:
                continue

            parts: list[str] = []
            if node.prev_node and node.prev_node.text:
                parts.append(node.prev_node.text)
            if node.text:
                parts.append(node.text)
            if node.next_node and node.next_node.text:
                parts.append(node.next_node.text)

            merged = "\n\n".join(part for part in parts if part)
            if len(merged) > RETRIEVAL_SETTINGS.max_merged_chars:
                merged = merged[: RETRIEVAL_SETTINGS.max_merged_chars]
            item.text = merged or item.text
