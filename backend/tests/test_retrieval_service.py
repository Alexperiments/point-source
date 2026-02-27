"""Unit tests for retrieval service behavior."""

import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest
from pydantic_ai import EmbeddingResult

from src.core.rag_config import RERANKER_SETTINGS
from src.schemas.reranking import Passage, RerankingModelResult
from src.schemas.retrieval import RetrievedChunk
from src.services.retrieval_service import RetrievalService


class _DummySession:
    async def execute(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        msg = "execute should not be called in this unit test"
        raise AssertionError(msg)


def _chunk(
    name: str,
    *,
    doi_url: str = "https://www.doi.org/10.48550/arXiv.paper-1",
    path: str | None = None,
) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{name}"),
        document_id=f"doc-{name}",
        doi_url=doi_url,
        path=path,
        text=f"text-{name}",
    )


@dataclass
class _RecordingReranker:
    def __post_init__(self) -> None:
        self.calls: list[tuple[str, list[Passage]]] = []

    def rerank(
        self,
        query: str,
        passages: list[Passage],
    ) -> list[RerankingModelResult]:
        self.calls.append((query, passages))
        return [
            RerankingModelResult(
                query=query,
                passage=passage,
                relevance_score=float(len(passages) - index),
            )
            for index, passage in enumerate(passages)
        ]


@dataclass
class _RecordingEmbedder:
    embedding: list[float]

    def __post_init__(self) -> None:
        self.calls: list[str] = []

    async def embed_query(self, query: str) -> EmbeddingResult:
        self.calls.append(query)
        return EmbeddingResult(
            embeddings=[self.embedding],
            inputs=[query],
            input_type="query",
            model_name="test-embedding-model",
            provider_name="test-provider",
        )


@pytest.mark.asyncio
async def test_retrieve_returns_empty_for_blank_queries(monkeypatch) -> None:
    reranker = _RecordingReranker()
    embedder = _RecordingEmbedder([0.1, 0.2])
    service = RetrievalService(
        session=_DummySession(),
        reranker=reranker,
        embedder=embedder,
    )

    hybrid_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(service, "_hybrid_candidates", hybrid_mock)

    chunks = await service.retrieve("  \n\t ")

    assert chunks == []
    assert embedder.calls == []
    hybrid_mock.assert_not_awaited()
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_retrieve_normalizes_query_and_calls_candidates(monkeypatch) -> None:
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", True)

    reranker = _RecordingReranker()
    embedder = _RecordingEmbedder([0.3, 0.7])
    service = RetrievalService(
        session=_DummySession(),
        reranker=reranker,
        embedder=embedder,
    )

    hybrid_mock = AsyncMock(return_value=[_chunk("a")])

    monkeypatch.setattr(service, "_hybrid_candidates", hybrid_mock)

    await service.retrieve("  graph\n   theory   ")

    assert embedder.calls == ["graph theory"]
    hybrid_mock.assert_awaited_once_with("graph theory", [0.3, 0.7])
    assert reranker.calls[0][0] == "graph theory"


@pytest.mark.asyncio
async def test_retrieve_applies_reranker_before_top_n(monkeypatch) -> None:
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", True)
    monkeypatch.setattr(RERANKER_SETTINGS, "top_k", 1)

    @dataclass
    class _ReverseReranker:
        def __post_init__(self) -> None:
            self.calls: list[list[str]] = []

        def rerank(
            self,
            query: str,
            passages: list[Passage],
        ) -> list[RerankingModelResult]:
            self.calls.append([item.passage for item in passages])
            return [
                RerankingModelResult(
                    query=query,
                    passage=passages[index],
                    relevance_score=float(len(passages) - rank),
                )
                for rank, index in enumerate(reversed(range(len(passages))))
            ]

    reranker = _ReverseReranker()
    service = RetrievalService(
        session=_DummySession(),
        reranker=reranker,
        embedder=_RecordingEmbedder([1.0]),
    )

    hybrid_mock = AsyncMock(return_value=[_chunk("a"), _chunk("b")])
    monkeypatch.setattr(service, "_hybrid_candidates", hybrid_mock)

    chunks = await service.retrieve("query")

    assert reranker.calls == [["text-a", "text-b"]]
    assert [chunk.chunk_id for chunk in chunks] == [_chunk("b").chunk_id]
    assert chunks[0].relevance_score is not None


@pytest.mark.asyncio
async def test_retrieve_skips_reranker_when_disabled(monkeypatch) -> None:
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", False)
    monkeypatch.setattr(RERANKER_SETTINGS, "top_k", 2)

    reranker = _RecordingReranker()
    service = RetrievalService(
        session=_DummySession(),
        reranker=reranker,
        embedder=_RecordingEmbedder([1.0]),
    )

    hybrid_mock = AsyncMock(return_value=[_chunk("a"), _chunk("b")])
    monkeypatch.setattr(service, "_hybrid_candidates", hybrid_mock)

    chunks = await service.retrieve("query")

    assert reranker.calls == []
    assert [chunk.chunk_id for chunk in chunks] == [_chunk("a").chunk_id, _chunk("b").chunk_id]
