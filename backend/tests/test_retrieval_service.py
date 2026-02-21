"""Unit tests for retrieval service behavior."""

import uuid
from dataclasses import dataclass
from unittest.mock import AsyncMock

import pytest

from src.core.rag_config import RETRIEVAL_SETTINGS
from src.schemas.retrieval import RetrievedChunk
from src.services.retrieval_service import RetrievalService


class _DummySession:
    async def execute(self, *_args, **_kwargs):  # noqa: ANN002, ANN003
        msg = "execute should not be called in this unit test"
        raise AssertionError(msg)


def _chunk(name: str, *, url: str = "https://arxiv.org/abs/paper-1", path: str | None = None) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid5(uuid.NAMESPACE_DNS, f"chunk-{name}"),
        document_id=f"doc-{name}",
        url=url,
        path=path,
        text=f"text-{name}",
    )


@dataclass
class _RecordingReranker:
    def __post_init__(self) -> None:
        self.calls: list[tuple[str, list[RetrievedChunk]]] = []

    def rerank(
        self,
        query: str,
        candidates: list[RetrievedChunk],
    ) -> list[RetrievedChunk]:
        self.calls.append((query, candidates))
        return candidates


@pytest.mark.asyncio
async def test_retrieve_returns_empty_for_blank_queries(monkeypatch) -> None:
    reranker = _RecordingReranker()
    service = RetrievalService(session=_DummySession(), reranker=reranker)

    embed_mock = AsyncMock(return_value=[0.1, 0.2])
    text_mock = AsyncMock(return_value=[])
    vector_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(service, "_embed_query_async", embed_mock)
    monkeypatch.setattr(service, "_text_candidates", text_mock)
    monkeypatch.setattr(service, "_vector_candidates", vector_mock)

    chunks = await service.retrieve("  \n\t ")

    assert chunks == []
    embed_mock.assert_not_awaited()
    text_mock.assert_not_awaited()
    vector_mock.assert_not_awaited()
    assert reranker.calls == []


@pytest.mark.asyncio
async def test_retrieve_normalizes_query_and_calls_candidates(monkeypatch) -> None:
    reranker = _RecordingReranker()
    service = RetrievalService(session=_DummySession(), reranker=reranker)

    embed_mock = AsyncMock(return_value=[0.3, 0.7])
    text_mock = AsyncMock(return_value=[])
    vector_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(service, "_embed_query_async", embed_mock)
    monkeypatch.setattr(service, "_text_candidates", text_mock)
    monkeypatch.setattr(service, "_vector_candidates", vector_mock)

    await service.retrieve("  graph\n   theory   ")

    embed_mock.assert_awaited_once_with("graph theory")
    text_mock.assert_awaited_once_with("graph theory")
    vector_mock.assert_awaited_once_with([0.3, 0.7])
    assert reranker.calls[0][0] == "graph theory"


def test_rrf_merge_dedupes_and_ranks(monkeypatch) -> None:
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "rrf_k", 60)
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "text_weight", 0.3)
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "semantic_weight", 0.7)

    service = RetrievalService(session=_DummySession(), reranker=_RecordingReranker())

    chunk_a = _chunk("a", url="https://arxiv.org/abs/doc-a", path="/a")
    chunk_b = _chunk("b", url="https://arxiv.org/abs/doc-b", path="/b")
    chunk_c = _chunk("c", url="https://arxiv.org/abs/doc-c", path="/c")

    merged = service._rrf_merge(
        text_chunks=[chunk_a, chunk_b],
        vector_chunks=[chunk_b, chunk_c],
    )

    assert [chunk.chunk_id for chunk in merged] == [
        chunk_b.chunk_id,
        chunk_c.chunk_id,
        chunk_a.chunk_id,
    ]

    by_id = {chunk.chunk_id: chunk for chunk in merged}
    assert by_id[chunk_b.chunk_id].text_rank == 2
    assert by_id[chunk_b.chunk_id].vector_rank == 1
    assert by_id[chunk_b.chunk_id].path == "/b"
    assert str(by_id[chunk_b.chunk_id].url) == "https://arxiv.org/abs/doc-b"

    assert by_id[chunk_b.chunk_id].rrf_score == pytest.approx(
        (0.3 / (60 + 2)) + (0.7 / (60 + 1)),
    )
    assert by_id[chunk_c.chunk_id].rrf_score == pytest.approx(0.7 / (60 + 2))
    assert by_id[chunk_a.chunk_id].rrf_score == pytest.approx(0.3 / (60 + 1))


@pytest.mark.asyncio
async def test_retrieve_applies_reranker_before_top_n(monkeypatch) -> None:
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "top_n", 1)

    @dataclass
    class _ReverseReranker:
        def __post_init__(self) -> None:
            self.calls: list[list[uuid.UUID]] = []

        def rerank(
            self,
            _query: str,
            candidates: list[RetrievedChunk],
        ) -> list[RetrievedChunk]:
            self.calls.append([item.chunk_id for item in candidates])
            return list(reversed(candidates))

    reranker = _ReverseReranker()
    service = RetrievalService(session=_DummySession(), reranker=reranker)

    embed_mock = AsyncMock(return_value=[1.0])
    text_mock = AsyncMock(return_value=[_chunk("a"), _chunk("b")])
    vector_mock = AsyncMock(return_value=[])

    monkeypatch.setattr(service, "_embed_query_async", embed_mock)
    monkeypatch.setattr(service, "_text_candidates", text_mock)
    monkeypatch.setattr(service, "_vector_candidates", vector_mock)

    chunks = await service.retrieve("query")

    assert reranker.calls == [[_chunk("a").chunk_id, _chunk("b").chunk_id]]
    assert [chunk.chunk_id for chunk in chunks] == [_chunk("b").chunk_id]


@pytest.mark.asyncio
async def test_retrieve_expands_neighbors_only_when_requested(monkeypatch) -> None:
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "use_prev_next", False)

    service = RetrievalService(
        session=_DummySession(),
        reranker=_RecordingReranker(),
    )

    embed_mock = AsyncMock(return_value=[1.0])
    text_mock = AsyncMock(side_effect=[[_chunk("target")], [_chunk("target")]])
    vector_mock = AsyncMock(return_value=[])
    expand_mock = AsyncMock()

    monkeypatch.setattr(service, "_embed_query_async", embed_mock)
    monkeypatch.setattr(service, "_text_candidates", text_mock)
    monkeypatch.setattr(service, "_vector_candidates", vector_mock)
    monkeypatch.setattr(service, "_expand_with_neighbors", expand_mock)

    await service.retrieve("query")
    await service.retrieve("query", use_prev_next=True)

    assert expand_mock.await_count == 1


def test_merge_neighbor_texts_truncates_by_setting(monkeypatch) -> None:
    monkeypatch.setattr(RETRIEVAL_SETTINGS, "max_merged_chars", 10)

    merged = RetrievalService._merge_neighbor_texts(
        current_text="text-target",
        prev_text="abcde",
        next_text="klmno",
    )

    assert merged == "abcde\n\ntex"
