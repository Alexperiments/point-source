"""Tests for reranking service."""

import uuid

from src.core.rag_config import RERANKER_SETTINGS
from src.schemas.protocols import Reranker
from src.schemas.retrieval import RetrievedChunk
from src.services.reranking_service import (
    BasicLexicalReranker,
    LiteLLMModelReranker,
    RerankingModelSettings,
    RerankingService,
    get_reranking_model,
)


def _make_chunk(index: int) -> RetrievedChunk:
    return RetrievedChunk(
        chunk_id=uuid.uuid4(),
        document_id=f"doc-{index}",
        url=f"https://arxiv.org/abs/source-{index}",
        text=f"text-{index}",
    )


class _FailingReranker(Reranker):
    def score(self, _query: str, _passages: list[str]) -> list[float]:
        msg = "score should not be called when reranker is disabled"
        raise AssertionError(msg)


class _StaticReranker(Reranker):
    def score(self, _query: str, passages: list[str]) -> list[float]:
        return [float(len(passages) - i) for i, _ in enumerate(passages)]


def test_rerank_returns_all_candidates_when_disabled(monkeypatch) -> None:
    """Reranking should be a no-op when disabled."""
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", False)

    service = RerankingService(model=_FailingReranker())
    candidates = [_make_chunk(index) for index in range(3)]

    reranked = service.rerank(query="test query", candidates=candidates)

    assert reranked == candidates


def test_rerank_limits_candidates_to_top_k(monkeypatch) -> None:
    """Reranking should trim candidates when enabled."""
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", True)
    monkeypatch.setattr(RERANKER_SETTINGS, "top_k", 2)

    service = RerankingService(model=_StaticReranker())
    candidates = [_make_chunk(index) for index in range(4)]

    reranked = service.rerank(query="test query", candidates=candidates)

    assert reranked == candidates[:2]


def test_rerank_populates_relevance_score(monkeypatch) -> None:
    """Reranking should attach model relevance scores to candidates."""
    monkeypatch.setattr(RERANKER_SETTINGS, "enabled", True)
    monkeypatch.setattr(RERANKER_SETTINGS, "top_k", 4)

    service = RerankingService(model=_StaticReranker())
    candidates = [_make_chunk(index) for index in range(4)]

    reranked = service.rerank(query="test query", candidates=candidates)

    assert [candidate.relevance_score for candidate in candidates] == [4.0, 3.0, 2.0, 1.0]
    assert [candidate.relevance_score for candidate in reranked] == [4.0, 3.0, 2.0, 1.0]


def test_get_reranking_model_defaults_to_model_backed_reranker() -> None:
    """Model names should resolve to the LiteLLM-backed reranker by default."""
    model = get_reranking_model(
        RerankingModelSettings(
            model_name="jina_ai/jina-reranker-v3",
            timeout_seconds=30,
        ),
    )

    assert isinstance(model, LiteLLMModelReranker)


def test_get_reranking_model_uses_lexical_for_mock_name() -> None:
    """Mock/local model names should map to the lexical fallback reranker."""
    model = get_reranking_model(
        RerankingModelSettings(
            model_name="mock-reranker",
            timeout_seconds=30,
        ),
    )

    assert isinstance(model, BasicLexicalReranker)
