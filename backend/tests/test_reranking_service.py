"""Tests for reranking service."""

import uuid

import pytest

from src.schemas.reranking import Passage, RerankingModelInput, RerankingModelResult
from src.services.reranking_service import RerankingService, RerankingServiceError


class _StubModel:
    def __init__(
        self,
        *,
        responses: list[RerankingModelResult] | None = None,
        error: Exception | None = None,
    ) -> None:
        self._responses = responses or []
        self._error = error
        self.calls: list[RerankingModelInput] = []

    def rerank(self, request: RerankingModelInput) -> list[RerankingModelResult]:
        self.calls.append(request)
        if self._error is not None:
            raise self._error
        return self._responses


def test_rerank_passes_request_to_model(monkeypatch) -> None:
    passage = Passage(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "passage-1"),
        passage="test passage",
    )
    expected = RerankingModelResult(
        query="test query",
        passage=passage,
        relevance_score=0.9,
    )
    model = _StubModel(responses=[expected])
    monkeypatch.setattr(
        "src.services.reranking_service.get_litellm_reranking_model",
        lambda _settings: model,
    )

    service = RerankingService()
    result = service.rerank(query="test query", passages=[passage])

    assert result == [expected]
    assert len(model.calls) == 1
    assert model.calls[0].query == "test query"
    assert model.calls[0].passages == [passage]


def test_rerank_wraps_model_errors(monkeypatch) -> None:
    model = _StubModel(error=RuntimeError("model failed"))
    monkeypatch.setattr(
        "src.services.reranking_service.get_litellm_reranking_model",
        lambda _settings: model,
    )

    service = RerankingService()
    passage = Passage(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "passage-1"),
        passage="test passage",
    )

    with pytest.raises(RerankingServiceError, match="Failed to run reranking"):
        service.rerank(query="test query", passages=[passage])
