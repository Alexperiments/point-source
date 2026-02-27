"""Tests for LiteLLM model provider builders."""

import uuid
from unittest.mock import MagicMock

from pydantic import BaseModel

from src.core.model_provider import litellm as litellm_module
from src.core.rag_config import EmbeddingSettings, RerankerSettings
from src.schemas.reranking import Passage, RerankingModelInput


def test_get_litellm_embedding_model_builds_openai_embedding_model(monkeypatch) -> None:
    """Embedder should wrap an OpenAIEmbeddingModel configured with LiteLLM provider."""
    provider_instance = MagicMock()
    provider_factory = MagicMock(return_value=provider_instance)
    embedding_model_instance = MagicMock()
    embedding_model_factory = MagicMock(return_value=embedding_model_instance)
    embedder_instance = MagicMock()
    embedder_factory = MagicMock(return_value=embedder_instance)

    monkeypatch.setattr(litellm_module, "LiteLLMProvider", provider_factory)
    monkeypatch.setattr(litellm_module, "OpenAIEmbeddingModel", embedding_model_factory)
    monkeypatch.setattr(litellm_module, "Embedder", embedder_factory)

    settings = EmbeddingSettings(
        model_name="jina_ai/jina-embeddings-v3",
        max_tokens=128,
        batch_size=4,
        query_instruction="test instruction",
        embedding_size=1024,
    )

    model = litellm_module.get_litellm_embedding_model(settings)

    assert model is embedder_instance
    provider_factory.assert_called_once_with(
        api_base=litellm_module.app_settings.litellm_base_url.unicode_string(),
        api_key=litellm_module.app_settings.litellm_api_key.get_secret_value(),
    )
    embedding_model_factory.assert_called_once_with(
        model_name=settings.model_name,
        provider=provider_instance,
        settings={"dimensions": settings.embedding_size},
    )
    embedder_factory.assert_called_once_with(model=embedding_model_instance)


class _StubRerankRow(BaseModel):
    index: int
    relevance_score: float


class _StubRerankResponse(BaseModel):
    id: str | None = None
    results: list[_StubRerankRow]


def test_litellm_rerank_score_accepts_pydantic_response(monkeypatch) -> None:
    """Reranker should accept LiteLLM pydantic response objects."""
    rerank_response = _StubRerankResponse(
        id="run-1",
        results=[
            _StubRerankRow(index=1, relevance_score=0.7),
            _StubRerankRow(index=0, relevance_score=0.2),
        ],
    )
    rerank_mock = MagicMock(return_value=rerank_response)
    monkeypatch.setattr(litellm_module.litellm, "rerank", rerank_mock)

    model = litellm_module.LiteLLMRerankingModel(RerankerSettings())
    first_passage = Passage(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "passage-1"),
        passage="first passage",
    )
    second_passage = Passage(
        id=uuid.uuid5(uuid.NAMESPACE_DNS, "passage-2"),
        passage="second passage",
    )
    request = RerankingModelInput(
        query="what is dark matter?",
        passages=[first_passage, second_passage],
    )

    scored = model.score(request)

    assert [item.passage.id for item in scored] == [
        second_passage.id,
        first_passage.id,
    ]
    assert [item.relevance_score for item in scored] == [0.7, 0.2]
    assert all(item.run_id == "run-1" for item in scored)
