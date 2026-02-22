"""Tests for embedding service behavior and provider routing."""

from __future__ import annotations

from collections.abc import Sequence
from unittest.mock import AsyncMock, MagicMock

import pytest
from pydantic_ai import EmbeddingResult

from src.core.rag_config import EmbeddingSettings
from src.services import embedding_service as embedding_module
from src.services.embedding_service import EmbeddingService


def _result(
    inputs: str | Sequence[str],
    *,
    input_type: str,
    embeddings: list[list[float]],
) -> EmbeddingResult:
    normalized_inputs = [inputs] if isinstance(inputs, str) else list(inputs)
    return EmbeddingResult(
        embeddings=embeddings,
        inputs=normalized_inputs,
        input_type=input_type,
        model_name="test-embedding-model",
        provider_name="test-provider",
    )


@pytest.fixture(autouse=True)
def clear_embedding_model_cache() -> None:
    """Ensure tests do not leak model cache state."""
    embedding_module._get_default_embedding_model.cache_clear()


@pytest.mark.asyncio
async def test_embedding_service_delegates_embed_query() -> None:
    query_result = _result("query", input_type="query", embeddings=[[0.1, 0.2]])
    docs_result = _result(["doc"], input_type="document", embeddings=[[0.3, 0.4]])

    model = MagicMock()
    model.embed_query = AsyncMock(return_value=query_result)
    model.embed_documents = AsyncMock(return_value=docs_result)

    service = EmbeddingService(model=model)
    result = await service.embed_query("query")

    assert result is query_result
    model.embed_query.assert_awaited_once_with("query")
    model.embed_documents.assert_not_awaited()


@pytest.mark.asyncio
async def test_embedding_service_delegates_embed_documents() -> None:
    query_result = _result("query", input_type="query", embeddings=[[0.1, 0.2]])
    docs_result = _result(
        ["doc a", "doc b"],
        input_type="document",
        embeddings=[[0.3, 0.4], [0.5, 0.6]],
    )

    model = MagicMock()
    model.embed_query = AsyncMock(return_value=query_result)
    model.embed_documents = AsyncMock(return_value=docs_result)

    service = EmbeddingService(model=model)
    result = await service.embed_documents(["doc a", "doc b"])

    assert result is docs_result
    model.embed_documents.assert_awaited_once_with(["doc a", "doc b"])
    model.embed_query.assert_not_awaited()


def test_embedding_service_builds_default_model_from_factory(monkeypatch) -> None:
    sentinel = MagicMock()
    factory = MagicMock(return_value=sentinel)
    monkeypatch.setattr(embedding_module, "get_embedding_model", factory)

    service = EmbeddingService()

    assert service.model is sentinel
    factory.assert_called_once_with()


@pytest.mark.parametrize(
    "model_name",
    [
        "mlx",
        "mlx/test-local-model",
        "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    ],
)
def test_get_embedding_model_routes_to_mlx(monkeypatch, model_name: str) -> None:
    mlx_embedder = MagicMock()
    mlx_factory = MagicMock(return_value=mlx_embedder)
    litellm_factory = MagicMock()
    monkeypatch.setattr(embedding_module, "get_mlx_embedding_model", mlx_factory)
    monkeypatch.setattr(
        embedding_module,
        "get_litellm_embedding_model",
        litellm_factory,
    )

    settings = EmbeddingSettings(
        model_name=model_name,
        max_tokens=128,
        batch_size=4,
        query_instruction="test instruction",
        embedding_size=1024,
    )
    model = embedding_module.get_embedding_model(settings)

    assert model is mlx_embedder
    mlx_factory.assert_called_once_with(settings)
    litellm_factory.assert_not_called()


def test_get_embedding_model_routes_to_litellm(monkeypatch) -> None:
    litellm_embedder = MagicMock()
    mlx_factory = MagicMock()
    litellm_factory = MagicMock(return_value=litellm_embedder)
    monkeypatch.setattr(embedding_module, "get_mlx_embedding_model", mlx_factory)
    monkeypatch.setattr(
        embedding_module,
        "get_litellm_embedding_model",
        litellm_factory,
    )

    settings = EmbeddingSettings(
        model_name="jina_ai/jina-embeddings-v3",
        max_tokens=128,
        batch_size=4,
        query_instruction="test instruction",
        embedding_size=1024,
    )
    model = embedding_module.get_embedding_model(settings)

    assert model is litellm_embedder
    litellm_factory.assert_called_once_with(settings)
    mlx_factory.assert_not_called()


def test_get_embedding_model_caches_default_instance(monkeypatch) -> None:
    litellm_factory = MagicMock(side_effect=[MagicMock(), MagicMock()])
    monkeypatch.setattr(
        embedding_module,
        "get_litellm_embedding_model",
        litellm_factory,
    )
    monkeypatch.setattr(embedding_module, "get_mlx_embedding_model", MagicMock())
    monkeypatch.setattr(
        embedding_module.EMBEDDING_SETTINGS,
        "model_name",
        "jina_ai/jina-embeddings-v3",
    )

    first = embedding_module.get_embedding_model()
    second = embedding_module.get_embedding_model()

    assert first is second
    litellm_factory.assert_called_once_with(embedding_module.EMBEDDING_SETTINGS)


def test_get_embedding_model_accepts_explicit_settings(monkeypatch) -> None:
    litellm_embedder = MagicMock()
    litellm_factory = MagicMock(return_value=litellm_embedder)
    monkeypatch.setattr(
        embedding_module,
        "get_litellm_embedding_model",
        litellm_factory,
    )
    monkeypatch.setattr(embedding_module, "get_mlx_embedding_model", MagicMock())

    settings = EmbeddingSettings(
        model_name="jina_ai/jina-embeddings-v3",
        max_tokens=64,
        batch_size=2,
        query_instruction="test instruction",
        embedding_size=768,
    )
    model = embedding_module.get_embedding_model(settings)

    assert model is litellm_embedder
    litellm_factory.assert_called_once_with(settings)
