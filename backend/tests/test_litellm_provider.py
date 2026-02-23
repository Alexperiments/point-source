"""Tests for LiteLLM model provider builders."""

from unittest.mock import MagicMock

from src.core.model_provider import litellm as litellm_module
from src.core.rag_config import EmbeddingSettings


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
