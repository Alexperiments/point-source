"""Agentic system utilities."""

from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from src.core.config import settings as app_settings
from src.core.rag_config import AGENT_SETTINGS


def get_chat_model(
    model_name: str,
) -> OpenAIChatModel:
    """Get a model from the model registry."""
    return OpenAIChatModel(
        model_name=model_name,
        settings={
            "temperature": AGENT_SETTINGS.temperature,
            "max_tokens": AGENT_SETTINGS.max_tokens,
        },
        provider=LiteLLMProvider(
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
        ),
    )
