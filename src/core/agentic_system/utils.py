"""Agentic system utilities."""

from pydantic_ai import ModelSettings
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.providers.litellm import LiteLLMProvider

from src.core.config import settings as app_settings
from src.core.rag_config import AGENT_SETTINGS


def build_model_settings() -> ModelSettings:
    """Build default model settings from rag_config."""
    return {
        "temperature": AGENT_SETTINGS.temperature,
        "max_tokens": AGENT_SETTINGS.max_tokens,
        "extra_body": {
            "chat_template_kwargs": {
                "enable_thinking": AGENT_SETTINGS.enable_thinking,
                "custom_llm_provider": AGENT_SETTINGS.custom_llm_provider,
            },
        },
    }


def get_chat_model(
    model_name: str,
) -> OpenAIChatModel:
    """Get a model from the model registry."""
    return OpenAIChatModel(
        model_name=model_name,
        settings=build_model_settings(),
        provider=LiteLLMProvider(
            api_base=app_settings.litellm_base_url.unicode_string(),
            api_key=app_settings.litellm_api_key.get_secret_value(),
        ),
    )
