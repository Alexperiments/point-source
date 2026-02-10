"""Service for LLM text generation via LiteLLM/OpenAI-compatible APIs."""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal, overload


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

import logfire
from openai import OpenAI

from src.core.config import settings as app_settings
from src.core.rag_config import AGENT_SETTINGS


Message = dict[str, str]


class GenerationService:
    """Generate text using an OpenAI-compatible API, with optional streaming."""

    def __init__(
        self,
        *,
        base_url: str | None = None,
        api_key: str | None = None,
        default_model: str | None = None,
    ) -> None:
        """Initialize the generation service."""
        self.base_url = base_url or app_settings.litellm_base_url.unicode_string()
        self.api_key = (
            api_key
            if api_key is not None
            else app_settings.litellm_api_key.get_secret_value()
        )
        self.default_model = default_model or AGENT_SETTINGS.model_name

    @overload
    def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        stream: Literal[False] = False,
        **kwargs: object,
    ) -> str: ...

    @overload
    def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        stream: Literal[True],
        **kwargs: object,
    ) -> Iterator[str]: ...

    def generate(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        stream: bool = False,
        **kwargs: object,
    ) -> str | Iterator[str]:
        """Generate text from messages.

        If stream=True, returns an iterator yielding token strings.
        Otherwise, returns the full response text.
        """
        if not stream:
            return self._generate_once(messages=messages, model=model, **kwargs)

        provider_stream = self._generate_stream(
            messages=messages,
            model=model,
            **kwargs,
        )
        return self._tokens_from_provider_stream(provider_stream)

    def _generate_once(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        **kwargs: object,
    ) -> str:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        with logfire.span("generation.request"):
            response = client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                stream=False,
                **kwargs,
            )
        return response.choices[0].message.content or ""

    def _generate_stream(
        self,
        *,
        messages: list[Message],
        model: str | None = None,
        **kwargs: object,
    ) -> Iterable[object]:
        client = OpenAI(api_key=self.api_key, base_url=self.base_url)
        with logfire.span("generation.stream_request"):
            return client.chat.completions.create(
                model=model or self.default_model,
                messages=messages,
                stream=True,
                **kwargs,
            )

    def _tokens_from_provider_stream(
        self,
        stream: Iterable[object],
    ) -> Iterator[str]:
        with logfire.span("generation.stream_consume"):
            for event in stream:
                token = self._extract_token(event)
                if token is not None:
                    yield token

    def _extract_token(self, event: object) -> str | None:
        choices = getattr(event, "choices", None)
        if not choices:
            return None
        delta = choices[0].delta
        return getattr(delta, "content", None)
