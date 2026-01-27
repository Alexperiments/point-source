"""Tokenizer service."""

from __future__ import annotations

import bisect
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from transformers import AutoTokenizer


class TokenizerProvider(str, Enum):
    """Enumeration of embedding providers."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class _TokenRange(NamedTuple):
    start: int
    end: int


@dataclass(frozen=True)
class TokenOffsets:
    """Instances represent a mapping between token indices and char spans for a specific text.

    offsets[i] = (char_start, char_end) for token i
    starts/ends are cached for fast bisect span->token range.
    """

    text: str
    offsets: list[tuple[int, int]]
    starts: list[int]
    ends: list[int]

    @classmethod
    def from_offsets(cls, text: str, offsets: list[tuple[int, int]]) -> TokenOffsets:
        """Build TokenOffsets with cached start/end arrays for fast span lookup."""
        return cls(
            text=text,
            offsets=offsets,
            starts=[s for s, _ in offsets],
            ends=[e for _, e in offsets],
        )

    def token_range_for_char_span(
        self,
        span_start: int,
        span_end: int,
    ) -> _TokenRange:
        """Return [start_token, end_token) that overlaps the char span.

        Uses precomputed ends/starts for O(log n) lookup.
        """
        if span_start >= span_end or not self.offsets:
            return _TokenRange(0, 0)

        start_token = bisect.bisect_right(self.ends, span_start)
        end_token = bisect.bisect_left(self.starts, span_end)
        end_token = max(end_token, start_token)
        return _TokenRange(start_token, end_token)

    def slice_text_by_token_range(
        self,
        start_token: int,
        end_token: int,
        clamp_start: int,
        clamp_end: int,
    ) -> str:
        """Convert a token span into a char slice, clamped to [clamp_start, clamp_end).

        This is the critical primitive enabling single-pass tokenization plus fast chunk slicing.
        """
        if start_token >= end_token or not self.offsets:
            return ""

        start_char = max(self.offsets[start_token][0], clamp_start)
        end_char = min(self.offsets[end_token - 1][1], clamp_end)
        if start_char >= end_char:
            return ""
        return self.text[start_char:end_char]


class Tokenizer:
    """Interface for providers that can produce token offset mappings in a single call."""

    provider: TokenizerProvider

    def tokenize_with_offsets(self, text: str) -> TokenOffsets:
        """Tokenize text and return offsets that map tokens to character spans."""
        raise NotImplementedError


class HuggingFaceTokenizer(Tokenizer):
    """Tokenizer implementation backed by a Hugging Face fast tokenizer."""

    provider = TokenizerProvider.HUGGINGFACE

    def __init__(self, model_name: str) -> None:
        """Initialize a tokenizer for the given Hugging Face model name."""
        self._model_name = model_name
        self._tokenizer = self._load_tokenizer(model_name)

    @staticmethod
    def _load_tokenizer(model_name: str) -> Any:  # noqa: ANN401
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_with_offsets(self, text: str) -> TokenOffsets:
        """Tokenize text and return offset mappings from the HF tokenizer."""
        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]
        return TokenOffsets.from_offsets(text=text, offsets=offsets)


class TokenizerFactory:
    """Centralized resolution of tokenizer provider based on model identifier.

    Current behavior:
    - HuggingFace: implemented.
    - OpenAI / Anthropic / Google: explicitly not implemented (raises NotImplementedError).
    """

    _CACHE: dict[str, Tokenizer] = {}

    @staticmethod
    def create(model_name: str) -> Tokenizer:
        """Return a cached tokenizer instance based on the inferred provider."""
        cache_key = (model_name or "").strip()
        if cache_key in TokenizerFactory._CACHE:
            return TokenizerFactory._CACHE[cache_key]

        provider = TokenizerFactory._infer_provider(model_name)

        if provider == TokenizerProvider.HUGGINGFACE:
            tokenizer = HuggingFaceTokenizer(model_name=model_name)
            TokenizerFactory._CACHE[cache_key] = tokenizer
            return tokenizer

        if provider in (
            TokenizerProvider.OPENAI,
            TokenizerProvider.ANTHROPIC,
            TokenizerProvider.GOOGLE,
        ):
            raise NotImplementedError(
                f"Tokenizer provider '{provider.value}' is not implemented yet for model '{model_name}'.",
            )

        raise ValueError(f"Unsupported tokenizer provider for model '{model_name}'.")

    @staticmethod
    def _infer_provider(model_name: str) -> TokenizerProvider:
        """Heuristics are intentionally explicit and easy to evolve.

        You can replace this with a more formal model registry later.
        """
        name = (model_name or "").strip()

        if "/" in name and not name.lower().startswith(
            ("openai:", "anthropic:", "google:"),
        ):
            return TokenizerProvider.HUGGINGFACE

        lower = name.lower()
        if lower.startswith(("openai:", "text-embedding-", "gpt-")):
            return TokenizerProvider.OPENAI
        if lower.startswith(("anthropic:", "claude-")):
            return TokenizerProvider.ANTHROPIC
        if lower.startswith(("google:", "textembedding-", "gemini-")):
            return TokenizerProvider.GOOGLE

        # Default to HF only if it looks like HF; otherwise treat as OpenAI-like for safety
        # (forces explicit NotImplementedError rather than silently tokenizing per-char).
        return TokenizerProvider.OPENAI
