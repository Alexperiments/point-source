"""Tokenizer utilities for efficient offset-aware chunking."""

from __future__ import annotations

from bisect import bisect_left, bisect_right
from dataclasses import dataclass
from enum import Enum
from typing import Any, NamedTuple

from transformers import AutoTokenizer


Offset = tuple[int, int]


class TokenizerProvider(str, Enum):
    """Enumeration of tokenizer backends."""

    HUGGINGFACE = "huggingface"
    OPENAI = "openai"
    ANTHROPIC = "anthropic"
    GOOGLE = "google"


class _TokenSpan(NamedTuple):
    """Half-open token span [start, end)."""

    start: int
    end: int


@dataclass(frozen=True, slots=True)
class TokenOffsets:
    """Token-to-character offsets for a specific text.

    offsets[i] = (char_start, char_end) for token i.
    starts/ends are cached for fast bisect span->token span lookup.
    """

    text: str
    offsets: list[Offset]
    starts: list[int]
    ends: list[int]

    @classmethod
    def from_offsets(cls, text: str, offsets: list[Offset]) -> TokenOffsets:
        """Build TokenOffsets with cached start/end arrays for fast lookup."""
        return cls(
            text=text,
            offsets=offsets,
            starts=[start for start, _ in offsets],
            ends=[end for _, end in offsets],
        )

    def token_range_for_char_span(self, span_start: int, span_end: int) -> _TokenSpan:
        """Return the token span [start, end) overlapping the char span."""
        if span_start >= span_end or not self.offsets:
            return _TokenSpan(0, 0)

        start_token = bisect_right(self.ends, span_start)
        end_token = bisect_left(self.starts, span_end)
        end_token = max(end_token, start_token)
        return _TokenSpan(start_token, end_token)

    def slice_text_by_token_range(
        self,
        start_token: int,
        end_token: int,
        clamp_start: int,
        clamp_end: int,
    ) -> str:
        """Convert a token span into a clamped character slice."""
        if start_token >= end_token or not self.offsets:
            return ""

        start_char = max(self.offsets[start_token][0], clamp_start)
        end_char = min(self.offsets[end_token - 1][1], clamp_end)
        if start_char >= end_char:
            return ""
        return self.text[start_char:end_char]


class Tokenizer:
    """Interface for tokenizers that can return offset mappings."""

    provider: TokenizerProvider

    def tokenize_with_offsets(self, text: str) -> TokenOffsets:
        """Tokenize text and return offsets that map tokens to character spans."""
        raise NotImplementedError


class HuggingFaceTokenizer(Tokenizer):
    """Tokenizer backed by a Hugging Face fast tokenizer."""

    provider = TokenizerProvider.HUGGINGFACE

    def __init__(self, model_name: str) -> None:
        """Initialize a tokenizer for the given Hugging Face model name."""
        self._model_name = model_name
        self._tokenizer = self._load_tokenizer(model_name)

    @staticmethod
    def _load_tokenizer(model_name: str) -> Any:  # noqa: ANN401
        """Load a fast Hugging Face tokenizer for the given model name."""
        return AutoTokenizer.from_pretrained(model_name, use_fast=True)

    def tokenize_with_offsets(self, text: str) -> TokenOffsets:
        """Tokenize text and return HF offset mappings."""
        encoded = self._tokenizer(
            text,
            return_offsets_mapping=True,
            add_special_tokens=False,
            truncation=False,
        )
        offsets = encoded["offset_mapping"]
        return TokenOffsets.from_offsets(text=text, offsets=offsets)


_NON_HF_PREFIXES = ("openai:", "anthropic:", "google:")
_OPENAI_PREFIXES = ("openai:", "text-embedding-", "gpt-")
_ANTHROPIC_PREFIXES = ("anthropic:", "claude-")
_GOOGLE_PREFIXES = ("google:", "textembedding-", "gemini-")


class TokenizerFactory:
    """Centralized tokenizer selection based on a model identifier.

    Current behavior:
    - HuggingFace: implemented.
    - OpenAI / Anthropic / Google: explicitly not implemented (raises NotImplementedError).
    """

    @staticmethod
    def create(model_name: str) -> Tokenizer:
        """Return a tokenizer instance for the inferred provider."""
        provider = TokenizerFactory._infer_provider(model_name)

        match provider:
            case TokenizerProvider.HUGGINGFACE:
                return HuggingFaceTokenizer(model_name=model_name)
            case (
                TokenizerProvider.OPENAI
                | TokenizerProvider.ANTHROPIC
                | TokenizerProvider.GOOGLE
            ):
                raise NotImplementedError(
                    f"Tokenizer provider '{provider.value}' is not implemented yet for model '{model_name}'.",
                )
            case _:
                raise ValueError(
                    f"Unsupported tokenizer provider for model '{model_name}'.",
                )

    @staticmethod
    def _infer_provider(model_name: str) -> TokenizerProvider:
        """Infer the tokenizer provider using explicit, easy-to-evolve heuristics."""
        name = (model_name or "").strip()
        lower = name.lower()

        if "/" in name and not lower.startswith(_NON_HF_PREFIXES):
            return TokenizerProvider.HUGGINGFACE

        if lower.startswith(_OPENAI_PREFIXES):
            return TokenizerProvider.OPENAI
        if lower.startswith(_ANTHROPIC_PREFIXES):
            return TokenizerProvider.ANTHROPIC
        if lower.startswith(_GOOGLE_PREFIXES):
            return TokenizerProvider.GOOGLE

        # Default to HF only if it looks like HF; otherwise treat as OpenAI-like for safety
        # (forces explicit NotImplementedError rather than silently tokenizing per-char).
        return TokenizerProvider.OPENAI
