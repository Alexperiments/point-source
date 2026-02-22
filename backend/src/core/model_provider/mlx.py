"""MLX embedding model provider."""

import asyncio
from collections.abc import Sequence

import mlx.core as mx
import numpy as np
from mlx_lm import load
from pydantic_ai import Embedder, EmbeddingResult
from pydantic_ai.embeddings.base import EmbeddingModel, EmbedInputType
from pydantic_ai.embeddings.settings import (
    EmbeddingSettings as PydanticAIEmbeddingSettings,
)
from transformers import AutoTokenizer

from src.core.rag_config import EmbeddingSettings as AppEmbeddingSettings


class MLXQwen3EmbeddingModel(EmbeddingModel):
    """Embedding model backed by local MLX."""

    def __init__(self, settings: AppEmbeddingSettings) -> None:
        """Initialize the MLX model, tokenizer, and query instruction."""
        super().__init__()
        self._app_settings = settings
        self._model_name = settings.model_name
        self.model, _ = load(self._model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self._model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.query_instruction = settings.query_instruction
        self.max_length = settings.max_tokens
        self.batch_size = settings.batch_size

    @property
    def model_name(self) -> str:
        """The embedding model name."""
        return self._model_name

    @property
    def system(self) -> str:
        """The embedding model provider."""
        return "mlx"

    @staticmethod
    def _normalize_inputs(inputs: str | Sequence[str]) -> list[str]:
        if isinstance(inputs, str):
            return [inputs]

        normalized = list(inputs)
        if any(not isinstance(item, str) for item in normalized):
            raise TypeError("All inputs must be strings.")
        return normalized

    def _encode(
        self,
        text_list: str | Sequence[str],
        max_length: int,
    ) -> np.ndarray:
        texts = self._normalize_inputs(text_list)
        n = len(texts)
        dim = self._app_settings.embedding_size

        out = mx.zeros((n, dim), dtype=mx.float32)

        for start in range(0, n, self.batch_size):
            batch_texts = texts[start : start + self.batch_size]

            tokenizer_output = self.tokenizer(
                batch_texts,
                max_length=max_length,
                padding=True,
                truncation=True,
                return_tensors="mlx",
            )

            input_ids = tokenizer_output["input_ids"]
            attention_mask = tokenizer_output["attention_mask"]

            hidden_states = self.model.model(input_ids)

            seq_len = mx.sum(attention_mask, axis=1) - 1
            idx = mx.maximum(seq_len, 0)

            bsz = hidden_states.shape[0]
            pooled = hidden_states[mx.arange(bsz), idx]

            norm = mx.linalg.norm(pooled, axis=1, keepdims=True)
            emb = pooled / mx.maximum(norm, 1e-9)

            out[start : start + bsz] = emb

        mx.eval(out)

        return np.array(out.tolist(), dtype=np.float32)

    async def embed(
        self,
        inputs: str | Sequence[str],
        *,
        input_type: EmbedInputType,
        settings: PydanticAIEmbeddingSettings | None = None,
    ) -> EmbeddingResult:
        """Embed query or document inputs."""
        normalized, _ = self.prepare_embed(inputs, settings)

        texts = (
            [
                f"Instruct: {self.query_instruction}\nQuery: {text}"
                for text in normalized
            ]
            if input_type == "query"
            else normalized
        )
        max_length = self.max_length + 15 if input_type == "query" else self.max_length

        embeddings = await asyncio.to_thread(self._encode, texts, max_length)
        return EmbeddingResult(
            embeddings=embeddings.tolist(),
            inputs=normalized,
            input_type=input_type,
            model_name=self.model_name,
            provider_name=self.system,
        )


def get_mlx_embedding_model(settings: AppEmbeddingSettings) -> Embedder:
    """Get an MLX embedder instance."""
    return Embedder(MLXQwen3EmbeddingModel(settings))
