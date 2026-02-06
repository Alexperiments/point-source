"""Embedding service."""

import logfire
import mlx.core as mx
import numpy as np
from mlx_lm import load
from transformers import AutoTokenizer

from src.core.chunking_config import CHUNKING_SETTINGS


embedded_chunks_metric = logfire.metric_counter(
    "embedded_chunks",
    unit="1",
    description="Number of embedded chunks.",
)


class MLXQwen3EmbeddingService:
    """Embedding service backed by MLX for Qwen3 models."""

    def __init__(self) -> None:
        """Initialize the MLX model, tokenizer, and query instruction."""
        self.model_name = "mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ"
        self.model, _ = load(self.model_name)
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True,
            use_fast=True,
        )
        self.query_instruction = "Given a web search query, retrieve relevant passages that answer the query."

    def _encode(
        self,
        text_list: list[str],
        batch_size: int = 32,
        max_length: int = CHUNKING_SETTINGS.max_tokens,
    ) -> np.ndarray:
        n = len(text_list)
        dim = 1024

        out = mx.zeros((n, dim), dtype=mx.float32)

        for start in range(0, n, batch_size):
            batch_texts = text_list[start : start + batch_size]

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

    def encode_document(
        self,
        text_list: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed documents using the MLX-backed Qwen3 model."""
        return self._encode(text_list, batch_size)

    def encode_query(
        self,
        text_list: list[str],
        batch_size: int = 8,
    ) -> np.ndarray:
        """Embed queries using the MLX-backed Qwen3 model."""
        prefixed = [f"Instruct: {self.query_instruction}\nQuery: " for t in text_list]
        return self._encode(
            prefixed,
            batch_size=batch_size,
            max_length=CHUNKING_SETTINGS.max_tokens + 15,
        )
