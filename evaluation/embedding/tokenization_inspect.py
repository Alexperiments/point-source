"""Inspect tokenizer outputs for embedding models."""

from __future__ import annotations

import logging
from functools import cache

from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

from evaluation.embedding.config import EMBEDDING_MODEL_SPECS


logger = logging.getLogger(__name__)
TEST_TEXT = r"ϵ ε \epsilon"


def _requires_trust_remote_code(name: str) -> bool:
    """Return True if the model requires trust_remote_code."""
    trusted_models = {
        "Alibaba-NLP/gte-large-en-v1.5",
    }
    return name in trusted_models


@cache
def get_embedding_model(name: str) -> SentenceTransformer:
    """Return a cached embedding model."""
    model = SentenceTransformer(
        name,
        trust_remote_code=_requires_trust_remote_code(name),
    )
    model.eval()
    return model


@cache
def get_tokenizer(name: str):  # noqa: ANN201
    """Return a cached tokenizer."""
    return AutoTokenizer.from_pretrained(
        name,
        use_fast=True,
        trust_remote_code=_requires_trust_remote_code(name),
    )


def inspect_model_tokens(model_name: str) -> None:
    """Print tokens (including special tokens) for a model."""
    tokenizer = get_tokenizer(model_name)

    encoding = tokenizer(
        TEST_TEXT,
        add_special_tokens=True,
        return_offsets_mapping=True,
    )
    token_ids = encoding["input_ids"]
    offsets = encoding["offset_mapping"]
    tokens = tokenizer.convert_ids_to_tokens(token_ids, skip_special_tokens=False)

    logger.info("Tokenizer: %s", tokenizer.__class__.__name__)
    logger.info("Special tokens map: %s", tokenizer.special_tokens_map)
    logger.info("Token count: %d", len(tokens))
    for index, (token, (start, end)) in enumerate(zip(tokens, offsets, strict=False)):
        text_segment = TEST_TEXT[start:end] if end > start else ""
        if not text_segment:
            text_segment = token
        logger.info("%4d  token=%r text=%r", index, token, text_segment)


def run_mojibake_round_trip_test(tokenizer) -> None:  # noqa: ANN001
    """Test if encode->decode round-trips for unicode symbols."""
    logger.info("\nTest 0 — Mojibake round-trip")
    samples = [
        "ϵ",
        "∑",
        "√",
        "∞",
        "\N{GREEK SMALL LETTER ALPHA}",
        "β",
        "\N{GREEK SMALL LETTER GAMMA}",
        "Ω",
        "ε",
        "μ",
        "π",
    ]
    sentence = (
        "Symbols: ϵ ∑ √ ∞ \N{GREEK SMALL LETTER ALPHA} β "
        "\N{GREEK SMALL LETTER GAMMA} Ω ε μ π; also <= >= != ~= +/- and ≤ ≥ ≠ ≈ ±."
    )
    for text in [*samples, sentence]:
        ids = tokenizer.encode(text, add_special_tokens=False)
        text2 = tokenizer.decode(ids)
        status = "OK" if text2 == text else "CHANGED"
        logger.info("%7s  text=%r  roundtrip=%r", status, text, text2)


def cosine_similarity_rows(vectors: list[list[float]]) -> list[list[float]]:
    """Compute cosine similarity matrix for already-normalized vectors."""
    sims: list[list[float]] = []
    for i, v_i in enumerate(vectors):
        row: list[float] = []
        for j, v_j in enumerate(vectors):
            if i == j:
                row.append(1.0)
            else:
                row.append(float(sum(a * b for a, b in zip(v_i, v_j, strict=False))))
        sims.append(row)
    return sims


def run_equivalence_test(model_name: str) -> None:
    """Test cosine similarity across equivalent representations."""
    logger.info("\nTest 1 — Embedding invariance under equivalent representations")
    embedding_model = get_embedding_model(model_name)
    equivalence_sets = {
        "epsilon": ["ϵ", "ε", "\\epsilon", "\\varepsilon"],
        "mu": ["\\mu", "μ"],
        "micron": ["\\mu m", "μm", "micron"],
        "omega": ["\\omega", "ω"],
        "leq": ["≤", "<="],
        "approx": ["≈", "~="],
        "times": ["\N{MULTIPLICATION SIGN}", "\\times", "*"],
    }
    for label, variants in equivalence_sets.items():
        embeddings = embedding_model.encode(
            variants,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )
        sims = cosine_similarity_rows(embeddings.tolist())
        logger.info("\n%s: %s", label, variants)
        for i, row in enumerate(sims):
            row_str = " ".join(f"{value:.4f}" for value in row)
            logger.info("  %d: %s", i, row_str)


def main() -> None:
    """Run tokenizer inspection and tests for selected embedding models."""
    target_keys = {"qwen3_embedding_0_6b", "embeddinggemma_300m"}
    for spec in EMBEDDING_MODEL_SPECS:
        if spec.key not in target_keys:
            continue
        logger.info("\n=== %s (%s) ===", spec.key, spec.model_name)
        if spec.provider != "sentence_transformers":
            logger.info(
                "Skipping provider %r (no local tokenizer).",
                spec.provider,
            )
            continue
        tokenizer = get_tokenizer(spec.model_name)
        run_mojibake_round_trip_test(tokenizer)
        inspect_model_tokens(spec.model_name)
        run_equivalence_test(spec.model_name)


if __name__ == "__main__":
    main()
