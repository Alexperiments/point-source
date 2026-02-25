"""Model-specific text preprocessing for embedding evaluation."""

from __future__ import annotations

from typing import Literal


InputType = Literal["query", "document"]

JINA_QUERY_INSTRUCTION = (
    "Given a web search query, retrieve relevant passages that answer the query."
)


def _is_e5_model(model_name: str) -> bool:
    lowered = model_name.lower()
    return "/e5" in lowered or lowered.startswith("e5")


def preprocess_texts(
    model_name: str,
    text_list: list[str],
    input_type: InputType | None,
) -> list[str]:
    """Apply model-specific input formatting before embedding."""
    if any(not isinstance(text, str) for text in text_list):
        raise TypeError("All embedding inputs must be strings.")
    if not text_list or input_type is None:
        return text_list

    lowered = model_name.lower()
    if "jina-embeddings-v3" in lowered and input_type == "query":
        return [
            f"Instruct: {JINA_QUERY_INSTRUCTION}\nQuery: {text}" for text in text_list
        ]

    if _is_e5_model(model_name):
        prefix = "query: " if input_type == "query" else "passage: "
        return [f"{prefix}{text}" for text in text_list]

    return text_list
