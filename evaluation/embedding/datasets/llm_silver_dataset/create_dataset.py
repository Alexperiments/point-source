"""Create baseline-3-curated-queries documents from a topical slice of chunks."""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

from sqlalchemy import func, select

from src.core.database.base import async_session
from src.models.node import TextNode


logger = logging.getLogger(__name__)

DATASET_NAME = "llm-silver-dataset"
SAMPLE_SIZE = 1000
OUTPUT_DIR = Path(__file__).parent
DOCS_PATH = OUTPUT_DIR / "documents.jsonl"


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


async def _fetch_chunks() -> list[TextNode]:
    async with async_session() as session:
        result = await session.execute(
            select(TextNode)
            .where(TextNode.text != "")
            .order_by(func.random())
            .limit(SAMPLE_SIZE),
        )
        chunks = list(result.scalars().all())

        if len(chunks) < SAMPLE_SIZE:
            result = await session.execute(
                select(TextNode)
                .where(TextNode.text != "")
                .order_by(func.random())
                .limit(SAMPLE_SIZE),
            )
            chunks = list(result.scalars().all())

    return chunks


def _build_documents(chunks: list[TextNode]) -> list[dict[str, object]]:
    return [
        {
            "source_id": str(chunk.id),
            "text": chunk.text,
            "metadata": {
                "dataset_name": DATASET_NAME,
            },
        }
        for chunk in chunks
    ]


async def main() -> None:
    """Create a JSONL dataset of curated text chunks."""
    chunks = await _fetch_chunks()
    if not chunks:
        raise ValueError("No chunks found with non-empty text.")

    documents = _build_documents(chunks)
    _write_jsonl(DOCS_PATH, documents)

    logger.info("Wrote %d documents to %s", len(documents), DOCS_PATH)


if __name__ == "__main__":
    asyncio.run(main())
