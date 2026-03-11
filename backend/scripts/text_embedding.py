"""Module to run embedding jobs."""

import asyncio
import os
from time import perf_counter
from uuid import UUID

import logfire
from sqlalchemy import select, update

from src.core.database.base import async_session
from src.core.telemetry import configure_logfire
from src.models.node import TextNode
from src.services.embedding_service import EmbeddingService


SLEEP_SECONDS_BETWEEN_BATCHES = float(
    os.getenv("TEXT_EMBEDDING_SLEEP_SECONDS", "0"),
)
DATABASE_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_DB_BATCH_SIZE", "1024"))
API_BATCH_SIZE = int(os.getenv("TEXT_EMBEDDING_API_BATCH_SIZE", "64"))
API_MAX_CONCURRENCY = int(
    os.getenv("TEXT_EMBEDDING_API_MAX_CONCURRENCY", "16"),
)

configure_logfire()
logfire.instrument_sqlalchemy()
logfire.instrument_pydantic_ai()


def _batch_ranges(total: int, batch_size: int) -> list[tuple[int, int]]:
    """Return inclusive-exclusive index ranges for chunking a sequence."""
    normalized_batch_size = max(1, batch_size)
    return [
        (start, min(start + normalized_batch_size, total))
        for start in range(0, total, normalized_batch_size)
    ]


async def _embed_api_batches(
    embedding_service: EmbeddingService,
    ids_list: list[UUID],
    text_list: list[str],
) -> tuple[list[dict[str, object]], set[UUID]]:
    """Embed a database batch using smaller concurrent API requests."""
    batch_ranges = _batch_ranges(len(text_list), API_BATCH_SIZE)
    semaphore = asyncio.Semaphore(max(1, API_MAX_CONCURRENCY))

    async def _embed_single_batch(
        batch_index: int,
        start: int,
        end: int,
    ) -> tuple[int, list[dict[str, object]], set[UUID]]:
        batch_ids = ids_list[start:end]
        batch_texts = text_list[start:end]
        batch_chars = sum(len(text) for text in batch_texts)
        started_at = perf_counter()

        async with semaphore:
            try:
                with logfire.span(
                    "Embedding API batch",
                    batch_index=batch_index + 1,
                    total_batches=len(batch_ranges),
                    batch_size=len(batch_texts),
                    batch_chars=batch_chars,
                ):
                    embedding_result = await embedding_service.embed_documents(
                        documents=batch_texts,
                    )

                params = [
                    {"id": chunk_id, "embedding": embedding}
                    for chunk_id, embedding in zip(
                        batch_ids,
                        embedding_result.embeddings,
                        strict=True,
                    )
                ]
            except Exception as exc:  # noqa: BLE001
                logfire.exception(
                    "Embedding API batch failed; skipping sub-batch.",
                    batch_index=batch_index + 1,
                    total_batches=len(batch_ranges),
                    batch_size=len(batch_texts),
                    batch_chars=batch_chars,
                    elapsed_seconds=perf_counter() - started_at,
                    error=str(exc),
                )
                return batch_index, [], set(batch_ids)
            else:
                return batch_index, params, set()

    results = await asyncio.gather(
        *[
            _embed_single_batch(batch_index, start, end)
            for batch_index, (start, end) in enumerate(batch_ranges)
        ],
    )

    params: list[dict[str, object]] = []
    failed_ids: set[UUID] = set()
    for _, batch_params, batch_failed_ids in sorted(results, key=lambda item: item[0]):
        params.extend(batch_params)
        failed_ids.update(batch_failed_ids)

    return params, failed_ids


@logfire.instrument("Embedding chunks job")
async def main() -> None:
    """Run an embedding job."""
    with logfire.span("Loading embedding model"):
        embedding_service = EmbeddingService()

    skipped_chunk_ids: set[UUID] = set()

    async with async_session() as session:
        while True:
            stmt = select(TextNode.id, TextNode.text).where(
                TextNode.text != "",
                TextNode.embedding.is_(None),
            )
            if skipped_chunk_ids:
                stmt = stmt.where(~TextNode.id.in_(skipped_chunk_ids))

            result = await session.execute(
                stmt.order_by(TextNode.id)
                .with_for_update(skip_locked=True)
                .limit(DATABASE_BATCH_SIZE),
            )
            rows = result.all()
            if not rows:
                break

            ids_list = [row.id for row in rows]
            text_list = [row.text for row in rows]

            try:
                with logfire.span(
                    f"Embedding {len(rows)} chunks.",
                    api_batch_size=max(1, API_BATCH_SIZE),
                    api_max_concurrency=max(1, API_MAX_CONCURRENCY),
                ):
                    params, failed_ids = await _embed_api_batches(
                        embedding_service,
                        ids_list,
                        text_list,
                    )

                with logfire.span("Inserting batch's embeddings in database"):
                    if params:
                        await session.execute(update(TextNode), params)

                await session.commit()
                if failed_ids:
                    skipped_chunk_ids.update(failed_ids)
                    logfire.error(
                        "One or more embedding API sub-batches failed.",
                        failed_chunks=len(failed_ids),
                        skipped_chunks_total=len(skipped_chunk_ids),
                    )
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                skipped_chunk_ids.update(ids_list)
                logfire.exception(
                    "Batch write failed; skipping batch for current run.",
                    batch_size=len(rows),
                    skipped_chunks_total=len(skipped_chunk_ids),
                    error=str(exc),
                )

            with logfire.span(
                "Sleeping before next batch",
                seconds=SLEEP_SECONDS_BETWEEN_BATCHES,
            ):
                await asyncio.sleep(SLEEP_SECONDS_BETWEEN_BATCHES)


if __name__ == "__main__":
    asyncio.run(main())
