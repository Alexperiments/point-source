"""Module to run embedding jobs."""

import asyncio

import logfire
from sqlalchemy import select, update

from src.core.database.base import async_session
from src.models.node import TextNode
from src.services.embedding_service import EmbeddingService


logfire.configure()
logfire.instrument_sqlalchemy()
logfire.instrument_pydantic_ai()

SLEEP_SECONDS_BETWEEN_BATCHES = 0
BATCH_SIZE = 2048


@logfire.instrument("Embedding chunks job")
async def main() -> None:
    """Run an embedding job."""
    with logfire.span("Loading embedding model"):
        embedding_service = EmbeddingService()

    skipped_chunk_ids: set = set()

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
                .limit(BATCH_SIZE),
            )
            rows = result.all()
            if not rows:
                break

            ids_list = [row.id for row in rows]
            text_list = [row.text for row in rows]

            try:
                with logfire.span(f"Embedding {len(rows)} chunks."):
                    embedding_result = await embedding_service.embed_documents(
                        documents=text_list,
                    )

                with logfire.span("Inserting batch's embeddings in database"):
                    params = [
                        {"id": chunk_id, "embedding": embedding}
                        for chunk_id, embedding in zip(
                            ids_list,
                            embedding_result.embeddings,
                            strict=True,
                        )
                    ]
                    await session.execute(update(TextNode), params)

                await session.commit()
            except Exception as exc:  # noqa: BLE001
                await session.rollback()
                skipped_chunk_ids.update(ids_list)
                logfire.exception(
                    "Batch failed; skipping batch for current run.",
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
