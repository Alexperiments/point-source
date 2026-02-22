"""Module to run embedding jobs."""

import logfire
from sqlalchemy import select, update

from src.core.database.base import async_session
from src.models.node import TextNode
from src.services.embedding_service import EmbeddingService


logfire.configure()


@logfire.instrument("Embedding chunks job")
async def main() -> None:
    """Run an embedding job."""
    with logfire.span(
        "Loading embedding model: mlx-community/Qwen3-Embedding-0.6B-4bit-DWQ",
    ):
        embedding_service = EmbeddingService()

    async with async_session() as session:
        while True:
            result = await session.execute(
                select(TextNode.id, TextNode.text)
                .where(TextNode.text != "", TextNode.embedding.is_(None))
                .order_by(
                    TextNode.id,
                )
                .with_for_update(skip_locked=True)
                .limit(512),
            )
            rows = result.all()
            if not rows:
                break

            with logfire.span("Embedding batch"):
                ids_list = [r.id for r in rows]
                text_list = [r.text for r in rows]

                embedding_result = await embedding_service.embed_documents(
                    documents=text_list,
                )

            with logfire.span("Inserting batch's embeddings in database"):
                params = [
                    {"id": i, "embedding": e}
                    for i, e in zip(ids_list, embedding_result.embeddings, strict=True)
                ]
                await session.execute(update(TextNode), params)

            await session.commit()


if __name__ == "__main__":
    import asyncio

    asyncio.run(main())
