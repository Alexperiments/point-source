"""Populate embedding benchmark tables and compute retrieval metrics (pgvector-backed, dataset-correct).

Key correction vs prior pgvector version:
- Retrieval is now constrained to the current dataset by joining the canonical
  evaluation.documents table (which has dataset_name) to the embedding table.
  This prevents “cross-dataset” retrieval leakage and restores comparability
  with the original in-memory benchmark.

Assumptions:
- Per-model embedding tables exist (via migrations) and have columns:
  (document_id|query_id, quantization, embedding, created_at, id).
- Canonical docs/queries live in evaluation.documents / evaluation.queries (your ORM models).
- Cosine distance is used; document embeddings are L2-normalized on insert.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import numpy as np
import sqlalchemy as sa
import typer
from pgvector.sqlalchemy import Vector
from sqlalchemy.dialects.postgresql import insert

from evaluation.embedding.config import (
    EMBEDDING_MODEL_SPECS,
    EmbeddingModelSpec,
    document_embedding_table_name,
)
from evaluation.embedding.models import (
    EmbeddingEvalAnalytics,
    EmbeddingEvalDocument,
    EmbeddingEvalPairMetric,
    EmbeddingEvalQuery,
)
from src.core.database.base import async_session
from src.services.embedding_service import EmbeddingService


app = typer.Typer(no_args_is_help=True)


# ---------------------------
# IO + metrics helpers
# ---------------------------


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    path = Path(path)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            stripped = line.strip()
            if not stripped:
                continue
            try:
                obj = json.loads(stripped)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSON on line {line_no} in {path}") from exc
            if not isinstance(obj, dict):
                raise TypeError(f"Expected object on line {line_no} in {path}")
            rows.append(obj)
    return rows


def _l2_normalize(arr: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(arr, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return arr / norms


def _normalize_relevance(raw: object | None) -> dict[str, int]:
    if raw is None:
        return {}
    if isinstance(raw, dict):
        return {str(k): int(v) for k, v in raw.items()}
    if isinstance(raw, list):
        ordered = [str(x) for x in raw if x]
        total = len(ordered)
        out: dict[str, int] = {}
        for i, doc_id in enumerate(ordered):
            if doc_id in out:
                continue
            out[doc_id] = total - i
        return out
    raise TypeError("Query 'relevance' must be dict(id->grade) or ordered list(ids)")


def _extract_relevance(row: dict[str, Any]) -> dict[str, int]:
    raw = row.get("relevance")
    md = row.get("metadata")
    if raw is None and isinstance(md, dict):
        raw = md.get("relevance")
    return _normalize_relevance(raw)


def _recall_at_k(relevant: set[str], ranked_ids: list[str], k: int) -> float:
    if not relevant:
        return 0.0
    return len(relevant.intersection(ranked_ids[:k])) / float(len(relevant))


def _dcg(rels: list[int]) -> float:
    s = 0.0
    for rank, rel in enumerate(rels, start=1):
        s += (2**rel - 1) / np.log2(rank + 1)
    return s


def _ndcg_at_k(relevance_by_id: dict[str, int], ranked_ids: list[str], k: int) -> float:
    rels = [relevance_by_id.get(doc_id, 0) for doc_id in ranked_ids[:k]]
    ideal = sorted(relevance_by_id.values(), reverse=True)[:k]
    dcg = _dcg(rels)
    idcg = _dcg(ideal)
    return 0.0 if idcg == 0 else dcg / idcg


def _mrr(relevance_by_id: dict[str, int], ranked_ids: list[str]) -> float:
    for rank, doc_id in enumerate(ranked_ids, start=1):
        if relevance_by_id.get(doc_id, 0) > 0:
            return 1.0 / float(rank)
    return 0.0


def _render_progress(label: str, step: int, total: int, detail: str) -> None:
    width = 28
    filled = int(width * step / total) if total else width
    bar = "#" * filled + "-" * (width - filled)
    suffix = f"{step}/{total}"
    if detail:
        suffix = f"{suffix} {detail}"
    typer.echo(f"\r{label} [{bar}] {suffix}", nl=False)
    if step >= total:
        typer.echo()


# ---------------------------
# DB table helpers
# ---------------------------


def _build_embedding_table(
    *,
    metadata: sa.MetaData,
    table_name: str,
    dimension: int,
    kind: str,  # "document" or "query"
) -> sa.Table:
    # Table handle only; actual DDL assumed to exist.
    return sa.Table(
        table_name,
        metadata,
        sa.Column(
            "id",
            sa.UUID(),
            primary_key=True,
            server_default=sa.text("gen_random_uuid()"),
        ),
        sa.Column(f"{kind}_id", sa.UUID(), nullable=False),
        sa.Column("quantization", sa.Text(), nullable=False),
        sa.Column("embedding", Vector(dimension), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        schema="evaluation",
    )


def _select_specs(model_keys: Iterable[str] | None) -> list[EmbeddingModelSpec]:
    specs = [s for s in EMBEDDING_MODEL_SPECS if s.enabled]
    if model_keys is not None and not isinstance(model_keys, Iterable):
        raise TypeError("model_keys must be an iterable of strings")
    if not model_keys:
        return specs
    lookup = {s.key: s for s in specs}
    missing = [k for k in model_keys if k not in lookup]
    if missing:
        raise ValueError(f"Unknown model keys: {', '.join(missing)}")
    return [lookup[k] for k in model_keys]


# ---------------------------
# Canonical data creation / loading
# ---------------------------


async def _load_or_insert_documents(
    dataset_name: str,
    documents_path: Path,
) -> tuple[list[EmbeddingEvalDocument], dict[str, str]]:
    async with async_session() as session:
        res = await session.execute(
            sa.select(EmbeddingEvalDocument)
            .where(EmbeddingEvalDocument.dataset_name == dataset_name)
            .order_by(EmbeddingEvalDocument.created_at, EmbeddingEvalDocument.id),
        )
        docs = list(res.scalars().all())

    if docs:
        source_to_id = {str(d.source_id or d.id): str(d.id) for d in docs}
        return docs, source_to_id

    raw_docs = _load_jsonl(documents_path)
    doc_rows: list[EmbeddingEvalDocument] = []
    for row in raw_docs:
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError("Document rows require a non-empty 'text' field")
        doc_rows.append(
            EmbeddingEvalDocument(
                dataset_name=dataset_name,
                source_id=row.get("source_id"),
                text=text,
                document_metadata=row.get("metadata") or None,
            ),
        )

    async with async_session() as session:
        session.add_all(doc_rows)
        await session.flush()
        await session.commit()

    source_to_id = {str(d.source_id or d.id): str(d.id) for d in doc_rows}
    return doc_rows, source_to_id


def _load_queries_from_file(
    queries_path: Path,
) -> tuple[list[str], list[dict[str, int]], list[dict[str, Any] | None]]:
    raw_queries = _load_jsonl(queries_path)
    query_texts: list[str] = []
    relevance_maps: list[dict[str, int]] = []
    query_metadatas: list[dict[str, Any] | None] = []

    for row in raw_queries:
        text = str(row.get("text", "")).strip()
        if not text:
            raise ValueError("Query rows require a non-empty 'text' field")

        md_val = row.get("metadata")
        if md_val is None:
            metadata: dict[str, Any] = {}
        elif isinstance(md_val, dict):
            metadata = dict(md_val)
        else:
            metadata = {}
        if "relevance" not in metadata and row.get("relevance") is not None:
            metadata["relevance"] = row.get("relevance")

        query_texts.append(text)
        query_metadatas.append(metadata or None)
        relevance_maps.append(_extract_relevance(row))

    return query_texts, relevance_maps, query_metadatas


async def _ensure_queries(
    *,
    dataset_name: str,
    query_texts: list[str],
    query_metadatas: list[dict[str, Any] | None],
) -> list[str]:
    if not query_texts:
        return []

    unique_texts = list(dict.fromkeys(query_texts))
    async with async_session() as session:
        res = await session.execute(
            sa.select(EmbeddingEvalQuery.id, EmbeddingEvalQuery.text).where(
                EmbeddingEvalQuery.dataset_name == dataset_name,
                EmbeddingEvalQuery.text.in_(unique_texts),
            ),
        )
        existing = {text: str(qid) for qid, text in res.all()}

        to_insert: list[EmbeddingEvalQuery] = []
        seen_new: set[str] = set()
        for text, metadata in zip(query_texts, query_metadatas, strict=True):
            if text in existing or text in seen_new:
                continue
            to_insert.append(
                EmbeddingEvalQuery(
                    dataset_name=dataset_name,
                    text=text,
                    query_metadata=metadata,
                ),
            )
            seen_new.add(text)

        if to_insert:
            session.add_all(to_insert)
            await session.flush()
            for query in to_insert:
                existing[query.text] = str(query.id)
            await session.commit()

    return [existing[text] for text in query_texts]


def _convert_relevance(
    relevance_maps: list[dict[str, int]],
    source_to_id: dict[str, str],
) -> list[dict[str, int]]:
    out: list[dict[str, int]] = []
    for rel in relevance_maps:
        mapped: dict[str, int] = {}
        for key, grade in rel.items():
            doc_id = source_to_id.get(str(key))
            if doc_id is not None:
                mapped[doc_id] = int(grade)
        out.append(mapped)
    return out


# ---------------------------
# Embedding upserts (simple)
# ---------------------------


async def _ensure_document_embeddings(
    *,
    session: sa.ext.asyncio.AsyncSession,
    doc_table: sa.Table,
    embedding_service: EmbeddingService,
    doc_ids: list[str],
    doc_text_by_id: dict[str, str],
    quantization: str,
    dimension: int,
    batch_size: int,
) -> None:
    res = await session.execute(
        sa.select(doc_table.c.document_id).where(
            doc_table.c.quantization == quantization,
            doc_table.c.document_id.in_(doc_ids),
            doc_table.c.embedding.is_not(None),
        ),
    )
    have = {str(x[0]) for x in res.all()}
    missing = [d for d in doc_ids if d not in have]
    if not missing:
        return

    texts = [doc_text_by_id[d] for d in missing]
    emb = np.asarray(
        embedding_service.encode(texts, batch_size=batch_size),
        dtype=np.float32,
    )
    if emb.ndim != 2 or emb.shape[1] != dimension:
        raise ValueError(
            f"Doc embedding dim mismatch: got {emb.shape}, expected (*,{dimension})",
        )

    emb = _l2_normalize(emb)

    rows = [
        {"document_id": d, "quantization": quantization, "embedding": emb[i].tolist()}
        for i, d in enumerate(missing)
    ]
    await session.execute(sa.insert(doc_table), rows)


# ---------------------------
# Retrieval (pgvector) — dataset-correct via join
# ---------------------------


async def _retrieve_topk_for_query(
    *,
    session: sa.ext.asyncio.AsyncSession,
    doc_table: sa.Table,
    query_vec: list[float],
    quantization: str,
    dataset_name: str,
    k: int,
) -> list[tuple[str, float]]:
    """Retrieve top-k documents for a single query vector, restricted to docs in dataset_name.

    This fixes the leakage issue:
    - doc_table has no dataset_name, so we JOIN evaluation.documents to constrain.
    """
    docs = EmbeddingEvalDocument.__table__  # evaluation.documents

    qparam = sa.bindparam("qvec", value=query_vec)
    dist_expr = doc_table.c.embedding.cosine_distance(qparam)

    stmt = (
        sa.select(
            doc_table.c.document_id,
            dist_expr.label("distance"),
        )
        .select_from(
            doc_table.join(
                docs,
                docs.c.id == doc_table.c.document_id,
            ),
        )
        .where(docs.c.dataset_name == dataset_name)
        .where(doc_table.c.quantization == quantization)
        .where(doc_table.c.embedding.is_not(None))
        .order_by(dist_expr)
        .limit(k)
    )

    res = await session.execute(stmt)
    return [(str(doc_id), float(dist)) for doc_id, dist in res.all()]


# ---------------------------
# CLI
# ---------------------------


@app.callback()
def main() -> None:
    """Embedding benchmark CLI."""


@app.command()
def run(
    documents_path: Path = typer.Option(..., help="Path to documents JSONL"),
    queries_path: Path = typer.Option(..., help="Path to queries JSONL"),
    dataset_name: str = typer.Option(..., help="Dataset name for grouping"),
    run_name: str = typer.Option(..., help="Run name for grouping"),
    model_keys: list[str] = typer.Option(None, help="Subset of model keys"),
    quantizations: list[str] = typer.Option(["fp32"], help="Quantization tags"),
    batch_size: int = typer.Option(64, help="Batch size for embedding"),
    retrieval_k: int = typer.Option(200, help="Retrieval cutoff"),
) -> None:
    """Populate embedding eval tables and compute metrics using pgvector retrieval."""
    asyncio.run(
        _run_async(
            documents_path=documents_path,
            queries_path=queries_path,
            dataset_name=dataset_name,
            run_name=run_name,
            model_keys=model_keys,
            quantizations=quantizations,
            batch_size=batch_size,
            retrieval_k=retrieval_k,
        ),
    )


async def _run_async(
    *,
    documents_path: Path,
    queries_path: Path,
    dataset_name: str,
    run_name: str,
    model_keys: list[str] | None,
    quantizations: list[str],
    batch_size: int,
    retrieval_k: int,
) -> None:
    # Clean previous results for this dataset+run
    async with async_session() as session:
        await session.execute(
            sa.delete(EmbeddingEvalPairMetric).where(
                EmbeddingEvalPairMetric.dataset_name == dataset_name,
                EmbeddingEvalPairMetric.run_name == run_name,
            ),
        )
        await session.execute(
            sa.delete(EmbeddingEvalAnalytics).where(
                EmbeddingEvalAnalytics.dataset_name == dataset_name,
                EmbeddingEvalAnalytics.run_name == run_name,
            ),
        )
        await session.commit()

    # Load or create canonical doc/query rows for THIS dataset
    docs, source_to_id = await _load_or_insert_documents(dataset_name, documents_path)
    query_texts, relevance_maps, query_metadatas = _load_queries_from_file(queries_path)
    query_ids = await _ensure_queries(
        dataset_name=dataset_name,
        query_texts=query_texts,
        query_metadatas=query_metadatas,
    )

    if not docs or not query_texts or not query_ids:
        raise ValueError("Documents and queries must be non-empty")

    doc_ids = [str(d.id) for d in docs]
    doc_text_by_id = {str(d.id): d.text for d in docs}

    relevance_by_query = _convert_relevance(relevance_maps, source_to_id)

    retrieval_k = min(retrieval_k, len(doc_ids))

    specs = _select_specs(model_keys)
    table_metadata = sa.MetaData()

    for spec in specs:
        if spec.provider != "sentence_transformers":
            typer.echo(
                f"Skipping {spec.key}: provider '{spec.provider}' not implemented",
                err=True,
            )
            continue

        embedding_service = EmbeddingService(embedding_model_name=spec.model_name)

        doc_table = _build_embedding_table(
            metadata=table_metadata,
            table_name=document_embedding_table_name(spec),
            dimension=spec.dimension,
            kind="document",
        )

        for quantization in quantizations:
            if quantization != "fp32":
                raise NotImplementedError(
                    f"Quantization '{quantization}' is not implemented",
                )

            recall_50: list[float] = []
            recall_100: list[float] = []
            recall_200: list[float] = []
            ndcg_10: list[float] = []
            mrr_vals: list[float] = []
            retrieval_latencies: list[float] = []
            metric_rows: list[dict[str, Any]] = []

            async with async_session() as session:
                # Ensure doc embeddings exist for docs in THIS dataset
                await _ensure_document_embeddings(
                    session=session,
                    doc_table=doc_table,
                    embedding_service=embedding_service,
                    doc_ids=doc_ids,
                    doc_text_by_id=doc_text_by_id,
                    quantization=quantization,
                    dimension=spec.dimension,
                    batch_size=batch_size,
                )

                # Retrieve per query, restricted to dataset_name via JOIN (critical fix)
                total_queries = len(query_texts)
                _render_progress(spec.key, 0, total_queries, "queries")
                for q_idx, query_text in enumerate(query_texts):
                    query_id = query_ids[q_idx]
                    rel_map = relevance_by_query[q_idx]
                    relevant_ids = {
                        doc_id for doc_id, grade in rel_map.items() if grade > 0
                    }

                    t1 = time.perf_counter()
                    query_emb = np.asarray(
                        embedding_service.encode([query_text], batch_size=1),
                        dtype=np.float32,
                    )
                    if query_emb.ndim != 2 or query_emb.shape[1] != spec.dimension:
                        raise ValueError(
                            f"Query embedding dim mismatch: got {query_emb.shape}, "
                            f"expected (*,{spec.dimension})",
                        )
                    query_vec = query_emb[0].tolist()
                    top = await _retrieve_topk_for_query(
                        session=session,
                        doc_table=doc_table,
                        query_vec=query_vec,
                        quantization=quantization,
                        dataset_name=dataset_name,
                        k=retrieval_k,
                    )
                    retrieval_ms = (time.perf_counter() - t1) * 1000.0

                    ranked_doc_ids = [doc_id for doc_id, _dist in top]

                    recall_50.append(_recall_at_k(relevant_ids, ranked_doc_ids, 50))
                    recall_100.append(_recall_at_k(relevant_ids, ranked_doc_ids, 100))
                    recall_200.append(_recall_at_k(relevant_ids, ranked_doc_ids, 200))
                    ndcg_10.append(_ndcg_at_k(rel_map, ranked_doc_ids, 10))
                    mrr_vals.append(_mrr(rel_map, ranked_doc_ids))

                    retrieval_latencies.append(retrieval_ms)
                    for rank, (doc_id, dist) in enumerate(top, start=1):
                        metric_rows.append(
                            {
                                "dataset_name": dataset_name,
                                "run_name": run_name,
                                "model_name": spec.model_name,
                                "quantization": quantization,
                                "query_id": query_id,
                                "document_id": doc_id,
                                "retrieval_rank": rank,
                                "retrieval_score": float(1.0 - dist),
                                "rerank_rank": None,
                                "rerank_score": None,
                                "relevance_grade": rel_map.get(doc_id),
                                "retrieval_latency_ms": retrieval_ms,
                                "rerank_latency_ms": None,
                            },
                        )
                    _render_progress(spec.key, q_idx + 1, total_queries, "queries")

                if metric_rows:
                    await session.execute(
                        sa.insert(EmbeddingEvalPairMetric),
                        metric_rows,
                    )

                analytics_row = {
                    "dataset_name": dataset_name,
                    "run_name": run_name,
                    "model_name": spec.model_name,
                    "quantization": quantization,
                    "recall_50_mean": float(np.mean(recall_50)) if recall_50 else None,
                    "recall_100_mean": float(np.mean(recall_100))
                    if recall_100
                    else None,
                    "recall_200_mean": float(np.mean(recall_200))
                    if recall_200
                    else None,
                    "ndcg_10_mean": float(np.mean(ndcg_10)) if ndcg_10 else None,
                    "mrr_mean": float(np.mean(mrr_vals)) if mrr_vals else None,
                    "retrieval_latency_p50_ms": float(
                        np.percentile(retrieval_latencies, 50),
                    )
                    if retrieval_latencies
                    else None,
                    "retrieval_latency_p95_ms": float(
                        np.percentile(retrieval_latencies, 95),
                    )
                    if retrieval_latencies
                    else None,
                    "retrieval_latency_p99_ms": float(
                        np.percentile(retrieval_latencies, 99),
                    )
                    if retrieval_latencies
                    else None,
                    "rerank_latency_p50_ms": None,
                    "rerank_latency_p95_ms": None,
                    "rerank_latency_p99_ms": None,
                }

                await session.execute(
                    insert(EmbeddingEvalAnalytics)
                    .values(**analytics_row)
                    .on_conflict_do_update(
                        index_elements=[
                            "dataset_name",
                            "run_name",
                            "model_name",
                            "quantization",
                        ],
                        set_=analytics_row,
                    ),
                )
                await session.commit()

            typer.echo(f"Model {spec.key} ({quantization}) results:")
            typer.echo(f"  Recall@50:  {np.mean(recall_50):.4f}")
            typer.echo(f"  Recall@100: {np.mean(recall_100):.4f}")
            typer.echo(f"  Recall@200: {np.mean(recall_200):.4f}")
            typer.echo(f"  nDCG@10:    {np.mean(ndcg_10):.4f}")
            typer.echo(f"  MRR:        {np.mean(mrr_vals):.4f}")
            typer.echo(
                f"  Retrieval p95/p99: {np.percentile(retrieval_latencies, 95):.2f} / "
                f"{np.percentile(retrieval_latencies, 99):.2f} ms",
            )


if __name__ == "__main__":
    app()
