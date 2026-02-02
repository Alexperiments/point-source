"""Embedding evaluation models."""

import uuid
from datetime import datetime

from sqlalchemy import (
    JSON,
    DateTime,
    Float,
    ForeignKey,
    Integer,
    Text,
    UniqueConstraint,
)
from sqlalchemy.dialects.postgresql import UUID
from sqlalchemy.orm import Mapped, mapped_column, relationship
from sqlalchemy.sql import func

from src.core.database.base import Base


class EmbeddingEvalDocument(Base):
    """Canonical document rows for embedding benchmarks."""

    __tablename__ = "documents"
    __table_args__ = {"schema": "evaluation"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    source_id: Mapped[str | None] = mapped_column(
        Text,
        nullable=True,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    document_metadata: Mapped[dict[str, object] | None] = mapped_column(
        JSON(),
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    pair_metrics: Mapped[list["EmbeddingEvalPairMetric"]] = relationship(
        "EmbeddingEvalPairMetric",
        back_populates="document",
        cascade="all, delete-orphan",
    )


class EmbeddingEvalQuery(Base):
    """Canonical query rows for embedding benchmarks."""

    __tablename__ = "queries"
    __table_args__ = {"schema": "evaluation"}

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    text: Mapped[str] = mapped_column(Text, nullable=False)
    query_metadata: Mapped[dict[str, object] | None] = mapped_column(
        JSON(),
        nullable=True,
        default=None,
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    pair_metrics: Mapped[list["EmbeddingEvalPairMetric"]] = relationship(
        "EmbeddingEvalPairMetric",
        back_populates="query",
        cascade="all, delete-orphan",
    )


class EmbeddingEvalPairMetric(Base):
    """Wide per-pair statistics for retrieval and reranking."""

    __tablename__ = "pair_metrics"
    __table_args__ = (
        UniqueConstraint(
            "run_name",
            "model_name",
            "quantization",
            "query_id",
            "document_id",
            name="uq_pair_metrics",
        ),
        {"schema": "evaluation"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    run_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    model_name: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        index=True,
    )
    quantization: Mapped[str] = mapped_column(
        Text,
        nullable=False,
        default="fp32",
        index=True,
    )
    query_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation.queries.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    document_id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        ForeignKey("evaluation.documents.id", ondelete="CASCADE"),
        nullable=False,
        index=True,
    )
    retrieval_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieval_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_rank: Mapped[int | None] = mapped_column(Integer, nullable=True)
    rerank_score: Mapped[float | None] = mapped_column(Float, nullable=True)
    relevance_grade: Mapped[int | None] = mapped_column(Integer, nullable=True)
    retrieval_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_latency_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )

    query: Mapped["EmbeddingEvalQuery"] = relationship(
        "EmbeddingEvalQuery",
        back_populates="pair_metrics",
    )
    document: Mapped["EmbeddingEvalDocument"] = relationship(
        "EmbeddingEvalDocument",
        back_populates="pair_metrics",
    )


class EmbeddingEvalAnalytics(Base):
    """Aggregated metrics per model/run for embedding evaluations."""

    __tablename__ = "analytics"
    __table_args__ = (
        UniqueConstraint(
            "dataset_name",
            "run_name",
            "model_name",
            "quantization",
            name="uq_analytics",
        ),
        {"schema": "evaluation"},
    )

    id: Mapped[uuid.UUID] = mapped_column(
        UUID(as_uuid=True),
        primary_key=True,
        default=uuid.uuid4,
    )
    dataset_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    run_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    model_name: Mapped[str] = mapped_column(Text, nullable=False, index=True)
    quantization: Mapped[str] = mapped_column(Text, nullable=False, index=True)

    recall_50_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_100_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    recall_200_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    ndcg_10_mean: Mapped[float | None] = mapped_column(Float, nullable=True)
    mrr_mean: Mapped[float | None] = mapped_column(Float, nullable=True)

    retrieval_latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    retrieval_latency_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_latency_p50_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_latency_p95_ms: Mapped[float | None] = mapped_column(Float, nullable=True)
    rerank_latency_p99_ms: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        server_default=func.now(),
        nullable=False,
    )
