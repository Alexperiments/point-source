"""create eval analytics table

Revision ID: 5f0c2f0e42f1
Revises: d1e7c8b1f9aa
Create Date: 2026-01-31 10:30:00.000000

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = "5f0c2f0e42f1"
down_revision = "d1e7c8b1f9aa"
branch_labels = None
depends_on = None

SCHEMA_NAME = "evaluation"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {SCHEMA_NAME}"))

    op.create_table(
        "analytics",
        sa.Column(
            "id",
            sa.UUID(),
            server_default=sa.text("gen_random_uuid()"),
            nullable=False,
        ),
        sa.Column("dataset_name", sa.Text(), nullable=False),
        sa.Column("run_name", sa.Text(), nullable=False),
        sa.Column("model_name", sa.Text(), nullable=False),
        sa.Column("quantization", sa.Text(), nullable=False),
        sa.Column("recall_50_mean", sa.Float(), nullable=True),
        sa.Column("recall_100_mean", sa.Float(), nullable=True),
        sa.Column("recall_200_mean", sa.Float(), nullable=True),
        sa.Column("ndcg_10_mean", sa.Float(), nullable=True),
        sa.Column("mrr_mean", sa.Float(), nullable=True),
        sa.Column("retrieval_latency_p50_ms", sa.Float(), nullable=True),
        sa.Column("retrieval_latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("retrieval_latency_p99_ms", sa.Float(), nullable=True),
        sa.Column("rerank_latency_p50_ms", sa.Float(), nullable=True),
        sa.Column("rerank_latency_p95_ms", sa.Float(), nullable=True),
        sa.Column("rerank_latency_p99_ms", sa.Float(), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.text("now()"),
            nullable=False,
        ),
        sa.PrimaryKeyConstraint("id"),
        sa.UniqueConstraint(
            "dataset_name",
            "run_name",
            "model_name",
            "quantization",
            name="uq_analytics",
        ),
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_analytics_dataset_name"),
        "analytics",
        ["dataset_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_analytics_run_name"),
        "analytics",
        ["run_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_analytics_model_name"),
        "analytics",
        ["model_name"],
        unique=False,
        schema=SCHEMA_NAME,
    )
    op.create_index(
        op.f("ix_analytics_quantization"),
        "analytics",
        ["quantization"],
        unique=False,
        schema=SCHEMA_NAME,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_analytics_quantization"),
        table_name="analytics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_analytics_model_name"),
        table_name="analytics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_analytics_run_name"),
        table_name="analytics",
        schema=SCHEMA_NAME,
    )
    op.drop_index(
        op.f("ix_analytics_dataset_name"),
        table_name="analytics",
        schema=SCHEMA_NAME,
    )
    op.drop_table("analytics", schema=SCHEMA_NAME)
