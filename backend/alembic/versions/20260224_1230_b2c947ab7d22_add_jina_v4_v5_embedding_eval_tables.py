"""add jina v4 v5 embedding eval tables

Revision ID: b2c947ab7d22
Revises: 4ad7c7b905e1
Create Date: 2026-02-24 12:30:00.000000

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "b2c947ab7d22"
down_revision = "4ad7c7b905e1"
branch_labels = None
depends_on = None

SCHEMA_NAME = "evaluation"
MODEL_SPECS: tuple[tuple[str, int], ...] = (
    ("jina_embeddings_v5_text_small", 1024),
    ("jina_embeddings_v4", 2048),
)


def _create_embedding_table(*, model_key: str, dim: int, kind: str, fk_table: str) -> None:
    table_prefix = "document" if kind == "document" else "query"
    table_name = f"{table_prefix}_embeddings_{model_key}"
    id_column = f"{kind}_id"

    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{table_name} (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                {id_column} uuid NOT NULL REFERENCES {SCHEMA_NAME}.{fk_table}(id) ON DELETE CASCADE,
                quantization text NOT NULL DEFAULT 'fp32',
                embedding vector({dim}) NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                CONSTRAINT uq_{table_prefix}_embeddings_{model_key} UNIQUE (
                    {id_column},
                    quantization
                )
            );
            """
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{model_key}_{id_column} "
            f"ON {SCHEMA_NAME}.{table_name} ({id_column});"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{model_key}_quantization "
            f"ON {SCHEMA_NAME}.{table_name} (quantization);"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA_NAME}.{table_name} "
            f"ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec({dim});"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA_NAME}.{table_name} "
            f"ADD COLUMN IF NOT EXISTS embedding_bit bit({dim});"
        )
    )


def upgrade() -> None:
    for model_key, dim in MODEL_SPECS:
        _create_embedding_table(
            model_key=model_key,
            dim=dim,
            kind="document",
            fk_table="documents",
        )
        _create_embedding_table(
            model_key=model_key,
            dim=dim,
            kind="query",
            fk_table="queries",
        )


def downgrade() -> None:
    for model_key, _ in reversed(MODEL_SPECS):
        op.execute(
            sa.text(
                f"DROP TABLE IF EXISTS {SCHEMA_NAME}.query_embeddings_{model_key} CASCADE;"
            )
        )
        op.execute(
            sa.text(
                f"DROP TABLE IF EXISTS {SCHEMA_NAME}.document_embeddings_{model_key} CASCADE;"
            )
        )
