"""add jina embeddings v3 eval tables

Revision ID: 4ad7c7b905e1
Revises: 9e3f7a2b1c4d
Create Date: 2026-02-24 12:00:00.000000

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "4ad7c7b905e1"
down_revision = "9e3f7a2b1c4d"
branch_labels = None
depends_on = None

SCHEMA_NAME = "evaluation"
MODEL_KEY = "jina_embeddings_v3"
DIMENSION = 1024


def _create_embedding_table(*, kind: str, fk_table: str) -> None:
    table_prefix = "document" if kind == "document" else "query"
    table_name = f"{table_prefix}_embeddings_{MODEL_KEY}"
    id_column = f"{kind}_id"

    op.execute(
        sa.text(
            f"""
            CREATE TABLE IF NOT EXISTS {SCHEMA_NAME}.{table_name} (
                id uuid PRIMARY KEY DEFAULT gen_random_uuid(),
                {id_column} uuid NOT NULL REFERENCES {SCHEMA_NAME}.{fk_table}(id) ON DELETE CASCADE,
                quantization text NOT NULL DEFAULT 'fp32',
                embedding vector({DIMENSION}) NULL,
                created_at timestamptz NOT NULL DEFAULT now(),
                CONSTRAINT uq_{table_prefix}_embeddings_{MODEL_KEY} UNIQUE (
                    {id_column},
                    quantization
                )
            );
            """
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{MODEL_KEY}_{id_column} "
            f"ON {SCHEMA_NAME}.{table_name} ({id_column});"
        )
    )
    op.execute(
        sa.text(
            f"CREATE INDEX IF NOT EXISTS ix_{table_prefix}_embeddings_{MODEL_KEY}_quantization "
            f"ON {SCHEMA_NAME}.{table_name} (quantization);"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA_NAME}.{table_name} "
            f"ADD COLUMN IF NOT EXISTS embedding_halfvec halfvec({DIMENSION});"
        )
    )
    op.execute(
        sa.text(
            f"ALTER TABLE {SCHEMA_NAME}.{table_name} "
            f"ADD COLUMN IF NOT EXISTS embedding_bit bit({DIMENSION});"
        )
    )


def upgrade() -> None:
    _create_embedding_table(kind="document", fk_table="documents")
    _create_embedding_table(kind="query", fk_table="queries")


def downgrade() -> None:
    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.query_embeddings_{MODEL_KEY} CASCADE;"
        )
    )
    op.execute(
        sa.text(
            f"DROP TABLE IF EXISTS {SCHEMA_NAME}.document_embeddings_{MODEL_KEY} CASCADE;"
        )
    )
