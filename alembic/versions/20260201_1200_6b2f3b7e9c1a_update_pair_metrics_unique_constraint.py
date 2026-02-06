"""update pair_metrics unique constraint to include dataset_name

Revision ID: 6b2f3b7e9c1a
Revises: 5f0c2f0e42f1
Create Date: 2026-02-01 12:00:00.000000

"""
from alembic import op


# revision identifiers, used by Alembic.
revision = "6b2f3b7e9c1a"
down_revision = "5f0c2f0e42f1"
branch_labels = None
depends_on = None

SCHEMA_NAME = "evaluation"


def upgrade() -> None:
    op.drop_constraint(
        "uq_pair_metrics",
        "pair_metrics",
        schema=SCHEMA_NAME,
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_pair_metrics",
        "pair_metrics",
        [
            "dataset_name",
            "run_name",
            "model_name",
            "quantization",
            "query_id",
            "document_id",
        ],
        schema=SCHEMA_NAME,
    )


def downgrade() -> None:
    op.drop_constraint(
        "uq_pair_metrics",
        "pair_metrics",
        schema=SCHEMA_NAME,
        type_="unique",
    )
    op.create_unique_constraint(
        "uq_pair_metrics",
        "pair_metrics",
        [
            "run_name",
            "model_name",
            "quantization",
            "query_id",
            "document_id",
        ],
        schema=SCHEMA_NAME,
    )
