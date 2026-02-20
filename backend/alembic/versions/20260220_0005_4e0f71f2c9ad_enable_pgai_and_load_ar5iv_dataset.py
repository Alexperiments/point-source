"""enable pgai extension and load ar5iv dataset into raw schema

Revision ID: 4e0f71f2c9ad
Revises: 9b825f90b44f
Create Date: 2026-02-20 00:05:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "4e0f71f2c9ad"
down_revision = "9b825f90b44f"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
DATASET_NAME = "marin-community/ar5iv-no-problem-markdown"
TABLE_NAME = "ar5iv_no_problem_markdown"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {RAW_SCHEMA}"))
    op.execute(sa.text("CREATE EXTENSION IF NOT EXISTS ai CASCADE"))

    # Warm Hugging Face metadata with a bounded non-streaming sample so pgai's
    # streaming loader has features populated for this dataset.
    op.execute(
        sa.text(
            f"""
            DO $$
            import site
            version = plpy.execute(
                "SELECT extversion FROM pg_extension WHERE extname = 'ai' LIMIT 1"
            )[0]["extversion"]
            site.addsitedir("/usr/local/lib/pgai/" + version)
            import datasets
            split_expr = "train[" + ":" + "1]"
            datasets.load_dataset(
                "{DATASET_NAME}",
                split=split_expr,
                streaming=False,
            )
            $$ LANGUAGE plpython3u;
            """
        )
    )

    op.execute(
        sa.text(
            f"""
            SELECT ai.load_dataset(
                '{DATASET_NAME}',
                table_name => '{TABLE_NAME}',
                schema_name => '{RAW_SCHEMA}'
            );
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f'DROP TABLE IF EXISTS {RAW_SCHEMA}."{TABLE_NAME}"'))
    op.execute(sa.text("DROP EXTENSION IF EXISTS ai"))
