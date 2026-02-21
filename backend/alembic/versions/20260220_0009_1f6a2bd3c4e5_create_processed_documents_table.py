"""create processed documents table filtered by astro-ph category

Revision ID: 1f6a2bd3c4e5
Revises: c4d9e1a7f3b2
Create Date: 2026-02-20 00:09:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "1f6a2bd3c4e5"
down_revision = "c4d9e1a7f3b2"
branch_labels = None
depends_on = None

CLEANED_SCHEMA = "cleaned"
PROCESSED_SCHEMA = "processed"
SOURCE_TABLE_NAME = "ar5iv_no_problem_markdown_with_metadata"
TABLE_NAME = "documents"


def upgrade() -> None:
    op.execute(
        sa.text(
            f"""
            CREATE TABLE {PROCESSED_SCHEMA}.{TABLE_NAME} AS
            SELECT *
            FROM {CLEANED_SCHEMA}.{SOURCE_TABLE_NAME}
            WHERE categories ILIKE '%astro-ph%'
            """
        )
    )


def downgrade() -> None:
    op.execute(sa.text(f"DROP TABLE IF EXISTS {PROCESSED_SCHEMA}.{TABLE_NAME}"))
