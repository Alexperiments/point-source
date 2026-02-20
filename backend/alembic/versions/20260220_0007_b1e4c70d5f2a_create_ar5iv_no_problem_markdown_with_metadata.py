"""create cleaned ar5iv markdown with metadata table

Revision ID: b1e4c70d5f2a
Revises: 6f97b7ec2b31
Create Date: 2026-02-20 00:07:00.000000

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "b1e4c70d5f2a"
down_revision = "6f97b7ec2b31"
branch_labels = None
depends_on = None

CLEANED_SCHEMA = "cleaned"
METADATA_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"
MARKDOWN_TABLE_NAME = "ar5iv_no_problem_markdown"
JOINED_TABLE_NAME = "ar5iv_no_problem_markdown_with_metadata"


def upgrade() -> None:
    op.execute(sa.text(f"CREATE SCHEMA IF NOT EXISTS {CLEANED_SCHEMA}"))

    op.create_table(
        JOINED_TABLE_NAME,
        sa.Column("id", sa.Text(), nullable=False),
        sa.Column("authors", sa.Text(), nullable=True),
        sa.Column("title", sa.Text(), nullable=True),
        sa.Column("comments", sa.Text(), nullable=True),
        sa.Column("journal_ref", sa.Text(), nullable=True),
        sa.Column("doi", sa.Text(), nullable=True),
        sa.Column("report_no", sa.Text(), nullable=True),
        sa.Column("categories", sa.Text(), nullable=True),
        sa.Column("license", sa.Text(), nullable=True),
        sa.Column("created", sa.DateTime(timezone=True), nullable=True),
        sa.Column("last_updated", sa.DateTime(timezone=True), nullable=True),
        sa.Column("text", sa.Text(), nullable=True),
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )

    op.execute(
        sa.text(
            f"""
            INSERT INTO {CLEANED_SCHEMA}.{JOINED_TABLE_NAME} (
                id,
                authors,
                title,
                comments,
                journal_ref,
                doi,
                report_no,
                categories,
                license,
                created,
                last_updated,
                text
            )
            SELECT
                metadata.id,
                metadata.authors,
                metadata.title,
                metadata.comments,
                metadata.journal_ref,
                metadata.doi,
                metadata.report_no,
                metadata.categories,
                metadata.license,
                metadata.created,
                metadata.last_updated,
                markdown.text
            FROM {CLEANED_SCHEMA}.{METADATA_TABLE_NAME} AS metadata
            JOIN {CLEANED_SCHEMA}.{MARKDOWN_TABLE_NAME} AS markdown
              ON markdown.id = metadata.id
            ON CONFLICT (id) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_table(JOINED_TABLE_NAME, schema=CLEANED_SCHEMA)
