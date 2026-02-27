"""create doi_url and drop legacy url columns

Revision ID: a6b1ff3abcbc
Revises: ed586ee27f65
Create Date: 2026-02-26 22:39:31.497478

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "a6b1ff3abcbc"
down_revision = "ed586ee27f65"
branch_labels = None
depends_on = None

PROCESSED_SCHEMA = "processed"
TABLE_NAME = "documents"
DOI_PREFIX = "https://www.doi.org/"
ARXIV_DOI_PREFIX = "10.48550/arXiv."
ARXIV_ABS_PREFIX = "https://arxiv.org/abs/"


def upgrade() -> None:
    op.add_column(
        TABLE_NAME,
        sa.Column("doi_url", sa.Text(), nullable=True),
        schema=PROCESSED_SCHEMA,
    )

    op.execute(
        sa.text(
            f"""
            UPDATE {PROCESSED_SCHEMA}.{TABLE_NAME}
            SET doi_url = CASE
                WHEN NULLIF(trim(doi), '') IS NOT NULL
                    THEN :doi_prefix || trim(doi)
                ELSE :arxiv_doi_prefix || id
            END
            """
        ).bindparams(
            doi_prefix=DOI_PREFIX,
            arxiv_doi_prefix=ARXIV_DOI_PREFIX,
        )
    )

    op.alter_column(
        TABLE_NAME,
        "doi_url",
        existing_type=sa.Text(),
        nullable=False,
        schema=PROCESSED_SCHEMA,
    )

    op.drop_column(TABLE_NAME, "url", schema=PROCESSED_SCHEMA)


def downgrade() -> None:
    op.add_column(
        TABLE_NAME,
        sa.Column("url", sa.Text(), nullable=True),
        schema=PROCESSED_SCHEMA,
    )
    op.execute(
        sa.text(
            f"""
            UPDATE {PROCESSED_SCHEMA}.{TABLE_NAME}
            SET url = :arxiv_abs_prefix || id
            """
        ).bindparams(arxiv_abs_prefix=ARXIV_ABS_PREFIX)
    )
    op.alter_column(
        TABLE_NAME,
        "url",
        existing_type=sa.Text(),
        nullable=False,
        schema=PROCESSED_SCHEMA,
    )
    op.drop_column(TABLE_NAME, "doi_url", schema=PROCESSED_SCHEMA)
