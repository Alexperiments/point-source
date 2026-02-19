"""create cleaned arxiv snapshot table

Revision ID: 9b825f90b44f
Revises: a0dcce305480
Create Date: 2026-02-20 00:04:00.798104

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "9b825f90b44f"
down_revision = "a0dcce305480"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
RAW_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"
CLEANED_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"


def upgrade() -> None:
    op.create_table(
        CLEANED_TABLE_NAME,
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
        sa.PrimaryKeyConstraint("id"),
        schema=CLEANED_SCHEMA,
    )

    op.execute(
        sa.text(
            f"""
            WITH source AS (
                SELECT COALESCE(raw_json->'root', raw_json) AS payload
                FROM {RAW_SCHEMA}.{RAW_TABLE_NAME}
            )
            INSERT INTO {CLEANED_SCHEMA}.{CLEANED_TABLE_NAME} (
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
                last_updated
            )
            SELECT
                payload->>'id' AS id,
                CASE
                    WHEN jsonb_typeof(payload->'authors_parsed') = 'array' THEN (
                        SELECT string_agg(
                            NULLIF(
                                trim(
                                    regexp_replace(
                                        concat_ws(
                                            ' ',
                                            NULLIF(
                                                trim(
                                                    regexp_replace(
                                                        author_item->>1,
                                                        '[[:space:]]*-[[:space:]]*',
                                                        '-',
                                                        'g'
                                                    )
                                                ),
                                                ''
                                            ),
                                            NULLIF(trim(author_item->>0), ''),
                                            NULLIF(trim(author_item->>2), '')
                                        ),
                                        '[[:space:]]+',
                                        ' ',
                                        'g'
                                    )
                                ),
                                ''
                            ),
                            ', ' ORDER BY ord
                        )
                        FROM jsonb_array_elements(payload->'authors_parsed')
                        WITH ORDINALITY AS author_parts(author_item, ord)
                    )
                    ELSE NULL
                END AS authors,
                payload->>'title' AS title,
                payload->>'comments' AS comments,
                payload->>'journal-ref' AS journal_ref,
                payload->>'doi' AS doi,
                payload->>'report-no' AS report_no,
                payload->>'categories' AS categories,
                payload->>'license' AS license,
                CASE
                    WHEN jsonb_typeof(payload->'versions') = 'array'
                     AND jsonb_array_length(payload->'versions') > 0 THEN
                        ((payload->'versions')->0->>'created')::timestamptz
                    ELSE NULL
                END AS created,
                CASE
                    WHEN jsonb_typeof(payload->'versions') = 'array'
                     AND jsonb_array_length(payload->'versions') > 0 THEN (
                        SELECT (version_item->>'created')::timestamptz
                        FROM jsonb_array_elements(payload->'versions')
                        WITH ORDINALITY AS version_parts(version_item, ord)
                        ORDER BY ord DESC
                        LIMIT 1
                    )
                    ELSE NULL
                END AS last_updated
            FROM source
            WHERE NULLIF(trim(payload->>'id'), '') IS NOT NULL
            ON CONFLICT (id) DO NOTHING
            """
        )
    )


def downgrade() -> None:
    op.drop_table(CLEANED_TABLE_NAME, schema=CLEANED_SCHEMA)
