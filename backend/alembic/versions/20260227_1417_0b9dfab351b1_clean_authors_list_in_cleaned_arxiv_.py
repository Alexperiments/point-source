"""clean authors list in cleaned arxiv metadata snapshot

Revision ID: 0b9dfab351b1
Revises: a6b1ff3abcbc
Create Date: 2026-02-27 14:17:56.099966

"""
import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "0b9dfab351b1"
down_revision = "a6b1ff3abcbc"
branch_labels = None
depends_on = None

RAW_SCHEMA = "raw"
CLEANED_SCHEMA = "cleaned"
TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"
RAW_TABLE_NAME = "arxiv_metadata_oai_snapshot_2026_02_19"


def _author_item_expr(index: str) -> str:
    if index == "1":
        return f"""
        NULLIF(
            trim(
                regexp_replace(
                    author_item->>{index},
                    '[[:space:]]*-[[:space:]]*',
                    '-',
                    'g'
                )
            ),
            ''
        )
        """
    return f"NULLIF(trim(author_item->>{index}), '')"


def _authors_expression(*, order: str) -> str:
    return f"""
    CASE
        WHEN jsonb_typeof(payload->'authors_parsed') = 'array' THEN (
            SELECT string_agg(author_name, ', ' ORDER BY ord)
            FROM (
                SELECT
                    NULLIF(
                        trim(
                            regexp_replace(
                                concat_ws(
                                    ' ',
                                    {_author_item_expr(order[0])},
                                    {_author_item_expr(order[1])},
                                    {_author_item_expr(order[2])}
                                ),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                    ) AS author_name,
                    ord
                FROM jsonb_array_elements(payload->'authors_parsed')
                WITH ORDINALITY AS author_parts(author_item, ord)
            ) AS normalized
            WHERE author_name IS NOT NULL
        )
        ELSE NULL
    END
    """


def _number_of_authors_expression(*, order: str) -> str:
    return f"""
    CASE
        WHEN jsonb_typeof(payload->'authors_parsed') = 'array' THEN (
            SELECT COUNT(*)
            FROM (
                SELECT
                    NULLIF(
                        trim(
                            regexp_replace(
                                concat_ws(
                                    ' ',
                                    {_author_item_expr(order[0])},
                                    {_author_item_expr(order[1])},
                                    {_author_item_expr(order[2])}
                                ),
                                '[[:space:]]+',
                                ' ',
                                'g'
                            )
                        ),
                        ''
                    ) AS author_name
                FROM jsonb_array_elements(payload->'authors_parsed')
                WITH ORDINALITY AS author_parts(author_item, ord)
            ) AS normalized
            WHERE author_name IS NOT NULL
        )::integer
        ELSE 0
    END
    """


def _refresh_authors(*, order: str, refresh_count: bool) -> None:
    number_of_authors_sql = _number_of_authors_expression(order=order)
    set_clause = "authors = source.authors"
    if refresh_count:
        set_clause += ", number_of_authors = source.number_of_authors"

    op.execute(
        sa.text(
            f"""
            WITH payload_source AS (
                SELECT COALESCE(raw_json->'root', raw_json) AS payload
                FROM {RAW_SCHEMA}.{RAW_TABLE_NAME}
            ),
            source AS (
                SELECT
                    payload->>'id' AS id,
                    {_authors_expression(order=order)} AS authors,
                    {number_of_authors_sql} AS number_of_authors
                FROM payload_source
                WHERE NULLIF(trim(payload->>'id'), '') IS NOT NULL
            )
            UPDATE {CLEANED_SCHEMA}.{TABLE_NAME} AS target
            SET {set_clause}
            FROM source
            WHERE target.id = source.id
            """
        )
    )


def upgrade() -> None:
    op.add_column(
        TABLE_NAME,
        sa.Column(
            "number_of_authors",
            sa.Integer(),
            nullable=False,
            server_default=sa.text("0"),
        ),
        schema=CLEANED_SCHEMA,
    )
    _refresh_authors(order="012", refresh_count=True)
    op.alter_column(
        TABLE_NAME,
        "number_of_authors",
        existing_type=sa.Integer(),
        server_default=None,
        schema=CLEANED_SCHEMA,
    )


def downgrade() -> None:
    _refresh_authors(order="102", refresh_count=False)
    op.drop_column(TABLE_NAME, "number_of_authors", schema=CLEANED_SCHEMA)
