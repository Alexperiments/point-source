"""add is_premium to users

Revision ID: 7e4a1f2c9b61
Revises: 5c7e9a1d2f34
Create Date: 2026-02-27 23:50:00.000000

"""

import sqlalchemy as sa
from alembic import op


# revision identifiers, used by Alembic.
revision = "7e4a1f2c9b61"
down_revision = "5c7e9a1d2f34"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "is_premium",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("false"),
        ),
    )


def downgrade() -> None:
    op.drop_column("users", "is_premium")
