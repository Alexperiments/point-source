"""add email verification and reset tokens

Revision ID: 2b3f4c5d6e71
Revises: 7e4a1f2c9b61
Create Date: 2026-03-17 00:01:00.000000

"""

from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql


# revision identifiers, used by Alembic.
revision = "2b3f4c5d6e71"
down_revision = "7e4a1f2c9b61"
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.add_column(
        "users",
        sa.Column(
            "email_verified",
            sa.Boolean(),
            nullable=False,
            server_default=sa.text("true"),
        ),
    )
    op.add_column(
        "users",
        sa.Column("email_verified_at", sa.DateTime(timezone=True), nullable=True),
    )
    op.create_index(
        op.f("ix_users_email_verified"),
        "users",
        ["email_verified"],
        unique=False,
    )

    op.create_table(
        "email_action_tokens",
        sa.Column(
            "id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column(
            "user_id",
            postgresql.UUID(as_uuid=True),
            nullable=False,
        ),
        sa.Column("purpose", sa.String(length=32), nullable=False),
        sa.Column("token_hash", sa.String(length=64), nullable=False),
        sa.Column("email", sa.String(length=255), nullable=False),
        sa.Column("expires_at", sa.DateTime(timezone=True), nullable=False),
        sa.Column("consumed_at", sa.DateTime(timezone=True), nullable=True),
        sa.Column(
            "created_at",
            sa.DateTime(timezone=True),
            server_default=sa.func.now(),
            nullable=False,
        ),
        sa.ForeignKeyConstraint(
            ["user_id"],
            ["users.id"],
            ondelete="CASCADE",
        ),
        sa.PrimaryKeyConstraint("id"),
    )
    op.create_index(
        op.f("ix_email_action_tokens_user_id"),
        "email_action_tokens",
        ["user_id"],
        unique=False,
    )
    op.create_index(
        op.f("ix_email_action_tokens_purpose"),
        "email_action_tokens",
        ["purpose"],
        unique=False,
    )
    op.create_index(
        op.f("ix_email_action_tokens_token_hash"),
        "email_action_tokens",
        ["token_hash"],
        unique=True,
    )
    op.create_index(
        op.f("ix_email_action_tokens_expires_at"),
        "email_action_tokens",
        ["expires_at"],
        unique=False,
    )
    op.create_index(
        op.f("ix_email_action_tokens_consumed_at"),
        "email_action_tokens",
        ["consumed_at"],
        unique=False,
    )


def downgrade() -> None:
    op.drop_index(
        op.f("ix_email_action_tokens_consumed_at"),
        table_name="email_action_tokens",
    )
    op.drop_index(
        op.f("ix_email_action_tokens_expires_at"),
        table_name="email_action_tokens",
    )
    op.drop_index(
        op.f("ix_email_action_tokens_token_hash"),
        table_name="email_action_tokens",
    )
    op.drop_index(
        op.f("ix_email_action_tokens_purpose"),
        table_name="email_action_tokens",
    )
    op.drop_index(
        op.f("ix_email_action_tokens_user_id"),
        table_name="email_action_tokens",
    )
    op.drop_table("email_action_tokens")

    op.drop_index(op.f("ix_users_email_verified"), table_name="users")
    op.drop_column("users", "email_verified_at")
    op.drop_column("users", "email_verified")
