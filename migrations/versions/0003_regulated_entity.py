"""regulated entity columns

Revision ID: 0003
Revises: 0002
"""

import sqlalchemy as sa
from alembic import op
from sqlalchemy.dialects.postgresql import ARRAY

revision = "0003"
down_revision = "0002"
branch_labels = None
depends_on = None


def upgrade() -> None:
    for table in ("documents", "chunks"):
        op.add_column(
            table,
            sa.Column(
                "regulated_entity",
                ARRAY(sa.Text()),
                nullable=False,
                server_default="{}",
            ),
        )


def downgrade() -> None:
    for table in ("chunks", "documents"):
        op.drop_column(table, "regulated_entity")
