"""Alembic setup. The database URL comes from our settings, the models are the source of truth."""

import logging

from sqlalchemy import create_engine

from alembic import context
from trailmate_search.config import get_settings
from trailmate_search.models import Base

logging.basicConfig(format="%(levelname)s [%(name)s] %(message)s")
logging.getLogger("alembic").setLevel(logging.INFO)

engine = create_engine(get_settings().sync_database_url)
with engine.connect() as connection:
    context.configure(
        connection=connection,
        target_metadata=Base.metadata,
        compare_type=True,
        # This service's own version table: the agent service will share the database later,
        # and two services must not fight over one `alembic_version` table.
        version_table="alembic_version_search",
    )
    with context.begin_transaction():
        context.run_migrations()
