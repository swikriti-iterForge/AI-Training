"""Alembic setup. The database URL comes from our settings (.env / environment),
and the models are the source of truth for `alembic revision --autogenerate`."""

import logging

from alembic import context
from sqlalchemy import create_engine

from bookverse.core.config import get_settings
from bookverse.models import Base

logging.basicConfig(format="%(levelname)s [%(name)s] %(message)s")
logging.getLogger("alembic").setLevel(logging.INFO)

engine = create_engine(get_settings().sync_database_url)
with engine.connect() as connection:
    context.configure(connection=connection, target_metadata=Base.metadata, compare_type=True)
    with context.begin_transaction():
        context.run_migrations()
