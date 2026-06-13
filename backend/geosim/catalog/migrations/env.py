"""Alembic migration environment for the catalog DB (doc 04 §2.4).

``target_metadata`` is the ORM ``Base.metadata`` so autogenerate diffs against the
doc 04 §2.4 tables. The DB URL is taken from ``GEOSIM_DB_URL`` when set (the
PostgreSQL primary, doc 04 §2.1), else the ``alembic.ini`` SQLite fallback. Tests
do NOT run migrations — they use ``geosim.catalog.db.create_all``; this env exists
so the production schema path is a real Alembic chain.
"""

from __future__ import annotations

import os

from alembic import context
from sqlalchemy import engine_from_config, pool

from geosim.catalog.models import Base

config = context.config

if (env_url := os.environ.get("GEOSIM_DB_URL")):
    config.set_main_option("sqlalchemy.url", env_url)

target_metadata = Base.metadata


def run_migrations_offline() -> None:
    url = config.get_main_option("sqlalchemy.url")
    context.configure(
        url=url,
        target_metadata=target_metadata,
        literal_binds=True,
        render_as_batch=url.startswith("sqlite"),  # SQLite ALTER support
        dialect_opts={"paramstyle": "named"},
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    connectable = engine_from_config(
        config.get_section(config.config_ini_section, {}),
        prefix="sqlalchemy.",
        poolclass=pool.NullPool,
    )
    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            render_as_batch=connection.dialect.name == "sqlite",
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
