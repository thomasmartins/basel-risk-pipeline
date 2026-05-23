# Legacy (Postgres / SQLAlchemy era)

Files here are kept for reference only. They predate the Phase 0 migration to
**DuckDB + Parquet** described in `docs/ARCHITECTURE.md` and are not on any
import path. None of the runtime code reads from this directory.

| File                         | Original role                                                |
|------------------------------|--------------------------------------------------------------|
| `models_sqlalchemy.py`       | SQLAlchemy declarative models (replaced by `basel_common.types`) |
| `init_db.py`                 | Postgres schema bootstrap via `Base.metadata.create_all`     |
| `generate_data_postgres.py`  | Generator that inserted into Postgres (replaced by `basel_ingestion.generate`) |
| `schema_postgres.sql`        | Hand-written PostgreSQL DDL                                  |
| `setup.py`                   | Pre-pyproject.toml install script                            |
| `requirements_legacy.txt`    | Pre-pyproject.toml dependency list                           |

If you need to recreate the old Postgres setup for any reason (regression test,
demo of "v1"), apply `schema_postgres.sql` to an empty Postgres, set
`DB_USER`/`DB_PASSWORD`/`DB_HOST`/`DB_PORT`/`DB_NAME` env vars, then run
`python -m legacy.generate_data_postgres` (will need a one-line PYTHONPATH hack
and a `pip install sqlalchemy psycopg2-binary python-dotenv`).
