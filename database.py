"""
FinGraph AI - Database Configuration & Session Management
Supports SQLite by default (with PRAGMA foreign_keys=ON) and PostgreSQL.
"""

import os
from pathlib import Path
from sqlalchemy import create_engine, event
from sqlalchemy.orm import declarative_base, sessionmaker

# Database URL: default to sqlite in data/fingraph.db
DEFAULT_DB_PATH = Path(__file__).resolve().parent / "data" / "fingraph.db"
DEFAULT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

DATABASE_URL = os.getenv("DATABASE_URL", f"sqlite:///{DEFAULT_DB_PATH.as_posix()}")

# Create SQLAlchemy engine
connect_args = {}
if DATABASE_URL.startswith("sqlite"):
    connect_args = {"check_same_thread": False}

engine = create_engine(
    DATABASE_URL,
    connect_args=connect_args,
    echo=False,
    future=True,
)

# Enforce foreign key constraints and check constraints in SQLite
if DATABASE_URL.startswith("sqlite"):
    @event.listens_for(engine, "connect")
    def set_sqlite_pragma(dbapi_connection, connection_record):
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine, future=True)

Base = declarative_base()


def get_db():
    """Dependency helper yielding database session."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def init_db_and_migrate():
    """Ensure all tables and new columns exist without data loss."""
    Base.metadata.create_all(bind=engine)
    if DATABASE_URL.startswith("sqlite"):
        import sqlite3
        db_path = DEFAULT_DB_PATH
        if db_path.exists():
            con = sqlite3.connect(str(db_path))
            cur = con.cursor()
            
            # Helper to add column if missing
            def ensure_col(table, col, col_def):
                cols = [r[1] for r in cur.execute(f"PRAGMA table_info({table})").fetchall()]
                if col not in cols:
                    try:
                        cur.execute(f"ALTER TABLE {table} ADD COLUMN {col} {col_def}")
                    except Exception:
                        pass

            ensure_col("articles", "fingerprint", "VARCHAR(64)")
            ensure_col("articles", "category", "VARCHAR(64)")
            ensure_col("articles", "entities", "JSON DEFAULT '[]'")
            ensure_col("articles", "event_id", "VARCHAR(64)")

            ensure_col("events", "category", "VARCHAR(64)")
            ensure_col("events", "source_url", "VARCHAR(1024)")
            ensure_col("events", "source_name", "VARCHAR(256)")
            ensure_col("events", "entities", "JSON DEFAULT '[]'")

            ensure_col("provider_status", "records_count", "INTEGER DEFAULT 0")
            ensure_col("provider_status", "last_success_at", "DATETIME")
            ensure_col("provider_status", "last_attempt_at", "DATETIME")

            con.commit()
            con.close()


# Run migration on module import
init_db_and_migrate()
