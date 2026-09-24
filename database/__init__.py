"""Central PostgreSQL configuration and connection helpers for the VMS backend."""

from __future__ import annotations

import os
import re
from pathlib import Path

from dotenv import load_dotenv
import psycopg

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")


def _require_env(name: str) -> str:
    value = os.getenv(name, "").strip()
    if not value:
        raise RuntimeError(f"PostgreSQL database connection failed: missing {name} in environment")
    return value


def get_postgres_config() -> dict:
    return {
        "host": _require_env("POSTGRES_HOST"),
        "port": int(_require_env("POSTGRES_PORT")),
        "dbname": _require_env("POSTGRES_DB"),
        "user": _require_env("POSTGRES_USER"),
        "password": _require_env("POSTGRES_PASSWORD"),
    }


def translate_sqlite_placeholders(query: str, params: tuple | list | None = None):
    params = tuple() if params is None else tuple(params)
    if "?" not in query:
        return query, params
    rewritten = re.sub(r"\?", "%s", query)
    return rewritten, params


class PostgresConnection:
    def __init__(self, connection):
        self._connection = connection

    def execute(self, query, params=()):
        rewritten_query, rewritten_params = translate_sqlite_placeholders(query, params)
        return self._connection.execute(rewritten_query, rewritten_params)

    def commit(self):
        self._connection.commit()

    def rollback(self):
        self._connection.rollback()

    def close(self):
        self._connection.close()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


def get_db_connection():
    config = get_postgres_config()
    connection = psycopg.connect(**config)
    return PostgresConnection(connection)


def validate_postgres_connection():
    try:
        with get_db_connection() as connection:
            connection.execute("SELECT 1").fetchone()
        return True
    except Exception as exc:  # pragma: no cover - runtime startup validation
        raise RuntimeError(f"PostgreSQL database connection failed: {exc}") from exc


__all__ = [
    "PROJECT_ROOT",
    "get_postgres_config",
    "translate_sqlite_placeholders",
    "PostgresConnection",
    "get_db_connection",
    "validate_postgres_connection",
]
