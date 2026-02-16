"""
Database connection for PostgreSQL (pgvector).
Uses DATABASE_URL from environment (.env).
"""
import os
from contextlib import contextmanager
from dotenv import load_dotenv

load_dotenv()


def get_connection():
    """Return a new psycopg2 connection. Caller must close it."""
    import psycopg2

    url = os.getenv("DATABASE_URL")
    if not url:
        raise ValueError("DATABASE_URL is not set in environment")
    return psycopg2.connect(url)


@contextmanager
def connection():
    """Context manager: yields a connection and closes it on exit."""
    conn = get_connection()
    try:
        yield conn
    finally:
        conn.close()
