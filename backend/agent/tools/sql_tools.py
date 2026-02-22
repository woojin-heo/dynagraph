"""
SQL execution tool for PostgreSQL.
Executes a read-only style query and returns results as text.
"""
from langchain_core.tools import tool

try:
    from backend.db import get_connection
except ImportError:
    from db import get_connection


@tool
def sql_execution(query: str) -> str:
    """
    Execute a SQL query on the PostgreSQL database and return the result.
    Args:
        query: The SQL query to execute (e.g. from SQL_GENERATION).
    Returns:
        A string representation of the query result, or an error message.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(query)
            rows = cur.fetchall()
            if not rows:
                return "(No rows returned)"
            # Simple tabular text output
            col_count = len(rows[0])
            lines = [" | ".join(str(c) for c in row) for row in rows]
            return "\n".join(lines)
    except Exception as e:
        return f"Error executing query: {e}"
    finally:
        conn.close()
