"""
PostgreSQL schema introspection for SQL generation prompts.
Uses information_schema to get table and column metadata.
"""
from typing import Optional, List

from .connection import get_connection


def get_schema_for_prompt(
    schema_name: str = "public",
    table_filter: Optional[List[str]] = None,
) -> str:
    """
    Return a human-readable description of PostgreSQL tables and columns
    suitable for LLM prompts (e.g. SQL_GENERATION).

    Args:
        schema_name: PostgreSQL schema (default "public").
        table_filter: If set, only include these table names (case-sensitive).

    Returns:
        String like "public.table_name: col1 (type), col2 (type), ..." per table.
    """
    sql = """
        SELECT t.table_schema, t.table_name, c.column_name, c.data_type, c.udt_name
        FROM information_schema.tables t
        JOIN information_schema.columns c
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE t.table_schema = %s
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            if table_filter:
                cur.execute(
                    sql + " AND t.table_name = ANY(%s)",
                    (schema_name, table_filter),
                )
            else:
                cur.execute(sql, (schema_name,))
            rows = cur.fetchall()
    finally:
        conn.close()

    # Build "schema.table: col1 (type), col2 (type), ..." per table
    from itertools import groupby

    lines = []
    for (table_schema, table_name), group in groupby(
        rows, key=lambda r: (r[0], r[1])
    ):
        cols = [f"{r[2]} ({r[3]})" for r in group]
        lines.append(f"{table_schema}.{table_name}: {', '.join(cols)}")
    return "\n".join(lines) if lines else ""
