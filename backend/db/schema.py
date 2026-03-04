"""
PostgreSQL schema introspection for SQL generation prompts.
Uses information_schema to get table/column metadata and foreign-key relationships.
"""
from itertools import groupby
from typing import Optional, List

from .connection import get_connection


def _fetch_columns(cur, schema_name: str, table_filter: Optional[List[str]]) -> list:
    sql = """
        SELECT t.table_schema, t.table_name, c.column_name, c.data_type, c.udt_name
        FROM information_schema.tables t
        JOIN information_schema.columns c
          ON t.table_schema = c.table_schema AND t.table_name = c.table_name
        WHERE t.table_schema = %s
          AND t.table_type = 'BASE TABLE'
        ORDER BY t.table_name, c.ordinal_position
    """
    if table_filter:
        cur.execute(sql + " AND t.table_name = ANY(%s)", (schema_name, table_filter))
    else:
        cur.execute(sql, (schema_name,))
    return cur.fetchall()


def _fetch_primary_keys(cur, schema_name: str, table_filter: Optional[List[str]]) -> set:
    """Return a set of (table_name, column_name) that are primary keys."""
    sql = """
        SELECT kcu.table_name, kcu.column_name
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema   = kcu.table_schema
        WHERE tc.constraint_type = 'PRIMARY KEY'
          AND tc.table_schema = %s
    """
    if table_filter:
        cur.execute(sql + " AND kcu.table_name = ANY(%s)", (schema_name, table_filter))
    else:
        cur.execute(sql, (schema_name,))
    return {(row[0], row[1]) for row in cur.fetchall()}


def _fetch_foreign_keys(cur, schema_name: str, table_filter: Optional[List[str]]) -> list:
    """Return (src_table, src_column, ref_table, ref_column) tuples."""
    sql = """
        SELECT
            kcu.table_name   AS src_table,
            kcu.column_name  AS src_column,
            ccu.table_name   AS ref_table,
            ccu.column_name  AS ref_column
        FROM information_schema.table_constraints tc
        JOIN information_schema.key_column_usage kcu
          ON tc.constraint_name = kcu.constraint_name
         AND tc.table_schema   = kcu.table_schema
        JOIN information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
         AND tc.table_schema   = ccu.table_schema
        WHERE tc.constraint_type = 'FOREIGN KEY'
          AND tc.table_schema = %s
        ORDER BY kcu.table_name, kcu.ordinal_position
    """
    if table_filter:
        cur.execute(sql + " AND kcu.table_name = ANY(%s)", (schema_name, table_filter))
    else:
        cur.execute(sql, (schema_name,))
    return cur.fetchall()


def get_schema_for_prompt(
    schema_name: str = "public",
    table_filter: Optional[List[str]] = None,
) -> str:
    """
    Return a human-readable description of PostgreSQL tables, columns,
    and foreign-key relationships suitable for LLM prompts.

    Args:
        schema_name: PostgreSQL schema (default "public").
        table_filter: If set, only include these table names (case-sensitive).

    Returns:
        Tables section  – "schema.table: col1 (type), col2 (type), ..."
        Relationships section – "table.column -> referenced_table.column"
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            col_rows = _fetch_columns(cur, schema_name, table_filter)
            pk_set = _fetch_primary_keys(cur, schema_name, table_filter)
            fk_rows = _fetch_foreign_keys(cur, schema_name, table_filter)
    finally:
        conn.close()

    # Build FK lookup: (table, column) -> "ref_table.ref_column"
    fk_map: dict[tuple[str, str], str] = {}
    for src_table, src_col, ref_table, ref_col in fk_rows:
        fk_map[(src_table, src_col)] = f"{ref_table}.{ref_col}"

    lines: list[str] = []

    for (tbl_schema, tbl_name), group in groupby(
        col_rows, key=lambda r: (r[0], r[1])
    ):
        lines.append(f"{tbl_schema}.{tbl_name}:")
        for row in group:
            col_name, data_type = row[2], row[3]
            markers = []
            if (tbl_name, col_name) in pk_set:
                markers.append("PK")
            fk_target = fk_map.get((tbl_name, col_name))
            if fk_target:
                markers.append(f"FK -> {fk_target}")
            marker_str = f" [{', '.join(markers)}]" if markers else ""
            lines.append(f"  - {col_name} ({data_type}){marker_str}")

    if fk_rows:
        lines.append("")
        lines.append("JOIN conditions (always use these exact conditions):")
        for src_table, src_col, ref_table, ref_col in fk_rows:
            lines.append(
                f"  {src_table}.{src_col} = {ref_table}.{ref_col}"
            )

    return "\n".join(lines) if lines else ""
