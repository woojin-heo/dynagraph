"""
PostgreSQL schema introspection for SQL generation prompts.
Uses information_schema to get table/column metadata and foreign-key relationships.
Outputs CREATE TABLE DDL so the LLM sees the exact column definitions.
"""
from itertools import groupby
from typing import Optional, List, Set

from .connection import get_connection

OPERATIONAL_TABLES: Set[str] = {
    "tenants",
    "conversations",
    "document_chunks",
    "checkpoint_blobs",
    "checkpoint_migrations",
    "checkpoint_writes",
    "checkpoints",
}


def _effective_filter(
    table_filter: Optional[List[str]],
) -> Optional[List[str]]:
    """If table_filter is given, return it as-is (user explicitly chose tables).
    Otherwise return None so the SQL queries fetch everything — operational
    tables are stripped later in Python so we don't need to know their names
    at query time."""
    return table_filter


def _exclude_operational(rows: list, table_name_index: int,
                         table_filter: Optional[List[str]]) -> list:
    """Remove rows belonging to OPERATIONAL_TABLES unless the caller
    explicitly asked for them via table_filter."""
    if table_filter:
        return rows
    return [r for r in rows if r[table_name_index] not in OPERATIONAL_TABLES]


def _fetch_columns(cur, schema_name: str, table_filter: Optional[List[str]]) -> list:
    sql = """
        SELECT t.table_schema, t.table_name,
               c.column_name, c.data_type, c.udt_name,
               c.is_nullable, c.column_default
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


_PG_TYPE_MAP = {
    "integer": "INTEGER",
    "bigint": "BIGINT",
    "smallint": "SMALLINT",
    "numeric": "NUMERIC",
    "real": "REAL",
    "double precision": "DOUBLE PRECISION",
    "boolean": "BOOLEAN",
    "text": "TEXT",
    "character varying": "VARCHAR",
    "character": "CHAR",
    "date": "DATE",
    "timestamp without time zone": "TIMESTAMP",
    "timestamp with time zone": "TIMESTAMPTZ",
    "time without time zone": "TIME",
    "time with time zone": "TIMETZ",
    "uuid": "UUID",
    "jsonb": "JSONB",
    "json": "JSON",
    "bytea": "BYTEA",
    "USER-DEFINED": None,
}


def _pg_type(data_type: str, udt_name: str) -> str:
    mapped = _PG_TYPE_MAP.get(data_type)
    if mapped is not None:
        return mapped
    if data_type == "USER-DEFINED":
        return udt_name.upper()
    return data_type.upper()


def get_schema_for_prompt(
    schema_name: str = "public",
    table_filter: Optional[List[str]] = None,
) -> str:
    """
    Return CREATE TABLE DDL for PostgreSQL tables suitable for LLM prompts.

    Operational tables (tenants, conversations, document_chunks, checkpoint_*)
    are excluded by default unless explicitly requested via table_filter.

    Args:
        schema_name: PostgreSQL schema (default "public").
        table_filter: If set, only include these table names (case-sensitive).

    Returns:
        DDL string with CREATE TABLE statements and foreign-key comments.
    """
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            col_rows = _fetch_columns(cur, schema_name, table_filter)
            pk_set = _fetch_primary_keys(cur, schema_name, table_filter)
            fk_rows = _fetch_foreign_keys(cur, schema_name, table_filter)
    finally:
        conn.close()

    col_rows = _exclude_operational(col_rows, 1, table_filter)
    fk_rows = _exclude_operational(fk_rows, 0, table_filter)

    fk_map: dict[tuple[str, str], str] = {}
    for src_table, src_col, ref_table, ref_col in fk_rows:
        fk_map[(src_table, src_col)] = f"{ref_table}({ref_col})"

    blocks: list[str] = []

    for (tbl_schema, tbl_name), group in groupby(
        col_rows, key=lambda r: (r[0], r[1])
    ):
        col_defs: list[str] = []
        pk_cols: list[str] = []
        fk_clauses: list[str] = []

        for row in group:
            col_name = row[2]
            data_type = row[3]
            udt_name = row[4]
            is_nullable = row[5]
            col_default = row[6]

            parts = [f"  {col_name}", _pg_type(data_type, udt_name)]
            if is_nullable == "NO":
                parts.append("NOT NULL")
            if col_default and "nextval" not in str(col_default):
                parts.append(f"DEFAULT {col_default}")

            if (tbl_name, col_name) in pk_set:
                pk_cols.append(col_name)

            fk_target = fk_map.get((tbl_name, col_name))
            if fk_target:
                fk_clauses.append(
                    f"  FOREIGN KEY ({col_name}) REFERENCES {fk_target}"
                )

            col_defs.append(" ".join(parts))

        if pk_cols:
            col_defs.append(f"  PRIMARY KEY ({', '.join(pk_cols)})")
        col_defs.extend(fk_clauses)

        ddl = f"CREATE TABLE {tbl_name} (\n"
        ddl += ",\n".join(col_defs)
        ddl += "\n);"
        blocks.append(ddl)

    return "\n\n".join(blocks) if blocks else ""
