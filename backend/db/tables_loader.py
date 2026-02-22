"""
Load the table definition document (tables.yaml) for planner and SQL generation.
Returns a human-readable string for use in prompts.
"""
import os
import yaml


def get_available_tables() -> str:
    """
    Read the table definition document (tables.yaml) and return its contents
    as a formatted string. Used by the planner to know which tables exist
    in the SQL database.

    Returns:
        Formatted table list and descriptions, or empty string if missing/unreadable.
    """
    try:
        base = os.path.dirname(os.path.abspath(__file__))
        path = os.path.join(base, "tables.yaml")
        if not os.path.isfile(path):
            return ""
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        if not data:
            return ""
        lines = []
        if isinstance(data.get("overview"), str):
            lines.append(f"Overview: {data['overview']}")
        tables = data.get("tables")
        if isinstance(tables, dict):
            lines.append("Tables:")
            for name, desc in tables.items():
                desc_str = desc if isinstance(desc, str) else str(desc)
                lines.append(f"  - {name}: {desc_str}")
        return "\n".join(lines) if lines else ""
    except Exception:
        return ""
