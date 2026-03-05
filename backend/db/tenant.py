"""
Multi-tenant support: tenants and conversations tables.

Tables:
  tenants       – id (uuid PK), name (unique), created_at
  conversations – id (uuid PK), tenant_id (FK), title, created_at, updated_at

Provides CRUD helpers and an ensure_tables() bootstrap that creates both tables
idempotently (IF NOT EXISTS).
"""
import uuid
from datetime import datetime, timezone
from typing import Optional, List, Dict, Any

from .connection import get_connection


def ensure_tables() -> None:
    """Create tenants and conversations tables if they don't already exist."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS tenants (
                    id         TEXT PRIMARY KEY,
                    name       TEXT NOT NULL UNIQUE,
                    created_at TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS conversations (
                    id          TEXT PRIMARY KEY,
                    tenant_id   TEXT NOT NULL REFERENCES tenants(id) ON DELETE CASCADE,
                    title       TEXT NOT NULL DEFAULT '',
                    created_at  TIMESTAMPTZ NOT NULL DEFAULT now(),
                    updated_at  TIMESTAMPTZ NOT NULL DEFAULT now()
                );
            """)
            cur.execute("""
                CREATE INDEX IF NOT EXISTS idx_conversations_tenant
                ON conversations(tenant_id, updated_at DESC);
            """)
        conn.commit()
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Tenant CRUD
# ---------------------------------------------------------------------------

def create_tenant(name: str, tenant_id: Optional[str] = None) -> Dict[str, Any]:
    """Insert a new tenant. Returns {"id", "name", "created_at"}."""
    tid = tenant_id or str(uuid.uuid4())
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "INSERT INTO tenants (id, name) VALUES (%s, %s) RETURNING id, name, created_at",
                (tid, name),
            )
            row = cur.fetchone()
        conn.commit()
        return {"id": row[0], "name": row[1], "created_at": row[2].isoformat()}
    finally:
        conn.close()


def get_tenant(tenant_id: str) -> Optional[Dict[str, Any]]:
    """Return tenant dict or None."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, created_at FROM tenants WHERE id = %s", (tenant_id,))
            row = cur.fetchone()
        if not row:
            return None
        return {"id": row[0], "name": row[1], "created_at": row[2].isoformat()}
    finally:
        conn.close()


def list_tenants() -> List[Dict[str, Any]]:
    """Return all tenants ordered by created_at."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT id, name, created_at FROM tenants ORDER BY created_at")
            rows = cur.fetchall()
        return [{"id": r[0], "name": r[1], "created_at": r[2].isoformat()} for r in rows]
    finally:
        conn.close()


def delete_tenant(tenant_id: str) -> bool:
    """Delete tenant and cascade conversations. Returns True if deleted."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM tenants WHERE id = %s", (tenant_id,))
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


# ---------------------------------------------------------------------------
# Conversation CRUD (scoped to tenant)
# ---------------------------------------------------------------------------

def create_conversation(tenant_id: str, conversation_id: Optional[str] = None,
                        title: str = "") -> Dict[str, Any]:
    """Insert a new conversation for a tenant."""
    cid = conversation_id or str(uuid.uuid4())
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """INSERT INTO conversations (id, tenant_id, title)
                   VALUES (%s, %s, %s)
                   RETURNING id, tenant_id, title, created_at, updated_at""",
                (cid, tenant_id, title),
            )
            row = cur.fetchone()
        conn.commit()
        return {
            "id": row[0], "tenant_id": row[1], "title": row[2],
            "created_at": row[3].isoformat(), "updated_at": row[4].isoformat(),
        }
    finally:
        conn.close()


def get_conversation_meta(conversation_id: str, tenant_id: str) -> Optional[Dict[str, Any]]:
    """Return conversation metadata if it belongs to the tenant."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, tenant_id, title, created_at, updated_at
                   FROM conversations WHERE id = %s AND tenant_id = %s""",
                (conversation_id, tenant_id),
            )
            row = cur.fetchone()
        if not row:
            return None
        return {
            "id": row[0], "tenant_id": row[1], "title": row[2],
            "created_at": row[3].isoformat(), "updated_at": row[4].isoformat(),
        }
    finally:
        conn.close()


def list_conversations(tenant_id: str) -> List[Dict[str, Any]]:
    """Return conversations for a tenant, most recent first."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """SELECT id, tenant_id, title, created_at, updated_at
                   FROM conversations
                   WHERE tenant_id = %s
                   ORDER BY updated_at DESC""",
                (tenant_id,),
            )
            rows = cur.fetchall()
        return [
            {"id": r[0], "tenant_id": r[1], "title": r[2],
             "created_at": r[3].isoformat(), "updated_at": r[4].isoformat()}
            for r in rows
        ]
    finally:
        conn.close()


def update_conversation_title(conversation_id: str, tenant_id: str, title: str) -> bool:
    """Update conversation title. Returns True if found and updated."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                """UPDATE conversations SET title = %s, updated_at = now()
                   WHERE id = %s AND tenant_id = %s""",
                (title, conversation_id, tenant_id),
            )
            updated = cur.rowcount > 0
        conn.commit()
        return updated
    finally:
        conn.close()


def touch_conversation(conversation_id: str, tenant_id: str) -> None:
    """Bump updated_at to now (call on each new message)."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "UPDATE conversations SET updated_at = now() WHERE id = %s AND tenant_id = %s",
                (conversation_id, tenant_id),
            )
        conn.commit()
    finally:
        conn.close()


def delete_conversation(conversation_id: str, tenant_id: str) -> bool:
    """Delete a conversation. Returns True if deleted."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "DELETE FROM conversations WHERE id = %s AND tenant_id = %s",
                (conversation_id, tenant_id),
            )
            deleted = cur.rowcount > 0
        conn.commit()
        return deleted
    finally:
        conn.close()


def conversation_belongs_to_tenant(conversation_id: str, tenant_id: str) -> bool:
    """Check if a conversation belongs to the given tenant."""
    conn = get_connection()
    try:
        with conn.cursor() as cur:
            cur.execute(
                "SELECT 1 FROM conversations WHERE id = %s AND tenant_id = %s",
                (conversation_id, tenant_id),
            )
            return cur.fetchone() is not None
    finally:
        conn.close()
