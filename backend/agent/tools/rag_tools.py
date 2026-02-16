"""
RAG tool: search internal documents via pgvector (document_chunks table).
Requires DATABASE_URL and OPENAI_API_KEY in environment.
"""
import os
from typing import Optional

from dotenv import load_dotenv
from langchain_core.tools import tool
from langchain_openai import OpenAIEmbeddings

load_dotenv()

# Embedding model must match document_chunks.embedding dimension (1536 for text-embedding-3-small)
EMBEDDING_MODEL = "text-embedding-3-small"
VECTOR_DIM = 1536


def _get_connection():
    """Lazy import to avoid loading psycopg2 at module import when not using RAG."""
    try:
        from backend.db import get_connection
    except ImportError:
        from db import get_connection
    return get_connection()


@tool
def search_document(query: str, top_k: int = 5) -> str:
    """
    Search internal documents (RAG) by semantic similarity.
    Use this when the user asks about internal docs, knowledge base, or stored documents.

    Args:
        query: Natural language search query.
        top_k: Maximum number of document chunks to return (default 5).

    Returns:
        A string of relevant document chunks with source info, or an error message.
    """
    if not query or not query.strip():
        return "Error: query is empty."

    url = os.getenv("DATABASE_URL")
    if not url:
        return "Error: DATABASE_URL is not set. Configure the database connection."

    try:
        embeddings = OpenAIEmbeddings(model=EMBEDDING_MODEL)
        query_embedding = embeddings.embed_query(query.strip())
    except Exception as e:
        return f"Error generating embedding: {e}"

    # Format vector for PostgreSQL: '[0.1, 0.2, ...]'
    vector_str = "[" + ",".join(str(x) for x in query_embedding) + "]"

    try:
        conn = _get_connection()
        conn.autocommit = True
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT content, source, chunk_index, metadata
                FROM document_chunks
                ORDER BY embedding <=> %s::vector
                LIMIT %s;
                """,
                (vector_str, top_k),
            )
            rows = cur.fetchall()
        conn.close()
    except Exception as e:
        return f"Error querying documents: {e}"

    if not rows:
        return "No matching documents found."

    parts = []
    for i, (content, source, chunk_index, metadata) in enumerate(rows, 1):
        source_label = source or "unknown"
        meta = metadata if isinstance(metadata, dict) else {}
        page = meta.get("page")
        attrs = f'source="{source_label}" chunk_index="{chunk_index}"'
        if page is not None:
            attrs += f' page="{page}"'
        parts.append(f'<Document {attrs}>\n{content}\n</Document>')
    return "\n\n-----\n\n".join(parts)
