"""
Insert, list, and delete documents in the RAG table (document_chunks).
Chunks text, embeds with OpenAI, and INSERTs into PostgreSQL pgvector.

Usage:
  - From code: index_text("Some long text...", source="my_doc")
  - From code: index_file("/path/to/file.txt"), index_pdf("/path/to/doc.pdf")
  - From code: list_documents(), delete_document(source)
  - From CLI: python -m backend.db.rag_indexing --list
  - From CLI: python -m backend.db.rag_indexing --delete <source> [source2 ...]
  - From CLI: python -m backend.db.rag_indexing <file1> [file2 ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from backend.db import get_connection

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def insert_chunks(
    chunks: list[dict],
    *,
    embedding_model: str = EMBEDDING_MODEL,
) -> int:
    """
    Embed and insert a list of chunks into document_chunks.
    Each chunk dict should have: content (str), and optionally source (str), metadata (dict).

    Returns the number of rows inserted.
    """
    if not chunks:
        return 0

    conn = get_connection()
    conn.autocommit = True
    embeddings = OpenAIEmbeddings(model=embedding_model)

    inserted = 0
    with conn.cursor() as cur:
        for i, c in enumerate(chunks):
            content = c.get("content") or c.get("text")
            if not content:
                continue
            source = c.get("source") or ""
            metadata = c.get("metadata") or {}
            meta_json = json.dumps(metadata)

            vec = embeddings.embed_query(content)
            vector_str = "[" + ",".join(str(x) for x in vec) + "]"

            cur.execute(
                """
                INSERT INTO document_chunks (content, embedding, source, chunk_index, metadata)
                VALUES (%s, %s::vector, %s, %s, %s::jsonb);
                """,
                (content, vector_str, source, i, meta_json),
            )
            inserted += 1
    conn.close()
    return inserted


def index_text(
    text: str,
    source: str = "unknown",
    metadata: dict | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Split text into chunks, embed, and insert into document_chunks.
    Returns the number of chunks inserted.
    """
    if not text or not text.strip():
        return 0
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    texts = splitter.split_text(text.strip())
    chunks = [
        {"content": t, "source": source, "metadata": metadata or {}}
        for t in texts
    ]
    return insert_chunks(chunks)


def index_pdf(
    path: str | Path,
    source_label: str | None = None,
    *,
    chunk_size: int = CHUNK_SIZE,
    chunk_overlap: int = CHUNK_OVERLAP,
) -> int:
    """
    Load a PDF, split into chunks per page (preserving page number in metadata), embed and insert.
    Each chunk's metadata will include "page" (1-based). Returns the number of chunks inserted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    try:
        from langchain_community.document_loaders import PyPDFLoader
    except ImportError:
        raise ImportError("Install pypdf and langchain-community: pip install pypdf langchain-community")

    loader = PyPDFLoader(str(path))
    docs = loader.load()
    if not docs:
        return 0

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        length_function=len,
    )
    source = source_label or path.name
    chunks: list[dict] = []

    for doc in docs:
        # PyPDFLoader often puts "page" in metadata (0-based); we store 1-based for display
        page_meta = doc.metadata if isinstance(doc.metadata, dict) else {}
        page_num = page_meta.get("page", 0)
        if isinstance(page_num, int):
            page_num = page_num + 1  # 1-based page number
        page_text = (doc.page_content or "").strip()
        if not page_text:
            continue
        page_chunks = splitter.split_text(page_text)
        for c in page_chunks:
            chunks.append({
                "content": c,
                "source": source,
                "metadata": {"page": page_num},
            })

    return insert_chunks(chunks)


def index_file(
    path: str | Path,
    source_label: str | None = None,
    metadata: dict | None = None,
    *,
    encoding: str = "utf-8",
) -> int:
    """
    Read a text file, split into chunks, embed, and insert.
    source_label defaults to the file path.
    Returns the number of chunks inserted.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(str(path))
    text = path.read_text(encoding=encoding)
    source = source_label or str(path)
    return index_text(text, source=source, metadata=metadata)


def list_documents() -> list[dict]:
    """
    Return list of uploaded documents (by source) with chunk count and latest created_at.
    Each dict has: source (str), chunk_count (int), created_at (datetime or None).
    """
    conn = get_connection()
    conn.autocommit = True
    out: list[dict] = []
    with conn.cursor() as cur:
        cur.execute(
            """
            SELECT source, COUNT(*) AS chunk_count, MAX(created_at) AS created_at
            FROM document_chunks
            GROUP BY source
            ORDER BY created_at DESC NULLS LAST, source;
            """
        )
        for row in cur.fetchall():
            source, chunk_count, created_at = row
            out.append({
                "source": source or "",
                "chunk_count": chunk_count,
                "created_at": created_at,
            })
    conn.close()
    return out


def delete_document(source: str) -> int:
    """
    Delete all chunks for the given source (e.g. file name or path).
    Returns the number of rows deleted.
    """
    if not source:
        return 0
    conn = get_connection()
    conn.autocommit = True
    with conn.cursor() as cur:
        cur.execute("DELETE FROM document_chunks WHERE source = %s;", (source,))
        deleted = cur.rowcount
    conn.close()
    return deleted


def main():
    """CLI: list/delete or index files.
    Usage:
      python -m backend.db.rag_indexing --list
      python -m backend.db.rag_indexing --delete <source> [source2 ...]
      python -m backend.db.rag_indexing <file1> [file2 ...]
    """
    if len(sys.argv) < 2:
        print("Usage: python -m backend.db.rag_indexing --list | --delete <source>... | <file1> [file2 ...]")
        sys.exit(1)
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL not set")
        sys.exit(1)

    args = sys.argv[1:]
    if args[0] == "--list":
        docs = list_documents()
        if not docs:
            print("No documents in RAG.")
            return
        for d in docs:
            created = d["created_at"].strftime("%Y-%m-%d %H:%M") if d.get("created_at") else "-"
            print(f"  {d['source']}\tchunks={d['chunk_count']}\tuploaded={created}")
        print(f"Total: {len(docs)} document(s)")
        return

    if args[0] == "--delete":
        if len(args) < 2:
            print("Usage: python -m backend.db.rag_indexing --delete <source> [source2 ...]")
            sys.exit(1)
        total = 0
        for source in args[1:]:
            n = delete_document(source)
            total += n
            print(f"{source}: deleted {n} chunks")
        print(f"Total deleted: {total} chunks")
        return

    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    total = 0
    for p in args:
        try:
            path = Path(p)
            if path.suffix.lower() == ".pdf":
                n = index_pdf(path)
            else:
                n = index_file(path)
            total += n
            print(f"{p}: inserted {n} chunks")
        except Exception as e:
            print(f"{p}: error - {e}")
    print(f"Total inserted: {total}")
