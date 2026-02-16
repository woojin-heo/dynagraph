"""
Insert documents into the RAG table (document_chunks).
Chunks text, embeds with OpenAI, and INSERTs into PostgreSQL pgvector.

Usage:
  - From code: index_text("Some long text...", source="my_doc")
  - From code: index_file("/path/to/file.txt")
  - From CLI: python -m backend.db.rag_indexing /path/to/file.txt
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

load_dotenv()

EMBEDDING_MODEL = "text-embedding-3-small"
CHUNK_SIZE = 1000
CHUNK_OVERLAP = 200


def _get_connection():
    try:
        from backend.db import get_connection
    except ImportError:
        from db import get_connection
    return get_connection()


def _chunk_text(text: str) -> list[str]:
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        length_function=len,
    )
    return splitter.split_text(text.strip())


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

    conn = _get_connection()
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


def main():
    """CLI: python -m backend.db.rag_indexing <file1> [file2 ...]"""
    if len(sys.argv) < 2:
        print("Usage: python -m backend.db.rag_indexing <file1> [file2 ...]")
        sys.exit(1)
    if not os.getenv("DATABASE_URL"):
        print("Error: DATABASE_URL not set")
        sys.exit(1)
    if not os.getenv("OPENAI_API_KEY"):
        print("Error: OPENAI_API_KEY not set")
        sys.exit(1)

    total = 0
    for p in sys.argv[1:]:
        try:
            n = index_file(p)
            total += n
            print(f"{p}: inserted {n} chunks")
        except Exception as e:
            print(f"{p}: error - {e}")
    print(f"Total inserted: {total}")
