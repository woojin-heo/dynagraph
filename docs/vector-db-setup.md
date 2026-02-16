# Vector DB Setup (PostgreSQL + pgvector)

This document describes how the vector database used for RAG (Retrieval Augmented Generation) was set up in this project.

## Overview

- **Database**: PostgreSQL with the [pgvector](https://github.com/pgvector/pgvector) extension for storing and querying embedding vectors.
- **Use case**: RAG over internal documents (e.g. `document_chunks` table) and optional SQL/vector search from the dynagraph agent.

## Prerequisites

- PostgreSQL installed (e.g. via Homebrew: `brew install postgresql@17`).
- pgvector extension available for your PostgreSQL version.  
  If you use Homebrew: `brew install pgvector`. Ensure the pgvector formula matches your Postgres major version (e.g. pgvector installs for Postgres 17/18; if you run Postgres 14, either upgrade Postgres or build pgvector from source for 14).

## 1. Enable the vector extension

In the database you will use (e.g. `postgres`), run:

```sql
CREATE EXTENSION vector;
```

You can do this from:

- **psql**: `psql postgres -c "CREATE EXTENSION vector;"`
- **pgAdmin 4**: Query Tool → run the SQL above.

Verify:

```sql
SELECT extname, extversion FROM pg_extension WHERE extname = 'vector';
```

## 2. Environment configuration

Create a `.env` file at the project root (or where the app loads it) and set the database URL:

```env
DATABASE_URL=postgresql://USERNAME:PASSWORD@HOST:PORT/DATABASE_NAME
```

Example for a local default instance:

```env
DATABASE_URL=postgresql://postgres:yourpassword@localhost:5432/postgres
```

Do not commit `.env`; add it to `.gitignore`.

## 3. RAG table schema

Create the table used for document chunks and their embeddings (e.g. in pgAdmin Query Tool or an init script):

```sql
CREATE TABLE IF NOT EXISTS document_chunks (
    id          BIGSERIAL PRIMARY KEY,
    content     TEXT NOT NULL,
    embedding   vector(1536) NOT NULL,
    source      TEXT,
    chunk_index INT DEFAULT 0,
    metadata    JSONB DEFAULT '{}',
    created_at  TIMESTAMPTZ DEFAULT now()
);

-- Optional: index for faster similarity search when the table has many rows
CREATE INDEX IF NOT EXISTS document_chunks_embedding_idx
ON document_chunks
USING ivfflat (embedding vector_cosine_ops)
WITH (lists = 100);
```

- **embedding**: `vector(1536)` matches OpenAI `text-embedding-3-small`. For another model, change the dimension (e.g. `vector(3072)` for `text-embedding-3-large`).
- **content**: Chunk text used for display and search context.
- **source**, **chunk_index**, **metadata**: Optional; useful for filtering and tracing back to original documents.

## 4. Verify connection and table

Use the project’s Jupyter notebook `db_connection_example.ipynb` at the repo root to:

- Load `DATABASE_URL` from `.env`.
- Connect with `psycopg2` and run `SELECT current_database(), current_user;`.
- Confirm the `vector` extension: `SELECT * FROM pg_extension WHERE extname = 'vector';`.
- List tables in `public` and confirm `document_chunks` exists; optionally inspect `information_schema.columns` for `document_chunks`.

Run the notebook from the project root so `.env` is found.

## 5. Python dependencies

For connection and (if needed) pgvector helpers from Python:

```bash
pip install psycopg2-binary
# Optional: pgvector Python package for type/helper support
pip install pgvector
```

These are listed in `backend/requirements.txt` where applicable.

## 6. Inserting documents (indexing)

To populate the RAG table so that `SEARCH_DOCUMENT` returns results, use the indexing module `backend.db.rag_indexing`. It chunks text, embeds with OpenAI `text-embedding-3-small`, and inserts into `document_chunks`.

**Requirements:** `DATABASE_URL` and `OPENAI_API_KEY` in `.env`.

### From Python

```python
from backend.db.rag_indexing import index_text, index_file

# Index a string (e.g. from a doc or API)
index_text("Your long document text here...", source="my_note")

# Index a file (path to .txt or any UTF-8 text file)
index_file("/path/to/document.txt")
index_file("/path/to/readme.md", source_label="readme")
```

### From the command line

From the project root (so `backend` is a package and `.env` is found):

```bash
python -m backend.db.rag_indexing path/to/file1.txt path/to/file2.txt
```

Each file is split into chunks (default 1000 chars, 200 overlap), embedded, and inserted. The number of chunks per file is printed.

### Lower-level: custom chunks

To insert a list of pre-built chunks with optional metadata:

```python
from backend.db.rag_indexing import insert_chunks

insert_chunks([
    {"content": "First chunk text", "source": "doc1", "metadata": {"page": 1}},
    {"content": "Second chunk text", "source": "doc1", "metadata": {"page": 2}},
])
```

After indexing, the agent’s `SEARCH_DOCUMENT` action will return these chunks when users ask about the stored documents.

## Summary

| Step | Action |
|------|--------|
| 1 | Install PostgreSQL and pgvector (version-matched). |
| 2 | Run `CREATE EXTENSION vector;` in the target database. |
| 3 | Set `DATABASE_URL` in `.env`. |
| 4 | Create `document_chunks` (and optional ivfflat index). |
| 5 | Verify with `db_connection_example.ipynb` and/or pgAdmin. |
| 6 | Index documents with `backend.db.rag_indexing` (see section 6). |

After this, the dynagraph backend can connect to the same database, and `SEARCH_DOCUMENT` will return chunks from `document_chunks` for user queries.
