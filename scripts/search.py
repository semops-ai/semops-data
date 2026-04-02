"""
Shared semantic search functions for the SemOps knowledge base.

All search functions accept a database connection and pre-computed query embedding.
They return plain dicts. Consumers (API, MCP, CLI) handle embedding generation,
error handling, and response formatting.

Used by:
    - api/mcp_server.py (MCP tools for Claude Code agents)
    - api/query.py (FastAPI endpoints)
    - scripts/semantic_search.py (CLI diagnostic tool)
"""

from __future__ import annotations

import psycopg

# ---------------------------------------------------------------------------
# Constants (single source of truth for embedding config)
# ---------------------------------------------------------------------------

EMBEDDING_MODEL = "text-embedding-3-small"
EMBEDDING_DIMENSIONS = 1536


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _build_entity_where(
    *,
    corpus: list[str] | None = None,
    content_type: list[str] | None = None,
    lifecycle_stage: list[str] | None = None,
    entity_type: list[str] | None = None,
) -> tuple[str, list]:
    """Build WHERE clause and params for entity table searches."""
    conditions = ["embedding IS NOT NULL"]
    params: list = []

    if entity_type:
        placeholders = ", ".join(["%s"] * len(entity_type))
        conditions.append(f"entity_type IN ({placeholders})")
        params.extend(entity_type)

    if corpus:
        placeholders = ", ".join(["%s"] * len(corpus))
        conditions.append(f"metadata->>'corpus' IN ({placeholders})")
        params.extend(corpus)

    if content_type:
        placeholders = ", ".join(["%s"] * len(content_type))
        conditions.append(f"metadata->>'content_type' IN ({placeholders})")
        params.extend(content_type)

    if lifecycle_stage:
        placeholders = ", ".join(["%s"] * len(lifecycle_stage))
        conditions.append(f"metadata->>'lifecycle_stage' IN ({placeholders})")
        params.extend(lifecycle_stage)

    return " AND ".join(conditions), params


def _build_chunk_where(
    *,
    corpus: list[str] | None = None,
) -> tuple[str, list]:
    """Build WHERE clause and params for document_chunk table searches."""
    conditions = ["embedding IS NOT NULL"]
    params: list = []

    if corpus:
        placeholders = ", ".join(["%s"] * len(corpus))
        conditions.append(f"corpus IN ({placeholders})")
        params.extend(corpus)

    return " AND ".join(conditions), params


# ---------------------------------------------------------------------------
# Public search functions
# ---------------------------------------------------------------------------


def search_entities(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    limit: int = 10,
    corpus: list[str] | None = None,
    content_type: list[str] | None = None,
    lifecycle_stage: list[str] | None = None,
    entity_type: list[str] | None = None,
) -> list[dict]:
    """Entity-level semantic search.

    Returns dicts with: id, title, corpus, content_type, summary, similarity,
    uri, filespec, metadata, ownership, entity_type.
    """
    where_clause, where_params = _build_entity_where(
        corpus=corpus,
        content_type=content_type,
        lifecycle_stage=lifecycle_stage,
        entity_type=entity_type,
    )
    params: list = [query_embedding] + where_params + [query_embedding, limit]

    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT
            id, title,
            entity_type,
            metadata->>'corpus' as corpus,
            metadata->>'content_type' as content_type,
            metadata->>'summary' as summary,
            1 - (embedding <=> %s::vector) as similarity,
            filespec->>'uri' as uri,
            filespec,
            metadata,
            attribution->>'concept_ownership' as ownership,
            metadata->>'date_created' as date_created,
            metadata->>'date_updated' as date_updated
        FROM entity
        WHERE {where_clause}
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        params,
    )

    return [
        {
            "id": row[0],
            "title": row[1],
            "entity_type": row[2],
            "corpus": row[3],
            "content_type": row[4],
            "summary": row[5],
            "similarity": round(row[6], 4),
            "uri": row[7],
            "filespec": row[8],
            "metadata": row[9],
            "ownership": row[10],
            "date_created": row[11],
            "date_updated": row[12],
        }
        for row in cursor.fetchall()
    ]


def search_chunks(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    limit: int = 10,
    corpus: list[str] | None = None,
    content_max_chars: int | None = None,
) -> list[dict]:
    """Chunk-level semantic search.

    Returns dicts with: chunk_id, entity_id, source_file, heading_hierarchy,
    content, corpus, content_type, similarity, chunk_index, total_chunks.

    Args:
        content_max_chars: Truncate content to this many chars. None for full content.
    """
    where_clause, where_params = _build_chunk_where(corpus=corpus)
    params: list = [query_embedding] + where_params + [query_embedding, limit]

    content_expr = f"LEFT(dc.content, {int(content_max_chars)})" if content_max_chars else "dc.content"
    # Qualify unaliased column references from _build_chunk_where for the joined query
    qualified_where = where_clause.replace("embedding", "dc.embedding").replace("corpus", "dc.corpus")

    cursor = conn.cursor()
    cursor.execute(
        f"""
        SELECT
            dc.id, dc.entity_id, dc.source_file, dc.heading_hierarchy,
            {content_expr} as content,
            dc.corpus, dc.content_type,
            1 - (dc.embedding <=> %s::vector) as similarity,
            dc.chunk_index, dc.total_chunks,
            e.metadata->>'date_created' as date_created
        FROM document_chunk dc
        LEFT JOIN entity e ON dc.entity_id = e.id
        WHERE {qualified_where}
        ORDER BY dc.embedding <=> %s::vector
        LIMIT %s
        """,
        params,
    )

    return [
        {
            "chunk_id": row[0],
            "entity_id": row[1],
            "source_file": row[2],
            "heading_hierarchy": row[3] or [],
            "content": row[4],
            "corpus": row[5],
            "content_type": row[6],
            "similarity": round(row[7], 4),
            "chunk_index": row[8],
            "total_chunks": row[9],
            "date_created": row[10],
        }
        for row in cursor.fetchall()
    ]


def search_hybrid(
    conn: psycopg.Connection,
    query_embedding: list[float],
    *,
    entity_limit: int = 10,
    chunks_per_entity: int = 3,
    corpus: list[str] | None = None,
    content_max_chars: int | None = None,
) -> list[dict]:
    """Two-stage hybrid search: top entities, then best chunks within each.

    Returns list of dicts with: entity (dict), chunks (list[dict]).
    """
    entities = search_entities(
        conn,
        query_embedding,
        limit=entity_limit,
        corpus=corpus,
    )

    content_expr = f"LEFT(content, {int(content_max_chars)})" if content_max_chars else "content"

    cursor = conn.cursor()
    results = []
    for entity in entities:
        cursor.execute(
            f"""
            SELECT
                id, entity_id, source_file, heading_hierarchy,
                {content_expr} as content, corpus, content_type,
                1 - (embedding <=> %s::vector) as similarity,
                chunk_index, total_chunks
            FROM document_chunk
            WHERE entity_id = %s AND embedding IS NOT NULL
            ORDER BY embedding <=> %s::vector
            LIMIT %s
            """,
            [query_embedding, entity["id"], query_embedding, chunks_per_entity],
        )

        chunks = [
            {
                "chunk_id": row[0],
                "entity_id": row[1],
                "source_file": row[2],
                "heading_hierarchy": row[3] or [],
                "content": row[4],
                "corpus": row[5],
                "content_type": row[6],
                "similarity": round(row[7], 4),
                "chunk_index": row[8],
                "total_chunks": row[9],
            }
            for row in cursor.fetchall()
        ]

        results.append({"entity": entity, "chunks": chunks})

    return results


def list_corpora(conn: psycopg.Connection) -> list[dict]:
    """List available corpora with entity counts."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT metadata->>'corpus' as corpus, count(*) as cnt
        FROM entity
        WHERE metadata->>'corpus' IS NOT NULL
        GROUP BY corpus
        ORDER BY cnt DESC
        """
    )
    return [{"corpus": row[0], "count": row[1]} for row in cursor.fetchall()]
