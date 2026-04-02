"""
SemOps Query API — Corpus-aware semantic search.

Local FastAPI endpoint for agents to query the knowledge base
with corpus filtering and semantic similarity search.

Usage:
    uvicorn api.query:app --port 8101 --reload

Endpoints:
    POST /search          Entity-level semantic search
    POST /search/chunks   Chunk-level semantic search (passage retrieval)
    POST /search/hybrid   Entity search → chunk retrieval within top entities
    GET  /graph/neighbors  Graph neighbors for an entity
    GET  /corpora         List available corpora with entity counts
    GET  /health          Health check
"""

from __future__ import annotations

import os
import sys
from contextlib import asynccontextmanager
from pathlib import Path

import psycopg
from fastapi import FastAPI, HTTPException
from openai import OpenAI
from pydantic import BaseModel, Field

# Shared utilities (after sys.path setup)
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))
import graph_queries as _graph  # noqa: E402
from db_utils import get_db_connection, load_env  # noqa: E402

import search as _search  # noqa: E402

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def _get_query_embedding(query: str) -> list[float]:
    """Generate embedding for a query string."""
    try:
        resp = openai_client.embeddings.create(
            model=_search.EMBEDDING_MODEL,
            input=query,
            dimensions=_search.EMBEDDING_DIMENSIONS,
        )
        return resp.data[0].embedding
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Embedding generation failed: {e}")


# ---------------------------------------------------------------------------
# Pydantic models
# ---------------------------------------------------------------------------


class SearchRequest(BaseModel):
    query: str = Field(..., description="Natural language search query")
    corpus: list[str] | None = Field(
        None,
        description="Corpus filter (e.g. ['core_kb', 'deployment'])",
    )
    content_type: list[str] | None = Field(
        None,
        description="Content type filter (e.g. ['concept', 'adr'])",
    )
    entity_type: list[str] | None = Field(
        None,
        description="Entity type filter (e.g. ['agent'], ['capability', 'content'])",
    )
    limit: int = Field(10, ge=1, le=100, description="Max results to return")


class SearchResult(BaseModel):
    id: str
    title: str | None
    corpus: str | None
    content_type: str | None
    summary: str | None
    similarity: float
    filespec: dict | None = None
    metadata: dict | None = None


class SearchResponse(BaseModel):
    query: str
    corpus_filter: list[str] | None
    content_type_filter: list[str] | None
    count: int
    results: list[SearchResult]


class ChunkResult(BaseModel):
    chunk_id: int
    entity_id: str | None
    source_file: str
    heading_hierarchy: list[str]
    content: str
    corpus: str | None
    content_type: str | None
    similarity: float
    chunk_index: int
    total_chunks: int


class ChunkSearchResponse(BaseModel):
    query: str
    corpus_filter: list[str] | None
    count: int
    results: list[ChunkResult]


class HybridResult(BaseModel):
    entity: SearchResult
    chunks: list[ChunkResult]


class HybridSearchResponse(BaseModel):
    query: str
    corpus_filter: list[str] | None
    entity_count: int
    chunk_count: int
    results: list[HybridResult]


class GraphNeighbor(BaseModel):
    id: str
    label: str
    relationship: str
    direction: str  # "outgoing" or "incoming"
    strength: float | None = None


class CorpusInfo(BaseModel):
    corpus: str
    count: int


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------

load_env()

openai_client: OpenAI | None = None
db_conn: psycopg.Connection | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global openai_client, db_conn
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY not set")
    openai_client = OpenAI(api_key=api_key)
    db_conn = get_db_connection(autocommit=True)
    yield
    if db_conn:
        db_conn.close()


app = FastAPI(
    title="SemOps Query API",
    description="Corpus-aware semantic search for Project SemOps knowledge base",
    version="0.3.0",
    lifespan=lifespan,
)


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/corpora", response_model=list[CorpusInfo])
def corpora():
    """List available corpora with entity counts."""
    results = _search.list_corpora(db_conn)
    return [CorpusInfo(corpus=r["corpus"], count=r["count"]) for r in results]


@app.post("/search", response_model=SearchResponse)
def search(req: SearchRequest):
    """Entity-level semantic search with optional corpus and content_type filters."""
    query_embedding = _get_query_embedding(req.query)

    try:
        results = _search.search_entities(
            db_conn,
            query_embedding,
            limit=req.limit,
            corpus=req.corpus,
            content_type=req.content_type,
            entity_type=req.entity_type,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Search query failed: {e}")

    return SearchResponse(
        query=req.query,
        corpus_filter=req.corpus,
        content_type_filter=req.content_type,
        count=len(results),
        results=[
            SearchResult(
                id=r["id"],
                title=r["title"],
                corpus=r["corpus"],
                content_type=r["content_type"],
                summary=r["summary"],
                similarity=r["similarity"],
                filespec=r["filespec"],
                metadata=r["metadata"],
            )
            for r in results
        ],
    )


@app.post("/search/chunks", response_model=ChunkSearchResponse)
def search_chunks(req: SearchRequest):
    """Chunk-level semantic search — returns passage-level results with heading context."""
    query_embedding = _get_query_embedding(req.query)

    try:
        results = _search.search_chunks(
            db_conn,
            query_embedding,
            limit=req.limit,
            corpus=req.corpus,
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Chunk search failed: {e}")

    return ChunkSearchResponse(
        query=req.query,
        corpus_filter=req.corpus,
        count=len(results),
        results=[ChunkResult(**r) for r in results],
    )


@app.post("/search/hybrid", response_model=HybridSearchResponse)
def search_hybrid(req: SearchRequest):
    """
    Hybrid search: find top entities, then retrieve best chunks within each.

    Two-stage retrieval:
    1. Entity search to find top-N relevant documents
    2. For each entity, retrieve top chunks by similarity
    """
    query_embedding = _get_query_embedding(req.query)

    results = _search.search_hybrid(
        db_conn,
        query_embedding,
        entity_limit=min(req.limit, 10),
        corpus=req.corpus,
    )

    total_chunks = sum(len(r["chunks"]) for r in results)

    return HybridSearchResponse(
        query=req.query,
        corpus_filter=req.corpus,
        entity_count=len(results),
        chunk_count=total_chunks,
        results=[
            HybridResult(
                entity=SearchResult(
                    id=r["entity"]["id"],
                    title=r["entity"]["title"],
                    corpus=r["entity"]["corpus"],
                    content_type=r["entity"]["content_type"],
                    summary=r["entity"]["summary"],
                    similarity=r["entity"]["similarity"],
                    filespec=r["entity"]["filespec"],
                    metadata=r["entity"]["metadata"],
                ),
                chunks=[ChunkResult(**c) for c in r["chunks"]],
            )
            for r in results
        ],
    )


@app.get("/graph/neighbors/{entity_id}", response_model=list[GraphNeighbor])
def graph_neighbors(entity_id: str):
    """Get graph neighbors for an entity from Neo4j."""
    try:
        results = _graph.get_neighbors(entity_id)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"Neo4j query failed: {e}")

    return [GraphNeighbor(**r) for r in results]
