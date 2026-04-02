# ADR-0008: Unified Ingestion and Retrieval Pipeline

> **Status:** Complete
> **Date:** 2026-02-02
> **Related Issue:** 
> **Related ADRs:** ADR-0005 (Ingestion Strategy and Corpus Architecture), ADR-0007 (Concept/Content Value Objects)
> **Design Doc:** [DD-0001](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md)

---

## Executive Summary

The knowledge base has three storage layers (entity, chunk, graph) served by three independent pipelines with different source configs, embedding models, and no cross-references. This ADR unifies them into a single ingestion pipeline that populates all layers from the same source configuration, using a single embedding model, with foreign key links between layers.

---

## Context

### Current State (as of 2026-02-02)

| Store | Pipeline | Embedding | Count |
|-------|----------|-----------|-------|
| Entity (Supabase pgvector) | `ingest_from_source.py` | OpenAI `text-embedding-3-small` (1536d) | 131 entities |
| Chunk (Supabase pgvector) | `chunk_markdown_docs.py` | Ollama `nomic-embed-text` (768d) | 103 chunks from 5 files |
| Graph (Neo4j) | Manual / unknown | N/A | 19 nodes, 14 edges |

**Problems:**
1. Chunks come from a deprecated repo (`project-ike-private`), not current sources
2. Graph has 19 nodes when 131+ entities exist with hundreds of detected edges
3. No FK between chunks and entities — hybrid retrieval impossible
4. Two embedding models with different dimensions — no vector space alignment
5. Chunk search and entity search return results from different content

### Cross-Repo Dependency

data-pr's research pipeline uses `text-embedding-3-small` (1536d) with Qdrant. Future lineage scoring requires the same embedding model across both repos. The chunk pipeline's `nomic-embed-text` is the outlier.

---

## Decision

1. **Single embedding model** — use OpenAI `text-embedding-3-small` (1536d) for all vectors across entity, chunk, and data-pr layers. Same vector space enables cross-layer similarity and cross-repo coherence scoring.
2. **Chunk-entity FK linking** — add `entity_id` column to `document_chunk` table, enabling hybrid query (entity search then chunk retrieval within top-N).
3. **Chunking integrated into entity ingestion** — extend `ingest_from_source.py` to chunk each file during ingestion (heading-based strategy with max token limit and overlap).
4. **Graph materialization from detected edges** — post-ingestion step writes `metadata.detected_edges` to Neo4j as a batch operation.
5. **Three query modes** — entity search (document-level), chunk search (passage-level), and hybrid search (entity then chunk two-stage). Query API extended with `/search/chunks`, `/search/hybrid`, `/graph/neighbors`.

---

## Consequences

### Positive

- Single source of truth for content: one ingestion populates all layers
- Hybrid retrieval enables precise passage citation in RAG responses
- Graph becomes usable for concept navigation and relationship discovery
- Same embedding space across all layers and repos enables coherence scoring

### Negative

- OpenAI API dependency for all embeddings (no local fallback)
- Ingestion becomes slower (chunking + more embeddings per file)
- More storage (chunk embeddings at 1536d vs current 768d)

### Risks

- Neo4j `detected_edges` reference `target_concept` IDs that may not correspond to actual entity IDs (concept slugs vs file-derived IDs). Need a concept resolution strategy.
- Re-embedding existing chunks with OpenAI changes their vectors — existing search results will differ

---

## References

- [ADR-0005: Ingestion Strategy and Corpus Architecture](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/decisions/ADR-0005-ingestion-strategy-corpus-architecture.md)
- 
- 
- 
- [USER_GUIDE.md](../USER_GUIDE.md) — Current query documentation
