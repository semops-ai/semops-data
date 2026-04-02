# ADR-0015: Ingestion Pipeline Architecture

> **Status:** Decided
> **Date:** 2026-04-01
> **Supersedes:** [ADR-0005](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/decisions/ADR-0005-ingestion-strategy-corpus-architecture.md) (content strategy — absorbed), [ADR-0008](./ADR-0008-unified-ingestion-and-retrieval.md) (pipeline unification — absorbed)
> **Builds On:** [ADR-0012](./ADR-0012-pattern-coherence-co-equal-aggregates.md) (Pattern + Coherence as co-equal aggregates), [ADR-0014](./ADR-0014-coherence-measurement-model.md) (Coherence Measurement Model)
> **Design Doc:** [DD-0001: Ingestion Pipeline Architecture](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md)
> **Related:** [ISSUE-101: Episode-Centric Provenance](./ISSUE-101-episode-centric-provenance.md), [ingestion-graph-edge-relationships analysis](../analysis/ingestion-graph-edge-relationships.md)

---

## Executive Summary

The ingestion pipeline serves three distinct consumers with conflicting requirements: RAG retrieval (quality), coherence measurement (deterministic stability), and agentic lineage (completeness). This ADR defines the pipeline architecture that serves all three by establishing a deterministic core with clearly separated enrichment layers, source-defined processing strategies, and agentic lineage as the telemetry layer that wraps everything.

This supersedes ADR-0005 (which defined *what* content goes *where*) and ADR-0008 (which unified the storage layers). Both decisions remain valid — this ADR absorbs them into a single pipeline architecture that also addresses *how* content is processed and *who* consumes the outputs.

---

## Context

### The Collision

Three initiatives have converged on the ingestion pipeline, each with different requirements:

| Consumer | What it needs | Key constraint |
|----------|--------------|----------------|
| **RAG Retrieval** | Chunks + embeddings + graph context for LLM inference | Quality — best possible retrieval, LLM enrichment welcome |
| **Coherence Measurement** | Deterministic outputs — same input always produces same graph | Stability — LLM outputs contaminate measurement |
| **Agentic Lineage** | Instrumented operations — episodes capture what happened and why | Completeness — every operation recorded with provenance |

These three consumers cannot be served by a single undifferentiated pipeline. The current pipeline (`ingest_from_source.py`) writes entities that serve all three purposes simultaneously, with no way to distinguish deterministic outputs from LLM-derived ones, and no instrumentation for lineage.

### What ADR-0005 and ADR-0008 Decided (Still Valid)

**ADR-0005** defined the content strategy:
- Four corpus types: `core_kb`, `published`, `deployment`, `research_*`
- Six-phase ingestion sequence (A through F)
- Content-type-to-corpus routing
- Inside-out ingestion ordering (self-model first)

**ADR-0008** unified the storage layers:
- Single embedding model (`text-embedding-3-small`, 1536d)
- Chunk-entity linking via `entity_id` FK on `document_chunk`
- Three stores: PostgreSQL (entities/episodes), Neo4j (graph), pgvector/Qdrant (vectors)

Both decisions are correct. What's missing is the pipeline architecture that connects them to the three consumers with clean separation.

### What the Current Pipeline Lacks

1. **No consumer separation** — same outputs serve RAG, measurement, and audit with no provenance distinction
2. **No source-defined processing** — chunking strategy and embedding model are hardcoded, not declared per source
3. **No deterministic/LLM boundary** — LLM-classified metadata and deterministic metadata are stored identically
4. **No entity recognition against manifest** — NER is ad-hoc LLM classification, not dictionary lookup against known concepts
5. **Coherence profiles are dead code** — declared on source configs but never enforced (correctly identified by )
6. **No lineage instrumentation** — operations are not wrapped in episodes

---

## Decision

Six decisions define the pipeline architecture. For full design detail including pipeline diagrams, stage specifications, YAML examples, and coherence measurement constraints, see [DD-0001](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md).

**D1: Three-consumer pipeline with deterministic core.** The pipeline produces two tagged output classes — deterministic and LLM-enriched. Consumers declare which class they read. Both output classes participate in coherence measurement, but differently: deterministic outputs are the foundation for structural measurement (Scopes 1-2), while LLM-enriched outputs serve relationship discovery, semantic comparison, and broader measurement scopes (Scopes 3-5). See [DD-0023](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0023-semantic-optimization-pipeline.md) for the scope-dependent measurement model.

**D2: Six pipeline stages.** Preprocessing → Entity Recognition → Chunking → Embedding → Relation Extraction → Graph Write. Each stage has a deterministic path; some stages optionally run LLM enrichment.

**D3: Source-defined processing strategies.** Source configs declare a `processing` block specifying chunking strategy, embedding, entity recognition method, relation extraction method, and LLM enrichment. The source config is a contract between content authors and the pipeline.

**D4: Scope-dependent measurement requirements.** Pipeline consistency requirements (same embedding model, same NER granularity, same chunking, same preprocessing) apply within a comparison scope, not universally. Core aggregate measurement (Scope 1) operates on deterministic outputs. Broader scopes incorporate LLM-enriched outputs with provenance-aware confidence weighting. All outputs carry `extraction_method` tags (explicit/derived/discovered) so downstream analytics know what kind of signal they are working with. See [DD-0023 Section 3](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0023-semantic-optimization-pipeline.md) for the full scope model.

**D5: Agentic lineage wraps everything.** Every pipeline operation is wrapped in an episode (ISSUE-101). Lineage replaces ad-hoc provenance — `coherence_profile`, `classified_by` metadata, and scattered JSONB fields are all replaced by structured episode records.

**D6: Metadata contract simplification.** The current `MetadataContract` (required_fields, expected_fields, coherence_profile) is replaced by the `processing` block (D3), episode provenance (D5), and fitness functions. `coherence_profile` is removed (dead code, per ).

---

## Consequences

### Positive

- **Clean consumer separation** — coherence measurement gets a stable substrate; RAG gets rich retrieval; lineage gets complete telemetry
- **Source authors control processing** — content owners declare what their content needs
- **Manifest-driven entity recognition** — concept inventory is a deliberate curatorial decision, not an artifact of NER model behavior
- **Provenance on everything** — every edge, entity, and classification carries its extraction method
- **Coherence measurement validity** — by excluding LLM outputs, measurement is stable and reproducible
- **Lineage replaces ad-hoc provenance** — episodes are the single mechanism for "why did this happen"

### Negative

- **Source configs need updating** — all 20 source configs need a `processing` block added (one-time migration)
- **Two-tier complexity** — maintaining the deterministic/LLM boundary requires discipline
- **Manifest curation is HITL** — the concept manifest does not grow automatically

### Risks

- **Manifest completeness** — if the manifest is incomplete, deterministic entity recognition misses concepts. Mitigation: LLM candidate discovery surfaces gaps; HITL review keeps the manifest growing.
- **Over-engineering before implementation** — this describes a target state ahead of the codebase. Mitigation: implementation is incremental — source-defined processing first, then provenance tagging, then episodes.

---

## References

### Design
- [DD-0001: Ingestion Pipeline Architecture](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md) — full design detail

### Superseded
- [ADR-0005: Ingestion Strategy and Corpus Architecture](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/decisions/ADR-0005-ingestion-strategy-corpus-architecture.md)
- [ADR-0008: Unified Ingestion and Retrieval Pipeline](./ADR-0008-unified-ingestion-and-retrieval.md)

### Foundation
- [ADR-0012: Pattern + Coherence as Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md)
- [ADR-0014: Coherence Measurement Model](./ADR-0014-coherence-measurement-model.md)

### Analysis
- [ingestion-graph-edge-relationships.md](../analysis/ingestion-graph-edge-relationships.md) — deterministic vs. LLM pipeline analysis
- [ISSUE-101: Episode-Centric Provenance](./ISSUE-101-episode-centric-provenance.md) �� agentic lineage design
