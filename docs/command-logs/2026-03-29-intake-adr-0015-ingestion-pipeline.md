# Intake Evaluation: ADR-0015 — Ingestion Pipeline Architecture

> **Date:** 2026-03-29
> **Input:** [ADR-0015: Ingestion Pipeline Architecture](../decisions/ADR-0015-ingestion-pipeline-architecture.md)
> **Evaluator:** /intake (agent)
> **Design Doc:** [ingestion-pipeline-design.md](../analysis/ingestion-pipeline-design.md)

## Tier Assessment

**Result: Tier 3 (Project)**

Promotion trigger check:
- Spans 3+ repos? **Yes** — semops-data (pipeline, schema, MCP), semops-orchestrator (ADR-0005 superseded, pattern docs, registry), plus all repos that produce source content.
- Requires ADR? **Yes** — ADR-0015 drafted. Supersedes ADR-0005 and ADR-0008.
- Has blocking dependencies? **Yes** — source config schema change (P1) blocks all downstream work.
- Needs sequenced execution? **Yes** — 7-step implementation sequence with hard dependencies.

This is a cross-cutting architectural change that affects the ingestion pipeline, coherence measurement, and agentic lineage simultaneously.

---

## Territory Map

### Three Consumer Goals

ADR-0015 identifies three distinct consumers of the ingestion pipeline, each with a pattern cluster.

#### Consumer 1: Ingestion Pipeline (shared substrate)

| Layer | Artifact | Status |
|-------|----------|--------|
| **Pattern table** | `semantic-ingestion` (1p) | REGISTERED — 9 capabilities, 5 repos |
| **Pattern edges** | `semantic-ingestion --[extends]--> etl` | PRESENT |
| **Pattern edges** | `semantic-ingestion --[extends]--> medallion-architecture` | PRESENT |
| **Pattern edges** | `semantic-ingestion --[extends]--> vector-embedding` | PRESENT |
| **Capability** | `ingestion-pipeline` | REGISTERED — 7 patterns, 1 repo |
| **Capability** | `corpus-collection` | REGISTERED — 3 patterns, 1 repo |
| **Capability** | `corpus-meta-analysis` | REGISTERED — 3 patterns, 1 repo |
| **Capability** | `structured-extraction` | REGISTERED — 3 patterns, 1 repo |
| **Source config schema** | `scripts/source_config.py` | PRESENT — missing `processing` block |
| **Source configs** | `config/sources/*.yaml` (20 files) | PRESENT — no processing declarations |
| **Pipeline scripts** | `ingest_from_source.py`, `generate_embeddings.py` | PRESENT — hardcoded processing |
| **Pattern doc** | `semantic-ingestion.md` | **ABSENT** — no domain pattern doc |

#### Consumer 2: RAG Retrieval (quality-optimized inference)

| Layer | Artifact | Status |
|-------|----------|--------|
| **Pattern table** | `agentic-rag` (3p) | REGISTERED — 7 capabilities, 4 repos |
| **Pattern edges** | `agentic-rag --[extends]--> retrieval-augmented-generation` | PRESENT |
| **Pattern edges** | `chain-of-thought --[broader]--> agentic-rag` | PRESENT |
| **Pattern edges** | `react-reasoning --[broader]--> agentic-rag` | PRESENT |
| **Pattern edges** | `tree-of-thought --[broader]--> agentic-rag` | PRESENT |
| **Pattern table** | `raptor` (3p) | REGISTERED — 2 capabilities, 2 repos |
| **Capability** | `internal-knowledge-access` | REGISTERED — 4 patterns, 1 repo |
| **Capability** | `context-engineering` | REGISTERED — 2 patterns, 1 repo |
| **Capability** | `input-optimization` | REGISTERED — 3 patterns, 1 repo |
| **MCP tools** | `search_knowledge_base`, `search_chunks`, `search_patterns` | PRESENT |
| **Search module** | `scripts/search.py` | PRESENT |

#### Consumer 3: Coherence Measurement (deterministic stability)

| Layer | Artifact | Status |
|-------|----------|--------|
| **Pattern table** | `semantic-coherence` (1p) | REGISTERED — 2 capabilities, 4 repos |
| **Pattern edges** | `semantic-coherence --[extends]--> ddd` | PRESENT |
| **Pattern edges** | `semantic-coherence --[extends]--> skos` | PRESENT |
| **Pattern edges** | `governance-as-strategy --[related]--> semantic-coherence` | PRESENT |
| **Pattern table** | `governance-as-strategy` (1p) | REGISTERED — 4 capabilities, 3 repos |
| **Pattern edges** | `governance-as-strategy --[extends]--> data-management` | PRESENT |
| **Pattern table** | `data-management` (3p) | REGISTERED — narrower: `data-lineage`, `data-quality`, `metadata-management` |
| **Capability** | `coherence-scoring` | REGISTERED — 5 patterns, 2 repos |
| **Capability** | `architecture-audit` | REGISTERED — 4 patterns, 1 repo |
| **Capability** | `pattern-governance` | REGISTERED — 2 patterns, 1 repo |
| **ADR** | ADR-0014 (Coherence Measurement Model) | DRAFT |
| **Fitness functions** | `schemas/fitness-functions.sql` (12 functions) | PRESENT |

#### Consumer 4: Agentic Lineage (telemetry layer)

| Layer | Artifact | Status |
|-------|----------|--------|
| **Pattern table** | `agentic-lineage` (1p) | REGISTERED — 1 capability, 2 repos |
| **Pattern edges** | `agentic-lineage --[extends]--> data-lineage` | PRESENT |
| **Pattern edges** | `agentic-lineage --[extends]--> episode-provenance` | PRESENT |
| **Pattern edges** | `agentic-lineage --[extends]--> open-lineage` | PRESENT |
| **Pattern edges** | `governance-as-strategy --[related]--> agentic-lineage` | PRESENT |
| **Pattern table** | `episode-provenance` (3p) | REGISTERED — 1 capability, 2 repos |
| **Pattern table** | `open-lineage` (3p) | REGISTERED — 1 capability, 2 repos |
| **Pattern table** | `provenance-first-design` (1p) | REGISTERED — 1 capability, 2 repos |
| **Capability** | `agentic-lineage` | REGISTERED — 6 patterns, 2 repos |
| **Design doc** | ISSUE-101-episode-centric-provenance.md | PRESENT |
| **Episode schema** | `schemas/migrations/001_episode_provenance.sql` | PRESENT — not deployed |
| **Episode code** | `scripts/lineage/` | PRESENT — not integrated |

### Cross-Consumer Connective Tissue

| Layer | Artifact | Status |
|-------|----------|--------|
| **Pattern** | `governance-as-strategy` | Hub — relates to both `semantic-coherence` and `agentic-lineage`, extends `data-management` |
| **Pattern** | `data-management` (3p) | Foundation — narrower: `data-lineage`, `data-quality`, `metadata-management` |
| **Pattern** | `explicit-architecture` (1p) | Resolution layer — 11 capabilities, 4 repos, 23 content entities |

---

## Delta: What ADR-0015 Changes

### New Architectural Concepts (not yet in domain model)

| # | Concept | Impact |
|---|---------|--------|
| 1 | **Deterministic/LLM output boundary** | Hard constraint — coherence measurement reads only deterministic outputs. Not encoded in any pattern definition or capability. |
| 2 | **Source-defined processing contracts** | Source configs declare chunking, embedding, entity recognition, relation extraction strategies. Currently hardcoded in pipeline scripts. |
| 3 | **Manifest-driven entity recognition** | Dictionary lookup against UL + pattern registry + capability registry replaces statistical NER. Inverts the traditional ingestion pipeline. |
| 4 | **Concept discovery queue (HITL)** | Separate slow process for growing the manifest. LLM surfaces candidates, human approves before manifest entry. |
| 5 | **`extraction_method` provenance tag** | Every output carries `deterministic` or `llm` tag. Consumers filter by trust level. |

### Extensions to Existing Patterns

| # | Pattern | What Changes | Direction |
|---|---------|-------------|-----------|
| 6 | `semantic-ingestion` | Definition should include deterministic/LLM output separation | Extends existing — definition update |
| 7 | `data-contracts` | Source `processing` block is a data contract between content authors and pipeline | Applies existing — new instance |
| 8 | `provenance-first-design` | `extraction_method` tag is provenance at the pipeline output level | Applies existing — new instance |
| 9 | `metadata-management` | Manifest as controlled vocabulary is metadata management | Applies existing — new instance |

### Fills Gaps

| # | Gap | How Filled |
|---|-----|-----------|
| 10 | `coherence_profile` on source configs is dead code | Removed — replaced by pipeline version tracking on episodes |
| 11 | LLM classification metadata mixed into entity JSONB with no provenance | Separated — LLM outputs tagged with `extraction_method: llm`, moved to episodes |
| 12 | No way to selectively re-ingest a corpus segment | Source-defined processing enables targeted re-ingestion by source |
| 13 | ADR-0005 and ADR-0008 are separate decisions covering overlapping territory | Superseded — ADR-0015 absorbs both into unified pipeline architecture |

---

## Capability-Pattern Coverage

### Existing Capabilities Affected

| Capability | Domain | Patterns | Current State | ADR-0015 Impact |
|-----------|--------|----------|---------------|-----------------|
| `ingestion-pipeline` | core | 7 | 1 repo (semops-data) | **Primary target** — source-defined processing, extraction method tagging |
| `coherence-scoring` | core | 5 | 2 repos | **Consumer** — reads deterministic layer only |
| `agentic-lineage` | core | 6 | 2 repos | **Wrapper** — episodes instrument all pipeline operations |
| `internal-knowledge-access` | core | 4 | 1 repo | **Consumer** — RAG reads both layers |
| `corpus-collection` | core | 3 | 1 repo | **Affected** — source processing contracts change collection behavior |
| `corpus-meta-analysis` | core | 3 | 1 repo | **Affected** — extraction method tags enable corpus quality analysis |
| `structured-extraction` | core | 3 | 1 repo | **Affected** — manifest-based entity recognition changes extraction |
| `context-engineering` | core | 2 | 1 repo | Low pattern coverage — may need attribution from `raptor`, `agentic-rag` |
| `pattern-governance` | core | 2 | 1 repo | Low pattern coverage — governance is central to coherence but under-attributed |

### Capability Gaps (new capabilities implied by ADR-0015)

| # | Proposed Capability | What It Covers | Patterns |
|---|-------------------|---------------|----------|
| 1 | **`manifest-curation`** | HITL concept discovery process — growing the controlled vocabulary | `metadata-management`, `semantic-ingestion` |
| 2 | **`source-config-management`** | Declaring and maintaining source processing contracts | `data-contracts`, `semantic-ingestion` |
| 3 | **`pipeline-provenance`** | Tagging outputs with extraction method and pipeline version | `provenance-first-design`, `episode-provenance` |

### Under-Decomposed Capability

`agentic-lineage` capability has 6 pattern links but is registered as a single capability. It spans:
- Episode tracking (recording operations)
- Provenance audit (querying decision context)
- Decision context capture (why the agent chose this action)

Consider decomposition into sub-capabilities if implementation reveals distinct delivery boundaries.

---

## Recommended Actions

### Immediate (ADR-0015 scope)

1. **Finalize ADR-0015** — review draft, address open questions, move to In Progress
2. **Update `semantic-ingestion` pattern definition** — add deterministic/LLM output boundary to definition
3. **Create domain pattern doc** `semantic-ingestion.md` — currently absent
4. **Create GitHub issue** for ADR-0015 implementation tracking

### Implementation sequence (from ADR-0015)

| Priority | Action | Patterns Exercised |
|----------|--------|-------------------|
| P1 | Add `processing` block to source config schema | `semantic-ingestion`, `data-contracts` |
| P2 | Tag existing outputs with `extraction_method` | `provenance-first-design`, `edge-predicates` |
| P3 | Manifest-based entity recognition | `semantic-ingestion`, `metadata-management` |
| P4 | Source-defined chunking strategies | `semantic-ingestion`, `data-contracts` |
| P5 | Episode instrumentation | `agentic-lineage`, `episode-provenance`, `open-lineage` |
| P6 | Coherence measurement on deterministic layer | `semantic-coherence`, `data-management` |
| P7 | Dead code removal (`coherence_profile`, ad-hoc provenance) | cleanup |
| P8 | Reclassify `governance-as-strategy` from pattern to concept entity | `semantic-object-pattern` |

### Pattern audit finding

**`governance-as-strategy` fails the three-property test** (fits a domain, implies capabilities, recognizable fit). It's a concept/principle that describes how `data-management` disciplines are applied strategically, not a pattern that independently implies capabilities. Was incorrectly promoted from concept to pattern in . Reclassification tracked in . See design doc Section 8 for full audit.

### Capability registration (after implementation starts)

- Register `manifest-curation` capability
- Register `source-config-management` capability
- Register `pipeline-provenance` capability
- Evaluate `agentic-lineage` decomposition

### Out of scope (defer)

- Qdrant collection routing (infrastructure, decided when needed)
- Research corpus (Phase F) — remains in PROJECT-24
- Project restructuring (P35 and related) — separate concern

---

## Evaluation Result

| Field | Value |
|-------|-------|
| **Tier** | 3 (Project) |
| **Classification** | Architectural refactor — pipeline architecture, consumer separation |
| **Complexity** | High — three consumers, 7-step implementation sequence, supersedes 2 ADRs |
| **Dependencies** | ADR-0014 (coherence model), ISSUE-101 (episode design), source config schema |
| **Patterns touched** | 16 (4 primary: `semantic-ingestion`, `semantic-coherence`, `agentic-lineage`, `agentic-rag`) |
| **Capabilities affected** | 9 existing + 3 proposed new |
| **Label applied** | `intake:evaluated` |
