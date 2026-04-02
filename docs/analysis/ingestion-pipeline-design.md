# Ingestion Pipeline Design Document

> **ADR:** [ADR-0015: Ingestion Pipeline Architecture](../decisions/ADR-0015-ingestion-pipeline-architecture.md)
> **Intake Log:** [2026-03-29-intake-adr-0015-ingestion-pipeline.md](../command-logs/2026-03-29-intake-adr-0015-ingestion-pipeline.md)
> **Date:** 2026-03-29
> **Status:** Draft — intake analysis complete, ready for review

---

## 1. Business Context

### Why This Matters

The ingestion pipeline is the entry point for all knowledge entering the SemOps system. Every downstream capability — agent reasoning, coherence measurement, pattern governance, published content — depends on what the pipeline produces and how it produces it.

Three strategic initiatives have converged on this pipeline, each with legitimate but conflicting requirements:

| Initiative | Business Goal | Pipeline Requirement |
|------------|--------------|---------------------|
| **RAG Retrieval** | Agents and LLMs can find and reason over organizational knowledge | Best possible retrieval quality — rich embeddings, LLM-enriched metadata, graph context |
| **Coherence Measurement** | Quantify how well reality matches intent (SC = (A×C×S)^(1/3)) | Deterministic stability — same input must always produce same output, or measurement is invalid |
| **Agentic Lineage** | Complete audit trail of what happened, why, and with what context | Every operation instrumented — nothing happens silently |

**The core tension:** RAG wants LLM enrichment everywhere (better retrieval). Coherence measurement needs LLM outputs excluded (measurement contamination). You cannot serve both from an undifferentiated pipeline.

### Business Consequences of Getting This Wrong

- **If coherence measurement uses LLM-derived data:** Measurements are non-reproducible. Same corpus re-measured tomorrow gives different results. SC scores become meaningless.
- **If RAG excludes LLM enrichment:** Retrieval quality drops. Agents get worse context. Downstream content quality suffers.
- **If lineage is absent:** No audit trail. "Why did the agent say this?" is unanswerable. Trust in the system degrades.

The pipeline must serve all three without compromise. This document defines how.

---

## 2. Pattern and Capability Mapping

### Intake Analysis: Patterns by Consumer

Each consumer goal maps to a distinct pattern cluster in the domain model. The patterns within each cluster have SKOS and adoption relationships that reveal the dependency structure.

#### Consumer 1: Ingestion Pipeline (the shared substrate)

| Pattern | Provenance | Role | SKOS Edges |
|---------|-----------|------|------------|
| **`semantic-ingestion`** | 1p | Core pipeline pattern — every byproduct becomes queryable | extends `etl`, extends `medallion-architecture`, extends `vector-embedding` |
| `etl` | 3p | Foundation — extract/transform/load | (adopted by semantic-ingestion) |
| `medallion-architecture` | 3p | Bronze/silver/gold layer model | (adopted by semantic-ingestion) |
| `vector-embedding` | 3p | Fixed embedding model, common vector space | (adopted by semantic-ingestion) |
| `edge-predicates` | 3p | Typed semantic relationships vocabulary | — |
| `dublin-core` | 3p | Metadata standard for entity attribution | — |

**Capabilities served:** `ingestion-pipeline` (7 patterns, 1 repo), `corpus-collection` (3 patterns, 1 repo), `corpus-meta-analysis` (3 patterns, 1 repo), `structured-extraction` (3 patterns, 1 repo)

**Key insight:** `semantic-ingestion` is defined as "every byproduct becomes a queryable knowledge artifact." This is the right framing — the pipeline doesn't just move data, it produces knowledge. But the current definition doesn't distinguish deterministic byproducts from LLM-derived ones. ADR-0015 adds that distinction.

#### Consumer 2: RAG Retrieval (quality-optimized inference)

| Pattern | Provenance | Role | SKOS Edges |
|---------|-----------|------|------------|
| **`agentic-rag`** | 3p | Autonomous agent retrieval strategies | extends `retrieval-augmented-generation` |
| `raptor` | 3p | Multi-level abstractive retrieval | — |
| `retrieval-augmented-generation` | 3p | Base RAG pattern | (extended by agentic-rag) |
| `chain-of-thought` | 3p | Reasoning pattern for retrieval planning | broader → agentic-rag |
| `react-reasoning` | 3p | Observe/reason/act for retrieval | broader → agentic-rag |
| `tree-of-thought` | 3p | Branching exploration for retrieval | broader → agentic-rag |

**Capabilities served:** `internal-knowledge-access` (4 patterns, 1 repo), `context-engineering` (2 patterns, 1 repo), `input-optimization` (3 patterns, 1 repo)

**Key insight:** RAG is entirely 3p patterns. SemOps doesn't innovate on retrieval — it adopts best practices. The 1p innovation is in what the retrieval *serves* (coherence, governance, lineage). This means the RAG layer should be treated as commodity infrastructure.

#### Consumer 3: Coherence Measurement (deterministic stability)

| Pattern | Provenance | Role | SKOS Edges |
|---------|-----------|------|------------|
| **`semantic-coherence`** | 1p | SC = (A×C×S)^(1/3) — the objective function | extends `ddd`, extends `semantic-optimization`, extends `skos` |
| `governance-as-strategy` | 1p | Governance as strategic capability, not compliance | extends `data-management`, related → `semantic-coherence`, related → `agentic-lineage` |
| `data-management` | 3p | Discipline hierarchy: lineage, quality, metadata, provenance | narrower → `data-lineage`, `data-quality`, `metadata-management` |
| `data-quality` | 3p | Operational correctness — SC generalizes this | broader → data-management |
| `skos` | 3p | Concept taxonomy standard | (adopted by semantic-coherence) |
| `explicit-architecture` | 1p | Queryable architecture graph — the resolution layer | — |

**Capabilities served:** `coherence-scoring` (5 patterns, 2 repos), `architecture-audit` (4 patterns, 1 repo), `pattern-governance` (2 patterns, 1 repo)

**Key insight:** Coherence measurement is where the 1p innovation lives. The pattern cluster reveals the dependency chain from ADR-0014: `governance-as-strategy` extends `data-management` (3p) and relates to both `semantic-coherence` and `agentic-lineage`. This is the "data management disciplines applied to semantic operations" thesis.

#### Consumer 4: Agentic Lineage (telemetry layer)

| Pattern | Provenance | Role | SKOS Edges |
|---------|-----------|------|------------|
| **`agentic-lineage`** | 1p | Lineage + agent decision context + trust provenance | extends `data-lineage`, extends `episode-provenance`, extends `open-lineage` |
| `episode-provenance` | 3p | Episodes as bounded activity sequences | (adopted by agentic-lineage) |
| `open-lineage` | 3p | Standard event model for pipeline observability | (adopted by agentic-lineage) |
| `data-lineage` | 3p | Flow and transformation tracking | broader → data-management |
| `prov-o` | 3p | W3C provenance ontology | (adopted by data-lineage) |
| `provenance-first-design` | 1p | 1P/2P/3P trust classification | extends prov-o |
| `derivative-work-lineage` | 1p | Content transformation chain tracking | — |

**Capabilities served:** `agentic-lineage` (6 patterns, 2 repos)

**Key insight:** Agentic lineage synthesizes three 3p standards (OpenLineage, Episode Provenance, Data Lineage) into a 1p innovation. The SKOS graph shows it's broader than data-lineage — it's lineage plus intent. This confirms it's the right layer to wrap the pipeline: it captures not just "what data moved where" but "why the agent chose this action."

### Cross-Consumer Pattern Dependencies

> **Note:** `governance-as-strategy` is a **concept** (principle/framing), not a pattern — it fails the three-property test (see [pattern audit](-pattern-audit-governance-as-strategy), ). The graph below shows the corrected structure with `governance-as-strategy` as a concept entity referenced by patterns, not a peer pattern.

```
governance-as-strategy (concept entity — strategic framing)
    ├── described_by ← semantic-coherence (1p pattern)  ←── CONSUMER 3
    ├── described_by ← agentic-lineage (1p pattern)  ←── CONSUMER 4
    └── described_by ← data-management (3p pattern)

data-management (3p)
    ├── narrower → data-lineage (3p)
    │       └── (extended by) agentic-lineage (1p)  ←── CONSUMER 4
    ├── narrower → data-quality (3p)
    │       └── (generalized by) semantic-coherence (1p)  ←── CONSUMER 3
    └── narrower → metadata-management (3p)

semantic-ingestion (1p)  ←── CONSUMER 1 (shared substrate)
    ├── extends → etl (3p)
    ├── extends → medallion-architecture (3p)
    └── extends → vector-embedding (3p)

agentic-rag (3p)  ←── CONSUMER 2
    └── extends → retrieval-augmented-generation (3p)
```

**The graph tells a clear story:**
- `governance-as-strategy` is a concept that provides the strategic framing — "governance is strategy execution, not compliance." It informs how the consumer patterns are applied but doesn't independently imply capabilities.
- `data-management` (3p) is the discipline hierarchy that all consumers extend
- `semantic-ingestion` is the shared substrate — it feeds all consumers
- `agentic-rag` is commodity retrieval — it consumes the substrate
- The 1p innovations (`semantic-coherence`, `agentic-lineage`, `semantic-ingestion`) are where SemOps adds value on top of 3p foundations

---

## 3. The Deterministic / LLM Boundary

### Why This Is an Architectural Constraint, Not a Preference

From the [ingestion analysis](ingestion-graph-edge-relationships.md):

> "A less capable but fully deterministic NER model that always extracts 'data lineage' as one node is more valuable to your architecture than a brilliant LLM that sometimes extracts it as one node and sometimes as three."

Cross-corpus coherence measurement requires identical pipeline behavior. Five properties must be constant:

| Property | Why |
|----------|-----|
| **Embedding model** | Different models produce incommensurable vector spaces |
| **Entity recognition** | Different NER granularity produces artificial incoherence |
| **Chunking strategy** | Different chunk sizes capture different semantic context per vector |
| **Preprocessing** | Normalization differences produce different embeddings from identical text |
| **Relation extraction method** | LLM-derived edges are non-reproducible |

An LLM call at temperature=0 is *nearly* deterministic but not guaranteed — model updates, context window differences, and API-side changes can shift outputs. For measurement, "nearly" is not enough.

### Two Output Layers

| Layer | What produces it | Properties | Consumers |
|-------|-----------------|------------|-----------|
| **Deterministic** | Dictionary lookup, dependency parsing, co-occurrence, fixed embeddings | Reproducible, auditable, consistent, controllable | All three consumers |
| **LLM-enriched** | Constrained LLM calls (classification, relation extraction, summarization) | High quality, non-reproducible, provenance-tagged | RAG retrieval only |

Both layers are written to the same stores (PostgreSQL, Neo4j, Qdrant). The distinction is metadata — every output carries `extraction_method: deterministic` or `extraction_method: llm` with full provenance.

### Coherence Measurement Reads Only the Deterministic Layer

This is the hard constraint. ADR-0014's measurement model operates on:

| ADR-0014 Concept | Pipeline Layer | Why Deterministic |
|-----------------|----------------|-------------------|
| Corpus alignment check | Pipeline config versioning | Same config = comparable outputs |
| Node set comparison | Manifest-derived entities | Manifest is the controlled vocabulary |
| Vector-level coherence | Fixed embedding model | Same model = same vector space |
| Edge structure coherence | Deterministic edges only | Reproducible topology |
| Granularity consistency | Manifest-controlled concepts | One concept = one node, always |

---

## 4. Source-Defined Processing Strategies

### The Problem

Currently, the source config declares *where* content goes (corpus routing) but not *how* it's processed. Processing is hardcoded in `ingest_from_source.py` — every source gets the same chunking, the same embedding, the same LLM classification call. This means:

- Session note metadata (entity-only, no retrieval needed) gets chunked and embedded anyway
- Structured ADRs (heading-based chunks are obvious) get the same token-window chunking as unstructured research
- Sources that don't need LLM classification still go through the LLM call

### The Design: Processing as Source Contract

Each source config declares a `processing` block that specifies what the pipeline does. The source author knows their content best — they declare its characteristics and the pipeline executes.

**Processing dimensions:**

| Dimension | Options | What It Controls |
|-----------|---------|-----------------|
| `chunking` | `document_structure`, `concept_anchored`, `fixed_window`, `none` | How text is split for embedding |
| `embedding` | `default`, `none` | Whether vectors are generated |
| `entity_recognition` | `manifest`, `llm`, `none` | How concepts are identified in text |
| `relation_extraction` | `deterministic`, `deterministic+llm`, `none` | How edges are derived |
| `llm_enrichment` | `true`, `false` | Whether LLM metadata classification runs |

**Example source profiles:**

| Content Type | Chunking | Embedding | Entity Recognition | Relation Extraction | LLM Enrichment | Rationale |
|-------------|----------|-----------|--------------------|--------------------|----------------|-----------|
| Framework theory docs | `document_structure` | `default` | `manifest` | `deterministic+llm` | `true` | Core knowledge — full RAG + measurement |
| ADRs | `document_structure` | `default` | `manifest` | `deterministic` | `false` | Structured docs — no LLM needed, derivable from headings |
| Session notes | `none` | `none` | `none` | `none` | `false` | Entity-only — metadata pointer to GitHub issue |
| Published READMEs | `document_structure` | `default` | `manifest` | `deterministic+llm` | `true` | Public content — full retrieval with enrichment |
| Pattern docs | `document_structure` | `default` | `manifest` | `deterministic` | `false` | Authoritative — deterministic only, no LLM interpretation |
| YAML catalogs | `none` | `default` | `none` | `deterministic` | `false` | Structured data — edges from schema, not text |
| Research papers | `concept_anchored` | `default` | `manifest` | `deterministic+llm` | `true` | External content — LLM enrichment for discovery |

### Manifest as NER Model

The concept manifest (Ubiquitous Language + pattern registry + capability registry) replaces statistical NER for known concepts:

```
Manifest → scan text for mentions → extract context → embed → link to existing nodes
```

This inverts the traditional pipeline:
```
text → chunk → embed → NER → discover concepts → build graph  (traditional)
manifest → scan text for mentions → extract context → embed → link  (manifest-driven)
```

**Candidate discovery** (for unknown concepts) is a separate, slow, HITL process:
1. LLM/NER surfaces candidate concepts not in manifest
2. Candidates enter a discovery queue
3. Human reviews and either promotes to manifest or discards
4. Manifest grows deliberately, not automatically

---

## 5. Agentic Lineage as Telemetry Layer

### What Lineage Replaces

Agentic lineage (episodes) is not a fourth system — it's the instrumentation that wraps the other three. Once implemented, it replaces several ad-hoc data structures:

| Current Structure | What It Tracks | Replaced By |
|------------------|---------------|-------------|
| `coherence_profile` on source configs | Expected metadata quality | Pipeline version tracking on episodes |
| `metadata->>'classified_by'` on entities | Which model classified this | Episode `model_name`, `prompt_hash` |
| LLM classification mixed into entity JSONB | Why this entity has these labels | Episode `context_pattern_ids`, `detected_edges` |
| Ad-hoc `extraction_method` fields | How an edge was created | Episode provenance with full context |

### Episode Model (from ISSUE-101)

Each pipeline operation produces an episode:

| Episode Type | Captures | Consumers |
|-------------|----------|-----------|
| Ingestion | Source, files, entities created, pipeline config version | Audit, coherence baseline |
| Classification | LLM model, prompt, context patterns, confidence | Audit, RAG quality |
| Relation extraction | Method (deterministic/LLM), edges proposed, edges written | Audit, measurement exclusion |
| Embedding | Model, input hash, vector dimensions | Audit, reproducibility |

The episode carries `coherence_score` — this provides the temporal signal for ADR-0014's Stability dimension without needing a separate `coherence_snapshot` table.

### SKOS Lineage

The pattern graph shows the dependency chain:

```
agentic-lineage (1p)
    ├── extends → open-lineage (3p)     — event model for pipeline observability
    ├── extends → episode-provenance (3p) — bounded activity sequences
    └── extends → data-lineage (3p)     — flow and transformation tracking
                    └── adopts → prov-o (3p)  — W3C provenance ontology
```

This is textbook SemOps: adopt 3p standards, innovate 1p on top. The 1p innovation is adding agent decision context and trust provenance to the established lineage standards.

---

## 6. Capability Gap Analysis

### Capabilities That Serve Each Consumer

| Consumer | Capability | Pattern Count | Repo Count | Gap? |
|----------|-----------|--------------|------------|------|
| **Ingestion** | `ingestion-pipeline` | 7 | 1 | Single repo — no redundancy |
| | `corpus-collection` | 3 | 1 | Single repo |
| | `corpus-meta-analysis` | 3 | 1 | Single repo |
| | `structured-extraction` | 3 | 1 | Single repo |
| **RAG** | `internal-knowledge-access` | 4 | 1 | Single repo |
| | `context-engineering` | 2 | 1 | Low pattern coverage |
| | `input-optimization` | 3 | 1 | Single repo |
| **Coherence** | `coherence-scoring` | 5 | 2 | OK — multi-repo |
| | `architecture-audit` | 4 | 1 | Single repo |
| | `pattern-governance` | 2 | 1 | Low pattern coverage |
| **Lineage** | `agentic-lineage` | 6 | 2 | OK — multi-repo, but **capability_count=1** in registry |

### Gaps Identified

1. **`agentic-lineage` capability has 6 pattern links but only 1 capability entry** — the registry lists it as a single capability, but it spans episode tracking, provenance, and audit. Consider whether this should decompose into sub-capabilities (e.g., `episode-tracking`, `provenance-audit`, `decision-context`).

2. **`context-engineering` has only 2 pattern links** — for a capability that's critical to RAG quality, this is thin. May need pattern attribution from `raptor`, `agentic-rag`, or reasoning patterns.

3. **`pattern-governance` has only 2 pattern links** — governance is central to coherence measurement but under-attributed.

4. **All ingestion capabilities are single-repo** — semops-data is the sole owner. This is architecturally correct (schema owner owns ingestion) but creates a bus factor of 1.

5. **No capability for "manifest curation"** — the HITL concept discovery process (Phase 1 in the pipeline) has no registered capability. This is a gap — it's a real activity that should be tracked.

6. **No capability for "source config management"** — declaring and maintaining source processing contracts is operational work that isn't registered.

---

## 7. Implementation Priorities

### Ordered by Dependency, Not Ambition

| Priority | What | Why First | Patterns Exercised |
|----------|------|-----------|-------------------|
| **P1** | Add `processing` block to source config schema | Everything else depends on knowing what each source needs | `semantic-ingestion`, `data-contracts` |
| **P2** | Tag existing outputs with `extraction_method` | Enables consumer separation immediately — no pipeline rewrite needed | `provenance-first-design`, `edge-predicates` |
| **P3** | Manifest-based entity recognition | Replaces ad-hoc LLM NER; makes concept inventory deterministic | `semantic-ingestion`, `metadata-management` |
| **P4** | Source-defined chunking strategies | Sources declare what they need; pipeline executes the contract | `semantic-ingestion`, `data-contracts` |
| **P5** | Episode instrumentation | Agentic lineage wraps operations; replaces ad-hoc provenance | `agentic-lineage`, `episode-provenance`, `open-lineage` |
| **P6** | Coherence measurement on deterministic layer | ADR-0014 implementation; consumes only tagged-deterministic outputs | `semantic-coherence`, `data-management` |
| **P7** | Dead code removal | Remove `coherence_profile`, ad-hoc provenance fields | cleanup |
| **P8** | Reclassify `governance-as-strategy` from pattern to concept entity | Pattern audit finding — fails three-property test  | `semantic-object-pattern` |

### What Can Be Done Now vs. What Requires Infrastructure

| Now (source config + tagging) | Near-term (manifest + chunking) | Later (episodes + measurement) |
|------------------------------|--------------------------------|-------------------------------|
| P1: Source config `processing` block | P3: Manifest entity recognition | P5: Episode instrumentation |
| P2: `extraction_method` tagging | P4: Chunking strategy dispatch | P6: Coherence measurement |
| P7: Dead code removal (independent) | | |

---

## 8. Pattern Audit: `governance-as-strategy`

During this design session, a pattern audit was triggered on `governance-as-strategy` — it appeared as the strategic hub connecting all three consumers. The audit found it should be a concept entity, not a pattern.

### Three-Property Test (Pattern v9.0.0)

| Property | Result | Reasoning |
|----------|--------|-----------|
| **Fits a domain** | **Fail** | Describes *how to think about* governance, not *what you do*. `data-management` (3p) describes practices and policies. `governance-as-strategy` describes a mindset — "governance is strategy execution, not compliance." That's a principle, not a domain practice. |
| **Implies capabilities** | **Fail** | 4 capabilities attributed, but all are more naturally implied by `explicit-architecture` or `data-management`. The capabilities exist because of those patterns; `governance-as-strategy` describes the *lens* through which they're applied. |
| **Recognizable fit** | **Fail** | When you recognize "governance-as-strategy" in practice, you're recognizing strategic application of `data-management` disciplines — not a separate, independent pattern. |

### History

- Promoted from concept to pattern in  (coherence remediation, 2026-03-15)
- That promotion created: pattern table row, pattern doc, SKOS edges, capability IMPLEMENTS edges, UL entry, registry.yaml entries
- This audit finds the promotion was incorrect

### Corrective Action

Tracked in :

- Remove from pattern table → create as concept entity
- Reattribute capabilities to `data-management`, `explicit-architecture`, `semantic-coherence`
- Replace IMPLEMENTS edges with DESCRIBED_BY edges (patterns reference the concept)
- Update ADR-0014, ADR-0015, registry.yaml references

### Impact on This Design

The cross-consumer dependency graph (Section 2) has been updated to show `governance-as-strategy` as a concept entity. The strategic framing it provides is real and valuable — it just isn't a pattern. The patterns it informs (`semantic-coherence`, `agentic-lineage`, `data-management`) carry the actual capability implications.

---

## 9. Open Questions

1. **Should `semantic-ingestion` pattern definition be updated?** Current definition doesn't mention the deterministic/LLM boundary. This is a significant enough architectural constraint to be part of the pattern definition.

2. **Qdrant collection routing** — ADR-0015 defers this. When does it become necessary? Only when coherence measurement needs collection-level isolation, or when RAG performance requires it.

3. **Manifest completeness threshold** — How complete must the manifest be before manifest-driven entity recognition is more valuable than LLM NER? Current state: 66 patterns, 45 capabilities, ~100 UL terms. Is that enough?

4. **Episode storage** — PostgreSQL (as ISSUE-101 proposes) or a separate telemetry store? Episodes are time-series data with different access patterns than entities.

5. **Migration path for existing data** — 1,077 entities exist without `extraction_method` tags. Backfill as `extraction_method: legacy` or attempt to classify retroactively?

---

## References

- [ADR-0015: Ingestion Pipeline Architecture](../decisions/ADR-0015-ingestion-pipeline-architecture.md) — the architectural decision
- [ADR-0014: Coherence Measurement Model](../decisions/ADR-0014-coherence-measurement-model.md) — what coherence measurement needs from the pipeline
- [ISSUE-101: Episode-Centric Provenance](../decisions/ISSUE-101-episode-centric-provenance.md) — agentic lineage design
- [ingestion-graph-edge-relationships.md](ingestion-graph-edge-relationships.md) — deterministic vs. LLM analysis
- [config/registry.yaml](../../config/registry.yaml) — capability and integration map authority
