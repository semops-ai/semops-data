# ADR-0016: Data Layer Boundaries

> **Status:** Draft
> **Date:** 2026-03-30
> **Related Issue:** 
> **Builds On:** [ADR-0009](./ADR-0009-strategic-tactical-ddd-refactor.md), [ADR-0012](./ADR-0012-pattern-coherence-co-equal-aggregates.md), [ADR-0015](./ADR-0015-ingestion-pipeline-architecture.md)
> **Design Doc:** [DD-0001](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md)

---

## Executive Summary

The system's data has three distinct concerns — stored data, metadata, and telemetry — but current implementation blurs the boundaries. This ADR establishes the conceptual model: a formality gradient where data representation follows business structurability, with the SQL domain model as the authoritative core. Corpus taxonomy is reframed from content origin to architectural position on this gradient.

---

## Context

During  (source config processing block), we identified a framing gap: DD-0001 treats ingestion as "markdown → chunks → vectors → graph," but the most important data in the system is the SQL domain model — patterns, capabilities, repositories, agents, and their edges.

The SemOps premise is that effective AI use requires encoding the whole company and its activities into data structures. Explicit Architecture is how we model business entities and structure into a fully-encoded DDD architecture, represented by actual schemas and data. This means **the architecture itself is data**, not documentation about data.

### The governed corpus problem

A well-governed corpus is epistemology infrastructure — the shared agreement on what counts as a reliable input to decisions. In an organizational setting, individuals should not control what enters the context window. Leaving context assembly to each agent or user means no version control on knowledge, no access control on sensitive documents, no expiration on superseded policy. Well-structured shared corpora are governance artifacts, not just retrieval optimizations.

This requires intentional corpus design. Every organization produces knowledge across distinct surfaces, whether they manage them or not:

| Surface type | Examples | Characteristics |
|---|---|---|
| **Transactional data** | CRM records, support tickets, financial data | Structured, queryable, captures operational reality |
| **Curated knowledge** | Architecture docs, domain models, style guides, policies | Low volume, high durability, represents settled understanding |
| **Operational artifacts** | Issues, PRs, ADRs, runbooks, incident reports | High volume, decays fast, captures decisions in motion |
| **Published content** | Blog posts, whitepapers, READMEs, API docs | Public-facing, version-controlled, represents external claims |
| **Research** | Literature reviews, vendor analysis, competitive intel | Mixed durability, captures evidence and hypotheses |
| **Aggregated analytics** | Normalized metrics, cohort analyses, KPI summaries | Machine-generated, pre-aggregated, captures quantitative state |

Each surface has a natural query pattern that determines its retrieval mode:

| Query pattern | Retrieval mode | Corpus type |
|---|---|---|
| "What do we know about X?" | Semantic search over prose | Semantic corpus — vector embeddings, chunk retrieval |
| "How many X in the last quarter?" | Structured query | Analytical store — SQL, materialized views |
| "What's related to X?" | Graph traversal | Graph corpus — entity-relationship edges |
| "What did we decide about X and why?" | Semantic + structured | Hybrid — semantic search with metadata filters |

### Where the current system falls short

The current system conflates three concerns:

- **YAML manifests** (registry.yaml, pattern_v1.yaml, agents.yaml) act as both metadata catalogs and data, creating confusion about what is authoritative
- **Corpus labels** (`core_kb`, `published`, `deployment`) describe content origin, not architectural significance — `core_kb` sounds like "the core domain model" but is actually just a label on content entities that describe theory
- **Telemetry** (episode provenance, `classified_by` fields, source config provenance) is scattered across entity JSONB fields with no unified surface
- The actual core — pattern table, capability/repository/agent entities, edge tables — has no corpus classification and isn't part of the ingestion pipeline framing

This confusion is not just organizational. It affects how agents reason about the system. When an agent retrieves from `core_kb`, it gets theory documents, not the actual domain model. The retrieval hierarchy agents actually follow is: SQL first (deterministic, authoritative), then vector search (semantic, fallback) — but the architecture doesn't make this explicit.

---

## Decision

### D1: The formality gradient is architectural, not technical

Data representation follows the structurability of what it represents in the business:

| Business layer | Structurability | Natural representation | Governance mode |
|---|---|---|---|
| Core domains, processes, capabilities | High — well-defined, stable | SQL schema (patterns, capabilities, repos, edges) | Schema constraints — hardest to change by design |
| Operational metadata, catalogs | Medium — structured but evolving | Metadata catalogs (currently YAML manifests) | Validation rules |
| Strategy, brand, conceptual content | Low — nuanced, contextual | Vector embeddings + graph edges | Cognitive agents |

The more core the aggregate is to the business, the more structured its data representation. SQL schema provides the most stability, the most semantic compression, and the strongest governance — because it is the hardest to change by design. This is built-in governance, not governance by process.

Vector and graph are not alternatives to SQL — they are the **bridge** that connects less structured entities (strategy, brand, research) to more structured entities (patterns, capabilities, domains). Cognitive agents do their heaviest work at the top of the gradient, where the business is hardest to formalize.

This gradient also determines query patterns: structured data is queried deterministically (SQL), unstructured data is queried semantically (vector search), and graph traversal bridges between them. Purpose-driven corpus design aligned to this gradient is what enables agentic reasoning patterns — chain-of-thought, tree-of-thought, and ReAct — to operate effectively across the full knowledge surface.

**This is a data architecture decision.** The ingestion pipeline follows the medallion architecture pattern: source ingestion (bronze), transformation (silver), and query surface (gold) are distinct layers. The formality gradient determines how much silver-layer transformation a source needs — structured YAML enters close to gold, unstructured research needs full transformation. Corpus design, retrieval hierarchy, and faceted classification are all gold layer concerns — how consumers access data — not pipeline concerns. See DD-0001 for the full medallion mapping.

### D2: Three data concerns with distinct boundaries

| Concern | What it is | Boundary |
|---|---|---|
| **Stored data** | The domain model — patterns, capabilities, entities, edges, surfaces, deliveries | SQL schema is the source of truth for business architecture |
| **Metadata** | How the system understands and manages its own data — processing contracts, routing, catalogs | Derived from stored data + telemetry; not a source of truth itself |
| **Telemetry** | Operational history — what happened, when, why | Agentic lineage (episodes) as the single telemetry surface |

Current YAML manifests are an implementation shortcut for context engineering. They serve as both metadata catalogs and data, which creates confusion. The target state is that manifests become pure metadata catalogs — views over the stored data layer, not an independent source of truth.

**Scope note:** Metadata management and telemetry are not mature enough for implementation decisions. This ADR establishes the model. Immediate action focuses on stored data and corpus taxonomy (D3).

### D3: Corpus taxonomy reflects architectural position, not content origin

The current corpus taxonomy (`core_kb`, `published`, `deployment`, `research_*`) classifies by where content came from. Under the formality gradient, corpus should reflect where data sits architecturally:

| Architectural layer | Current corpus | What it actually contains |
|---|---|---|
| SQL domain model (patterns, capabilities, repos, agents, edges) | *(none — outside corpus taxonomy)* | The actual core of the system |
| Structured domain knowledge (theory, methodology, definitions) | `core_kb` | Content entities that describe the domain |
| Operational artifacts (deployment configs, infrastructure) | `deployment` | Content entities from CI/CD and infrastructure |
| Published output (READMEs, sites, public content) | `published` | Content entities delivered to external audiences |
| Research input (analysis, exploration, external sources) | `research_*` | Content entities from research and discovery |

The key insight: the SQL domain model is the true core, and it has no corpus classification because the corpus concept was designed for content entities only. **Source is an infrastructure detail** — how content gets into the system — not an architectural classification.

This ADR does not rename or restructure the existing corpora. It establishes that the taxonomy should be understood as architectural layers, and that the SQL domain model is the foundational layer that the content corpora describe, reference, and connect to.

The pattern type system already encodes this corpus variety within the SQL domain model:

| Pattern type | What it captures | Corpus affinity |
|---|---|---|
| **Domain** | Domain capabilities and process capabilities — the business itself | Curated knowledge, transactional data |
| **Implementation** | Technical and infrastructure patterns | Operational artifacts, deployment |
| **Analytic** | Compound metrics — quantitative measurement | Aggregated analytics |

Concepts are entities with edges to patterns, but are not patterns themselves — they represent the less structured knowledge that the graph connects to the pattern layer.

The knowledge surfaces from the Context section map to these architectural layers:

| Knowledge surface | Architectural layer | Query mode |
|---|---|---|
| Transactional data | SQL domain model | Structured query |
| Curated knowledge | `core_kb` | Hybrid (semantic + structured) |
| Operational artifacts | `deployment` | Hybrid (semantic + structured) |
| Published content | `published` | Semantic search |
| Research | `research_*` | Semantic search |
| Aggregated analytics | *(not yet represented)* | Structured query |

**Future consideration: faceted classification.** Architectural position is one retrieval axis, but not the only one. Lifecycle state (active/planned/deprecated) and provenance (1P/2P vs. 3P) create additional query dimensions. For example: 1P published content and 1P structured concept docs share a provenance boundary that matters for competitive analysis against 3P research. Lifecycle state matters for roadmap-to-production evaluation against goals. Whether these facets require separate corpora or composable query filters over entity metadata is an implementation investigation (see DD-0001 Open Question 6).

---

## Consequences

**Positive:**

- Agents can reason about data significance by architectural position, not just corpus labels
- The retrieval hierarchy (SQL → vector → graph) becomes an explicit architectural constraint, not an emergent behavior
- Clear separation of concerns prevents manifests from drifting into a shadow source of truth
- Aligns the ingestion framing in DD-0001 with how the system actually works

**Negative:**

- DD-0001 needs revision to reflect the expanded framing (this is the work in )
- Existing documentation and tooling that references corpus labels by content origin needs conceptual alignment over time

**Risks:**

- Metadata and telemetry boundaries are intentionally deferred — risk of the model staying conceptual without implementation pressure. Mitigated by: the stored data and corpus changes are actionable now, and each future implementation step (episode consolidation, manifest-to-catalog migration) can reference this model.

---

## Pattern and Capability Impact

| Type | ID | Impact | Action |
|---|---|---|---|
| Pattern | `explicit-architecture` | Extends — formality gradient makes the "fully-encoded architecture" concept concrete | Update definition |
| Pattern | `data-management` | Extends — three-concern boundary model (stored/metadata/telemetry) is a data management principle | No action |
| Pattern | `semantic-ingestion` | Extends — ingestion now explicitly covers all data surfaces, not just content → vector | Update DD-0001 |
| Capability | `ingestion-pipeline` | Modified — scope expands from content ingestion to all-surface ingestion | Update registry description |

---

## References

- [DD-0001: Ingestion Pipeline Architecture](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md)
- [ADR-0009: Strategic/Tactical DDD Refactor](./ADR-0009-strategic-tactical-ddd-refactor.md) — three-layer architecture
- [ADR-0012: Pattern + Coherence Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md) — aggregate map
- [ADR-0014: Coherence Measurement Model](./ADR-0014-coherence-measurement-model.md) — deterministic output constraint
- [ADR-0015: Ingestion Pipeline Architecture](./ADR-0015-ingestion-pipeline-architecture.md) — three-consumer model
- [ISSUE-203 session notes](../session-notes/ISSUE-203-source-config-processing-block.md) — where the insight originated
