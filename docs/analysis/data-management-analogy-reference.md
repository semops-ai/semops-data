# Coherence Analysis: data-management-analogy (Discovery Mode)

> **Source:** `publisher-pr/docs/drafts/semops-framework/data-management-analogy.md`
> **Analysis Date:** 2026-03-15
> **Mode:** Discovery — analyzing a new viewpoint for pattern extraction, not auditing a registered pattern

---

## 1. Document Thesis

The document makes a single core argument across three layers:

1. **Data management is not the modern data stack.** The pipeline (ingestion, transform, warehouse) is one concern. Governance — collecting metadata across tool boundaries, structuring it into a queryable graph, governing through that graph — is a fundamentally different concern that sits *above* the stack.

2. **SemOps is the governance layer, applied to knowledge and agent operations.** We're not building a data stack. We're building the metadata graph, lineage tracking, and quality contracts — applied to organizational knowledge instead of data pipelines.

3. **Governance creates a deterministic substrate that reduces the surface area where non-deterministic inference carries the load.** Routine operations follow the graph. Novel reasoning gets pushed to the edges.

---

## 2. Concept Extraction — What the Document Names

### 3P Disciplines (established data management)

| Concept | Document Section | Status in Pattern Registry |
|---------|-----------------|---------------------------|
| Data Orchestration | "Core Concept: The DAG" — scheduling, dependency management, state tracking, retry/failure handling | **NOT REGISTERED** — no pattern for orchestration |
| Data Lineage | "Data Lineage" — tracking data flow across tool boundaries, stitching lineage fragments into end-to-end graph | **REGISTERED** as `data-lineage` (3p) |
| Data Quality | "Data Quality" — schema contracts, freshness thresholds, value range enforcement, assertion layer | **NOT REGISTERED** — no pattern for data quality/contracts |
| Metadata Management | "Metadata Management" — catalog/registry, making data asset relationships explicit, queryable, enforceable | **NOT REGISTERED** — no pattern for metadata management/data catalog |
| Transformation Governance | "Transformation Governance (dbt)" — `ref` as dependency declaration, manifest.json as governance artifact | **NOT REGISTERED** — tool-level, may not warrant its own pattern |
| OpenLineage | Referenced as standard for pipeline lineage | **REGISTERED** as `open-lineage` (3p) |

### 1P Extensions (SemOps innovations)

| Concept | Document Section | Status in Pattern Registry |
|---------|-----------------|---------------------------|
| Agentic Lineage | "Agentic Lineage" — extending governance disciplines to non-deterministic agents | **REGISTERED** as `agentic-lineage` (1p) |
| Structural → Causal Lineage | "From Structural to Causal Lineage" — tracking not just what happened but why | Part of `agentic-lineage` — not separate pattern |
| Decision Documents + Episode Schema | "Decision Documents + Episode Schema" — event-sourcing for agent reasoning | **REGISTERED** as `episode-provenance` (3p) — but doc frames it as 1P synthesis |
| Semantic Coherence as Data Quality | "Semantic Coherence as Data Quality" — SC score as agentic data quality | **REGISTERED** as `semantic-coherence` (1p) |
| Governance as Deterministic Substrate | "Governance as a Deterministic Substrate" — reducing non-deterministic inference surface | **NOT REGISTERED** — this is the core 1P thesis |
| Absorbing Change | "Absorbing Change Instead of Breaking" — metadata graph as change absorption mechanism | **NOT REGISTERED** — consequence/mechanism of deterministic substrate |

### Existing Pattern Connections (mentioned but not the focus)

| Pattern | How Referenced | Registered? |
|---------|--------------|-------------|
| `governance-as-strategy` | Title and framing — this doc IS the strategic rewrite governance-as-strategy needs | CONCEPT ONLY  |
| `explicit-architecture` | "Explicit Architecture where every layer is explicit, schema'd, wired into the same metadata graph" | REGISTERED (1p) |
| `mirror-architecture` | Listed in implementation patterns | REGISTERED (1p) |
| `scale-projection` | Listed in implementation patterns | Need to check |
| `semantic-coherence-measurement` | Listed in implementation patterns | Need to check vs `semantic-coherence` |

---

## 3. Pattern Mapping — The Adoption-Innovation Chain

The document reveals a clear 3P→1P progression that follows the Semantic Optimization Loop:

```
3P ADOPTED DISCIPLINES              1P SEMOPS EXTENSIONS
─────────────────────               ─────────────────────
Data Orchestration       ──────►    (agentic orchestration — implicit in doc)
Data Lineage             ──────►    Agentic Lineage (causal, bidirectional)
Data Quality             ──────►    Semantic Coherence (SC score)
Metadata Management      ──────►    Explicit Architecture (queryable arch graph)
Episode/Provenance       ──────►    Decision Documents + Episode Schema
                    ╲               ╱
                     ╲             ╱
                      ▼           ▼
               GOVERNANCE AS DETERMINISTIC SUBSTRATE
               (the overarching 1P synthesis)
```

### What's new here that doesn't exist in the registry:

**1. Data Orchestration (3p candidate)**
- DAG-based scheduling, dependency management, state tracking, retry/failure handling
- Tools: Airflow, Dagster
- SemOps relevance: agentic operations follow DAG-like patterns; the doc frames agents as dynamic DAG constructors
- Registration question: Is this pattern-level or just background knowledge?

**2. Data Quality / Data Contracts (3p candidate)**
- Schema contracts, freshness thresholds, value ranges, referential integrity
- Tools: Great Expectations, dbt tests, Soda, Monte Carlo
- SemOps relevance: `semantic-coherence` is explicitly positioned as the agentic extension of data quality. But the 3P foundation isn't registered.
- Registration question: If `semantic-coherence` extends `data-quality`, should `data-quality` be registered as its 3P broader pattern?

**3. Metadata Management / Data Catalog (3p candidate)**
- Catalog/registry layer — what assets exist, who owns them, how they relate
- Tools: DataHub, Atlan, OpenMetadata, Amundsen, Apache Atlas
- SemOps relevance: This is exactly what the entity/edge model does. `explicit-architecture` is the 1P extension. But the 3P foundation isn't registered.
- Registration question: Same logic — if `explicit-architecture` extends metadata management, should the 3P pattern exist?

**4. Governance as Deterministic Substrate (1p candidate — the big one)**
- "Make the environment agents operate in structured enough that they are organizing and reconciling more than reasoning new decisions"
- "Reduce the surface area where non-deterministic inference has to carry the load"
- Four mechanisms: governed corpus, decision documents, episodic lineage, metadata density
- This is the overarching thesis that ties `governance-as-strategy`, `explicit-architecture`, `agentic-lineage`, and `semantic-coherence` together
- Registration question: Is this a pattern, or is this the framework thesis itself?

**5. Absorbing Change (1p mechanism)**
- "Change is absorbed, surfaced, and solved. The audit layer that maintains metadata is continuously reconciling the real world against schema'd expectations."
- "When drift is detected, it's a metadata update event, not a breakage event"
- Closely related to `stable-core-flexible-edge` — the flexible edge absorbs change, the stable core provides the reference point
- Registration question: May be a mechanism within `governance-as-strategy` rather than its own pattern

---

## 4. Type 1 — Structural Coherence (against existing registry)

### Patterns the document depends on — registry status

| Pattern | Status | Coverage | Gap? |
|---------|--------|----------|------|
| `data-lineage` | Registered (3p) | 3 content, 1 cap, 1 repo | Low coverage |
| `open-lineage` | Registered (3p) | 0 content, 1 cap, 2 repos | No content entities |
| `agentic-lineage` | Registered (1p) | 1 content, 1 cap, 2 repos | Low coverage |
| `semantic-coherence` | Registered (1p) | 19 content, 1 cap, 2 repos | Good content, low cap |
| `episode-provenance` | Registered (3p) | 0 content, 1 cap, 2 repos | No content entities |
| `explicit-architecture` | Registered (1p) | 6 content, 9 cap, 4 repos | Good |
| `governance-as-strategy` | CONCEPT ONLY | 0 (no pattern row) | **Full gap — ** |
| Data Orchestration | NOT REGISTERED | — | **Missing 3P foundation** |
| Data Quality | NOT REGISTERED | — | **Missing 3P foundation** |
| Metadata Management | NOT REGISTERED | — | **Missing 3P foundation** |

### SKOS edges that should exist but don't

The document implies these relationships:

| Source | Predicate | Target | Exists? |
|--------|-----------|--------|---------|
| `agentic-lineage` | `broader` | `data-lineage` | Need to check |
| `semantic-coherence` | `broader` | data-quality (unregistered) | **NO** — 3P not registered |
| `explicit-architecture` | `broader` | metadata-management (unregistered) | **NO** — 3P not registered |
| `governance-as-strategy` | `narrower` | `agentic-lineage` | **NO** — governance not registered |
| `governance-as-strategy` | `narrower` | `semantic-coherence` | **NO** — governance not registered |
| `governance-as-strategy` | `narrower` | `explicit-architecture` | **NO** — governance not registered |
| `episode-provenance` | `related` | `agentic-lineage` | Need to check |

---

## 5. Type 2 — Semantic Coherence

### Framing Comparison: Document vs. Existing Definitions

| Concept | Document Framing | Registry Definition | Alignment |
|---------|-----------------|-------------------|-----------|
| Agentic Lineage | Table comparing deterministic vs. agentic extensions across 4 disciplines; structural→causal lineage; bidirectional traversal | "Lineage tracking extended with agent decision context and trust provenance" | **Doc is richer** — registry definition is abstract, doc has concrete mechanism table |
| Semantic Coherence | "Extends data quality to measure whether knowledge artifacts are aligned, consistent, and stable" — positioned explicitly as agentic data quality | "Measurable signal that quantifies how well a system's reality matches its domain model" | **Complementary** — doc frames the WHY (agentic data quality), registry has the WHAT (formula, measurement) |
| Explicit Architecture | "Every layer is explicit, schema'd, and wired into the same metadata graph" — positioned as the SemOps version of metadata management | "Making architectural decisions explicit, queryable, and traceable" | **Doc adds the data management lineage** — registry doesn't mention metadata management roots |
| Episode Provenance | "Event-sourcing pattern to keep the lineage graph lean while capturing rich reasoning context" — episodes + decision documents with reuse, versioning, auditability | "Extension of PROV-O that groups agent actions into episodes" | **Doc is significantly richer** — adds decision documents as separate artifacts, reuse model, versioning |
| Governance as Strategy | "We're building the governance layer — metadata graph, lineage tracking, quality contracts — applied to knowledge and agent operations" | Concept-pattern map: "Reframing governance from compliance overhead to strategic capability" | **Doc delivers what the concept-pattern map definition promises** — this IS the strategic rewrite |

### Key Semantic Findings

**Finding 1: This document IS the governance-as-strategy source doc rewrite**

The current `governance-as-strategy` source doc is a data engineering primer on provenance/lineage. This document does what that doc should do — frames governance as a strategic system, positions the 3P disciplines as foundation, shows the 1P extensions, and lands the "deterministic substrate" thesis. This is the content that should replace or restructure the current source doc.

**Finding 2: Three missing 3P foundations reveal a SKOS gap**

The document makes explicit that SemOps' 1P patterns extend specific 3P data management disciplines. But those 3P disciplines (data orchestration, data quality, metadata management) aren't registered. Without them, the SKOS `broader` edges from 1P innovations to their 3P foundations can't be expressed. The "adopt 3P → innovate 1P" lineage is invisible in the registry.

**Finding 3: "Governance as Deterministic Substrate" may be the actual 1P pattern**

The concept-pattern map registered `governance-as-strategy` with a definition about "reframing governance from compliance to strategic capability." But this document articulates something more specific and powerful: governance as a mechanism that reduces the surface area where agents need to reason from scratch. This might be a better pattern definition, or it might be a separate pattern that `governance-as-strategy` is `broader` than.

**Finding 4: The 4-discipline table is a publishable differentiation artifact**

The table comparing deterministic pipelines to agentic extensions across lineage, quality, metadata, and orchestration is the most concrete articulation of SemOps' value proposition against the data management landscape. This doesn't exist in any registered pattern doc or published content.

---

## 6. Remediation / Next Steps

### Pattern Registry Actions

| Action | Type | Pattern | Rationale |
|--------|------|---------|-----------|
| Register | 3p | `data-quality` | Foundation for `semantic-coherence` SKOS broader edge |
| Register | 3p | `metadata-management` | Foundation for `explicit-architecture` SKOS broader edge |
| Consider | 3p | `data-orchestration` | Foundation for agentic orchestration; lower priority |
| Promote | 1p | `governance-as-strategy` | Already planned  — but definition should incorporate "deterministic substrate" thesis |
| Consider | 1p | `deterministic-substrate` or fold into `governance-as-strategy` | The overarching thesis — decide if it's a separate pattern or the definition of governance-as-strategy |

### SKOS Edges to Create (once patterns registered)

| Source | Predicate | Target |
|--------|-----------|--------|
| `semantic-coherence` | `broader` | `data-quality` |
| `explicit-architecture` | `broader` | `metadata-management` |
| `agentic-lineage` | `broader` | `data-lineage` |
| `governance-as-strategy` | `narrower` | `agentic-lineage` |
| `governance-as-strategy` | `narrower` | `semantic-coherence` |
| `governance-as-strategy` | `narrower` | `explicit-architecture` |

### Content Actions

| Action | Detail |
|--------|--------|
| **Governance-as-strategy rewrite** | This document's framing should drive the source doc restructure planned in . Lead with "Data management is not the modern data stack" → SemOps is the governance layer → deterministic substrate thesis. |
| **Agentic lineage pattern doc enrichment** | The 4-discipline comparison table should be in the `agentic-lineage` pattern doc — it's the clearest articulation of what agentic lineage extends and why. |
| **Episode provenance enrichment** | Decision documents as separate, reusable, versioned artifacts — this level of detail isn't in the current pattern definition. |

### Open Question: Pattern Scope

Is "governance as deterministic substrate" the *definition* of `governance-as-strategy`, or is it a *separate* 1P pattern?

Arguments for **same pattern (update definition):**
- The concept-pattern map definition ("structures that generate insight, enable autonomy, align execution") is basically a softer version of "deterministic substrate"
- The source doc is already named `governance-as-strategy`
- One pattern with a richer definition is simpler than two overlapping patterns

Arguments for **separate pattern:**
- "Governance as strategy" is a broad organizational philosophy
- "Deterministic substrate" is a specific architectural mechanism
- They might have different `pattern_type` values (concept vs. implementation)
- The 3P→1P chain suggests governance-as-strategy is the philosophy and deterministic-substrate is the implementation

**Recommendation:** Start with one pattern (`governance-as-strategy`) with the enriched definition. If the distinction proves load-bearing during capability mapping, split later.

---

## 7. Remediation Executed

### Source doc rewrite (2026-03-15)

Rewrote `docs-pr: STRATEGIC_DATA/governance-as-strategy.md` to lead with strategic thesis:

1. **Governance is strategy, not compliance** — opening reframe
2. **Data management is not the modern data stack** — the 3P discipline being adopted
3. **Five governance disciplines** — lineage, quality, metadata management, orchestration, provenance (condensed from primer to reference)
4. **SemOps extensions table** — deterministic pipeline vs. SemOps extension for each discipline, with corrections:
   - `agentic-lineage` is a measurement technique (DataHub modified for agents), not a sub-concept
   - `semantic-coherence` is a deeper analysis using the governance layer for goal alignment, not just data quality extended
5. **Governance as deterministic substrate** — the 1P thesis (4 mechanisms)
6. **Absorbing change** — metadata graph as change absorption, tradeoff tracking

Also fixed ID inconsistency: `governance-as-a-strategy` → `governance-as-strategy` in frontmatter.

### Pattern hierarchy clarified

```
governance-as-strategy (1p, overarching concept)
    ├── adopts: data-management (3p — to be registered)
    │       ├── narrower: data-quality, metadata-management, data-lineage, etc.
    │
    ├── narrower: explicit-architecture (1p) — extends metadata-management
    └── ...

semantic-coherence (1p) — co-equal, reads the governance layer
agentic-lineage (1p) — implementation technique, DataHub for agents
    ├── broader: data-lineage
    └── related: governance-as-strategy
```
