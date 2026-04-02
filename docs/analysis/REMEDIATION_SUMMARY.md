# Coherence Remediation Summary

> **Period:** 2026-03-15 to 2026-03-16
> **Scope:** 8 patterns analyzed, 120+ remediation actions across 9 issues
> **Model:** [ADR-0014: Coherence Measurement Model](../decisions/ADR-0014-coherence-measurement-model.md)
> **Walkthrough:** [sc-walkthrough-explicit-architecture.md](sc-walkthrough-explicit-architecture.md)

---

## SC Scorecard

| Pattern | Verdict | A Pre | A Post | C Pre | C Post | S | SC Pre | SC Post | Delta |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| explicit-architecture | Active → Near Complete | 0.46 | 0.96 | 0.38 | 0.75 | 0.80 | **0.52** | **0.83** | +0.31 |
| governance-as-strategy | Concept Only → Active | 0.38 | 0.94 | 0.23 | 0.95 | 0.90 | **0.43** | **0.93** | +0.50 |
| data-system-classification | Active → Near Complete | 0.58 | 0.96 | 0.32 | 1.00 | 0.92 | **0.55** | **0.96** | +0.41 |
| scale-projection | Active-Incomplete | 0.57 | — | 0.38 | — | 0.88 | **0.58** | — | — |
| mirror-architecture | Retired → Development | 0.45 | 0.75 | 0.37 | 0.90 | 0.80 | **0.50** | **0.82** | +0.32 |
| system-primitive-decomposition | Pending (planned) | 0.65 | 0.85 | 0.35 | 0.75 | 0.82 | **0.57** | **0.81** | +0.24 |
| semantic-optimization | Concept Only (identity resolved) | 0.31 | 0.31 | 0.25 | 0.65 | 0.85 | **0.40** | **0.57** | +0.17 |
| data-management-analogy | Discovery (draft analysis) | 0.55 | 0.82 | 0.53 | 0.75 | 0.85 | **0.63** | **0.81** | +0.18 |

**Average SC improvement (remediated patterns): +0.30**
**Highest final: data-system-classification (0.96)** — ghost cleanup + SKOS lineage correction
**Largest delta: governance-as-strategy (+0.50)** — full concept-to-active promotion
**Unremediated baseline: scale-projection (0.58)** — validates pre-remediation methodology

---

## Remediation by Issue

### Issue : Coherence Analysis — Explicit Architecture
**Pattern:** explicit-architecture | **Status:** Complete

Created the coherence analysis methodology:
- [explicit-architecture-reference.md](explicit-architecture-reference.md) — 620-line cross-repo artifact assembly
- [COHERENCE_ANALYSIS_TEMPLATE.md](COHERENCE_ANALYSIS_TEMPLATE.md) — reusable 2-type analysis process
- Validated template across all 8 patterns
- Identified structural and semantic gaps deferred to –

### Issue : DESCRIBED_BY Edges (Pattern Aggregate Linking)
**Pattern:** all patterns | **Status:** Complete

Schema and tooling for linking patterns to their concept entities:
- Added `described_by` predicate to `pattern_edge` CHECK constraint (schema v8.3.0)
- Updated `pattern_coverage` view to count described_by edges
- Created `scripts/create_described_by_edges.py` — generalized workflow
- Pilot: 14 edges for explicit-architecture
- Deployment: 66 additional edges across all patterns (80 total)
- Enhanced `get_pattern` query and MCP tool with `include_described_by` parameter
- Retired 3 stale entities: `readme-old`, `symbiotic-architecture`, `symbiotic-enterprise`

### Issue : Remediation — Explicit Architecture
**Pattern:** explicit-architecture, viable-systems-model | **Status:** Complete

Type 1 (structural):
- Registered `viable-systems-model` as 3P pattern
- Created `explicit-architecture --extends--> viable-systems-model` edge
- Added 3 EA references to ARCHITECTURE.md
- Created `check_explicit_architecture_coverage` fitness function (MEDIUM severity)

Type 2 (semantic):
- Updated enriched definition: "governance as projection over entity/edge graph"
- Enhanced pattern doc v1.0.0 → v1.1.0: Origin section, Adoption Path, VSM derivation
- Added EA definition paragraph to STRATEGIC_DDD.md

### Issue : Governance-as-Strategy Coherence (Concept → Pattern)
**Patterns:** governance-as-strategy, data-management, data-quality, metadata-management | **Status:** Complete

The largest single remediation — promoted 1 concept pattern and registered 3 new 3P patterns:
- Registered 3 3P patterns: `data-management`, `data-quality`, `metadata-management`
- Updated `governance-as-strategy` definition with deterministic substrate thesis
- Created 4 pattern docs in `docs/patterns/`
- Created 12 SKOS edges (broader/narrower/adoption/related)
- Mapped 4 patterns across 6 capabilities in registry.yaml
- Updated UL v10.1.0: governance-as-strategy entry, Strategic Data pillar
- Updated STRATEGIC_DDD.md capability tables
- Ingested 4 pattern docs as KB entities with embeddings
- Created 4 DESCRIBED_BY edges
- Materialized to Neo4j: 59 pattern edges, 113 implements edges
- Created 4 graph enrichment edges
- Updated semops-docs public mirror
- Rewrote source doc in docs-pr (strategic thesis lead, governance disciplines)

### Issue : Register Pattern — semantic-optimization
**Pattern:** semantic-optimization | **Status:** In Progress

Identity resolution completed (parent pattern with broader edges to children). Registration actions defined but not yet executed:
- Pattern table row insertion (pending)
- 2 SKOS broader edges (pending)
- Capability implements edge (pending)
- Pattern doc creation (pending)
- Entity primary_pattern_id backfill (pending)

### Issue : Revive Pattern — mirror-architecture
**Pattern:** mirror-architecture | **Status:** In Progress

Lifecycle reversal from retired to development:
- Updated pattern lifecycle_stage: retired → development
- Created pattern doc: `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/mirror-architecture.md`
- Updated registry.yaml: 3 capabilities mapped
- Created 3 SKOS edges (extends→ddd, related→scale-projection, related→explicit-architecture)
- Updated entity lifecycle to development
- Pending: ingest_architecture.py re-run, semops-orchestrator stub cleanup

### Issue : Coherence Remediation — system-primitive-decomposition
**Capability:** system-primitive-decomposition | **Status:** In Progress

Scope and naming clarification for planned capability:
- Linked governance issue in registry.yaml
- Clarified criteria to product-level scope
- Added UL named term with canonical name and aliases
- Added ARCHITECTURE.md cross-repo capability reference
- Pending: worked example 

### Issue : Data-System-Classification Coherence
**Pattern:** data-system-classification | **Status:** Complete

Ghost pattern cleanup and 3P lineage correction:
- Deleted `four-data-system-types` ghost pattern (wrong definition)
- Migrated 5 edges from ghost to canonical pattern
- Registered `fundamentals-of-data-engineering` as 3P pattern
- Corrected SKOS: `extends dbt` → `extends fundamentals-of-data-engineering`
- Added `dbt extends fundamentals-of-data-engineering` (tool → framework)
- Added `related → medallion-architecture` edge
- Updated pattern doc with corrected SKOS edges
- Added ARCHITECTURE.md references and UL entry
- Ingested pattern doc as KB entity

### Issue : Registry Sync — VSM + Unregistered Capabilities
**Patterns:** viable-systems-model | **Capabilities:** attention-management, domain-reference-architecture, research | **Status:** Complete

Authority source alignment:
- Added `viable-systems-model` to pattern_v1.yaml (authority source)
- Registered 3 capabilities in registry.yaml
- Neo4j sync: 66 patterns, 61 edges
- Fitness check: 0 new violations

---

## Aggregate Statistics

| Metric | Count |
| --- | --- |
| Reference analyses produced | 8 |
| Issues created | 9 (–) |
| Issues completed | 6 |
| Issues in progress | 3 (, , ) |
| Patterns registered (new) | 5 (data-management, data-quality, metadata-management, viable-systems-model, fundamentals-of-data-engineering) |
| Pattern docs created | 7 |
| SKOS edges created | 30+ |
| DESCRIBED_BY edges created | 80 |
| Capability mappings added | 10+ |
| KB entities ingested | 8+ |
| Stale entities retired | 3 |
| Ghost patterns deleted | 1 (four-data-system-types) |
| Fitness functions created | 1 (check_explicit_architecture_coverage) |
| Schema version bumps | 1 (v8.3.0) |
| UL version bumps | 1 (v10.1.0) |

---

## Remaining Work

| Pattern | What's Left | Issue |
| --- | --- | --- |
| semantic-optimization | Full registration (pattern row, SKOS edges, pattern doc, entity backfill) |  |
| mirror-architecture | ingest_architecture.py re-run, semops-orchestrator stub cleanup |  |
| system-primitive-decomposition | Worked example  |  |
| scale-projection | No remediation started — 5 structural gaps, 5 semantic findings | None |

**scale-projection is the only analyzed pattern with no remediation issue.** Its pre-remediation SC (0.58) is mid-range, with strong pattern doc and SKOS alignment but weak graph connectivity (zero DESCRIBED_BY edges, zero content count).

---

## Remediation vs. Ingestion Pipeline 

The manual effort in this remediation cycle is a case study for . Many actions were repetitive across patterns and would be automated or simplified by a unified ingestion entry point.

### Actions that would be automated by `--target` re-ingestion

| Remediation Action | Occurrences |  Target | Current Method |
| --- | --- | --- | --- |
| Ingest pattern doc as KB entity | 8+ | `--target patterns` or `--target source:semops-data-patterns` | Manual `ingest_from_source.py` per doc |
| Neo4j graph sync after edge changes | 4+ | `--target architecture` (includes Neo4j sync) | Manual `sync_neo4j.py` |
| Re-run `ingest_architecture.py` after registry.yaml changes | 3+ | `--target architecture` | Manual script invocation |
| Pattern registration from pattern_v1.yaml | 5 new patterns | `--target patterns` | Manual `ingest_domain_patterns.py` |
| DESCRIBED_BY edge creation | 80 edges | `--target patterns` (as post-ingestion step) | Manual `create_described_by_edges.py` |
| Coverage view refresh | After every structural change | Automatic post-step in unified pipeline | Manual verification |

### Actions that remain manual (HITL or one-time)

| Action | Why Manual | Could Be Templated? |
| --- | --- | --- |
| SKOS edge definition | Requires domain judgment (broader/narrower/related) | Yes — SKOS edges could be declared in pattern_v1.yaml |
| Pattern doc authoring | Requires domain knowledge | Partially — template generation from registry.yaml |
| Enriched definition writing | Requires synthesis | Partially — LLM draft from pattern doc |
| UL entry authoring | Requires domain vocabulary | No — human defines the language |
| STRATEGIC_DDD.md updates | Requires architectural judgment | Partially — could flag where references are missing |
| Source doc rewrites (docs-pr) | Requires editorial judgment | No |

### Key finding

**~60% of remediation actions were ingestion or sync operations that a unified pipeline would handle.** The remaining ~40% were HITL content creation (pattern docs, UL entries, SKOS decisions) that require domain judgment. A unified `--target all` after a remediation session would eliminate the manual sync steps and ensure everything propagates.

The remediation cycle also revealed a missing ingestion path: **pattern docs in `docs/patterns/` have no source config.** Each was ingested ad-hoc. A source config for pattern docs (`config/sources/semops-data-pattern-docs.yaml`) would make this repeatable.

---

## What This Validates

This remediation cycle is the first operational use of the coherence measurement model (ADR-0014). Key validations:

1. **Goal-type continuum works.** Rule execution goals (structural checklists) found deterministic gaps. Criteria-based goals (semantic comparison) found drift. Both produced actionable remediation.
2. **A drives early-stage variance.** Pre-remediation A ranged 0.31–0.65 while S clustered 0.80–0.92. Structural availability is the most impactful dimension to improve first.
3. **Remediation is mostly deterministic.** ~85% of actions were Type 1 (create edge, ingest doc, register pattern). The hard semantic work was in analysis, not remediation.
4. **Discovery compounds.** The data-management-analogy analysis triggered 3 new pattern registrations that improved scores across multiple patterns simultaneously.
5. **The model differentiates meaningfully.** SC ranges from 0.40 (concept-only) to 0.96 (near-complete) — the spread reflects real maturity differences.
