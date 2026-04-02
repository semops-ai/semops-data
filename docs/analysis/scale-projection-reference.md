# Coherence Analysis: `scale-projection`

> **Date:** 2026-03-15
> **Pattern ID:** `scale-projection`
> **Pattern Label:** Scale Projection
> **Provenance:** 1P
> **Pattern Type:** concept
> **Lifecycle Stage:** active

---

## Phase 0: Reference Assembly

### 0.1 Pattern Identity

```
Pattern ID:    scale-projection
Label:         Scale Projection
Provenance:    1P
Pattern Type:  concept
Status:        active
Definition:    Validates domain coherence by projecting architecture to scale.
               Manual human-in-the-loop processes intentionally generate structured
               ML training data, creating a path from current-state to autonomous
               execution. Synthesizes RLHF, SECI, and data profiling with
               domain-model-aware scaling.
```

### 0.2 Canonical Layer (semops-data)

| Artifact | Location | Status |
|----------|----------|--------|
| Pattern doc | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md` | PRESENT (v2.1.0, 459 lines) |
| Registry capabilities | `config/registry.yaml` | PRESENT (3 capabilities implement) |
| STRATEGIC_DDD references | `docs/STRATEGIC_DDD.md` | PRESENT (6 references) |
| ARCHITECTURE.md references | `docs/ARCHITECTURE.md` | NOT FOUND |
| UL entry | `schemas/UBIQUITOUS_LANGUAGE.md` | PRESENT (pattern table + type example) |
| Fitness functions | `schemas/fitness-functions.sql` | NOT FOUND |

### 0.3 KB Layer (PostgreSQL + Qdrant + Neo4j)

#### Pattern Table

| Field | Value |
|-------|-------|
| `id` | `scale-projection` |
| `preferred_label` | Scale Projection |
| `provenance` | 1p |
| `metadata.$schema` | `pattern_registry_v1` |
| `metadata.pattern_type` | concept |
| `metadata.lifecycle_stage` | active |
| `metadata.documentation.primary` | `semops-data/docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md` |
| `metadata.documentation.related` | `semops-orchestrator/docs/decisions/ADR-0004-mirror-architecture.md` |
| `created_at` | 2026-02-10 |
| `updated_at` | 2026-03-10 |

#### Pattern Edges (SKOS / Adoption)

| Source | Predicate | Target |
|--------|-----------|--------|
| `scale-projection` | `extends` | `rlhf` |
| `scale-projection` | `extends` | `seci` |
| `scale-projection` | `extends` | `data-profiling` |
| `scale-projection` | `extends` | `ddd` |
| `agents-as-runtime` | `extends` | `scale-projection` |

#### Entity Table

The `scale-projection` entity exists as `entity_type: capability`.

| Field | Value |
|-------|-------|
| `id` | `scale-projection` |
| `title` | Scale Projection |
| `entity_type` | capability |
| `filespec` | `{}` (empty) |
| `metadata.status` | in_progress |
| `metadata.domain_classification` | core |
| `metadata.delivered_by_repos` | `[data-pr, semops-orchestrator]` |
| `metadata.implements_patterns` | `[scale-projection, rlhf, seci]` |

#### Entity Edges

| Source | Predicate | Target |
|--------|-----------|--------|
| `entity:scale-projection` | `delivered_by` | `entity:data-pr` |
| `entity:scale-projection` | `delivered_by` | `entity:semops-orchestrator` |
| `entity:scale-projection` | `delivered_by` | `entity:publisher-pr` |
| `entity:scale-projection` | `implements` | `pattern:scale-projection` |
| `entity:scale-projection` | `implements` | `pattern:rlhf` |
| `entity:scale-projection` | `implements` | `pattern:seci` |
| `entity:autonomous-execution` | `implements` | `pattern:scale-projection` |
| `entity:synthesis-simulation` | `implements` | `pattern:scale-projection` |
| `entity:style-learning` | `implements` | `pattern:scale-projection` |

#### Coverage Views

**pattern_coverage:**

| Field | Value |
|-------|-------|
| `content_count` | 0 |
| `capability_count` | 4 |
| `repo_count` | 3 |

**capability_coverage:**

| Field | Value |
|-------|-------|
| `domain_classification` | core |
| `primary_pattern_id` | None |
| `pattern_count` | 3 |
| `repo_count` | 3 |

#### DESCRIBED_BY Edges

None found.

#### Pattern Doc as KB Entity

The pattern doc (`docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md`) is NOT ingested as a content entity in PostgreSQL. However, the Qdrant vector store contains chunks from the docs-pr source copy (`docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/SEMANTIC_OPTIMIZATION/scale-projection.md`) with entity_id `scale-projection` — but this is associated with the **capability** entity, not a separate content entity for the pattern doc.

### 0.4 Related Content Entities

12 entities reference Scale Projection in their title or ID:

| Entity ID | Title | Type |
|-----------|-------|------|
| `issue-148-scale-vectors-resourcing-methodology` | Issue : Scale Projection: define scale vectors and resourcing methodology | content (session_note) |
| `issue-149-manual-scale-projection` | Issue : Scale Projection — Manual Projection Workflow | content (session_note) |
| `issue-157-repos-role-explicit-enterprise-scale` | Issue : GitHub (Forgejo) Repos role in `Explicit Enterprise` and Scale Projection | content (session_note) |
| `issue-96-scale-projection-feature-set` | Issue : Scale Projection Feature-Set (Coordination) | content (session_note) |
| `semops-orchestrator-issue-163` | Issue : Infrastructure tier taxonomy | content (issue) |
| `semops-orchestrator-issue-157` | Issue : Repos role in Explicit Enterprise and Scale Projection | content (issue) |
| `semops-orchestrator-issue-149` | Issue : Scale Projection: manual projection workflow | content (issue) |
| `semops-orchestrator-issue-148` | Issue : Scale Projection: define scale vectors and resourcing methodology | content (issue) |
| `semops-orchestrator-issue-147` | Issue : Scale Projection: absorb generate-proposed / promote-proposed | content (issue) |
| `semops-orchestrator-issue-108` | Issue : Scale Projection to Public Repos | content (issue) |
| `semops-orchestrator-issue-96` | Issue : Scale Projection: domain coherence validation | content (issue) |

### 0.5 Cross-Repo Content

#### docs-pr (6 files)

- `docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/SEMANTIC_OPTIMIZATION/scale-projection.md` (source doc)
- `docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/scale-projection.md` (cross-reference)
- `docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-systems-architecture-map.md`
- `docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/README.md`
- `docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/SEMANTIC_OPTIMIZATION/working-with-patterns.md`
- `docs/examples/intake-pattern-project-run.md`

#### sites-pr (1 file)

- `apps/semops/content/pages/explicit-architecture.mdx`

#### publisher-pr (13 files)

Includes source framework copies, drafts, blog content, and ARCHITECTURE.md references.

### 0.6 Timeline

| Date | Event | Track |
|------|-------|-------|
| 2026-01-26 | : Refine Scale-Projection Pattern (issue created) | Concept |
| 2026-02-07 | : Scale Projection feature-set (issue created) | Implementation |
| 2026-02-10 | Pattern row created in DB | Implementation |
| 2026-02-20 | Issue : Scale vectors and resourcing methodology | Concept |
| 2026-02-21 | Capability entity created in DB | Implementation |
| 2026-02-21 | : Define scale vectors (issue created) | Concept |
| 2026-02-21 | : Manual projection workflow (issue created) | Implementation |
| 2026-02-22 | : CRM as worked example | Implementation |
| 2026-02-26 | : Repos role in Scale Projection | Concept |
| 2026-02-27 | : Scale-vector-to-repo mapping | Implementation |
| 2026-02-28 | : Infrastructure tier taxonomy | Concept |
| 2026-03-08 | Pattern doc received in semops-data (24 domain-pattern docs) | Convergence |
| 2026-03-10 | Registry moved to YAML authority (registry.yaml) | Implementation |

**Promotion type:** Recognition promotion — the concept was explored in issues and session notes well before formal pattern registration.

**Convergence point:** 2026-03-08, when the pattern doc was received into semops-data alongside the YAML registry migration.

---

## Phase 2: Type 1 — Structural Coherence

### Structural Checklist

| # | Artifact | Status | Notes |
|---|----------|--------|-------|
| 1 | `pattern` table row | **PRESENT** | Full definition, provenance 1p, pattern_type concept, lifecycle active |
| 2 | `pattern_edge` SKOS edges | **PRESENT** | 4 outgoing `extends` edges (rlhf, seci, data-profiling, ddd). 1 incoming (agents-as-runtime extends scale-projection) |
| 3 | Capability IMPLEMENTS edges | **PRESENT** | 4 capabilities implement: scale-projection (self), autonomous-execution, synthesis-simulation, style-learning |
| 4 | `pattern_coverage` view | **PRESENT** | content_count=0, capability_count=4, repo_count=3 |
| 5 | Pattern doc file | **PRESENT** | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md` (v2.1.0, comprehensive) |
| 6 | Pattern doc as KB entity | **GAP** | Not ingested as a content entity. Chunks exist in Qdrant from docs-pr source, but no dedicated content entity in PG for the semops-data pattern doc |
| 7 | Registry.yaml capabilities | **PRESENT** | 3 capabilities listed: `scale-projection`, `synthesis-simulation`, `style-learning` |
| 8 | STRATEGIC_DDD.md | **PRESENT** | 6 references across capability tables, pattern type examples, and the infrastructure-up/domain-down framing |
| 9 | ARCHITECTURE.md | **GAP** | No references found in `docs/ARCHITECTURE.md` |
| 10 | UL entry | **PRESENT** | Listed in pattern table and as pattern_type example |
| 11 | DESCRIBED_BY edges | **GAP** | Zero DESCRIBED_BY edges. No links from pattern to content entities |
| 12 | Stale entities check | **PRESENT** | No stale/ghost entities found. All entity IDs resolve cleanly |
| 13 | Registry/DB alignment | **MISMATCH** | Registry: `delivered_by: [data-pr, semops-orchestrator]`. DB edges: `delivered_by` to `[data-pr, semops-orchestrator, publisher-pr]`. DB has an extra publisher-pr edge not in registry |
| 14 | content_count | **GAP** | pattern_coverage shows content_count=0. Despite 11+ content entities referencing Scale Projection, none are linked via edges |
| 15 | capability_coverage.primary_pattern_id | **GAP** | `primary_pattern_id` is None. Should be `scale-projection` given the capability shares its name |

### SKOS Edge Alignment

Pattern doc `Derives From`: `ddd`, `rlhf`, `seci`, `data-profiling`
DB `extends` edges: `ddd`, `rlhf`, `seci`, `data-profiling`

**Verdict: ALIGNED** — all four derivation links match.

### Registry/DB Capability Alignment

Registry capabilities implementing `scale-projection`:

| Capability | Registry | DB IMPLEMENTS edge |
|------------|----------|-------------------|
| `scale-projection` | Yes | Yes |
| `synthesis-simulation` | Yes | Yes |
| `style-learning` | Yes | Yes |
| `autonomous-execution` | No (not in registry as implementing scale-projection) | Yes |

**Mismatch:** `autonomous-execution` has an IMPLEMENTS edge to `scale-projection` in the DB but is not listed as implementing `scale-projection` in registry.yaml. Check: the UL entry lists autonomous-execution as a capability for scale-projection, but registry.yaml does not include `scale-projection` in autonomous-execution's `implements_patterns`.

### Structural Verdict

**ACTIVE — INCOMPLETE**

The pattern is fully registered, actively developed (Project P25), has strong SKOS lineage, and 4 capabilities implementing it. However, it has 5 structural gaps:

1. Pattern doc not ingested as KB content entity
2. No ARCHITECTURE.md references
3. Zero DESCRIBED_BY edges (content_count=0 in coverage)
4. Registry/DB `delivered_by` mismatch (publisher-pr in DB, not in registry)
5. `autonomous-execution` IMPLEMENTS edge in DB not reflected in registry

**Gap count: 5**

---

## Phase 3: Type 2 — Semantic Coherence

### Framing Comparison

| Artifact | Core Framing | Audience | Concreteness |
|----------|-------------|----------|-------------|
| Pattern doc (v2.1.0) | Validation technique: project architecture to scale, measure gap. Infrastructure delta = coherence diagnostic | Internal/technical | Very high — worked example, scale vectors, tier taxonomy, K8s manifests |
| UL entry | "Validate domain coherence by projecting architecture to scale. Manual HITL processes generate ML training data." | Domain vocabulary | Medium — definition plus capability mapping |
| STRATEGIC_DDD | "Scale projection works infrastructure-up — from actual implementation, project to scale and check" | Architecture governance | High — positioned as complement to reference architecture (domain-down) |
| KB chunks (docs-pr source) | "Uses the path of infrastructure scaling as a diagnostic — testing and validating core business logic" | Theory/research | Medium — overview-level, pre-v2.1.0 depth |
| Website (semops.ai) | Referenced in explicit-architecture page | External/accessible | Low — cross-reference only |
| Session notes | Operational work logs: scale vectors, repo-as-scaling-unit, infrastructure tiers | Internal process | Very high — concrete discoveries during projection work |

### Alignment Dimensions

| Dimension | Status | Assessment |
|-----------|--------|------------|
| **Thesis alignment** | **ALIGNED** | All artifacts consistently frame Scale Projection as domain coherence validation through infrastructure projection. The "infrastructure delta should be wrapper and plumbing" thesis is consistent across pattern doc, UL, STRATEGIC_DDD, issues, and session notes |
| **Scope consistency** | **ALIGNED** | Pattern doc v2.1.0 has absorbed Mirror Architecture, Scale Vectors, Infrastructure Tier Taxonomy, Repo as Scaling Unit, and Resourcing Methodology. This scope expansion is deliberate (per ) and documented. All artifacts track the expanded scope |
| **Concrete examples** | **ALIGNED** | Pattern doc includes worked example (semops-data projection, 2026-03-08). Session notes contain additional worked examples. GAPS.md template and decision criteria template provided |
| **Cross-pattern relationships** | **MINOR DRIFT** | Pattern doc lists `Derives From: ddd, rlhf, seci, data-profiling`. DB pattern_edges match. But the UL entry lists capabilities `style-learning, synthesis-simulation, autonomous-execution` while the pattern doc does not reference these downstream capabilities explicitly. The pattern doc is self-contained as theory; the architecture binding happens only in STRATEGIC_DDD and registry |
| **Implementation binding** | **GAP** | ARCHITECTURE.md does not reference scale-projection at all. STRATEGIC_DDD covers it well, but ARCHITECTURE.md — the primary structural document for this repo — is silent |
| **Session note mining** | **CONCEPT AHEAD** | Session notes from , , ,  contain rich operational insights (Four Data System Types lens, OLTP/OLAP split, human/agent scaling distinction) that HAVE been folded into the pattern doc v2.1.0. This is a success case — session note mining was already done effectively |
| **Audience appropriateness** | **MINOR DRIFT** | The docs-pr source copy (`SEMANTIC_OPTIMIZATION/scale-projection.md`) appears to be an older version of the pattern doc. The semops-data copy is v2.1.0 with significant additions (worked example, infrastructure tiers). The docs-pr version may be stale |

### Drift Findings

**Finding 1: ARCHITECTURE.md silence**

- **What:** `docs/ARCHITECTURE.md` contains zero references to `scale-projection`. Given that scale-projection is a core 1P concept with 4 capabilities implementing it, this is a significant gap.
- **Where:** `semops-data/docs/ARCHITECTURE.md`
- **Direction:** Implementation ahead of concept — the pattern is registered, has edges, capabilities, and a comprehensive pattern doc, but the repo's primary architecture document does not mention it
- **Impact:** Agents reading ARCHITECTURE.md to understand semops-data's role will not discover scale-projection as a governing concept. The worked example in the pattern doc projects semops-data's capabilities, but ARCHITECTURE.md doesn't acknowledge this relationship
- **Remediation:** Add scale-projection reference to ARCHITECTURE.md, likely in the context of how semops-data's architecture was validated via projection

**Finding 2: docs-pr source copy may be stale**

- **What:** docs-pr contains `scale-projection.md` in TWO locations: `SEMANTIC_OPTIMIZATION/` and `EXPLICIT_ARCHITECTURE/`. The semops-data copy is v2.1.0 with extensive content (scale vectors, infrastructure tiers, worked example). The docs-pr copy may not have received these updates
- **Where:** `docs-pr/docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/SEMANTIC_OPTIMIZATION/scale-projection.md` and `EXPLICIT_ARCHITECTURE/scale-projection.md`
- **Direction:** Concept ahead in semops-data; docs-pr may be behind
- **Impact:** Published content and KB chunks derived from docs-pr source will present an older, less complete version of the pattern
- **Remediation:** Sync docs-pr copies with semops-data v2.1.0. Resolve the dual-location question (SEMANTIC_OPTIMIZATION vs EXPLICIT_ARCHITECTURE — which is canonical?)

**Finding 3: delivered_by mismatch (publisher-pr)**

- **What:** The DB has a `delivered_by` edge from `scale-projection` to `publisher-pr`, but registry.yaml lists `delivered_by: [data-pr, semops-orchestrator]`. publisher-pr delivers `style-learning` which implements `scale-projection`, but the `scale-projection` capability itself is not delivered by publisher-pr
- **Where:** `edge` table vs `config/registry.yaml`
- **Direction:** Lateral drift — DB has an edge that registry does not
- **Impact:** Coverage views show repo_count=3 instead of 2, overstating delivery scope
- **Remediation:** Either remove the publisher-pr `delivered_by` edge from DB, or add publisher-pr to the registry if the delivery relationship is intentional (via style-learning's transitive delivery)

**Finding 4: autonomous-execution IMPLEMENTS edge without registry backing**

- **What:** `autonomous-execution` has an IMPLEMENTS edge to `pattern:scale-projection` in the DB, but `autonomous-execution` in registry.yaml lists `implements_patterns: [explicit-enterprise]` — no mention of `scale-projection`
- **Where:** `edge` table vs `config/registry.yaml`
- **Direction:** DB ahead of registry
- **Impact:** Overstates pattern reach in coverage views (capability_count=4 vs registry's 3)
- **Remediation:** Either update registry.yaml to add `scale-projection` to `autonomous-execution.implements_patterns`, or remove the stale edge from DB. The UL entry does list `autonomous-execution` as a capability for scale-projection, so this may be an intentional relationship that registry hasn't caught up with

**Finding 5: content_count = 0 despite rich content corpus**

- **What:** pattern_coverage shows `content_count: 0`, meaning no content entities are linked to the pattern via edges. Yet there are 11+ content entities (session notes, issues) that are semantically about scale-projection
- **Where:** `edge` table — missing `DESCRIBED_BY` or similar edges from content entities to pattern
- **Direction:** Implementation behind concept — the content exists but isn't connected
- **Impact:** Agents querying "what content describes scale-projection?" via graph traversal get nothing. Only semantic search (vector) discovers these relationships. This defeats the deterministic query surface
- **Remediation:** Create `DESCRIBED_BY` edges from the 11 content entities to the `scale-projection` pattern. This would bring content_count from 0 to 11+

---

## Phase 4: Summary

### Structural Verdict

**ACTIVE — INCOMPLETE**

### Counts

| Metric | Count |
|--------|-------|
| Structural gaps | 5 |
| Semantic drift findings | 5 |
| Capabilities implementing | 4 (DB) / 3 (registry) |
| Content entities (linked) | 0 |
| Content entities (unlinked) | 11+ |
| Cross-repo file references | 20+ |

### Key Findings

1. **The pattern doc is excellent.** At v2.1.0 with 459 lines, it is one of the most comprehensive pattern docs in the system. It includes worked examples, scale vectors, infrastructure tier taxonomy, K8s projection methodology, and resourcing framework. Session note insights have been effectively mined and folded back.

2. **Graph connectivity is weak.** Despite rich content, zero DESCRIBED_BY edges mean the pattern is an island in the graph. Capabilities connect to it, but no content does. This makes scale-projection invisible to graph-traversal queries.

3. **Registry/DB drift exists in two places.** publisher-pr delivered_by edge and autonomous-execution IMPLEMENTS edge are in DB but not registry. The registry is supposed to be the YAML authority (per ADR from 2026-03-10 migration), so either DB needs pruning or registry needs updating.

4. **ARCHITECTURE.md is silent.** The repo's primary architecture document does not mention scale-projection despite it being a core 1P pattern with a worked example projecting this very repo's capabilities.

5. **Dual source copies in docs-pr.** The pattern exists in both SEMANTIC_OPTIMIZATION/ and EXPLICIT_ARCHITECTURE/ directories in docs-pr, and may be stale relative to the semops-data v2.1.0 copy.

### Remediation Items

#### Fix in place (minor)

- [ ] Add `primary_pattern_id: scale-projection` to the capability entity metadata

#### Spin off issues (significant work)

- [ ] **Create DESCRIBED_BY edges** — link 11+ content entities to scale-projection pattern (content_count: 0 -> 11+)
- [ ] **Ingest pattern doc as KB content entity** — semops-data/docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md should be searchable as content_type "pattern"
- [ ] **Add ARCHITECTURE.md references** — mention scale-projection in semops-data's ARCHITECTURE.md
- [ ] **Resolve registry/DB drift** — reconcile publisher-pr delivered_by edge and autonomous-execution IMPLEMENTS edge between registry.yaml and DB
- [ ] **Sync docs-pr source copies** — update docs-pr scale-projection.md to v2.1.0 and resolve dual-location (SEMANTIC_OPTIMIZATION vs EXPLICIT_ARCHITECTURE)

#### Defer

- [ ] Fitness function coverage for scale-projection (no fitness function currently checks this pattern specifically)
- [ ] Website content expansion beyond cross-reference mention
