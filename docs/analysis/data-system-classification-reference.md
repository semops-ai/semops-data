# Data System Classification — Cross-Layer Reference Document

> **Purpose:** Assemble all content related to "Data System Classification" across every layer for coherence analysis
> **Generated:** 2026-03-15
> **Organization:** Timeline → Canonical → Source → Published → KB → Governance
> **Note:** This analysis covers BOTH `data-system-classification` (canonical) and `four-data-system-types` (pre-rename ghost)
> **HITL corrections applied:** (1) `dbt` is wrong 3P parent — should be `fundamentals-of-data-engineering` (Reis & Housley); (2) `reference-generation` capability was missing `data-system-classification` — fixed in registry.yaml this session; (3) `mirror-architecture` is a revived pattern (not a capability) — already wired via 

---

## 0. Artifact Timeline

### Source material era (Jan–Feb 2026)

| Date | Event | Repo | Artifact |
|------|-------|------|----------|
| 2026-01-31 | Source docs authored | docs-pr | `STRATEGIC_DATA/data-system-classification.md` — primary concept document |
| 2026-02-17 | Source docs renamed | docs-pr | Directory rearrangements for Explicit Architecture rename |
| 2026-02-21 | DDD validation authored | docs-pr | `data-system-classification-ddd-validation.md` — full DDD bounded context validation  |

### Pattern registration + rename (Mar 2026)

| Date | Event | Repo | Artifact |
|------|-------|------|----------|
| 2026-03-05 | 3P lineage patterns registered | semops-orchestrator | Issue  — `togaf`, `dbt` registered as parents for `data-system-classification` |
| 2026-03-07 | Rename decision | semops-orchestrator | Issue : `four-data-system-types` → `data-system-classification` |
| 2026-03-07 | Session note rename | semops-data | 19 text replacements across 7 files (ISSUE-169 session notes) |
| 2026-03-08 | Pattern doc migrated | semops-data | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md` received from semops-orchestrator  |
| 2026-03-08 | Pattern registered in pattern_v1.yaml | semops-orchestrator | `data-system-classification` added (Issue ) |
| 2026-03-09 | Publisher references updated | semops-data | Updated all docs-pr references to publisher-pr  |
| 2026-03-09 | Rename session note created | semops-data | `ISSUE-169-rename-four-data-system-types.md` |
| 2026-03-10 | Registry YAML created | semops-data | `registry.yaml` — 3 capabilities mapped to `data-system-classification` |
| 2026-03-10 | STRATEGIC_DDD references | semops-data | Pattern referenced in capability tables (, , ) |

### Concept-to-Pattern Promotion Lineage

**Promotion type: Recognition promotion** — concept material existed from Jan 31 as `four-data-system-types`; formally registered as pattern `data-system-classification` on Mar 8.

#### Two tracks

**Concept track** — `data-system-classification.md`, `data-system-classification-ddd-validation.md`, Strategic Data README, surface analysis, data silos docs, data-systems-essentials. Describes the four types, their physics, and DDD validation. Audience: humans and agents learning the framework.

**Implementation track** — `data-system-classification.md` pattern doc, `registry.yaml` capabilities, `STRATEGIC_DDD.md` references, `concept-pattern-map.yaml` mappings, stand-in connector typing. Pattern operating as architecture. Audience: the system and agents querying it.

**Convergence point:** 2026-03-08 — pattern registered in `pattern_v1.yaml` AND pattern doc created in `docs/patterns/`.

---

## 1. Canonical Layer (semops-data)

| Artifact | Location | Status |
|----------|----------|--------|
| Pattern doc | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md` | PRESENT (v2.0.0, 162 lines) |
| Registry capabilities | `config/registry.yaml` — 3 capabilities + bolt-on typing comment | PRESENT |
| Concept-pattern map | `config/mappings/concept-pattern-map.yaml` — 7 entries | PRESENT |
| STRATEGIC_DDD references | `docs/STRATEGIC_DDD.md` — 8 references | PRESENT |
| ARCHITECTURE.md references | `docs/ARCHITECTURE.md` | **GAP** — not referenced |
| UL entry | `schemas/UBIQUITOUS_LANGUAGE.md` | **GAP** — not present |
| Fitness functions | `schemas/fitness-functions.sql` | **GAP** — not referenced |

### Registry capabilities (3)

| Capability | Patterns (including data-system-classification) |
|------------|------------------------------------------------|
| `data-due-diligence` | ddd, data-modeling, explicit-architecture, business-domain, togaf, dama-dmbok, dcam, apqc-pcf, **data-system-classification** |
| `business-model-synthesis` | bizbok, bizbok-ddd-derivation, **data-system-classification**, ddd |
| `system-primitive-decomposition` | explicit-architecture, explicit-enterprise, **data-system-classification** |

All 3 delivered by `semops-research`.

---

## 2. Source Material Layer

### docs-pr (20+ files referencing)

Primary source documents:

| File | Content Type | Notes |
|------|-------------|-------|
| `STRATEGIC_DATA/data-system-classification.md` | concept | Primary concept document |
| `STRATEGIC_DATA/data-system-classification-ddd-validation.md` | concept | Full DDD bounded context validation |
| `STRATEGIC_DATA/README.md` | concept | Strategic Data pillar hub — references four types |
| `STRATEGIC_DATA/data-systems-essentials.md` | concept | Analytics Data Systems — child concept |
| `STRATEGIC_DATA/analytics-systems-evolution.md` | concept | OLTP/OLAP evolution history |
| `STRATEGIC_DATA/data-silos.md` | concept | Data silos — references four types table |
| `STRATEGIC_DATA/surface-analysis.md` | concept | Surface patterns per data system type |
| `STRATEGIC_DATA/data-is-organizational-challenge.md` | concept | Related links to four types |
| `STRATEGIC_DATA/data-systems-architecture-map.md` | concept | Architecture map |
| `STRATEGIC_DATA/business-analytics-patterns.md` | concept | Business analytics patterns |
| `EXPLICIT_ARCHITECTURE/scale-projection.md` | concept | Scale projection references |
| `EXPLICIT_ARCHITECTURE/ddd-data-architecture.md` | concept | DDD data architecture |
| `EXPLICIT_ARCHITECTURE/explicit-enterprise.md` | concept | Enterprise system type references |

### publisher-pr (20+ files — mirrors docs-pr source)

All source material mirrored under `docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/`.

### semops-research (significant presence)

| File | Content Type | Notes |
|------|-------------|-------|
| `docs/research/data-system-classification.md` | research | Industry diagnostic calibration |
| `docs/research/data-due-diligence-method.md` | research | Due diligence methodology |
| `docs/research/business-model-synthesis.md` | research | Business model analysis |
| `docs/research/business-analytics-patterns.md` | research | Analytics pattern research |
| `docs/research/ddd-data-architecture.md` | research | DDD data architecture research |
| `docs/research/decision-matrix.md` | research | Decision matrix |
| `docs/research/data-systems-vendor-comparison.md` | research | Vendor comparison |
| `src/research_toolkit/diligence/models.py` | code | Data model (system type classification in code) |
| `src/research_toolkit/diligence/lookups.py` | code | Lookup tables |
| `src/research_toolkit/diligence/reference.py` | code | Reference data |

---

## 3. Published Layer

### sites-pr

| File | Notes |
|------|-------|
| `apps/semops/content/pages/strategic-data.mdx` | References four data system types in page content |
| `.next/` build artifacts (4 files) | Built/cached versions |

### semops-docs (public repo)

| File | Notes |
|------|-------|
| `STRATEGIC_DATA/README.md` | Published mirror — references four types (chunk found in `published` corpus) |

---

## 4. KB Layer

### Pattern table — DUAL REGISTRATION (key finding)

| Pattern ID | Preferred Label | Definition | Provenance | Coverage | Edges |
|------------|----------------|------------|------------|----------|-------|
| `data-system-classification` | Data System Classification | Correct (Analytics, Application, Enterprise Work, Enterprise Record) | 1p | 0 content, 3 capabilities, 1 repo | 3 EXTENDS (dbt, ddd, togaf) |
| `four-data-system-types` | Four Data System Types | **WRONG** ("operational, analytical, streaming, and ML/AI systems") | 1p | 4 content, 0 capabilities, 0 repos | **NONE** |

**Critical:** The old `four-data-system-types` pattern row still exists in the database but is NOT in `pattern_v1.yaml` (the YAML authority). Its definition is incorrect — it describes "streaming" and "ML/AI" systems instead of the actual four types (Analytics, Application, Enterprise Work, Enterprise Record). This is a ghost entity from a pre-rename auto-discovery ingestion.

### Graph neighbors

**`data-system-classification`** (6 neighbors — clean):
- 3 outgoing EXTENDS → `dbt`, `ddd`, `togaf` (all Pattern)
- 3 incoming IMPLEMENTS ← `system-primitive-decomposition`, `business-model-synthesis`, `data-due-diligence` (all Capability)

**`four-data-system-types`** (39 neighbors — rich but orphaned):
- 5 outgoing Concept edges: `semantic-operations`, `domain-driven-design` (×2), `ddd-acl-governance-aas`, `semantic-coherence`, `data-silos`
- 34 incoming Entity edges: issues from data-pr, docs-pr, semops-research, semops-orchestrator + content entities (`data-systems-essentials`, `symbiotic-enterprise`, `everything-is-data`, `data-silos`, `data-is-organizational-challenge`, `surface-analysis`)

The old pattern node is the **hub** of a rich concept graph. The new pattern node has only implementation edges. The concept-to-implementation linkage is broken.

### Entity search

- `four-data-system-types` entity found as `content` type in `core_kb` corpus (similarity 0.68)
- `four-data-system-types-ddd-validation` entity found (similarity 0.61)
- `data-system-classification` pattern doc is **NOT** found via `content_type: "pattern"` search — the pattern doc entity may not be ingested

### Chunk search (passage-level)

Strong hits from:
- `four-data-system-types.md` (0.65) — "Analytics Data System Type, Application Data System Type, Enterprise Work System Type, Enterprise Record System Type"
- `guide-zettelkasten-hubs.md` (0.65) — "Four Systems of Source Data" section references the types
- `data-silos.md` (0.62) — Links to four types with governance table
- `surface-analysis.md` (0.62) — "Data System Types Lens" section maps surfaces to types
- `strategic-data README.md` (0.57) — Hub document tables

---

## 5. Governance History

### GitHub Issues

| Issue | Repo | Role |
|-------|------|------|
|  | semops-orchestrator | Register togaf, dbt as 3P lineage parents |
|  | semops-orchestrator | Rename `four-data-system-types` → `data-system-classification` |
|  | semops-orchestrator | Domain pattern docs migration to semops-data |

### Session Notes (cross-repo)

| File | Repo |
|------|------|
| `ISSUE-169-rename-four-data-system-types.md` | semops-data, docs-pr, semops-research, semops-orchestrator, data-pr |
| `ISSUE-138-bridge-docs-pr-to-pattern-layer.md` | semops-data |
| `ISSUE-159-register-capabilities-pattern-updates.md` | semops-data |
| `ISSUE-164-audit-pattern-type-lifecycle.md` | semops-data |

### Related graph entities (from `four-data-system-types` neighbors)

Issues linked in graph: ``, ``, ``, ``, ``, ``, ``, ``, ``, ``, ``, ``

---

## 6. Type 1 — Structural Coherence

| Artifact | Check | Status |
|----------|-------|--------|
| `pattern` table row | `data-system-classification` exists with correct definition | **PRESENT** |
| `pattern_edge` SKOS | 3 EXTENDS edges (dbt, ddd, togaf) — but `dbt` is wrong parent (see Finding 6) | **MISMATCH** |
| Capability IMPLEMENTS | 3 edges — matches registry.yaml count (3) | **PRESENT** |
| `pattern_coverage` view | Expected: 0 content, 3 capabilities, 1 repo | **PRESENT** (but 0 content is a gap) |
| Pattern doc file | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md` exists (v2.0.0) | **PRESENT** |
| Pattern doc as KB entity | NOT found via content_type "pattern" search | **GAP** — pattern doc not ingested as entity |
| Registry.yaml entries | 3 capabilities listed | **PRESENT** |
| STRATEGIC_DDD.md | 8 references | **PRESENT** |
| ARCHITECTURE.md | Not referenced | **GAP** |
| UL entry | Not present | **GAP** |
| DESCRIBED_BY edges | No DESCRIBED_BY edges from pattern to concept entities | **GAP** — concept content is linked to ghost `four-data-system-types` instead |
| Stale entities | `four-data-system-types` ghost pattern with wrong definition, 39 graph edges, 4 content entities | **MISMATCH** — critical cleanup needed |
| Registry/DB alignment | 3 capabilities match | **PRESENT** |
| `skos:related` to `medallion-architecture` | Declared in pattern doc but NOT in `pattern_edge` table | **MISMATCH** — missing edge |

### Structural Verdict: **ACTIVE — INCOMPLETE**

The pattern is registered, has capabilities, SKOS lineage edges, and appears in STRATEGIC_DDD and registry.yaml. However, 7 structural issues exist:

1. **Ghost pattern** — `four-data-system-types` still in DB with wrong definition and 39 orphaned graph edges
2. **Broken concept linkage** — all concept content (4 entities, 34 issue/entity edges) points to ghost pattern, not canonical `data-system-classification`
3. **Wrong SKOS parent** — `dbt` (tool) is listed as parent; should be `fundamentals-of-data-engineering` (Reis & Housley framework, 3P, unregistered)
4. **Pattern doc not ingested** — `data-system-classification.md` not findable via KB search as pattern content
5. **Missing ARCHITECTURE.md reference**
6. **Missing UL entry**
7. **Missing `skos:related` edge** to `medallion-architecture` (declared in doc but not in `pattern_edge`)

---

## 7. Type 2 — Semantic Coherence

### Framing Comparison

| Artifact | Core framing | Audience | Concreteness |
|----------|-------------|----------|-------------|
| Pattern doc (`data-system-classification.md`) | 4 bounded contexts with distinct DDD physics; industry mix diagnostic | Internal/technical | High — DDD validation table, SSOT cascade, integration patterns |
| Source doc (`data-system-classification.md`) | 4 system types with query interfaces | Theory/research | Medium — definitions + related links |
| DDD validation doc | Bounded context proof via differential DDD mapping | Theory/research | Very high — per-type DDD concept mapping |
| `STRATEGIC_DDD.md` | Capabilities and bolt-on system typing | Architecture governance | Implementation — capability × pattern matrix |
| `DATA_ARCHITECTURE.md` | Macro-level vocabulary, 4 system categories | Architecture | Medium — positional (where types fit in data architecture layers) |
| Ghost pattern definition | "operational, analytical, streaming, ML/AI" | KB agents | **WRONG** — completely inaccurate type names |
| Website (`strategic-data.mdx`) | Strategic Data pillar | External/accessible | Low — pillar-level framing |

### Alignment Dimensions

| Dimension | Status | Notes |
|-----------|--------|-------|
| Thesis alignment | **MISALIGNED** | Ghost definition says "streaming and ML/AI systems"; canonical says "Enterprise Work and Enterprise Record". Two different frameworks. |
| Scope consistency | **ALIGNED** | All canonical artifacts agree on scope — macro classification framework |
| Concrete examples | **ALIGNED** | Pattern doc and DDD validation doc both provide SemOps-specific examples (which repo = which type) |
| Cross-pattern relationships | **PARTIAL** | `medallion-architecture` declared related in pattern doc but no edge. `stand-in-connector` correctly cross-references. `dbt` listed as parent but should be `fundamentals-of-data-engineering`. |
| Implementation binding | **GAP** | ARCHITECTURE.md does not name the pattern |
| Session note mining | **ALIGNED** | ISSUE-164 flagged the dedup need; ISSUE-169 tracked the rename. Both captured. |
| Audience-appropriate versions | **INTENTIONAL** | Theory docs explain the framework; pattern doc adds DDD validation; STRATEGIC_DDD shows operational mapping. This is intentional layering. |

### Drift Findings

**Finding 1: Ghost pattern with wrong definition (CRITICAL)**

- **What:** `four-data-system-types` still exists as a registered pattern in PostgreSQL with definition "operational, analytical, streaming, and ML/AI systems" — this is factually wrong. The actual types are Analytics, Application, Enterprise Work, Enterprise Record. The ghost definition appears to be from an early auto-discovery ingestion before the framework was finalized.
- **Where:** `pattern` table, Neo4j graph (39 edges)
- **Direction:** Implementation ahead of concept — the rename happened but the old DB/graph entries weren't cleaned up
- **Impact:** Any agent querying `four-data-system-types` gets wrong types. The 4 concept content entities and 34 issue/entity graph edges point to the ghost, making them invisible from the canonical pattern. The rich concept graph is orphaned.
- **Remediation:**
  1. Delete `four-data-system-types` from `pattern` table
  2. Re-point all 4 content entity edges to `data-system-classification`
  3. Migrate or re-create the 39 Neo4j graph edges to point to `data-system-classification`
  4. Re-run ingestion for concept content with correct pattern attribution

**Finding 2: Pattern doc not ingested as KB entity**

- **What:** `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md` exists as a file but is not findable via `search_knowledge_base` with content_type "pattern"
- **Where:** KB / Qdrant
- **Direction:** Implementation ahead of concept — the doc exists but wasn't ingested
- **Impact:** Agents searching for pattern documentation won't find it. Content count is 0 for the canonical pattern.
- **Remediation:** Ingest the pattern doc into the KB

**Finding 3: Missing ARCHITECTURE.md reference**

- **What:** `data-system-classification` is not mentioned in `docs/ARCHITECTURE.md` despite being a structural pattern that types bolt-on systems
- **Where:** ARCHITECTURE.md
- **Direction:** Implementation ahead of concept
- **Impact:** Architecture doc doesn't reflect the full set of architectural patterns in use
- **Remediation:** Add reference to ARCHITECTURE.md where bolt-on systems or data architecture layers are discussed

**Finding 4: Missing UL entry**

- **What:** No entry for `data-system-classification` in `schemas/UBIQUITOUS_LANGUAGE.md` despite being a 1P classification framework with specific domain vocabulary (the four type names, "system mix", "SSOT failure cascade")
- **Where:** UBIQUITOUS_LANGUAGE.md
- **Direction:** Concept ahead of implementation — the vocabulary is defined in the pattern doc but not formalized in the UL
- **Impact:** The UL is incomplete; agents using UL for term resolution won't find the types
- **Remediation:** Add entry with definition and the four type names as sub-terms

**Finding 5: Missing `skos:related` edge to `medallion-architecture`**

- **What:** Pattern doc declares `skos:related → medallion-architecture` but this edge doesn't exist in `pattern_edge` table
- **Where:** `pattern_edge` table
- **Direction:** Concept ahead of implementation — doc says it, DB doesn't have it
- **Impact:** Minor — the relationship exists conceptually but not navigable via graph queries
- **Remediation:** Add `skos:related` edge in `pattern_v1.yaml` and re-ingest

**Finding 6: Wrong 3P parent — `dbt` should be `fundamentals-of-data-engineering` (HITL correction)**

- **What:** Pattern doc lists `skos:broader → dbt` but `dbt` is the *tool* that implements Reis & Housley's analytics anatomy. The actual 3P parent is the *framework* — "Fundamentals of Data Engineering" (Reis & Housley) — which provides the 7 Components × 5 Lifecycle × 6 Undercurrents anatomy that `data-system-classification` builds on. The concept content already exists as `data-engineering-core-framework` (entity in docs-pr) and `three-five-seven-data-systems` (entity in docs-pr), but the 3P pattern itself (`fundamentals-of-data-engineering`) is not registered.
- **Where:** `pattern_edge` table, `pattern_v1.yaml`, pattern doc SKOS edges
- **Direction:** Lateral drift — implementation names the wrong parent at the wrong abstraction level
- **Impact:** SKOS lineage is incorrect. `dbt` is a tool that implements the framework; the framework is what `data-system-classification` actually extends. Agents reasoning about lineage will trace to the tool instead of the intellectual foundation.
- **Remediation:**
  1. Register `fundamentals-of-data-engineering` as 3P pattern in `pattern_v1.yaml` (source: Reis & Housley book)
  2. Replace `skos:broader → dbt` with `skos:broader → fundamentals-of-data-engineering` on `data-system-classification`
  3. Add `dbt adopts fundamentals-of-data-engineering` edge (dbt implements the framework)
  4. Update pattern doc SKOS edges section
  5. Link `data-engineering-core-framework` and `three-five-seven-data-systems` concept entities to the new pattern via DESCRIBED_BY
- **Note:** These concepts were all just theory until semops-research built out the due diligence analysis features — at which point the concept-to-implementation lineage solidified. The `dbt` attribution was likely set during early registration when the distinction between tool and framework wasn't yet clear.

---

## 8. Remediation Summary

### From Type 1 (structural) — 7 items

| # | Item | Severity | Action |
|---|------|----------|--------|
| S1 | Ghost pattern `four-data-system-types` in DB | HIGH | Delete pattern row, migrate edges, clean up graph |
| S2 | 39 orphaned Neo4j edges on ghost node | HIGH | Re-point to `data-system-classification` or delete stale |
| S3 | Wrong 3P parent (`dbt` → should be `fundamentals-of-data-engineering`) | HIGH | Register new 3P pattern, fix SKOS edges |
| S4 | Pattern doc not ingested as KB entity | MEDIUM | Run ingestion for `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md` |
| S5 | Missing ARCHITECTURE.md reference | LOW | Add reference |
| S6 | Missing UL entry | MEDIUM | Add `data-system-classification` to UBIQUITOUS_LANGUAGE.md |
| S7 | Missing `skos:related` edge to `medallion-architecture` | LOW | Add to `pattern_v1.yaml` |

### From Type 2 (semantic) — 3 items

| # | Item | Severity | Action |
| --- | ------ | -------- | ------ |
| D1 | Ghost definition says wrong type names | HIGH | Resolved by S1 (deleting ghost) |
| D2 | 4 content entities linked to ghost, not canonical | HIGH | Resolved by S1+S2 (edge migration) |
| D3 | `dbt` SKOS lineage traces to tool instead of intellectual framework | HIGH | Resolved by S3 (register `fundamentals-of-data-engineering`, re-point edge) |

### Recommended approach

1. **Fix in place (this session):** S5 (ARCHITECTURE.md ref), S6 (UL entry), S7 (pattern edge)
2. **Spin off issue:** S1+S2+D1+D2 — "Clean up `four-data-system-types` ghost pattern and migrate edges to `data-system-classification`" — requires DB operations and Neo4j graph surgery
3. **Spin off issue:** S3+D3 — "Register `fundamentals-of-data-engineering` 3P pattern, fix `data-system-classification` SKOS lineage" — requires pattern_v1.yaml update + pattern doc update + re-ingestion
4. **Spin off issue:** S4 — "Ingest domain-pattern docs as KB entities" — likely a batch operation affecting all pattern docs, not just this one
