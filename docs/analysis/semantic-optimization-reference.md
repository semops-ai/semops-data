# Semantic Optimization — Cross-Layer Coherence Analysis

> **Purpose:** Full coherence analysis for `semantic-optimization` across all layers
> **Generated:** 2026-03-15 (updated with identity resolution)
> **Organization:** Phase 0 (Reference) + Phase 2 (Structural) + Phase 3 (Semantic)

## Identity Resolution (2026-03-15)

The coherence analysis flagged an "identity crisis" (pillar vs. loop). This has been resolved:

**`semantic-optimization` is a parent pattern** with SKOS `broader` relationships to two child patterns:

```
semantic-optimization (parent/broader)
├── semantic-object-pattern (narrower) — the target (what we optimize toward)
└── semantic-coherence (narrower) — the loss function (how we measure the gap)
```

This resolves all semantic drift findings:

- **Finding 1 (pillar vs. loop):** RESOLVED — it's a parent pattern. The "loop" framing (adopt → innovate → measure → repeat) describes the optimization process that emerges from the interplay of its two children. The "pillar" framing is the organizational container — both are valid at different levels.
- **Finding 2 (circular derivation):** RESOLVED — `semantic-coherence.md` saying "Derives From: semantic-optimization" is a SKOS `narrower` relationship, not circular. The source doc framing "coherence is a component of optimization" is consistent with broader/narrower.
- **Finding 4 (ADR-0012 framing):** CONFIRMED — "Pattern = target, Coherence = loss function, Optimization = control loop" maps directly to the parent/child structure.

### Registration parameters

- **provenance:** 1P
- **pattern_type:** concept (framing pattern — contains/organizes other patterns)
- **SKOS edges:** `broader` to `semantic-object-pattern`, `broader` to `semantic-coherence`
- **definition basis:** ADR-0012 — "The control loop that minimizes the gap between pattern targets and coherence measurements"

### Revised remediation

The original analysis identified 7 gaps, 3 mismatches, and 4 drift findings. With identity resolved:

- **Drift findings reduced:** 4 → 1 (only Finding 3 — concept inventory definition — remains as genuine drift)
- **Structural gaps unchanged:** Pattern row, doc, edges, coverage all still need creation
- **HITL decisions resolved:** Identity and circular derivation no longer block registration
- **Next step:** `/intake semantic-optimization` to wire capabilities and complete registration

---

## Phase 0: Reference Document

### 0.1 Pattern Identity

```
Pattern ID:    semantic-optimization
Pattern Label: Semantic Optimization
Provenance:    1p
Pattern Type:  concept (parent pattern)
DB Row:        ABSENT — not registered in the pattern table
SKOS:          broader → semantic-object-pattern, semantic-coherence
```

### 0.2 Artifacts by Layer

#### Canonical layer (semops-data)

| Artifact | Location | Status |
|----------|----------|--------|
| Pattern table row | `pattern` table | **ABSENT** — no row exists (61 other patterns registered) |
| Pattern doc | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/semantic-optimization.md` | **ABSENT** — file does not exist. Identified in ISSUE-134 as gap; planned but never created |
| Registry capabilities | `config/registry.yaml` line 193 | **PRESENT** — listed under `coherence-scoring` capability |
| Concept-pattern map | `config/mappings/concept-pattern-map.yaml` | **PRESENT** — 4 occurrences, max strength 0.9, mapped as exact match |
| STRATEGIC_DDD.md | Lines 156, 191, 629 | **PRESENT** — referenced in 3 places: optimization loop description, capability table, pattern audit table |
| ARCHITECTURE.md | — | **ABSENT** — zero mentions |
| UL entry | `schemas/UBIQUITOUS_LANGUAGE.md` | **PARTIAL** — referenced as pillar name and "Semantic Optimization Loop" (7 mentions) but no standalone definition block like other patterns |
| Fitness functions | `schemas/fitness-functions.sql` | **ABSENT** — zero mentions |
| PATTERNS.md | Line 418 | **PRESENT** — "The Semantic Optimization Loop as Engine" section |
| `pattern_edge` | DB query | **ABSENT** — zero rows (not possible without pattern row) |
| `edge` table | DB query | **ABSENT** — zero edges with `semantic-optimization` as src or dst pattern. Entity-level edges exist only for content entities |
| `pattern_coverage` view | DB query | **EMPTY** — no coverage row (no pattern row to cover) |

#### Entity layer (KB — PostgreSQL)

7 entities found with `semantic-optimization` in ID or title:

| Entity ID | Title | Type | Source |
|-----------|-------|------|--------|
| `semantic-optimization` | Semantic Optimization | content | `docs-pr/.../SEMANTIC_OPTIMIZATION/README.md` |
| `pub-semantic-optimization` | Semantic Optimization | content | `semops-docs/.../SEMANTIC_OPTIMIZATION/README.md` |
| `semantic-optimization-implementation` | Semantic Optimization Implementation | content | `docs-pr/.../semantic-optimization-implementation.md` |
| `docs-pr-issue-102` | Issue : Semantic flywheel vs. Semantic Optimization | content | `docs-pr/issues/102` |
| `docs-pr-issue-45` | Issue : Semantic Optimization focus | content | `docs-pr/issues/45` |
| `issue-99-semantic-optimization-page` | Issue : Semantic Optimization Page | content | `publisher-pr/docs/session-notes/...` |
| `publisher-pr-issue-99` | Issue : Semantic Optimization Page | content | `publisher-pr/issues/99` |

**Critical:** All 7 entities have `primary_pattern_id = NULL` — none are linked to a pattern, because the pattern row does not exist.

Entity-level edges (content → pattern, via `documents`/`related_to`):

| Source Entity | Predicate | Target Pattern | Strength |
|---------------|-----------|----------------|----------|
| semantic-optimization | documents | ddd | 0.70 |
| semantic-optimization | documents | semantic-coherence | 0.95 |
| semantic-optimization | related_to | semantic-funnel | 0.85 |
| semantic-optimization | related_to | strategic-data | 0.90 |
| semantic-optimization | related_to | prov-o | 0.60 |
| semantic-optimization-implementation | documents | ddd | 0.80 |
| semantic-optimization-implementation | documents | semantic-coherence | 0.95 |
| semantic-optimization-implementation | documents | prov-o | 0.60 |
| semantic-optimization-implementation | documents | skos | 0.70 |

**Capability status:**

The `coherence-scoring` capability in the DB has 3 edges:
- `coherence-scoring --[delivered_by]--> data-pr`
- `coherence-scoring --[delivered_by]--> semops-data`
- `coherence-scoring --[implements]--> semantic-coherence`

**MISMATCH:** registry.yaml declares `coherence-scoring` implements BOTH `semantic-optimization` and `semantic-coherence`, but the DB only has an IMPLEMENTS edge to `semantic-coherence`. The `semantic-optimization` IMPLEMENTS edge is missing (impossible to create without the pattern row).

#### Source material layer (docs-pr)

| Artifact | Location | Status |
|----------|----------|--------|
| Hub doc (README) | `docs/SEMOPS_DOCS/.../SEMANTIC_OPTIMIZATION/README.md` | PRESENT — 150 lines, comprehensive framework description |
| Implementation doc | `docs/SEMOPS_DOCS/.../SEMANTIC_OPTIMIZATION/semantic-optimization-implementation.md` | PRESENT — 58K, detailed implementation guide |
| Pattern Operations | `docs/SEMOPS_DOCS/.../SEMANTIC_OPTIMIZATION/pattern-operations.md` | PRESENT |
| Patterns doc | `docs/SEMOPS_DOCS/.../SEMANTIC_OPTIMIZATION/patterns.md` | PRESENT |
| Coherence Measurement | `docs/SEMOPS_DOCS/.../SEMANTIC_OPTIMIZATION/semantic-coherence-measurement.md` | PRESENT |
| Legacy doc | `docs/_legacy/SEMANTIC_OPERATIONS/semantic-optimization.md` | PRESENT (legacy) |
| 37 files referencing pattern | Various | Rich concept-track content |

#### Published layer

| Artifact | Location | Status |
|----------|----------|--------|
| Website page (semops.ai) | `sites-pr/apps/semops/content/pages/what-is-semops.mdx` | PRESENT — references semantic-optimization |
| Publisher staging page | `publisher-pr/content/pages/framework/semantic-optimization.md` | PRESENT |
| Publisher session notes | `publisher-pr/docs/session-notes/ISSUE-99-semantic-optimization-page.md` | PRESENT |
| 51 files in publisher-pr | Various (source, legacy, drafts, templates) | Rich content footprint |
| 7 files in sites-pr | Various (IA, architecture, ADRs) | PRESENT |

#### Governance history (semops-data)

| Source | Files | Key References |
|--------|-------|----------------|
| ADR-0004 | `docs/decisions/ADR-0004-schema-phase2-pattern-aggregate-root.md` | "NEEDS WORK" flag, listed in reclassification candidates |
| ADR-0006 | `docs/decisions/ADR-0006-v1-documentation-plan.md` | "Wired semantic-operations with 4 pillars: ...semantic-optimization..." |
| ADR-0012 | `docs/decisions/ADR-0012-pattern-coherence-co-equal-aggregates.md` | "When semantic-optimization becomes operational, coherence scoring IS the objective function" |
| ISSUE-47 | `docs/decisions/ISSUE-47-CONCEPT-STRATEGY.md` | Listed as narrower concept of semantic-operations |
| ISSUE-48 | `docs/decisions/ISSUE-48-FOUNDATION-CATALOG.md` | Cataloged at depth 2 under semantic-operations |
| ISSUE-115 | `docs/session-notes/ISSUE-115-pattern-type-schema.md` | Identified as one of the core patterns that should have pattern_type |
| ISSUE-118 | `docs/session-notes/ISSUE-118-domain-pattern-ingestion.md` | Listed as "0 doc coverage" — pattern with no domain-patterns doc |
| ISSUE-134 | `docs/session-notes/ISSUE-134-phase-b-cleanup.md` | Explicitly planned: "semantic-optimization.md — was pointing to docs-pr/..." — but never created |
| ISSUE-142 | `docs/session-notes/ISSUE-142-pattern-coherence-co-equal-aggregates.md` | "When semantic-optimization becomes operational, coherence scoring IS the objective function" |
| ISSUE-146 | `docs/session-notes/ISSUE-146-validate-query-precision.md` | Surfaces in semantic search for "semantic-coherence" queries |
| Concept inventory | `docs/concept-inventory/definition-review.md` | "The process of maintaining stable coherence... flag: needs work" |
| Definition parking lot | `docs/concept-inventory/definition-parking-lot.md` | "Needs voice refinement. Note: spurring vs spuring typo" |

### 0.3 Artifact Timeline

**First mention in semops-data:** 2025-11-24 — "language schema update"

**Concept track:**

| Date | Event | Repo |
|------|-------|------|
| 2025-11-24 | First mention in language schema | semops-data |
| 2025-11-25 | Concept schema, foundation catalog | semops-data |
| 2026-01-31 | Source docs directory created | docs-pr |
| 2026-02-06 | Website about pages | sites-pr |
| 2026-02-08 | Pattern doc frontmatter: `pattern: semantic-optimization` | docs-pr |
| 2026-02-13 | Framework IA on website | sites-pr |
| 2026-02-17 | Concept-pattern map created | semops-data |
| 2026-02-19 | Publisher staging page | publisher-pr |
| 2026-02-26 | Enriched definitions (via ) | semops-data |
| 2026-03-03 | Entity ingested to KB | semops-data (DB) |

**Implementation track:**

| Date | Event | Repo |
|------|-------|------|
| 2026-02-08 | ADR-0004 flags "NEEDS WORK" | semops-data |
| 2026-02-17 | Concept-pattern map: exact match, action: map | semops-data |
| 2026-03-10 | registry.yaml: coherence-scoring implements semantic-optimization | semops-data |
| 2026-03-10 | STRATEGIC_DDD.md: capability table references | semops-data |

**Convergence point:** 2026-03-10 — when `registry.yaml` was created and `STRATEGIC_DDD.md` was updated to reference `semantic-optimization` as a pattern implemented by `coherence-scoring`. However, the critical gap is that no pattern table row was ever created, so the convergence is **partial** — the implementation track references a pattern that doesn't exist in the database.

**Promotion type:** This is a **stalled recognition promotion** — the concept has been extensively documented and the implementation track references it, but the actual pattern registration step was never completed.

---

## Phase 2: Type 1 — Structural Coherence

### Structural Checklist

| # | Artifact | Expected | Actual | Score |
|---|----------|----------|--------|-------|
| 1 | `pattern` table row | Row with id, definition, provenance | No row exists | **GAP** |
| 2 | `pattern_edge` SKOS edges | Edges matching "Derives From" relationships | No edges (no row) | **GAP** |
| 3 | Capability IMPLEMENTS edge | `coherence-scoring --[implements]--> semantic-optimization` | Edge exists only to `semantic-coherence`, not to `semantic-optimization` | **GAP** |
| 4 | `pattern_coverage` view | Row with content/capability/repo counts | Empty (no pattern row) | **GAP** |
| 5 | Pattern doc file | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/semantic-optimization.md` | File does not exist | **GAP** |
| 6 | Pattern doc as KB entity | Entity with content_type "pattern" | Entity exists as content type "file" with `primary_pattern_id = NULL` | **MISMATCH** |
| 7 | Registry.yaml entries | Capabilities listed | PRESENT — coherence-scoring lists it | **PRESENT** |
| 8 | STRATEGIC_DDD.md | Pattern referenced in capability tables | PRESENT — 3 references | **PRESENT** |
| 9 | ARCHITECTURE.md | Pattern named | No mentions | **GAP** |
| 10 | UL entry | Standalone definition | Referenced as pillar name and loop, but no pattern definition block | **PARTIAL** |
| 11 | DESCRIBED_BY edges | Pattern to concept entities | No DESCRIBED_BY edges | **GAP** |
| 12 | Stale entities | Pre-rename/superseded entities retired | 7 entities with null pattern_id; legacy docs in docs-pr `_legacy/` | **MISMATCH** |
| 13 | Registry/DB alignment | Capability IMPLEMENTS count matches | Registry says 1 capability, DB has 0 IMPLEMENTS edges to this pattern | **MISMATCH** |

**Gap count: 7 GAPs, 3 MISMATCHes, 1 PARTIAL, 2 PRESENTs**

### Structural Verdict

## **CONCEPT ONLY**

Despite extensive source material across 4+ repos (37 files in docs-pr, 51 in publisher-pr, 7 in sites-pr), `semantic-optimization` has never been registered as a pattern in the database. The pattern row does not exist. All downstream structural artifacts that depend on the pattern row (pattern_edge, IMPLEMENTS edges, pattern_coverage, DESCRIBED_BY) are consequently absent.

The registry.yaml and STRATEGIC_DDD.md reference it as if it were a registered pattern, creating a **referential integrity gap** — the implementation track points to a pattern that the data layer does not contain.

This is the most significant structural gap found: `semantic-optimization` is a **framework pillar** (one of three: Strategic Data, Explicit Architecture, Semantic Optimization), yet it is the only pillar without a pattern table row.

For comparison:
- `strategic-data` — registered pattern (row exists)
- `explicit-architecture` — registered pattern (row exists, full analysis in `explicit-architecture-reference.md`)
- `semantic-optimization` — **NOT registered** (no row)

---

## Phase 3: Type 2 — Semantic Coherence

### Framing Comparison

| Artifact | Core Framing | Audience | Concreteness |
|----------|-------------|----------|-------------|
| UL (pillar description) | "Elevate your organization to operate like well-designed software: agent-ready, self-validating, expanding through patterns" | Domain vocabulary | Abstract — aspirational framing |
| UL ("Semantic Optimization Loop") | "Adopt 3P → innovate 1P → link via SKOS → measure coherence → repeat" | Architecture governance | Concrete — specific process steps |
| Source doc hub (docs-pr README) | "Business growth through agentic execution with rich semantic Patterns, while continually aligning semantic coherence" | Theory/research | Mixed — combines aspirational + concrete mapping to DIKW funnel |
| STRATEGIC_DDD.md | "Coherence scoring becomes the objective function — Pattern sets the target, Coherence measures the gap, the optimization loop minimizes the gap" | Architecture governance | Concrete — operational role in system |
| ADR-0012 | "Pattern = target, Coherence = loss function, Optimization = control loop that minimizes the gap" | Architecture decisions | Concrete — ML/optimization framing |
| Concept inventory definition | "The process of maintaining stable coherence between agents while spurring growth through new patterns and change" | Internal reference | Abstract — flagged "needs work" |
| PATTERNS.md | "The full cycle — adopt 3P → innovate 1P → link via SKOS → measure coherence → repeat — is the engine" | Technical audience | Concrete |
| Website (what-is-semops) | References as framework pillar | External/accessible | Abstract |
| Publisher staging page | Framework page | Public narrative | Unknown (not read in detail) |

### Alignment Dimensions

| Dimension | Assessment | Status |
|-----------|-----------|--------|
| **Thesis alignment** | Two competing framings: (1) "Semantic Optimization" as a **framework pillar** (alongside Strategic Data and Explicit Architecture) — a broad category; (2) "Semantic Optimization Loop" as a **specific process** (adopt 3P → innovate 1P → measure coherence → repeat). These are related but distinct concepts being conflated under one ID. | **DRIFT** |
| **Scope consistency** | Inconsistent. The source doc hub treats it as a broad pillar encompassing Semantic Coherence + Patterns. The UL and PATTERNS.md treat "Semantic Optimization Loop" as a specific mechanism. ADR-0012 uses optimization/loss-function language. The concept inventory definition is vague ("maintaining stable coherence while spurring growth"). | **DRIFT** |
| **Concrete examples** | The source doc hub maps to DIKW funnel stages. ADR-0012 provides ML-style framing (objective function, loss function, control loop). PATTERNS.md gives the adopt/innovate/link/measure cycle. No implementation examples yet exist because coherence-scoring is `in_progress`. | **PARTIAL** |
| **Cross-pattern relationships** | Source doc lists: derives from `ddd`, `semantic-coherence`, related to `semantic-funnel`, `strategic-data`, `prov-o`. The entity edges reflect this. But the `semantic-coherence` pattern doc says it "Derives From: semantic-optimization" — a circular dependency that should be `broader`/`narrower` instead. | **DRIFT** |
| **Implementation binding** | ARCHITECTURE.md has zero mentions. STRATEGIC_DDD.md references it. Registry.yaml lists it. But the DB has no pattern row, so there is no actual binding — only documentation-level references to a non-existent pattern. | **GAP** |
| **Session note mining** | ISSUE-134 identified the gap (no doc, 0 coverage) and planned to fix it — but the fix was never applied. ISSUE-142 provides the clearest framing: "Pattern = target, Coherence = loss function, Optimization = control loop." This framing has not been propagated to the source docs. | **DRIFT** |
| **Audience appropriateness** | The different framings appear to be accidental drift rather than intentional audience targeting. The UL pillar description ("operate like well-designed software") and the concept inventory definition ("maintaining stable coherence while spurring growth") are vague compared to the concrete ADR-0012/PATTERNS.md formulations. | **DRIFT** |

### Drift Findings

**Finding 1: Pillar vs. Loop identity crisis**

- **What:** `semantic-optimization` serves two distinct semantic roles: (a) a framework pillar (organizational container for Semantic Coherence + Patterns content), and (b) a specific process mechanism (the optimization loop: adopt → innovate → measure → repeat). These are conflated under one ID.
- **Where:** UL (pillar), source doc hub (pillar), PATTERNS.md (loop), ADR-0012 (loop), STRATEGIC_DDD.md (both)
- **Direction:** Lateral drift — both framings evolved independently
- **Impact:** When agents or users search for `semantic-optimization`, they get the framework pillar. But when the system references it as a pattern implemented by capabilities, it means the loop. This ambiguity makes coherence measurement unreliable.
- **Remediation:** Decide whether `semantic-optimization` is the pillar or the loop. If pillar: it should be a concept (broader than patterns), not a pattern. If loop: the pillar content in docs-pr should be reorganized. The ADR-0012 "loss function" framing suggests the loop is the correct pattern identity. The pillar function is better served by the directory structure in docs-pr (`SEMANTIC_OPTIMIZATION/`).

**Finding 2: Circular derivation with semantic-coherence**

- **What:** The `semantic-coherence` pattern doc states "Derives From: `semantic-optimization`", but the source doc hub for `semantic-optimization` frames Semantic Coherence as a *component* of Semantic Optimization ("two components: Semantic Coherence creates a stable knowledge state, and Patterns provide stable knowledge growth"). This creates a circular dependency: coherence derives from optimization, but optimization contains coherence.
- **Where:** `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/analytics/semantic-coherence.md` (line 7), `docs-pr/.../SEMANTIC_OPTIMIZATION/README.md` (line 53)
- **Direction:** Concept ahead of implementation — the SKOS relationship was assigned before the semantic relationship was resolved
- **Impact:** If `semantic-optimization` were registered, the SKOS edges would encode a circular derivation. This violates the SKOS broader/narrower hierarchy assumption.
- **Remediation:** Clarify the relationship: `semantic-optimization` is `broader` than `semantic-coherence` (pillar contains component), OR `semantic-coherence` is a peer pattern and the "derives from" should be `related` instead.

**Finding 3: Concept inventory definition flagged but never resolved**

- **What:** The definition in `docs/concept-inventory/definition-review.md` was flagged as "needs work" and has a typo ("spuring" for "spurring"). The definition-parking-lot.md notes "Needs voice refinement." This was identified early (2025 timeframe) and never addressed.
- **Where:** `docs/concept-inventory/definition-review.md`, `docs/concept-inventory/definition-parking-lot.md`
- **Direction:** Implementation ahead of concept — the pattern is referenced in architecture docs without a clean definition
- **Impact:** Low immediate impact since these are legacy concept inventory files, but the lack of a crisp definition propagates to the UL pillar description, which is vague.
- **Remediation:** Write a proper pattern definition. The ADR-0012 framing ("Pattern sets the target, Coherence measures the gap, the optimization loop minimizes the gap") is the clearest articulation and should be the basis.

**Finding 4: ADR-0012 framing not propagated**

- **What:** The most concrete and operationally useful framing of semantic optimization exists in ADR-0012 and ISSUE-142 session notes ("Pattern = target, Coherence = loss function, Optimization = control loop that minimizes the gap"). This framing has not been propagated to the source docs, UL, or pattern definition.
- **Where:** ADR-0012 (sections 10, Phase 9), ISSUE-142 session notes vs. docs-pr source hub, UL
- **Direction:** Implementation ahead of concept — the operational understanding has evolved past the documentation
- **Impact:** Agents and users reading the source docs get a weaker, more abstract understanding than what the architecture decisions have established.
- **Remediation:** Update the source doc hub and UL to incorporate the loss-function/control-loop framing.

**Finding 5: Seven unlinked entities**

- **What:** 7 KB entities related to semantic-optimization exist with `primary_pattern_id = NULL`. They are discoverable via semantic search but not linked to any pattern, making pattern_coverage impossible to compute.
- **Where:** PostgreSQL entity table
- **Direction:** Implementation gap
- **Impact:** The pattern has invisible content. Coverage views and fitness functions cannot measure it. Semantic search finds it but structural queries cannot.
- **Remediation:** Register the pattern row first, then backfill `primary_pattern_id` on relevant entities.

---

## Phase 4: Summary

### Verdicts

| Assessment | Result |
|------------|--------|
| **Structural verdict** | **CONCEPT ONLY** |
| **Gap count** | 7 structural gaps |
| **Mismatch count** | 3 structural mismatches |
| **Drift count** | 4 semantic drift findings |
| **Partial items** | 2 (UL entry, concrete examples) |

### Key Findings (revised)

1. **`semantic-optimization` is the only framework pillar without a pattern table row.** Strategic Data and Explicit Architecture are both registered. This is the single most important gap.

2. ~~**Identity crisis**~~ **RESOLVED:** `semantic-optimization` is a parent/concept pattern, `broader` than `semantic-object-pattern` (the target) and `semantic-coherence` (the loss function). No ambiguity.

3. ~~**Circular derivation**~~ **RESOLVED:** `semantic-coherence` "Derives From" is a SKOS `narrower` relationship. Update the label in the pattern doc but the relationship direction is correct.

4. **The best definition exists in ADR-0012** — "Pattern = target, Coherence = loss function, Optimization = control loop." Should be the basis for the pattern definition.

5. **Registry/DB referential integrity gap** — `config/registry.yaml` declares `coherence-scoring` implements `semantic-optimization`, but the DB cannot honor this because the pattern row does not exist.

### Remediation: `/intake semantic-optimization`

Run intake to handle registration, capability wiring, and SKOS edges in one structured pass. Intake has sufficient material from ADR-0012, source docs, and the identity resolution above.

#### Remaining manual items

- [ ] **Update `semantic-coherence.md`** — change "Derives From" label to SKOS `narrower` (relationship direction is correct, label is wrong)
- [ ] **Update UL entry** — add proper pattern definition block based on ADR-0012 framing
- [ ] **Propagate ADR-0012 framing** to source doc hub

#### Defer

- [ ] **Stale entity cleanup** — legacy entities in `docs-pr/_legacy/` are low priority
- [ ] **Fitness function coverage** — meaningful only after pattern row exists and coherence-scoring capability is operational
- [ ] **DESCRIBED_BY edges** — create after pattern row exists
