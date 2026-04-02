# Coherence Analysis: mirror-architecture

> **Date:** 2026-03-15 (updated with revival context)
> **Structural Verdict:** ACTIVE — INCOMPLETE (revived from retirement)
> **Gaps:** 5 (revival registration gaps)
> **Drift Findings:** 1
> **Pattern Status:** Revived — retirement was premature; active pattern with PROJECT-30

## Revival Context (2026-03-15)

`mirror-architecture` was retired in February 2026 , absorbed into `scale-projection` on the grounds that it was "a 1P label for Scale Projection's implementation details." This retirement was **premature**. Mirror Architecture is a distinct pattern with its own project board ([Mirror Data Architecture](https://github.com/users/semops-ai/projects/30)), 16 work items across `data-pr`, `semops-research`, and `publisher-pr`, and a concrete agentic pipeline (discover → populate → validate → overlay).

**Key distinction from scale-projection:** Scale Projection is about projecting a domain model across infrastructure tiers (infrastructure-up meets domain-down). Mirror Architecture is about standing up a running mirror of an existing data system and measuring the gap between reference model and reality. They share DDD foundations but address different problems.

### What was "correct all along"

Several findings from the original analysis (below) were flagged as drift or incomplete cleanup, but are actually correct for a revived pattern:

- Entity `lifecycle_stage: active` — correct
- `extends→ddd` pattern_edge — valid relationship
- UL example listing — appropriate (it IS an implementation pattern)
- publisher-pr draft references — active content for an active pattern

### Remediation for revival

| # | Item | Action | Priority |
|---|------|--------|----------|
| R1 | Pattern table lifecycle | Update `lifecycle_stage` from `retired` to `development` | HIGH |
| R2 | Pattern doc | Create new `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/mirror-architecture.md` (not the retired stub) | HIGH |
| R3 | Registry.yaml | Re-add `mirror-architecture` with capability mappings | HIGH |
| R4 | Retired stub | Remove or redirect `semops-orchestrator/docs/patterns/retired/mirror-architecture.md` | MEDIUM |
| R5 | SKOS edges | Review — may need more than just `extends→ddd` (e.g., `related→scale-projection`, `related→explicit-architecture`) | MEDIUM |
| R6 | ARCHITECTURE.md | Add mirror-architecture reference | LOW |

### Original analysis below (pre-revival context)

---

---

## Phase 0: Reference Document

### 0.1 Pattern Identity

```
Pattern ID:      mirror-architecture
Preferred Label: Mirror Architecture
Provenance:      1P
Pattern Type:    infrastructure (DB metadata says "infrastructure")
Lifecycle Stage: retired (pattern table), active (entity metadata -- MISMATCH)
Superseded By:   scale-projection
```

### 0.2 DB State

**Pattern table row:**
- `id`: mirror-architecture
- `preferred_label`: Mirror Architecture
- `provenance`: 1p
- `metadata.pattern_type`: infrastructure
- `metadata.lifecycle_stage`: retired
- `metadata.documentation.primary`: semops-orchestrator/docs/patterns/retired/mirror-architecture.md
- `metadata.documentation.related`: [semops-orchestrator/docs/decisions/ADR-0004-mirror-architecture.md]

**Entity table row:**
- `id`: mirror-architecture
- `entity_type`: content
- `metadata.corpus`: core_kb
- `metadata.content_type`: pattern
- `metadata.lifecycle_stage`: active  <-- MISMATCH with pattern table's "retired"
- `filespec.uri`: github://semops-ai/semops-orchestrator/docs/patterns/retired/mirror-architecture.md

**Pattern edges:**
- `mirror-architecture` --extends--> `ddd` (still present, should have been cleaned)

**Entity edges:** None

**Pattern coverage:** 0 content, 0 capabilities, 0 repos (all zeros -- expected for retired)

### 0.3 KB Search Results

**search_knowledge_base("mirror architecture")** -- top results:
1. `adr-0004-mirror-architecture` (content, deployment, adr, 0.69) -- ADR still ingested
2. `mirror-architecture` (content, core_kb, pattern, 0.68) -- retired pattern doc still ingested as active entity
3. `issue-65-progressive-productization` (content, deployment, session_note, 0.45) -- semops-orchestrator session note

**search_chunks("mirror architecture repo structure domain boundaries")** -- key findings:
1. semops-orchestrator Issue  chunk (0.62): "Mirror Architecture is not an industry term -- it's a 1P label for Scale Projection's implementation details... Maintaining two overlapping pattern files creates drift"
2. ADR-0004 chunk (0.62): Implementation plan for creating pattern doc and registering in pattern_v1.yaml
3. ADR-0005 chunk (0.62): Cross-reference to ADR-0004 for private-to-public corpus mapping

### 0.4 Graph Neighbors

MCP tools were unavailable. DB query shows:
- **Pattern edges:** 1 outgoing: `extends --> ddd`
- **Entity edges:** None (no DESCRIBED_BY, no IMPLEMENTS)

### 0.5 File-Level Checks

| Artifact | Location | Status |
|----------|----------|--------|
| Pattern doc (semops-data) | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/mirror-architecture.md` | DOES NOT EXIST |
| Pattern doc (semops-orchestrator) | `docs/patterns/retired/mirror-architecture.md` | EXISTS (retired stub) |
| Registry.yaml | `config/registry.yaml` | NOT PRESENT (cleaned) |
| STRATEGIC_DDD.md | `docs/STRATEGIC_DDD.md` | NOT PRESENT |
| ARCHITECTURE.md | `docs/ARCHITECTURE.md` | NOT PRESENT |
| UL entry | `schemas/UBIQUITOUS_LANGUAGE.md` line 181 | PRESENT -- listed as `implementation` type example |
| Fitness functions | `schemas/fitness-functions.sql` | NOT PRESENT |

### 0.6 Cross-Repo Content

**semops-orchestrator (17 files):**
- `docs/patterns/retired/mirror-architecture.md` -- retired stub
- `docs/decisions/ADR-0004-mirror-architecture.md` -- original ADR (still active in KB)
- `docs/session-notes/ISSUE-65-progressive-productization.md` -- origin session
- `docs/session-notes/ISSUE-148-scale-vectors-resourcing-methodology.md` -- absorption decision
- `docs/session-notes/ISSUE-134-expand-publication-scope.md` -- rename/cleanup session
- `docs/session-notes/ISSUE-100-register-strategic-ddd-patterns.md`
- `docs/session-notes/ISSUE-75-domain-pattern-ingestion.md`
- `docs/session-notes/ISSUE-146-architecture-documentation-alignment.md`
- `docs/session-notes/ISSUE-151-domain-model-governance-lifecycle.md`
- `docs/project-specs/PROJECT-30-mirror-data-architecture.md`
- `docs/templates/mirror-structure/README.md` -- template still exists
- `docs/PATTERN_AUDIT.md`, `docs/ADR_INDEX.md`
- `docs/patterns/README.md`

**publisher-pr (3 files):**
- `docs/drafts/semops-framework/data-management-analogy.md` -- lists mirror-architecture as active implementation pattern
- `docs/drafts/semops-framework/semops-consulting-view1.md` -- mentions mirror architecture
- `docs/drafts/semops-framework/semops-consulting-view2.md` -- has full "Mirror Architecture" section

**docs-pr:** No matches
**sites-pr:** No matches

### 0.7 Timeline

| Date | Event | Repo |
|------|-------|------|
| 2026-01-22 | ADR-0004: Mirror Architecture created | semops-orchestrator |
| 2026-01-23 | Pattern doc `mirror-architecture.md` created | semops-orchestrator |
| 2026-02-14 | : mirror-architecture renamed to scale-projection in registry; cleanup removes pattern + extends->ddd edge from DB | semops-data |
| 2026-02-20 | : Scale Projection v2.0 absorbs Mirror Architecture; pattern doc moved to `retired/` | semops-orchestrator |
| 2026-02-21 | : lifecycle alignment; pattern doc overhaul | semops-orchestrator |
| 2026-02-27 | Entity re-ingested (filespec.last_checked date) | semops-data |

**Promotion type:** Recognition promotion -- the concept existed as ADR-0004, then got registered, then was absorbed into scale-projection before it matured.

---

## Phase 2: Type 1 -- Structural Coherence

| # | Artifact | Expected | Actual | Score |
|---|----------|----------|--------|-------|
| 1 | `pattern` table row | Row with retired lifecycle | Present, lifecycle_stage=retired | PRESENT |
| 2 | `pattern_edge` SKOS | Should be empty for retired pattern | 1 edge: extends-->ddd still present | MISMATCH |
| 3 | Capability IMPLEMENTS edges | None expected for retired | None | N/A |
| 4 | `pattern_coverage` view | Zeros expected | 0/0/0 | PRESENT |
| 5 | Pattern doc file | Retired stub in retired/ dir | Exists in semops-orchestrator/docs/patterns/retired/ | PRESENT |
| 6 | Pattern doc as KB entity | Should be retired or removed | Entity exists with lifecycle_stage=**active** | MISMATCH |
| 7 | Registry.yaml entries | Not present (cleaned) | Not present | PRESENT |
| 8 | STRATEGIC_DDD.md | Not present | Not present | N/A |
| 9 | ARCHITECTURE.md | Not present | Not present | N/A |
| 10 | UL entry | Should not list retired pattern as example | Listed as `implementation` type example | GAP |
| 11 | DESCRIBED_BY edges | None expected | None | N/A |
| 12 | Stale entities | Entity should be marked retired | Entity metadata says lifecycle_stage=active | GAP |
| 13 | Registry/DB alignment | Pattern table says retired, entity says active | Inconsistent | MISMATCH |

**Additional structural findings:**

| # | Finding | Score |
|---|---------|-------|
| 14 | ADR-0004 still ingested as active KB entity | GAP |
| 15 | publisher-pr drafts still reference mirror-architecture as active pattern | GAP |

### Structural Verdict: RETIRED -- INCOMPLETE CLEANUP

The pattern was correctly retired in the pattern table and removed from registry.yaml. However, cleanup was incomplete:
- The entity row's `lifecycle_stage` metadata was never updated from "active" to "retired"
- The `pattern_edge` extends-->ddd still exists (was reportedly cleaned in  but is back -- possibly re-ingested)
- UL still uses mirror-architecture as an example of the `implementation` pattern type
- publisher-pr draft content still references it as active

---

## Phase 3: Type 2 -- Semantic Coherence

### Framing Comparison

| Artifact | Core Framing | Status |
|----------|-------------|--------|
| Retired pattern doc | Stub pointing to scale-projection | Correct |
| ADR-0004 | Original framing: 8-level continuum, GAPS.md, directory layout as architecture | Historical (correct to preserve) |
| scale-projection.md | Absorbed all Mirror Architecture concepts; references ADR-0004 as "original framing" | Correct |
| UL (line 181) | Lists mirror-architecture as example of `implementation` type | STALE -- should use scale-projection |
| publisher-pr drafts | Present mirror-architecture as a current, distinct concept with its own section | STALE |
| Session notes (, ) | Document the retirement process | Correct |

### Alignment Dimensions

| Dimension | Question | Status |
|-----------|----------|--------|
| Thesis alignment | Do artifacts agree on what this pattern IS? | PARTIAL -- retired docs say "absorbed"; UL and publisher drafts still treat it as independent |
| Scope consistency | Consistent scope? | N/A -- pattern is retired |
| Concrete examples | Reflect current state? | GAP -- publisher drafts have detailed "Mirror Architecture" sections that should reference scale-projection |
| Cross-pattern relationships | Consistent? | GAP -- extends-->ddd edge orphaned; scale-projection carries the relationship now |
| Implementation binding | Named in ARCHITECTURE.md? | N/A -- correctly absent |
| Session note mining | Insights captured? | PRESENT --  and  document the retirement clearly |
| Audience appropriateness | Intentional vs. accidental framing differences? | DRIFT -- publisher drafts are accidental (pre-retirement content never updated) |

### Drift Findings

**Finding 1: Entity lifecycle_stage mismatch**

- **What:** The `entity` table row for mirror-architecture has `metadata.lifecycle_stage = "active"` while the `pattern` table row has `metadata.lifecycle_stage = "retired"`. The entity is searchable as an active core_kb pattern entity.
- **Where:** Entity table vs. pattern table metadata
- **Direction:** Implementation ahead of concept (pattern was retired but entity ingestion didn't propagate the status)
- **Impact:** KB searches for patterns return mirror-architecture as an active result. Agents consulting the KB may reference a retired pattern. The explicit-architecture coherence analysis already flagged this as noise (rank 7 result at 0.49 similarity).
- **Remediation:** Update entity metadata to set `lifecycle_stage: retired`. Consider adding entity-level retirement to the pattern cleanup workflow.

**Finding 2: UL still uses mirror-architecture as pattern_type example**

- **What:** `schemas/UBIQUITOUS_LANGUAGE.md` line 181 lists `mirror-architecture` in the examples for the `implementation` pattern type.
- **Where:** UL pattern_type table
- **Direction:** Concept behind implementation (UL not updated when pattern was retired)
- **Impact:** Agents reading the UL will see mirror-architecture as a canonical example of an implementation pattern, despite it being retired. This is a governance document -- it should only reference active patterns.
- **Remediation:** Replace `mirror-architecture` with an active implementation pattern example (e.g., `containerization` or `platform-engineering`).

**Finding 3: publisher-pr draft content references mirror-architecture as active**

- **What:** Three publisher-pr draft files reference mirror-architecture as a current, active implementation pattern. `semops-consulting-view2.md` has an entire "Mirror Architecture" section describing it as a novel approach. `data-management-analogy.md` lists it alongside scale-projection as if they are separate active patterns.
- **Where:** publisher-pr/docs/drafts/semops-framework/
- **Direction:** Concept behind implementation (draft content predates retirement, never updated)
- **Impact:** If these drafts are published or used as source material for content generation, they will propagate a retired concept. The data-management-analogy analysis document already flagged this as "REGISTERED (1p)" without noting retirement.
- **Remediation:** Update drafts to reference scale-projection. Remove or redirect the Mirror Architecture section in semops-consulting-view2.md.

---

## Phase 4: Remediation Summary

### From Type 1 (Structural) -- 5 gaps

| # | Item | Action | Priority |
|---|------|--------|----------|
| S1 | Entity lifecycle_stage = "active" (should be "retired") | Update entity metadata | HIGH -- causes KB noise |
| S2 | pattern_edge extends-->ddd still present | Delete orphaned edge | MEDIUM |
| S3 | UL lists mirror-architecture as implementation example | Replace with active pattern | MEDIUM |
| S4 | ADR-0004 entity still active in KB | Mark as historical/retired or add superseded_by metadata | LOW |
| S5 | publisher-pr drafts reference mirror-architecture as active | Update draft content | LOW |

### From Type 2 (Semantic) -- 3 drift findings

| # | Item | Action | Priority |
|---|------|--------|----------|
| D1 | Entity/pattern lifecycle mismatch | Same as S1 | HIGH |
| D2 | UL stale example | Same as S3 | MEDIUM |
| D3 | publisher-pr draft content stale | Same as S5 | LOW |

### Recommended Actions

1. **Fix in place (S1, S2, S3):** Update entity metadata, delete orphaned pattern_edge, update UL example. These are small, deterministic changes.
2. **Defer (S4):** ADR-0004 is historical record; marking it is nice-to-have but low impact.
3. **Defer (S5/D3):** publisher-pr drafts are pre-publication; update when the drafts are next edited.

### Process Improvement

The retirement of mirror-architecture reveals a gap in the pattern retirement workflow:
- Pattern table lifecycle was updated correctly
- Registry.yaml was cleaned correctly
- Pattern doc was moved to retired/ correctly
- **But:** entity metadata, pattern_edge cleanup, and UL cross-references were missed

**Recommendation:** Add to the pattern retirement checklist:
1. Update pattern table lifecycle_stage
2. Remove or annotate pattern_edges
3. Update entity metadata lifecycle_stage
4. Grep UL for pattern-id references in examples
5. Cross-repo grep for stale references in draft/published content

---

## Appendix: Artifact Inventory

### All files referencing mirror-architecture

**semops-data:**
- `schemas/UBIQUITOUS_LANGUAGE.md` (line 181)
- `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/process/scale-projection.md` (lines 19, 26, 452 -- correctly references as predecessor)
- `docs/session-notes/ISSUE-134-phase-b-cleanup.md` (retirement documentation)
- `docs/session-notes/ISSUE-164-audit-pattern-type-lifecycle.md` (lifecycle audit)
- `docs/session-notes/ISSUE-115-pattern-type-schema.md` (reclassification history)
- `docs/search-results-example.md` (correctly notes "retired -> scale-projection")
- `docs/analysis/explicit-architecture-reference.md` (flagged as noise)
- `docs/analysis/data-management-analogy-reference.md` (noted as "REGISTERED (1p)" without retirement flag)

**semops-orchestrator (17 files):** See Phase 0.6 above.

**publisher-pr (3 files):** See Phase 0.6 above.
