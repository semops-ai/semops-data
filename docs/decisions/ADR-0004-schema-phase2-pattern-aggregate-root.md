# ADR-0004: Schema Phase 2 - Pattern as Aggregate Root

## Status
**COMPLETE** - Schema implemented (2025-12-22)

## Related
- **Project:** [Schema Phase 2](https://github.com/users/semops-ai/projects/8)
- **Issue:**  - Implement Phase 2 schema: Pattern as aggregate root

## Summary

Pattern is the aggregate root for Project Ike's semantic operations system. A Pattern is an applied unit of meaning with a business purpose, measured for semantic coherence and optimization.

---

## Historical Context: Atom Definition Refinement Session

> The following documents the discovery process that led to this architectural decision.

### Status
Paused after full review (2025-12-06)

## Context

Following completion of 60 new atoms across 8 priorities, we began a bottom-up review of atom definitions to ensure quality and consistency before wiring edges.

### Previous Session Work
- Created 60 atoms (Priority 1-8) in `
- Deleted 4 stub files (ike-bottom-up, ike-top-down, project-ike, ike-framework)
- Added definitions to 11 3p DDD atoms that were missing frontmatter definitions

### Current Session State
- **Review doc**: `
- **ALL SECTIONS REVIEWED** (1-9)
- Tim added inline comments with decisions

## Review Summary

### Section 1: First Principles (3p)
| Atom | Decision |
|------|----------|
| `information-theory` | ✏️ Add: "Surprise is information." |
| All others | ✅ OK |

### Section 2-4: DDD Patterns (3p)
- ✅ All definitions approved as-is

### Section 5: Understanding Theory (1p)
| Atom | Decision |
|------|----------|
| `understanding-as-process` | ✏️ Change "exists" → "emerges" |
| `understanding-components` | ❌ DELETE - not an atom |
| `causal-understanding` | ❌ DELETE - not an atom |
| `shared-understanding` | ✅ OK |
| `energy-barrier-problem` | ✏️ RENAME → `understanding-min-energy` |
| `suspended-understanding` | ❓ NEEDS CONTEXT |
| `understanding-maturity-ladder` | ❌ DELETE - not an atom |

### Section 6: Hard Problem of AI (1p)
| Atom | Decision |
|------|----------|
| `regression-paradox` | ✏️ REWRITE (lead with consensus patterns, connect to Shannon) |
| `runtime-emergence` | ✏️ Change "Transformation" → "unexpected change and novel solutions" |
| `semantic-decoherence` | ✏️ REWROTE INLINE |
| `paradigmatic-lock-in` | ⚠️ 3p not 1p |
| `measurement-fallacy` | ❌ 3p and not an atom |
| `hard-problem-of-ai` | ✏️ REWROTE INLINE |

### Section 7: Real Data Framework (1p)
| Atom | Decision |
|------|----------|
| `seven-core-capabilities` | ✏️ REWROTE INLINE |
| `three-forces` | ✅ OK |
| `source-instrumentation` | ❌ not 1p, not atom |
| `source-ownership` | ❌ not 1p, not atom |
| `system-up-thinking` | ✏️ REWROTE INLINE |
| `no-free-lunch-data` | ❌ not atom |
| `thermodynamics-of-data` | ✏️ REWROTE INLINE |
| `deterministic-transformations` | ✏️ REWROTE INLINE |
| `rtfm-principle` | ✏️ REWROTE INLINE |
| `self-validating-systems` | ⚠️ not 1p |
| `semantic-layer` | ⚠️ not 1p |
| `real-data` | ✏️ REWROTE INLINE |

### Section 8: Semantic Operations Framework (1p)
| Atom | Decision |
|------|----------|
| `knowledge-corpus` | ⚠️ not 1p |
| `knowledge-artifacts` | ⚠️ not 1p |
| `corpus-artifact-delta` | 🚧 NEEDS WORK |
| `progressive-semantics` | ✏️ REWROTE INLINE |
| `flexible-edge` | ⚠️ not 1p |
| `hard-schema` | ❌ not 1p, not atom |
| `semantic-coherence` | 🚧 NEEDS WORK (rewrote inline) |
| `semantic-optimization` | 🚧 NEEDS WORK (rewrote inline) |
| `semantic-operations` | 🚧 NEEDS WORK (rewrote inline) |

### Section 9: Architecture Patterns (1p)
| Atom | Decision |
|------|----------|
| `stable-core-flexible-edge` | ⚠️ not 1p, but important 3p |
| `semantic-containers` | 🚧 NEEDS WORK |
| `semantic-governance` | ✅ OK |
| `semantic-contracts` | ✅ OK |
| `intentional-architecture` | ❓ check if 1p or DDD |
| `semantic-invariants` | 🚧 NEEDS WORK ("constitution"/"principles"?) |
| `global-architecture` | ❌ not 1p, not atom |
| `domain-driven-architecture` | 🚧 NEEDS WORK (same as global-architecture - pick one) |
| `model-everything` | ✅ OK |

## Action Items

### 1. DELETE (not atoms)
- `understanding-components.md`
- `causal-understanding.md`
- `understanding-maturity-ladder.md`
- `measurement-fallacy.md`
- `source-instrumentation.md`
- `source-ownership.md`
- `no-free-lunch-data.md`
- `hard-schema.md`
- `global-architecture.md`

### 2. RENAME
- `energy-barrier-problem.md` → `understanding-min-energy.md`

### 3. CHANGE OWNERSHIP (1p → 3p)
- `paradigmatic-lock-in`
- `self-validating-systems`
- `semantic-layer`
- `knowledge-corpus`
- `knowledge-artifacts`
- `flexible-edge`
- `stable-core-flexible-edge`

### 4. APPLY INLINE REWRITES
Update definitions in these atoms from the review doc:
- `semantic-decoherence`
- `hard-problem-of-ai`
- `seven-core-capabilities`
- `system-up-thinking`
- `thermodynamics-of-data`
- `deterministic-transformations`
- `rtfm-principle`
- `real-data`
- `progressive-semantics`
- `semantic-coherence`
- `semantic-optimization`
- `semantic-operations`

### 5. NEEDS WORK (parking lot - don't block progress)
These atoms are correctly classified but definitions need refinement:
- `corpus-artifact-delta`
- `semantic-coherence`
- `semantic-optimization`
- `semantic-operations`
- `semantic-containers`
- `semantic-invariants`
- `domain-driven-architecture`

### 6. NEEDS CLARIFICATION
- `suspended-understanding` - Tim to clarify concept
- `intentional-architecture` - check if 1p or DDD 3p
- `semantic-containers` vs `bounded-context` - overlapping?
- `domain-driven-architecture` vs `global-architecture` - pick one name

## Key Insights

1. **Atoms vs Frameworks**: Atoms are stable concepts. Maturity ladders, component lists, and frameworks USE atoms but aren't atoms themselves.

2. **1p vs 3p**: Many concepts marked 1p are actually industry-standard (3p). Being honest about ownership matters for credibility.

3. **"Needs Work" ≠ Blocking**: Definition quality issues are tracked but don't block atom classification. These go in a parking lot for later refinement.

## Files

- **Review document**: `docs/concept-inventory/definition-review.md` (has all inline notes)
- **Atoms location**: `
- **Previous inventory**: `docs/concept-inventory/consolidated-concept-inventory.md`

## How to Resume

1. ~~**Apply deletes** (Action Item 1)~~ ✅ DONE
2. ~~**Apply rename** (Action Item 2)~~ ✅ DONE
3. ~~**Update ownership** in frontmatter (Action Item 3)~~ ✅ DONE
4. ~~**Copy inline rewrites** from review doc to atom files (Action Item 4)~~ ✅ DONE
5. ~~**Log "needs work"** items to separate tracking doc (Action Item 5)~~ ✅ DONE (see definition-parking-lot.md)
6. **Discuss clarifications** with Tim (Action Item 6) - IN PROGRESS

---

## Architecture Decision: Pattern as Aggregate Root

**Date:** 2025-12-22 (evolved from 2025-12-06)
**Status:** PARTIALLY SUPERSEDED — The "Pattern as sole aggregate root" framing is superseded by [ADR-0012](ADR-0012-pattern-coherence-co-equal-aggregates.md). Pattern remains the aggregate root of the Pattern Aggregate and the core domain concept, but it is not the sole aggregate root for the entire system. Coherence Assessment is a co-equal core aggregate. See ADR-0012 for the full domain model.

### Evolution

The decision evolved through three stages:
1. **Atom** (initial) - Too granular, atoms are components within patterns
2. **Concept** (2025-12-06) - Better, but still abstract
3. **Pattern** (2025-12-22) - Final: applied unit of meaning with business purpose

### Decision

**Pattern is the aggregate root.** A Pattern is an applied unit of meaning with a business purpose, measured for semantic coherence and optimization.

Key insight: What we were calling "concepts" and "domain patterns" are the same thing - patterns. `semantic-coherence` is a pattern. `content-classify-pattern` is a pattern. `DDD` is a pattern we adopt.

### Definition

> **Pattern:** An applied unit of meaning with a business purpose, measured for semantic coherence and optimization. Patterns are stable semantic structures that can be adopted, extended, or modified.

### Structure

```
Pattern (aggregate root - core domain)
    │
    ├── pattern_edge (SKOS: broader, narrower, related)
    │                (Adoption: adopts, extends, modifies)
    │
    └── Entity (DAM - supporting domain)
            │
            ├── edge (PROV-O: derived_from, cites, etc.)
            │
            └── Delivery (publishing - with approval_status, visibility)
```

### Key Changes from Concept Model

1. **Pattern replaces Concept** - Same SKOS properties, but emphasizes applied business purpose
2. **approval_status moved to Delivery** - Per-surface governance, not on entity
3. **visibility moved to Delivery** - Per-surface access control
4. **provenance stays on Pattern** - 1p/2p/3p indicates whose semantic structure
5. **Adoption predicates added** - adopts, extends, modifies (in addition to SKOS)

### Pattern Examples

| Pattern | Provenance | Type |
|---------|------------|------|
| `semantic-operations` | 1p | Methodology |
| `content-classify-pattern` | 1p | Workflow |
| `DDD` | 3p | Standard we adopt |
| `SKOS` | 3p | Standard we adopt |
| `semantic-coherence` | 1p | Core concept |

### Rationale

1. **Applied meaning**: Patterns have business purpose, not just semantic definition
2. **Measurable**: Patterns are measured for semantic coherence and optimization
3. **Unifies concepts and workflows**: No artificial distinction between "semantic concepts" and "domain patterns"
4. **Per-surface governance**: approval_status on Delivery allows different states per publication target
5. **Clean adoption model**: 3p patterns are adopted/extended/modified to become 1p

### Pattern Adoptions

| Domain | Pattern | Standard |
|--------|---------|----------|
| Core (Patterns) | SKOS | W3C |
| Supporting (DAM) | Digital Asset Management | Industry |
| Supporting (Lineage) | PROV-O | W3C |
| Generic (Attribution) | Dublin Core | W3C |

### Implementation

- [x] Create phase2-schema.sql with pattern table
- [x] Update UBIQUITOUS_LANGUAGE.md (v7.0.0)
- [x] Update SCHEMA_CHANGELOG.md (v7.0.0 entry)
- [x] Archive phase1-schema.sql to schemas/archive/
- [x] Apply schema to Supabase (v7.0.0 deployed 2025-12-23)

See: [phase2-schema.sql](../../schemas/phase2-schema.sql) for implementation.

**Note:** Pattern-enforcing ingestion is a separate concern tracked in .
