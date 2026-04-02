# Scale Projection — KB Results

## Core Concept Entity

| Entity | Corpus | Type | Source |
|--------|--------|------|--------|
| `scale-projection` | core_kb | concept | `docs-pr: SEMANTIC_OPTIMIZATION/scale-projection.md` |

> *Scale Projection uses the path of infrastructure scaling as a diagnostic — testing and validating your core business logic and architecture by sketching what changes at the next order of magnitude.*

## ADRs

| ADR | Repo | Relationship |
|-----|------|-------------|
| **ADR-0004: Mirror Architecture** | semops-orchestrator | Superseded — constructs absorbed into Scale Projection via  |
| **ADR-0009: Three-Layer Architecture** | semops-data | Scale Projection sits in the Pattern layer (WHY) |
| **ADR-0011: Agent Governance Model** | semops-data | § Scale Projection levels for transition authority |

## Session Notes (this repo)

| File | Issue | Key Content |
|------|-------|-------------|
| `ISSUE-112-lifecycle-stage-design.md` |  | Coherence = retrospective, Scale Projection = prospective. Together form the SemOps quality gate. |
| `ISSUE-122-strategic-tactical-ddd-refactor.md` |  | `scale-projection` as cross-cutting 1P pattern across style-learning, synthesis-simulation, autonomous-execution |
| `ISSUE-134-phase-b-cleanup.md` |  | mirror-architecture renamed to scale-projection; pattern registry cleanup |
| `ISSUE-145-pattern-refinement.md` |  | "Reference arch = domain-down, Scale projection = infrastructure-up, Coherence = where they meet" |
| `ISSUE-149-manual-scale-projection.md` |  | Four Data System Types lens is essential; type-driven projection produces architectural insights |
| `ISSUE-96-scale-projection-feature-set.md` |  | Feature-set coordination (projection structure, review process, tooling) |

## GitHub Issues (from ingested issues)

| Issue | Repo | Title |
|-------|------|-------|
|  | semops-orchestrator | Refine Scale-Projection Pattern (definition) |
|  | semops-orchestrator | Feature-set development (tooling & process) |
|  | semops-orchestrator | Absorb generate-proposed/promote-proposed into Scale Projection |
|  | semops-orchestrator | Define scale vectors and resourcing methodology |
|  | semops-orchestrator | Manual projection workflow — run scenarios against repo docs |
|  | data-pr | Synthetic Data Generation from Domain Models (@scale-projection project) |

## Related Patterns (via chunk cross-references)

`rlhf`, `seci`, `data-profiling`, `synthetic-data`, `containerization`, `mirror-architecture` (retired → scale-projection)

## Other Docs

- **GAPS.md** — the live gap-tracking document for current deployment
- **STRATEGIC_DDD.md** — scale-projection mapped to 6 capabilities across 3 repos
- **PATTERNS.md** — narrative description of the infrastructure-up direction

---

## Pattern Layer (SKOS/Adoption)

```
scale-projection (1P)
  ──extends──► rlhf (3P)
  ──extends──► seci (3P)
  ──extends──► data-profiling (3P)
  ──extends──► ddd (3P)
```

All four are `extends` edges at strength 1.0 — the 1P pattern synthesizes these 3P foundations.

## Architecture Layer (Capabilities → Patterns)

Four capabilities implement `scale-projection`:

```
scale-projection (capability)  ──implements──► scale-projection (pattern)
                               ──implements──► rlhf (pattern)
                               ──implements──► seci (pattern)
style-learning (capability)    ──implements──► scale-projection (pattern)
synthesis-simulation (cap.)    ──implements──► scale-projection (pattern)
autonomous-execution (cap.)    ──implements──► scale-projection (pattern)
```

## Delivery Layer (Capabilities → Repos)

```
scale-projection (capability)
  ──delivered_by──► semops-orchestrator
  ──delivered_by──► publisher-pr
  ──delivered_by──► data-pr

style-learning ──delivered_by──► publisher-pr
synthesis-simulation ──delivered_by──► data-pr
autonomous-execution ──delivered_by──► semops-orchestrator
```

## Full Traversal (Composite)

```
                        ┌─────── rlhf (3P) ◄──implements── scale-projection (cap)
                        │                   ◄──extends───── scale-projection (pat)
                        │
                        ├─────── seci (3P) ◄──implements── scale-projection (cap)
                        │                  ◄──extends───── scale-projection (pat)
                        │
scale-projection (1P) ──┤─────── data-profiling (3P) ◄──extends
                        │
                        └─────── ddd (3P) ◄──extends

   Implementing Capabilities          Delivering Repos
   ─────────────────────────          ────────────────
   scale-projection (cap) ──────────► semops-orchestrator, publisher-pr, data-pr
   style-learning ──────────────────► publisher-pr
   synthesis-simulation ────────────► data-pr
   autonomous-execution ────────────► semops-orchestrator
```

The pattern has strong coverage: 4 capabilities across 3 repos — consistent with its role as a cross-cutting 1P innovation pattern.
