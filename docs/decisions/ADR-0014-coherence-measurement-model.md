# ADR-0014: Coherence Measurement Model

> **Status:** Draft
> **Date:** 2026-03-15
> **Related Issue:** , 
> **Builds On:** [ADR-0012: Pattern + Coherence as Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md)
> **Design Doc:** [DD-0006: Coherence Measurement Model](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md)

---

## Executive Summary

This ADR defines the canonical measurement model for Semantic Coherence in SemOps. Coherence measures how well reality matches intent — and all coherence measurement is **goal-driven**. What varies is how the goal is defined: from rules encoded in the architecture graph (deterministic) to structured acceptance criteria (low reasoning) to qualitative directional vectors (higher reasoning). The model uses **SC = (A × C × S)^(1/3)** as its measurement dimensions, operates at two cadences (operational and batch), and assigns every entity a coherence score from the moment it exists — including unclassified work at the flexible edge (SC ≈ 0). As metadata density and agentic lineage mature, more goals shift toward the deterministic end of the spectrum without changing the model.

---

## Goal

Coherence measurement answers one question: **does reality match intent?**

Intent comes from goals — acceptance criteria, strategic directions, metric targets, or simply "the rules are satisfied." Goals can be expressed at any level of abstraction, from a fitness function to a qualitative aspiration. What makes coherence measurement work is that **every goal must be interpreted down to the level of the explicit architecture** to be evaluated.

The explicit architecture — the pattern graph, capability mappings, SKOS edges, provenance classifications, integration map — is the **resolution layer**. It is not the source of all intent, but it is the structure against which all intent is assessed. The inference cost of any coherence evaluation is the distance between how the goal is stated and the architecture graph. A rule encoded as a fitness function has zero distance. A directional goal like "more concrete, less theoretical" requires interpretation steps to decompose it into architecture-level checks.

Pattern provides the reference structure — the semantic graph that makes measurement possible. Coherence measures the gap between goals and what actually exists. The two are co-equal core aggregates of SemOps (ADR-0012) — Pattern provides structure, Coherence evaluates alignment. Together they form the **Semantic Optimization Loop**:

```text
Goals ──interpreted down to──→ Architecture (target state)
    │                                │
    │                    Coherence ──measures──→ gap from target (loss)
    │                                │
    └── Optimization ←──minimizes────┘
```

Coherence findings drive action: coverage gaps increase loss, newly formalized alignments decrease loss (compounding), and regressions — previously coherent things that broke — trigger corrective action. This ADR specifies how alignment is measured.

The implementation pattern for reducing inference distance — making goals easier to evaluate against the architecture — is [governance-as-strategy](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/governance-as-strategy.md), which extends [data management](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-management.md) disciplines (metadata, lineage, quality, provenance) from operational data to semantic operations. See [DD-0006 — Governance as Strategy](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#governance-as-strategy-the-implementation-pattern) for the full implementation pattern.

### What Coherence Is Not

- **Not a gate.** Coherence is audit, not permission. Aggregate root invariants (SKOS hierarchy, provenance rules) protect the stable core. The flexible edge is free to exist. Coherence reports the cost of that freedom.
- **Not operational health.** Error rates, latency, uptime are infrastructure signals that may *feed* coherence (as stability inputs), but coherence measures semantic alignment, not system health. This is the `data-quality` → `semantic-coherence` distinction: operational health is a necessary foundation, not the thing being measured.
- **Not a one-time assessment.** Coherence is continuous — two audit layers running perpetually, compounding their own effectiveness.

---

## Decisions

Full design detail for each decision is in [DD-0006](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md).

- **D1.** All coherence measurement is goal-driven; goals vary by inference distance to the architecture graph along a four-type continuum (rule execution → criteria-based → directional → metric-driven). [DD-0006 §Goal-Driven Coherence](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#goal-driven-coherence-the-core-principle)
- **D2.** SC is measured as the geometric mean of three dimensions: Availability (A), Consistency (C), Stability (S) — SC = (A × C × S)^(1/3). [DD-0006 §Measurement Dimensions](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#measurement-dimensions-sc--a-x-c-x-s13)
- **D3.** Scoring is scaffolded by patterns: goals and analytics are first-class pattern types whose dependency chains make scoring self-diagnosing. [DD-0006 §Pattern-Scaffolded Scoring](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#pattern-scaffolded-scoring)
- **D4.** Goal type is earned by measurement capability maturity, not chosen at goal definition time. [DD-0006 §Goal-Type Continuum Is Earned](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#goal-type-continuum-is-earned-not-chosen)
- **D5.** Every entity participates from creation; the flexible edge scores SC ≈ 0 (accurate measurement, not failure). [DD-0006 §Flexible Edge](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#universal-participation-flexible-edge--sc-approximately-0)
- **D6.** Lifecycle transition to `active` serves as the stability baseline — no separate snapshot ceremony. [DD-0006 §Lifecycle as Stability Baseline](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#lifecycle-as-stability-baseline)
- **D7.** Governance-as-strategy is the implementation pattern: four data management disciplines (metadata, lineage, quality, provenance) extended from operational data to semantic operations. [DD-0006 §Governance as Strategy](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#governance-as-strategy-the-implementation-pattern)
- **D8.** Two operational cadences: operational/runtime (triggered by work) and batch/scheduled (periodic sweeps), using the same model. [DD-0006 §Operational Cadences](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#operational-cadences)
- **D9.** Implementation trajectory proceeds in three stages: document-based → manifest/metadata-driven → graph-driven, each absorbing the previous. [DD-0006 §Implementation Trajectory](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md#implementation-trajectory)

---

## Consequences

### Positive

- **Single canonical reference** for coherence measurement — eliminates ambiguity across session notes, specs, and ADRs
- **Goal-driven unification** — all coherence measurement follows the same model regardless of goal type; rule execution, criteria-based, and directional goals are one continuum, not separate systems
- **Universal participation** — no special cases for unmeasured entities; everything has a score from creation
- **Incremental implementation** — start with rule execution goals (partially implemented via fitness functions and `/arch-sync`), add criteria-based and directional as infrastructure matures
- **Self-improving** — metadata density and agentic lineage push goals toward determinism over time; each formalized finding becomes a new rule execution check
- **Pattern-anchored** — coherence is always relative to intent, never a free-floating quality metric
- **Extensible** — new goal types (metric-driven, analytics-based) slot into the continuum without model changes

### Negative

- **Criteria-based and directional goals are largely unimplemented** — the model is ahead of the infrastructure
- **Assessment unit granularity** (pattern × capability × domain) may be too fine for early implementation — may need to start at pattern level and decompose later

### Risks

- **Over-modeling before data exists.** Mitigation: implementation phases in PROJECT-18 start with what's already instrumented (rule execution goals, Layer 1) and expand incrementally.
- **Geometric mean sensitivity.** A single zero in any dimension collapses the score. Mitigation: this is by design (each dimension is necessary), but scoring needs to handle "unmeasurable" differently from "measured at zero."

---

## References

### ADRs

- [ADR-0012: Pattern + Coherence as Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md) — foundation: establishes the two core aggregates
- [ADR-0009: Three-Layer Architecture](./ADR-0009-strategic-tactical-ddd-refactor.md) — layers as organizational modules

### Design Doc

- [DD-0006: Coherence Measurement Model](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md) — full design detail

### Pattern Documents

- [governance-as-strategy.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/governance-as-strategy.md) — 1P pattern; governance as strategic execution, not compliance
- [data-management.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-management.md) — 3P adopted; the discipline hierarchy this model extends
- [data-quality.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-quality.md) — 3P narrower of data-management; operational quality that SC generalizes
- [semantic-coherence.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/analytics/semantic-coherence.md) — 1P pattern; describes *why* coherence measurement exists
- [data-system-classification.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md) — 1P; four data system types with distinct coherence characteristics

### Implementation

- [fitness-functions.sql](../../schemas/fitness-functions.sql) — current Layer 1 sensors
- [UBIQUITOUS_LANGUAGE.md](../../schemas/UBIQUITOUS_LANGUAGE.md) — domain definitions

### External

-  — design session that produced this model
-  — pattern types for scoring, dependency chain, self-diagnosing model
- [PROJECT-18: Semantic Coherence Measurement](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/project-specs/PROJECT-18-coherence-measurement.md) — execution plan and acceptance criteria
