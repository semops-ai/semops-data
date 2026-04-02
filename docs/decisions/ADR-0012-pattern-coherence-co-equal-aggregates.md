# ADR-0012: Pattern + Coherence as Co-Equal Core Aggregates

> **Status:** Draft
> **Date:** 2026-02-19
> **Related Issue:** 
> **Supersedes:** ADR-0004 § "Pattern as sole aggregate root" framing
> **Related:** ADR-0009 (three-layer architecture),  (bounded context alignment)
> **Design Doc:** [DD-0005: Domain Model — Patterns and Coherence](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0005-domain-model-patterns-and-coherence.md)

## Executive Summary

The current domain model declares Pattern as the sole aggregate root for the entire SemOps bounded context. This conflates "most important domain concept" with "aggregate root" (a DDD transactional consistency boundary). The schema already implements multiple independent aggregates. More fundamentally, Pattern alone captures only the prescriptive force of SemOps. The evaluative/directive force — coherence measurement — has no first-class domain object.

This ADR introduces **Coherence Assessment** as a co-equal core aggregate alongside Pattern, formalizes the multiple aggregates that already exist implicitly in the schema, and reframes the Semantic Optimization Loop as the Pattern ⇄ Coherence feedback loop.

## Context

### What ADR-0004 got right

1. **Pattern as the core domain concept** — the stable meaning everything traces to
2. **SKOS for taxonomy** — broader/narrower/related relationships
3. **Provenance** (1P/2P/3P) — whose semantic structure is this
4. **Adoption lineage** — adopts/extends/modifies relationships

### What needs revision

**"Pattern is the aggregate root of the domain model — the single entity through which all access to the aggregate occurs."**

In Evans' DDD, an aggregate root is the transactional consistency boundary leader for its aggregate — not the most important concept in the system. The schema already has multiple independent lifecycles:

- You create Deliveries without going through Pattern
- You register Surfaces independently
- You create Brands and Products with their own lifecycle
- Capabilities and Repositories have independent identity

These are separate aggregates, not children of Pattern.

### The missing force

SemOps described briefly: **business optimization with AI by operating on optimal semantic objects — using semantically rich objects to address large chunks of a domain, then constantly auditing and realigning to ensure executability.**

This has two forces:

1. **Prescriptive** — Pattern defines "what should we look like?" (adopt 3P → innovate 1P)
2. **Evaluative/Directive** — Coherence measures "does reality match intent?" and DRIVES changes

The prescriptive force is well-modeled (Pattern + SKOS + provenance). The evaluative force exists only as:
- A pattern (`semantic-coherence`) — the *idea* of measurement
- A capability (`coherence-scoring`) — the *ability* to measure
- Views (`pattern_coverage`, `capability_coverage`) — computed snapshots
- Fitness functions — binary pass/fail checks

There is no domain object for the measurement itself — no way to track "pattern X had coherence score Y as of date Z, triggered by change W, with these gaps identified."

### Design session insight: the CRM lifecycle

A concrete example revealed the gap:

1. **Adopt**: Register "Zendesk" as a 3P CRM pattern (reverse-engineer API to core components)
2. **Implement**: Build capabilities aligned to the pattern
3. **Achieve coherence**: System reaches a stable, aligned state — *this moment has no domain representation*
4. **Evolve**: After months, diverge from Zendesk. Create 1P CRM pattern with `extends` → Zendesk (SKOS preserved)
5. **Coherence drops**: Changes affect financial-pipeline, attention-management, publishing — cross-cutting impact
6. **Assess and realign**: Measure gaps, act on them, restore coherence
7. **Traceability**: Can always "run it back" via SKOS chain to original CRM functions

Steps 3, 5, and 6 have no first-class domain objects. Pattern handles 1, 4, and 7. Entity handles 2. Coherence Assessment would handle 3, 5, and 6.

### Coherence is directive, not just evaluative

A critical insight: coherence assessment doesn't just report status — it drives action. A coherence assessment can trigger:

| Coherence signal | Action |
|-----------------|--------|
| Gap detected (missing coverage) | Adopt new pattern or create capability |
| Misalignment (conflict between patterns) | Modify or constrain a pattern |
| Regression (new pattern broke existing coherence) | **Revert or remove the pattern** |
| Drift (implementation diverged from intent) | Realign implementation OR evolve pattern to match reality |

Coherence can command pattern reversal or feature removal just as easily as it can prompt additions. This makes it a true peer to Pattern, not a subordinate measurement.

### Earlier framing: "Pattern and Provenance"

An earlier analysis identified the core pair as "Pattern and Provenance." This was close but incomplete — Provenance (1P/2P/3P) is a *property* of Pattern that feeds into coherence assessment as one signal among several. It answers "whose structure is this?" — important, but not the full evaluative force.

## Decision

Design detail has been extracted to [DD-0005: Domain Model — Patterns and Coherence](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0005-domain-model-patterns-and-coherence.md). The decisions are:

- **D1.** Pattern and Coherence Assessment are co-equal core aggregates — Pattern pushes (prescriptive), Coherence aligns (evaluative/directive). See DD-0005 §1.
- **D2.** Classify all domain objects using DDD building blocks — core aggregates, supporting aggregates, entities, value objects. See DD-0005 §2.
- **D3.** Coherence is audit, not gate — aggregate root invariants protect the stable core; coherence audits the gap between core and flexible edge. See DD-0005 §3.
- **D4.** Three modes of coherence: governance, discovery, regression. Deferred from operational workflow  but preserved as conceptual framework. See DD-0005 §4.
- **D5.** Existing fitness functions and coverage views are coherence sensors feeding into assessments. See DD-0005 §5.
- **D6.** Full traceability chain: Pattern → Capability → Script/Infrastructure. Every break is an audit finding. See DD-0005 §6.
- **D7.** Revise aggregate root language — Pattern is aggregate root of Pattern Aggregate, not of the whole system. See DD-0005 §7.
- **D8.** Coherence Assessment aggregate shape: trigger, scope, measurements (A/C/S), SC score, gaps, actions, lifecycle state. See DD-0005 §8.
- **D9.** `semantic-coherence` pattern describes its own aggregate — clean recursion. See DD-0005 §9.
- **D10.** When optimization becomes operational, coherence scoring is the objective function. See DD-0005 §10.

## Consequences

### What actually changes

The schema is already correct — multiple independent aggregates are already implemented. The change is primarily **vocabulary and framing**, not infrastructure.

#### Nothing changes (schema is already right)

- `pattern` + `pattern_edge` tables — Pattern Aggregate, already correct
- `entity` table with type discriminator — still works for content, capability, repository
- `edge` table with all predicates — still works
- `surface` + `surface_address` — Surface Aggregate, already correct
- `delivery` table — already correct (reclassified from separate aggregate to child of Content)
- `brand` + `product` + `brand_relationship` — Brand Aggregate, already correct
- All views, fitness functions, indexes, scripts — no changes

#### Documentation updates (framing changes)

- UBIQUITOUS_LANGUAGE.md: "Pattern as sole aggregate root" → "Pattern as aggregate root of Pattern Aggregate + core domain concept"
- UBIQUITOUS_LANGUAGE.md: Add Coherence Assessment definition
- UBIQUITOUS_LANGUAGE.md: Reclassify Capability as Entity, Repository as Value Object
- STRATEGIC_DDD.md: Add Coherence Assessment to domain model
- ADR-0004: Mark "aggregate root" section as superseded by ADR-0012

#### Future work (deferred)

- Coherence Assessment table/schema — when coherence scoring is operational
- PIM/CRM pattern registration — when CRM is built out
- Schema.org pattern registration — when Brand aggregate is active

### Positive

- **Completes the Semantic Optimization Loop** — both forces have first-class domain representation
- **Formalizes implicit aggregates** — no more claiming one aggregate root governs everything
- **Low implementation cost** — schema doesn't change; this is a vocabulary correction
- **Enables temporal coherence tracking** — assessments over time show whether changes helped or hurt
- **Coherence becomes actionable** — gaps and recommended actions are domain objects, not just dashboard metrics
- **Cleaner DDD alignment** — aggregate boundaries match transactional realities
- **Pattern prescribes aggregate structure** — each supporting aggregate traces to a 3P pattern (DAM → Content, Schema.org → Brand), demonstrating that Pattern produces structure at every level

### Negative

- **Schema addition required (deferred)** — Coherence Assessment needs a new table eventually
- **ADR-0004 framing needs revision** — "Pattern as aggregate root" language in multiple docs

### Risks

- **Over-modeling** — Coherence Assessment could become complex before there's data to populate it. Mitigation: start with the aggregate shape, defer schema until coherence scoring is operational.

## References

- [DD-0005: Domain Model — Patterns and Coherence](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0005-domain-model-patterns-and-coherence.md) — full design detail
- [ADR-0004: Pattern as Aggregate Root](ADR-0004-schema-phase2-pattern-aggregate-root.md) — foundational, partially superseded
- [ADR-0009: Three-Layer Architecture](ADR-0009-strategic-tactical-ddd-refactor.md) — layers as organizational modules
- [UBIQUITOUS_LANGUAGE.md](../../schemas/UBIQUITOUS_LANGUAGE.md) — domain definitions
- [STRATEGIC_DDD.md](../STRATEGIC_DDD.md) — capability registry
-  — tracking issue
-  — bounded context alignment (related)
- Evans, Eric. *Domain-Driven Design* (2003) — Aggregates, Ch. 6
