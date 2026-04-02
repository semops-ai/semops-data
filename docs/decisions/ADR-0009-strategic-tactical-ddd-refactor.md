# ADR-0009: Strategic/Tactical DDD Refactor — Three-Layer Architecture

> **Status:** In Progress
> **Date:** 2026-02-07
> **Related Issue:** 
> **Supersedes:** Portions of ADR-0004 Section "Structure" and ADR-0005 Section 3 "DDD Core Three Layers"
> **Blocks:** , 
> **Design Doc:** [DD-0005](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0005-domain-model-patterns-and-coherence.md)

---

## Executive Summary

Strategic DDD was never formally defined — we jumped straight to Tactical DDD (Pattern as aggregate root, Entity as DAM layer). This created scope creep in the DAM Entity layer, missing strategic concepts (capabilities, integration relationships), and overloading of `pattern_type`. This ADR introduces a three-layer architecture that separates **Pattern** (core domain — the WHY), **Architecture** (strategic design — the WHAT/WHERE), and **Content** (DAM publishing — the output).

---

## Context

### What worked (ADR-0004 decisions that remain valid)

1. **Pattern as aggregate root** — Stable, proven, no change needed.
2. **SKOS for pattern relationships** — Working well for taxonomy.
3. **PROV-O for entity edges** — Sound provenance model.
4. **Per-surface governance on Delivery** — Correct separation of concerns.
5. **Provenance on Pattern** (1p/2p/3p) — Right place for ownership signal.

### What's broken

1. **DAM Entity layer scope creep** — DAM became a catch-all for "everything that isn't a Pattern," including architecture metadata (repos, data flows, subdomains) that doesn't belong there.
2. **Missing Strategic DDD** — Subdomains, context map, and integration patterns are documented in prose but not formalized in the domain model.
3. **Pattern aggregate root overloading** — `pattern_type` tries to classify concepts, domain patterns, architecture metadata, AND topology under one aggregate.
4. **Naming collision** — Schema "Entity" means "DAM content artifact," not "DDD Entity," causing confusion in domain modeling discussions.

---

## Decision

1. **Three-layer architecture**: Pattern (core domain, the WHY), Architecture (strategic design, the WHAT/WHERE), Content (DAM publishing, the output). Each layer links upward to Pattern.
2. **Single entity table with type discriminator**: `content` (DAM atoms, rolled up via Surface/Delivery/Brand/Product), `capability` (what the system delivers, traces to >=1 Pattern), `repository` (where implementation lives, delivers capabilities).
3. **Every capability must trace to >=1 pattern** — capability-to-pattern coverage is a measurable coherence signal.
4. **Data flows are emergent from capabilities**, not explicitly modeled.
5. **Subdomains are groupings of capabilities**, not repo boundaries. Repos are agent role boundaries.
6. **SemOps has one bounded context** (single UBIQUITOUS_LANGUAGE.md).
7. **Integration relationships are rich, first-class edges** between repos, typed by DDD integration patterns (shared-kernel, conformist, etc.).
8. **pattern_type reduced** to `concept` and `domain` only — `architecture` and `topology` removed (become entity_type: capability/repository).
9. **DDD is the primary architecture** — DAM, SKOS, PROV-O bolt on as adopted 3P patterns serving specific roles within DDD.

---

## Consequences

### Positive

- **Clear separation of concerns** — Each layer has a defined purpose. No more "is this a pattern or an entity?"
- **Strategic DDD formalized** — Capabilities, repos, and integration patterns become queryable.
- **Coherence measurement** — Capability-to-pattern coverage provides a measurable quality signal.
- **DAM layer recovers original intent** — Content publishing artifacts only, as designed.
- **Pattern table cleaned up** — Architecture/topology concepts move to entity layer where they belong.

### Negative

- **Schema migration required** — Existing entities may need `entity_type` backfill.
- **Edge predicate expansion** — New predicates (`implements`, `delivered_by`, `integration`) need to be added.
- **Multi-document update** — UBIQUITOUS_LANGUAGE.md, GLOBAL_ARCHITECTURE.md, ADR-0005 all need revision.

### Risks

- **Over-modeling** — Capability/repo entities could become maintenance overhead if not populated from real data.
- **primary_pattern_id insufficiency** — Capabilities may need many-to-many pattern relationships. Mitigation: use edges with `implements` predicate in addition to `primary_pattern_id`.

---

## References

- [ADR-0004: Schema Phase 2 — Pattern as Aggregate Root](ADR-0004-schema-phase2-pattern-aggregate-root.md) (foundational, still valid)
- [UBIQUITOUS_LANGUAGE.md v8.0.0](../../schemas/UBIQUITOUS_LANGUAGE.md)
- [SCHEMA_REFERENCE.md v8.0.0](../../schemas/SCHEMA_REFERENCE.md)
- [GLOBAL_ARCHITECTURE.md v3.1.0](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/GLOBAL_ARCHITECTURE.md)
-  — design session log
- Coordination: 
