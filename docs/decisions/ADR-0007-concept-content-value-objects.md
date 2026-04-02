# ADR-0007: Concept Content Value Objects

> **Status:** Proposed
> **Date:** 2025-12-07
> **Related Issue:**  - Schema Option: Concept Content Value Objects
> **Builds On:** [ADR-0004-concept-aggregate-root](./ADR-0004-concept-aggregate-root.md)
> **Design Doc:** None — Proposed, not implemented. Content relates to DD-0005 domain model but is a separate proposed schema change.

---

## Executive Summary

Proposes a schema pattern where **Concept** remains the aggregate root, with **Content Value Objects** representing composable pieces (atoms, lenses, examples, etc.) that belong to the concept. Surfaces provide governance boundaries for where content is delivered.

---

## Context

The current schema has:
- `concept` - aggregate root with definition, provenance, SKOS relationships
- `concept_edge` - relationships between concepts
- `entity` - independent content with own lifecycle
- `surface` / `delivery` - publication destinations

**Problem:** Where do composable content pieces live? Examples of content that belongs to a concept but isn't the concept itself: atoms (canonical markdown), lenses (perspectives/framings), code examples, DIKW tables, diagrams. These aren't independent entities — they have no meaning without their parent concept.

---

## Decision

1. Model content pieces as **value objects** in a `concept_content` table with composite PK `(concept_id, content_type, version)` — no independent identity.
2. Value objects are immutable (version rather than mutate) and cascade-delete with the parent concept.
3. Add a `governance` column to `surface` table with four tiers: `strict` (source of truth), `loose` (flag for review on drift), `ephemeral` (snapshot, don't track), `frozen` (intentionally out of date).
4. Rejected alternatives: everything in Entity table (entities have independent lifecycle), JSONB array on concept (harder to query/version), separate tables per type (proliferates tables).

---

## Consequences

### Positive
- Clear ownership: content belongs to concept, not floating independently
- Composable: hubs can query `concept_content` to assemble views
- Versioned: content evolves without losing history
- Governance tiers: different surfaces have appropriate drift policies

### Negative
- More complex than flat `entity` table
- Requires content type registry (what types are valid?)
- Version management overhead

### Risks
- Over-engineering if content types don't stabilize
- May need to split large concepts if they accumulate too much content

---

## References

- [ADR-0004: Concept Aggregate Root](./ADR-0004-concept-aggregate-root.md)
- Vernon, V. "Implementing Domain-Driven Design" - Value Object patterns
- Evans, E. "Domain-Driven Design" - Aggregate design
