# ADR-0006: V1 Documentation Plan

> **Status:** In Progress
> **Date:** 2025-12-07
> **Related Issues:** , 
> **Builds On:** [ADR-0004-concept-aggregate-root](./ADR-0004-concept-aggregate-root.md)
> **Design Doc:** None — planning ADR

---

## Executive Summary

Consolidates scattered planning docs into a single execution plan for V1 documentation. Defines three phases: Architecture Documentation, Source Document Cleanup , and Publisher Integration .

---

## Context

Multiple overlapping planning documents existed:
- `concept-promotion-plan.md` (phases 1-4 complete)
- `ISSUE-62-SOURCE-INGESTION-PIPELINE.md` (complete)
- `ADR-0004-schema-phase2-pattern-aggregate-root.md` (complete)
- Various open GitHub issues with unclear priorities

This created confusion about what to work on next. The infrastructure is built (61 concepts, 195 edges, 183 classifications, Neo4j synced) but documentation explaining the system is incomplete.

---

## Decision

1. Create a single ADR that tracks V1 documentation work and close/defer scattered issues.
2. Execute in order: Phase 1 (Architecture Documentation) then Phase 2 ( Source Document Cleanup) then Phase 3 ( Publisher Integration).
3. Defer Advanced RAG , 3P Ingestion , Semantic Coherence , MCP Server , and Concept Promotion — none block V1.

---

## Consequences

**Positive:**
- Single source of truth for V1 planning — no more scattered docs
- Clear execution order with explicit deferral of non-blocking work
- Closed 4 superseded issues (, , , )

**Negative:**
- Phases 1 and 3 remain incomplete
- Publisher integration  blocked until architecture docs are done

---

## References

- 
- 
- [ADR-0004: Concept Aggregate Root](./ADR-0004-concept-aggregate-root.md)
