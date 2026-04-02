# ADR-0011: Agent Governance Model

> **Status:** Complete
> **Date:** 2026-02-14
> **Related Issue:** 
> **Extends:** ADR-0009 (three-layer architecture), ADR-0004 (pattern as aggregate root)
> **Spec:** [schemas/GOVERNANCE_MODEL.md](../../schemas/GOVERNANCE_MODEL.md)
> **Design Doc:** [DD-0010 Agent Governance Model](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md)

## Executive Summary

Defines how SemOps governs agent-produced and agent-managed artifacts. Introduces a 5-stage universal lifecycle (draft/active/stable/deprecated/archived), two governance modes (self-correcting andon cord for internal operations, hard gates for public boundary), and four driving principles that formalize the transition from manual governance to automated agent governance.

## Context

### What existed before

- `lifecycle_stage` with 3 values (draft/active/retired) — implemented in source_config.py and source configs
- `delivery.approval_status` — exists in schema but not enforced
- Ingestion episodes (ingestion_episode table) — already tracking agent operations
- Manual governance via session notes, ADRs, commit messages, architecture sync

### What was missing

- No universal lifecycle model across all entity types
- No definition of what agents can do autonomously
- No spec for how manual governance transitions to automated
- No formal connection between coherence scoring and lifecycle transitions
- `active` doing double duty (operational AND coherence baseline)
- `retired` conflating deprecated (still visible) and archived (gone)

## Decision

1. **Five-stage universal lifecycle** — Replace 3-value lifecycle with DRAFT/ACTIVE/STABLE/DEPRECATED/ARCHIVED. See [DD-0010 — Five-Stage Universal Lifecycle](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md#five-stage-universal-lifecycle).
2. **Universal governance matrix** — Same lineage/coherence/search treatment for all entity types. See [DD-0010 — Universal Governance Matrix](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md#universal-governance-matrix).
3. **Two governance modes** — Andon cord (internal, self-correcting) and hard gate (public boundary). See [DD-0010 — Governance Modes](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md#governance-modes).
4. **Sticky lifecycle + episode versioning** — Lifecycle stage does not reset on mutation; episode chain is the version history. See [DD-0010 — Sticky Lifecycle](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md#sticky-lifecycle-and-episode-based-versioning).
5. **Scale projection levels (L1-L5)** — Five levels from human-driven to full agentic governance. See [DD-0010 — Scale Projection](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0010-agent-governance-model.md#scale-projection-manual-to-agentic).

## Consequences

### Positive

- Universal lifecycle eliminates per-type special cases
- Coherence scoring gains a formal role in governance (quality gate)
- Clear path from manual (L1) to autonomous (L5) governance
- Draft entities become proactively useful (forecast zone)
- Episode chain provides full version history without versioning table

### Negative

- Migration needed: existing `retired` entities must map to `deprecated` or `archived`
- `stable` stage requires defining promotion criteria per entity type
- Governance model doc adds another schema-level document to maintain

### Risks

- Coherence scoring does not yet exist at the level needed for auto-recovery (L4-L5)
- Per-entity governance specs needed to operationalize the universal matrix

## References

- [schemas/GOVERNANCE_MODEL.md](../../schemas/GOVERNANCE_MODEL.md) — full governance model spec
-  — design discussion
-  — alignment/reframing
- [ADR-0009](ADR-0009-strategic-tactical-ddd-refactor.md) — three-layer architecture
- [ADR-0004](ADR-0004-schema-phase2-pattern-aggregate-root.md) — pattern as aggregate root
