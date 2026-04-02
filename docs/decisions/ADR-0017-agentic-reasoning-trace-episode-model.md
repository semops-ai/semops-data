# ADR-0017: Agentic Reasoning Trace in Episode Model

> **Status:** Draft
> **Date:** 2026-03-30
> **Related Issue:** 
> **Builds On:** [ADR-0014](./ADR-0014-coherence-measurement-model.md) (Coherence Measurement Model), [ADR-0015](./ADR-0015-ingestion-pipeline-architecture.md) (Ingestion Pipeline Architecture)
> **Design Doc:** [DD-0019: Agentic Lineage Graph Infrastructure](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0019-agentic-lineage-graph-infrastructure.md)

---

## Executive Summary

Extend the `ingestion_episode` table to capture agentic reasoning strategy metadata — how agents reason, not just what data they retrieve. This enables SC measurement to correlate coherence scores with reasoning patterns (CoT, ReAct, ToT, etc.) and context assembly quality, turning the episode model from a provenance ledger into a context engineering measurement surface.

---

## Context

### The Gap

The episode model (schema v7.1.0, migration 001) captures **what** happened — operation type, target, context pattern/entity IDs, agent identity, token usage. It does not capture **how** the agent reasoned about that context:

- Was it sequential (Chain of Thought)?
- Did it branch and evaluate (Tree of Thought)?
- Did it interleave observation and action (ReAct)?
- How much of the assembled context was actually used?

Without reasoning strategy metadata, SC measurement can answer "what was the coherence score?" but not "does the reasoning approach affect coherence?" This is the difference between measurement and diagnosis.

### Five-Primitive Isomorphism

The five agent primitives (Model, Tools, Memory, Context, Orchestration) — established by  and classified by  — map directly to SemOps constructs and SC dimensions:

| Agent Primitive | SemOps Equivalent | SC Dimension |
|----------------|-------------------|--------------|
| Model | Script (execution unit) | — |
| Tools | Capability (what it can do) | Availability |
| Memory | Pattern registry + episodic store | Stability |
| Context | Assessment unit (pattern × capability × domain) | Consistency |
| Orchestration | Workflow (single/multi-agent) | — |

Planning is not a sixth primitive — it is emergent behavior from Model + Prompt + Memory interaction, implemented through prompting techniques. The `reasoning_pattern` enum captures these techniques.

### SC as Context Engineering Measurement

SC dimensions align with Anthropic's context engineering guidance:

- **Availability** = Can the pattern be retrieved? → "provide the right information"
- **Consistency** = Does it contradict other knowledge? → "avoid conflicting instructions"
- **Stability** = Has it changed over time? → "maintain consistent context across sessions"

SC provides quantitative measurement; context engineering provides the optimization target. Reasoning traces connect the two — they explain *why* a given context assembly produced the coherence score it did.

---

## Decision

### D1: Extend `ingestion_episode` with reasoning strategy columns (not a new table)

Add columns to the existing `ingestion_episode` table rather than creating a separate `reasoning_trace` table. Rationale: reasoning metadata is 1:1 with episodes — it describes the same operation, not a separate entity. A new table would force joins on every lineage query for no modeling benefit.

### D2: Use the  reasoning pattern vocabulary

The `reasoning_pattern` column references the canonical classification from : `workflow`, `cot`, `react`, `tree-of-thoughts`, `reflexion`, `llm-p`. This is a CHECK constraint, not an enum type, consistent with existing episode schema conventions.

Alternatives rejected:
- **Free-text field** — no aggregation possible, vocabulary drift inevitable
- **PostgreSQL ENUM type** — harder to extend; CHECK constraints are the established pattern in this schema

### D3: Capture context assembly metadata alongside reasoning strategy

Two complementary signal sets:

1. **Reasoning strategy** (`reasoning_pattern`, `chain_depth`, `branching_factor`, `observation_action_cycles`) — how the agent processed context
2. **Context assembly quality** (`context_assembly_method`, `context_token_count`, `context_utilization`) — how context was constructed and how much was actually used

Together these enable the diagnostic queries the issue specifies: "What reasoning patterns correlate with high coherence?" and "How much context was wasted?"

### D4: All new columns are nullable — additive, non-breaking

Existing episodes and instrumentation continue to work unchanged. New columns are populated when reasoning metadata is available. This is a MINOR schema change.

---

## Consequences

**Positive:**
- SC scoring can group/filter by reasoning pattern, enabling "which approach works best?" analysis
- Context utilization metric (tokens loaded vs. tokens referenced) provides a direct context engineering quality signal
- Lineage queries gain diagnostic depth without additional table joins
- Aligns episode model with the five-primitive isomorphism, making the SemOps→agent mapping concrete

**Negative:**
- Episode table grows wider — 7 new nullable columns. Acceptable given the 1:1 relationship and query pattern (these columns are only read in analytical/diagnostic contexts, not in hot-path ingestion)
- Instrumentation burden — existing `@emit_lineage` callers must be updated to populate new fields. Mitigated by nullable columns (opt-in, not breaking)

**Risks:**
- Reasoning pattern vocabulary may evolve as agent techniques mature — mitigated by CHECK constraint (easy ALTER) rather than ENUM type
- Context utilization is hard to compute precisely for all agent types — mitigated by making it nullable and documenting it as best-effort initially

---

## Pattern and Capability Impact

| Type | ID | Impact | Action |
|------|----|--------|--------|
| Pattern | `agentic-lineage` | Extends — reasoning trace adds a new signal dimension | No action (pattern already registered) |
| Pattern | `semantic-coherence` | Extends — new correlation surface for SC measurement | No action (pattern already registered) |
| Capability | `coherence-measurement` | Modified — gains reasoning-pattern grouping/filtering | Update registry description |

---

## References

- [DD-0019: Agentic Lineage Graph Infrastructure](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0019-agentic-lineage-graph-infrastructure.md) — design detail for episode infrastructure
- [DD-0006: Coherence Measurement Model](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0006-coherence-measurement-model.md) — SC formula and dimensions
- [DD-0013: Coherence Scoring Pipeline](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0013-coherence-scoring-pipeline.md) — scoring algorithms that consume this data
-  — canonical agent classification vocabulary (reasoning_pattern enum source)
-  — five-primitive agent model
-  — context quality measurement (density, relevance, clarity)
