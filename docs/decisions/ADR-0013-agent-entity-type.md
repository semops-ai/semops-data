# ADR-0013: Agent as Entity Type

> **Status:** Decided
> **Date:** 2026-03-10
> **Related Issue:** 
> **Builds On:** [ADR-0009: Strategic/Tactical DDD Refactor](./ADR-0009-strategic-tactical-ddd-refactor.md), [ADR-0012: Pattern + Coherence Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md)
> **Design Doc:** None

---

## Executive Summary

Add `agent` as a fourth entity type in the entity table, representing the application/runtime layer of the DDD architecture. Agents are the autonomous actors (slash commands, MCP tools, API endpoints) that exercise capabilities — completing the traceability chain from pattern to execution.

---

## Context

The current entity_type discriminator supports three values:

| entity_type | DDD Layer | Role |
|-------------|-----------|------|
| `capability` | Domain | What the system delivers |
| `repository` | Domain | Where code lives |
| `content` | *(separate aggregate)* | Publishing artifacts (DAM) |

This maps the domain layer and the content/DAM bounded context, but the **application layer is missing**. The runtime is entirely agentic — every execution path is an agent:

- **Slash commands** (`/arch-sync`, `/intake`, `/research`) — prompt-defined Claude Code skills
- **MCP tools** (`search_knowledge_base`, `get_pattern`, `graph_neighbors`) — programmatic agent access
- **Query API endpoints** — same queries, REST surface

These are not thin wrappers. Each is an autonomous agent with its own prompt, tool access, and execution context. They exercise capabilities but are not capabilities themselves — a capability describes *what* the system delivers; an agent is *who* executes it.

The traceability chain today stops at capability:

```
Pattern → Capability → ???
(why)      (what)       (missing)
```

Adding agent completes it:

```
Pattern → Capability → Agent
(why)      (what)       (who executes)
```

### Why not a value object on Capability?

Repositories are value objects on capabilities (identity doesn't matter, role does). But agents have meaningful identity — `/arch-sync` is a specific, auditable actor with its own prompt, tool access, and execution context. Agents can exercise multiple capabilities, and multiple agents can exercise the same capability. This many-to-many relationship requires entity status.

### Why not `content`?

Content entities are DAM publishing artifacts with `asset_type` (file/link), `filespec`, and corpus routing. Agents have none of these properties. Overloading content would conflate two bounded contexts.

---

## Decision

1. Add `agent` to the entity_type check constraint on the entity table
2. Define `agent_metadata_v1` JSONB schema for agent-specific metadata
3. Create an Agent Registry in STRATEGIC_DDD.md (authority source, same pattern as Capability Registry)
4. Extend `ingest_architecture.py` to parse and ingest agent entities
5. Use `implements` predicate for agent → capability edges (agent implements the capability at the application layer)
6. Use `delivered_by` predicate for agent → repository edges (same as capabilities)

### Entity Type Model (after this ADR)

| entity_type | DDD Layer | Metadata Schema | Examples |
|-------------|-----------|----------------|----------|
| `capability` | Domain | `capability_metadata_v1` | architecture-audit, coherence-scoring |
| `repository` | Domain (value object) | `repository_metadata_v1` | semops-data, semops-orchestrator |
| `agent` | Application | `agent_metadata_v1` | /arch-sync, search_knowledge_base |
| `content` | Content/DAM | `content_metadata_v1` | blog posts, pattern docs |

### agent_metadata_v1 Schema

All agent-specific classification is soft metadata in JSONB — no hard constraints beyond entity_type.

```json
{
  "$schema": "agent_metadata_v1",
  "agent_type": "skill | mcp_tool | api_endpoint",
  "surface": "cli | mcp | rest",
  "exercises_capabilities": ["capability-id-1", "capability-id-2"],
  "delivered_by_repo": "semops-orchestrator",
  "lifecycle_stage": "planned | draft | in_progress | active | retired",
  "layer": "operations | orchestration | acquisition | measurement-and-memory"
}
```

`agent_type`, `surface`, and `layer` are classification tags (like `domain_classification` on capabilities), not schema-enforced constraints. They can evolve without migrations.

### Edge Predicates (no schema changes needed)

- `agent → capability` via `implements` (agent implements capability at application layer)
- `agent → repository` via `delivered_by` (where the agent definition lives)

---

## Consequences

**Positive:**
- Completes the DDD layer model — domain, application, and content all represented
- Enables agent-level queries: "which agents exercise coherence-scoring?", "what MCP tools are available for the acquisition layer?"
- Agents become searchable via semantic search (with embeddings)
- Supports coherence auditing of the application layer (agents without capabilities = unjustified; capabilities without agents = unimplemented)
- Same authority-source pattern as capabilities (STRATEGIC_DDD.md → ingestion → KB)
- **Enables agentic-lineage tracking with stable actor IDs.** Currently, PROV-O lineage episodes track operations but the actor is implicit (a script name or function string). With agent entities in the KB, lineage episodes can reference `prov:wasAssociatedWith` an actual agent entity ID — closing the full audit loop: what happened → who did it → what capability was exercised → why that capability exists. This is the `agentic-lineage` pattern (extends `open-lineage`, `episode-provenance`) realized at the data model level.

**Negative:**
- Schema migration required (ALTER TABLE check constraint)
- New registry to maintain in STRATEGIC_DDD.md
- Fitness functions may need extension for agent-specific invariants
- `/arch-sync` and `/global-arch-sync` need updates to validate agent registry

**Risks:**
- Agent proliferation — need clear criteria for when something is an agent vs. a utility function
- Cross-repo agent definitions (skills live in semops-orchestrator, MCP tools live in semops-data) — authority source spans repos

---

## Implementation Plan

### Phase 1: Schema + Registry
- [x] ALTER TABLE entity check constraint to add 'agent'
- [x] Update SCHEMA_REFERENCE.md with agent_metadata_v1
- [x] Update SCHEMA_CHANGELOG.md (v8.2.0)
- [x] Draft Agent Registry section in STRATEGIC_DDD.md (20 skills + 12 MCP tools)
- [x] Update issue  acceptance criteria

### Phase 2: Ingestion
- [x] Extend `ingest_architecture.py` to parse Agent Registry table
- [x] Create agent entities with agent_metadata_v1
- [x] Create agent → capability edges (implements)
- [x] Create agent → repository edges (delivered_by)
- [x] Materialize to Neo4j (Agent nodes + relationships)

**Result:** 71 entities (8 repos, 31 capabilities, 32 agents), 207 edges ingested.

### Phase 3: Governance
- [x] Agent entity embeddings for semantic search (32 agents embedded)
- [x] `entity_type` filter on search pipeline (search.py, MCP, Query API)
- [x] Fitness function: `check_agent_capability_coverage` — all agents pass
- [x] Fitness function: `check_capability_agent_coverage` — 25 coverage gaps (expected)
- [x] `/arch-sync` — ARCHITECTURE.md v3.4.0, INFRASTRUCTURE.md v2.2.0 (agent refs, MCP tool counts, missing tools)
- [ ] `/global-arch-sync` — propagate agent layer to GLOBAL_ARCHITECTURE.md

---

## Session Log

### 2026-03-10: Design + Implementation (Phases 1–2)
**Status:** Phases 1–2 Complete
**Tracking Issue:** 

**Design Decisions:**
- Identified that "commands" are actually agents — autonomous actors, not thin wrappers
- Recognized the missing application layer in the entity model
- Distinguished agent from capability (who executes vs. what is delivered)
- Distinguished agent from content (runtime actor vs. publishing artifact)
- Confirmed agents need entity status (many-to-many with capabilities, meaningful identity)
- Designed agent_metadata_v1 schema (soft JSONB metadata, not hard constraints)
- Confirmed no new edge predicates needed (implements + delivered_by reuse)

**Implementation:**
- Schema migration applied (entity_type CHECK constraint + agent_metadata_v1)
- Agent Registry created in STRATEGIC_DDD.md (20 slash commands, 12 MCP tools)
- `ingest_architecture.py` extended: agent parsing, edge building, Neo4j materialization
- Fixed capability parser section_end regex (was incorrectly stopping at H4 sub-headings)
- Ingestion verified: 71 entities, 207 edges in PostgreSQL + Neo4j

**Key Insight:** The runtime layer is unified — slash commands, MCP tools, and API endpoints are all agents at different access points. The interface doesn't matter; the agent is the actor.

**Next Session Should Start With:**

1. `/global-arch-sync` to propagate agent layer to GLOBAL_ARCHITECTURE.md
2. Address capability-agent coverage gaps as new agents are registered

### 2026-03-10: Governance + Doc Sync (Phases 3–4)

**Status:** Phase 3 Complete, Phase 4 Complete

**Governance:**

- 32 agent embeddings generated (pgvector, text-embedding-3-small)
- `entity_type` filter added to full search pipeline (search.py, MCP, Query API, CLI)
- 2 fitness functions added: `check_agent_capability_coverage` (CRITICAL), `check_capability_agent_coverage` (MEDIUM)
- SEARCH_GUIDE.md v2.7.0 updated with entity_type filter docs, agent examples, graph stats

**Doc Sync (arch-sync):**

- ARCHITECTURE.md v3.4.0: entity table description, ingest_architecture.py description, fitness count 10→12, MCP diagram + tool list (added `get_pattern_alternatives`, `graph_neighbors`)
- INFRASTRUCTURE.md v2.2.0: MCP tool count 10→12, added missing tools + graph traversal category

---

## References

- [STRATEGIC_DDD.md § Agent Registry](../STRATEGIC_DDD.md) — authority source for agent definitions
- [ADR-0009: Strategic/Tactical DDD Refactor](./ADR-0009-strategic-tactical-ddd-refactor.md) — three-layer architecture
- [ADR-0012: Pattern + Coherence Co-Equal Aggregates](./ADR-0012-pattern-coherence-co-equal-aggregates.md) — aggregate model
- [phase2-schema.sql](../../schemas/phase2-schema.sql) — current entity_type constraint
-  — Agentic lineage: link episode actor to agent entity ID (downstream enablement)

---

**End of Document**
