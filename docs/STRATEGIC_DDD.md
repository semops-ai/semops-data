# Strategic DDD: Capabilities, Repos, and Integration

> **Version:** 2.0.0
> **Last Updated:** 2026-03-07
> **Status:** Draft (ADR-0009, ADR-0012)
> **Related:** [ADR-0009](decisions/ADR-0009-strategic-tactical-ddd-refactor.md) | [ADR-0012](decisions/ADR-0012-pattern-coherence-co-equal-aggregates.md) |  | 

This document formalizes the Strategic DDD layer for SemOps. It defines **capabilities** (what the system delivers), **repositories** (where implementation lives), and **integration patterns** (how repos relate).

These concepts were previously captured only in GLOBAL_ARCHITECTURE.md prose. This document makes them structured, queryable, and traceable to patterns.

---

## Principles

1. **Repos are functionally aligned to model organizational roles** and manage agent boundaries and context.
2. **Repos are recognizable nodes in an "Agentic Enterprise"** — each scopes what an AI agent needs to do useful work, simulating team structure.
3. **Every capability traces to >=1 pattern.** If it can't, either a pattern is missing or the capability lacks domain justification.
4. **Data flows are emergent** from shared capability participation, not explicitly modeled.
5. **Integration relationships are first-class** — rich edges with DDD integration pattern typing.
6. **SemOps has one bounded context** with a single ubiquitous language.
7. **Capabilities decompose into scripts** — small, focused, identifiable files rather than buried in large applications (anti-monolith). Each script is a bounded piece of executable functionality that can be replaced and audited independently.
8. **Model depth earns intake freedom.** The domain model must be deep enough (patterns, lineage, coherence signals) that new ideas can enter without upfront classification. Don't gate ideation; audit coherence. Semantic measurement bridges the gap between "I have an idea" and "here's where it fits." (See )

### Repos, Bounded Contexts, and the DDD Repository Pattern

Three uses of "repository" coexist in SemOps — understanding the distinction is important for reading this document and for DDD alignment:

1. **Git repositories (repos)** — the physical code boundaries listed in the [Repository Registry](#repository-registry) below. These are **agent role boundaries**, not bounded contexts. Each repo scopes what an AI agent (or human) needs in context to do useful work, simulating team structure in a one-person operation. Repos can be reorganized — merged, split, renamed — without changing the domain model. When that happens, the `repository` entity mappings in this document update, but `capability` and `pattern` entities don't.

2. **Bounded Contexts** — the semantic boundaries where a particular domain model and ubiquitous language apply. SemOps Core (semops-orchestrator, semops-data, data-pr, semops-research) is the product, organized into four sublayers within the DDD layered architecture (see [SemOps Layers](#semops-layers)) — always present regardless of domain. **Bolt-on systems** (content, sites, enterprise record) attach via stand-in connectors, typed by data-system-classification. They consume Core but own their own domain logic. The key boundary rule: Core never depends on a bolt-on. Bolt-ons depend on Core.

3. **DDD Repository pattern** — the data access abstraction that mediates between the domain model and the transactional data layer (OLTP), responsible for retrieving and persisting Aggregates. Currently this lives in **semops-data** — the ingestion scripts, entity builders, and edge creation logic that persist domain objects to PostgreSQL. This is distinct from **data-pr**, which is the analytics layer (OLAP) — it reads from the domain model for coherence scoring and profiling, but doesn't persist aggregates. Research RAG and data due diligence were extracted to **semops-research** (Extraction layer) via . The Repository pattern handles concerns like tenant isolation at the infrastructure layer, keeping the domain model unaware of which tenant or deployment context it operates within.

This OLTP/OLAP distinction maps to the [Data System Classification](../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-system-classification.md) framework: semops-data is the **Application Data System** (transactional, DDD-governed), data-pr is the **Analytics Data System** (read-heavy, dimensional). GitHub issues, projects, and session notes are the **Enterprise Work System** (unstructured knowledge artifacts). semops-backoffice's financial pipeline is an **Enterprise Record System** (canonical truth, double-entry constraints). Each system type has different scaling physics, governance approaches, and integration patterns — understanding which type you're operating in determines which DDD patterns apply.

The relationship between these three is itself a [Scale Projection](../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/scale-projection.md) proof: because the mapping from domain model (capabilities, patterns) to physical boundaries (repos) is explicit and queryable in this document, changing the physical structure is an infrastructure decision. The architecture — what this document captures — remains stable.

---

## SemOps Layers

SemOps uses four named sublayers within the standard DDD layered architecture (Evans, Ch. 4). The DDD layers provide the structural frame; the sublayer names are the operational vocabulary specific to SemOps.

| DDD Layer | SemOps Sublayers | Responsibility |
|-----------|-----------------|----------------|
| **Application** | Operations, Orchestration | Agent-driven coordination — slash commands, workflows, cross-repo (heavy agent compute) |
| **Domain** | Data, Extraction | The semantic operations model — patterns, schema, coherence, research, ingestion |
| **Infrastructure** | *(semops-data services)* | PostgreSQL, Qdrant, Neo4j, Ollama, Docling, Docker |

**Dependency rule:** Application depends on Domain. Domain depends on Infrastructure. No layer depends on a layer above it. Bolt-on systems attach at the Application layer via stand-in connectors.

These sublayers replace the previous three-layer domain model (Pattern / Architecture / Content), which had a false separation between Pattern and Architecture — patterns ARE architecture. Content is a bolt-on, not a core layer.

Agents operate at every sublayer (agents-as-runtime). Operations and Orchestration are the most agent-dense.

| Sublayer | What It Does | Agent Density | Primary Repos |
|----------|-------------|---------------|---------------|
| **Extraction** | Research, outside signal, pattern discovery, competitive coherence, catalog building | Medium | semops-research |
| **Data** | Ingestion, catalog, knowledge graph, coherence scoring | Medium | semops-data, data-pr |
| **Operations** | Enrichment, lineage, drift detection, domain processing, audit commands | Heavy | data-pr, semops-data, semops-orchestrator |
| **Orchestration** | Cross-repo coordination, workflow management, scale projection | Heavy | semops-orchestrator |

**Flow:** Extraction finds things → Data ingests and stores them → Operations processes them → Orchestration coordinates across everything → Orchestration directs Extraction.

**Bolt-on systems** (content, sites, enterprise record) attach via stand-in connectors, typed by data-system-classification. They are not SemOps layers — they are external systems that consume SemOps services. See [SEMOPS_LAYERS.md](../../semops-orchestrator/docs/SEMOPS_LAYERS.md) for diagrams.

**DDD classification still applies:** Core = capabilities in the four sublayers. Generic = bolt-on capabilities. Supporting = empty for now.

> **Gap noted:** Many Operations layer capabilities exist as implemented slash commands/skills but are not yet registered in the Capability Registry. See .

---

## Domain Model: Aggregates and Building Blocks

The SemOps domain model contains multiple aggregates, each with its own root, lifecycle, and invariants (ADR-0012). The two **core aggregates** form the Semantic Optimization Loop — the feedback cycle that makes SemOps more than a knowledge graph.

### Core Aggregates

| Aggregate | Root | Children / Value Objects | Invariants |
| --- | --- | --- | --- |
| **Pattern** | `pattern` | `pattern_edge` | Valid SKOS hierarchy, provenance rules, unique preferred_label |
| **Coherence Assessment** | *(deferred — schema when operational)* | measurements, gaps, actions | Must reference >=1 pattern, lifecycle state machine |

**Pattern** is the prescriptive force — what we should look like. It defines stable meaning via SKOS taxonomy, provenance tiers (1P/2P/3P), and adoption lineage (adopts/extends/modifies).

**Coherence Assessment** is the evaluative/directive force — how well reality matches intent, and what to do about it. It audits the Pattern → Capability → Script chain across all aggregates and can drive pattern evolution, reversal, or realignment. Coherence is audit by default, not a gate — the flexible edge is free to exist. Aggregate root invariants protect the stable core. See ADR-0012 §3 for details.

Coherence findings are emergent — the audit process (territory map + delta identification) surfaces what's aligned, what's missing, and what conflicts. These findings don't need upfront classification; the evaluation itself reveals the appropriate action. When coherence measurement is automated, it can derive categories from audit data.

### Supporting Aggregates

Each supporting aggregate traces to a 3P pattern that prescribes its structure.

| Aggregate | Root | Children | 3P Pattern |
| --- | --- | --- | --- |
| **Content** (DAM) | `entity` (content) | `delivery` (publication records), edges | DAM, Dublin Core |
| **Surface** | `surface` | `surface_address` | DAM (channels) |
| **Brand** (PIM/CRM) | `brand` | `product`, `brand_relationship` | Schema.org, PIM *(unregistered)* |

### DDD Building Block Classifications

| Concept | DDD Building Block | Rationale |
| --- | --- | --- |
| **Capability** | Entity | Has identity and lifecycle. Produced by Pattern decisions, audited by Coherence. Implements multiple patterns — can't be a child of any single Pattern. Exists in the space between both core aggregates, owned by neither. |
| **Repository** | Value Object | Identity doesn't matter — role and delivery mapping do. Repos can be reorganized (merged, split, renamed) without changing the domain model. |

### Lifecycle Model

All domain model entities use the same 5-state lifecycle, sourced from the [Backstage Software Catalog](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/implementation/backstage-software-catalog.md) `spec.lifecycle` mapping.

| State | Meaning |
|-------|---------|
| `planned` | Identified as relevant, not yet evaluated |
| `draft` | Being evaluated or researched for adoption |
| `in_progress` | Actively being adopted or implemented |
| `active` | Adopted/implemented and operational |
| `retired` | No longer used (superseded or dropped) |

**Patterns and capabilities have independent lifecycles connected by the `implements` edge.**

- A pattern can be `active` while the capabilities implementing it range across all states (e.g., `ddd` is `active`; `bounded-context-extraction` which implements it is `planned`)
- A capability can be `planned` while multiple `draft` patterns are being evaluated for it (e.g., a planned `financial-pipeline` might evaluate 3 candidate accounting patterns)
- A pattern can be `retired` while the capability it served remains `active` via a replacement pattern

**Lifecycle governance rules:**

| Entity State | Coherence Signals | Orphan Detection | Ingestion | Audit Checks |
|---|---|---|---|---|
| `planned` | Skip — intent only | Skip — expected to be unmapped | Yes (seed for future) | Verify exists in registry |
| `draft` | Discovery mode only | Skip — evaluation in progress | Yes | Verify pattern linkage attempted |
| `in_progress` | Full audit | Full audit | Yes | Full consistency checks |
| `active` | Full audit | Full audit | Yes | Full consistency checks |
| `retired` | Skip | Skip — intentionally disconnected | Optional (historical) | Verify retirement documented |

**Where lifecycle is declared:**

| Entity | Authority | Field |
|--------|-----------|-------|
| Capability | This document (Capability Registry, Status column) | `status` |
| Pattern | `pattern_v1.yaml` | `status` |
| Repository | `REPOS.yaml` | `lifecycle_stage` |
| Pattern doc | Doc header | `Status:` |

### The Semantic Optimization Loop

```text
Pattern ──produces──→ Capability
   ↑                      ↓
   └──── Coherence ←──audits──┘
         (informs)

Pattern pushes. Coherence aligns.
```

When `semantic-optimization` becomes operational, coherence scoring becomes the objective function — Pattern sets the target, Coherence measures the gap, the optimization loop minimizes the gap. The existing agentic lineage system (episodes with `coherence_score` fields) provides the telemetry layer. See ADR-0012 §10.

---

## Capability Registry

> **Authority:** [`config/registry.yaml`](../config/registry.yaml) — structured data for all capabilities, agents, and integrations.
> Tables below are illustrative summaries. For programmatic access, always use the YAML registry.

A **capability** is what the system delivers. It implements one or more Patterns. It is delivered by one or more repos. Capability-to-pattern coverage is a measurable coherence signal.

The most frequently referenced pattern in the tables below is [`explicit-architecture`](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/explicit-architecture.md): architecture modeled as queryable data (entities + typed edges) so that governance is a set of projections over that model. Derives from Viable Systems Model (homeostatic feedback) and DDD (bounded contexts as structural vocabulary). The 1P innovation is coverage views as passive sensors — the andon cord is a `SELECT` query.

### Core Domain Capabilities

These are the differentiating capabilities aligned with the [Semantic Operations Framework](../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/README.md) — what makes SemOps unique. Organized by [SemOps Layer](#semops-layers).

#### Extraction Layer

| ID | Capability | Status | Implements Patterns | Delivered By | Project |
|----|-----------|--------|-------------------|--------------|---------|
| `corpus-meta-analysis` | Corpus Meta-Analysis | active | `semantic-ingestion`, `raptor`, `agentic-rag` | semops-research | P24, P35 |
| `data-due-diligence` | Data Due Diligence | active | `togaf`, `dcam`, `apqc-pcf` | semops-research | P31 |
| `reference-generation` | Reference Generation | active | `mirror-architecture`, `business-domain`, `agentic-rag`, `bizbok`, `data-system-classification` | semops-research | P24 |
| `business-model-synthesis` | Business Model Synthesis | active | `bizbok-ddd-derivation`, `data-system-classification` | semops-research | P24 |
| `agentic-ddd-derivation` | DDD Architecture Derivation | active | `bizbok-ddd-derivation`, `ddd-agent-boundary-encoding` | semops-research | P24 |
| `synthesis-simulation` | Synthesis and Simulation | in_progress | `mirror-architecture`, `data-lineage` | data-pr, semops-research | P25 |
| `reference-catalog` | Reference Catalog | planned | `unified-catalog`, `backstage-software-catalog`, `semantic-ingestion`, `agentic-rag` | semops-research | P31 |
| `vendor-decomposition` | Vendor Decomposition | in_progress | `system-primitive-decomposition`, `explicit-architecture`, `explicit-enterprise`, `data-system-classification` | semops-research | P31 |
| `business-model-decomposition` | Business Model Decomposition | active | `system-primitive-decomposition`, `business-domain` | semops-research | P31 |
| `code-decomposition` | Code Decomposition | planned | `system-primitive-decomposition`, `explicit-architecture` | semops-research | P31 |
| `corpus-collection` | Corpus Collection | active | `semantic-ingestion`, `agentic-rag`, `osint` | semops-research | P31 |
| `tech-stack-profiling` | Tech Stack Profiling | active | `system-primitive-decomposition`, `data-system-classification`, `osint` | semops-research | P31 |
| `structured-extraction` | Structured Extraction | active | `semantic-ingestion`, `data-modeling`, `osint` | semops-research | P31 |
| `sentiment-extraction` | Sentiment Extraction | in_progress | `semantic-ingestion`, `osint` | semops-research | P31 |
| `research` | Current Research and Trends | active | `raptor`, `semantic-ingestion` | data-pr | — |
| `autonomous-research` | Autonomous Comparative Research | planned | `semantic-coherence`, `seci`, `agentic-rag` | semops-research, semops-orchestrator | P40 |

#### Data Layer

| ID | Capability | Status | Implements Patterns | Delivered By | Project |
|----|-----------|--------|-------------------|--------------|---------|
| `internal-knowledge-access` | Internal Knowledge Access | active | `agentic-rag`, `tree-of-thought`, `react-reasoning` | semops-data | P13 |
| `coherence-scoring` | Coherence Scoring | in_progress | `semantic-coherence`, `data-management`, `data-quality`, `data-contracts` | data-pr, semops-data | P18 |
| `ingestion-pipeline` | Ingestion Pipeline | in_progress | `semantic-ingestion`, `data-quality` | semops-data | P13, P35 |
| `agent-observability` | Agent Observability | planned | `open-lineage`, `episode-provenance` | semops-data | P50, P15 |
| `operational-metrics` | Operational Metrics | planned | `agentic-lineage` | semops-data | P50, P15 |
| `reasoning-lineage` | Reasoning Lineage | planned | `agentic-lineage`, `derivative-work-lineage` | semops-data | P50, P15 |
| `pattern-governance` | Pattern Governance | active | `explicit-architecture`, `backstage-software-catalog` | semops-orchestrator | P39 |
| `pattern-registry` | Pattern Registry | active | `semantic-object-pattern`, `arc42`, `metadata-management` | semops-data, semops-orchestrator | P39 |
| `agentic-data-science` | Agentic Data Science | planned | `data-profiling`, `data-modeling`, `rlhf` | data-pr | P41 |

##### Ingestion Pipeline Detail

> Implementation detail relocated to [semops-data ARCHITECTURE.md § Retrieval Pipeline](ARCHITECTURE.md#retrieval-pipeline). The 1P innovation (`semantic-ingestion`) is that every byproduct — classifications, detected edges, coherence scores, embeddings — is captured and queryable, not discarded.

#### Operations Layer

| ID | Capability | Status | Implements Patterns | Delivered By | Project |
|----|-----------|--------|-------------------|--------------|---------|
| `architecture-audit` | Architecture Audit | active | `explicit-architecture`, `backstage-software-catalog`, `data-catalog`, `data-management` | semops-orchestrator | P39 |
| `input-optimization` | Input Optimization | active | `semantic-ingestion`, `explicit-enterprise`, `semantic-object-pattern` | semops-orchestrator | — |
| `bounded-context-extraction` | Bounded Context Extraction | planned | `explicit-architecture` | semops-data, semops-orchestrator | P27 |
| `capacity-planning` | Capacity Planning | planned | `scale-projection` | data-pr, semops-orchestrator | P25 |
| `capex-planning` | Capex Planning | planned | `scale-projection` | data-pr, semops-orchestrator | P25 |
| `architecture-validation` | Architecture Validation | planned | `scale-projection` | data-pr, semops-orchestrator | P25 |
| `vendor-evaluation` | Vendor Evaluation | planned | `scale-projection` | data-pr, semops-orchestrator | P25 |

#### Orchestration Layer

| ID | Capability | Status | Implements Patterns | Delivered By | Project |
|----|-----------|--------|-------------------|--------------|---------|
| `orchestration` | Orchestration | active | `explicit-enterprise`, `explicit-architecture` | semops-orchestrator | P19 |
| `context-engineering` | Context Engineering | active | `explicit-enterprise` | semops-orchestrator | P19 |
| `autonomous-execution` | Autonomous Execution | in_progress | `explicit-enterprise` | semops-orchestrator | P29 |

---

### Capability Traceability

The system enforces a full traceability chain from stable meaning to executable code:

```text
Pattern → Capability → Script
(why)      (what)       (where it runs)
```

- **Pattern → Capability** — every capability must trace to at least one pattern. Gaps indicate missing patterns or unjustified capabilities. Tracked in this document (Capability Registry above).
- **Capability → Script** — capabilities decompose into small, focused scripts (Principle 7). Each script is a bounded piece of executable functionality that can be identified, replaced, and audited independently. Tracked in per-repo `ARCHITECTURE.md` "Key Components" sections.
- **Lineage** — git provides the change history for scripts. Comments and docstrings provide intent. No separate registry is needed at current scale.

This chain is the primary audit domain of Coherence Assessment (ADR-0012 §6). Every break in the chain is a coherence finding — the fix can go in either direction (create a capability, delete a script, or discover that a script is already part of an existing capability). See [Coherence Signals](#coherence-signals) for how this is measured today.

> **Capability registry:** This document (above)
> **Per-repo script inventory:** Each repo's `ARCHITECTURE.md` § Key Components
> **Library → Capability crosswalk:** [GLOBAL_INFRASTRUCTURE.md § Stack Ecosystem](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/GLOBAL_INFRASTRUCTURE.md#stack-ecosystem)
> **Audit:** `/arch-sync` Step 5 (per-repo), `/global-arch-sync` Step 4e (cross-repo)

---

### Agent Registry (ADR-0013)

> **Authority:** [`config/registry.yaml § agents`](../config/registry.yaml) — structured data for agent registration.

The application layer is entirely agentic. Every execution path — slash commands, MCP tools, API endpoints — is an autonomous agent that exercises one or more capabilities. Agent is an entity type (`entity_type = 'agent'`), ingested from the registry into the KB.

**Slash Command Agents** (skills — prompt-defined, CLI surface)

| ID | Agent | Type | Exercises Capability | Layer | Delivered By |
|----|-------|------|---------------------|-------|-------------|
| `agent-arch-sync` | /arch-sync | skill | `architecture-audit` | operations | semops-orchestrator |
| `agent-global-arch-sync` | /global-arch-sync | skill | `architecture-audit` | operations | semops-orchestrator |
| `agent-generate-proposed` | /generate-proposed | skill | `architecture-audit` | operations | semops-orchestrator |
| `agent-promote-proposed` | /promote-proposed | skill | `architecture-audit` | operations | semops-orchestrator |
| `agent-pattern-audit` | /pattern-audit | skill | `pattern-governance` | operations | semops-orchestrator |
| `agent-register-repo` | /register-repo | skill | `architecture-audit` | operations | semops-orchestrator |
| `agent-intake` | /intake | skill | `input-optimization` | operations | semops-orchestrator |
| `agent-project-create` | /project-create | skill | `orchestration` | orchestration | semops-orchestrator |
| `agent-project-review` | /project-review | skill | `orchestration` | orchestration | semops-orchestrator |
| `agent-status` | /status | skill | `orchestration` | orchestration | semops-orchestrator |
| `agent-issue` | /issue | skill | `orchestration` | orchestration | semops-orchestrator |
| `agent-plan` | /plan | skill | `orchestration` | orchestration | semops-orchestrator |
| `agent-prime` | /prime | skill | `context-engineering` | orchestration | semops-orchestrator |
| `agent-prime-global` | /prime-global | skill | `context-engineering` | orchestration | semops-orchestrator |
| `agent-kb` | /kb | skill | `context-engineering` | orchestration | semops-orchestrator |
| `agent-adr` | /adr | skill | `context-engineering` | orchestration | semops-orchestrator |
| `agent-session` | /session | skill | `context-engineering` | orchestration | semops-orchestrator |
| `agent-prep-experiment` | /prep-experiment | skill | `autonomous-execution` | orchestration | semops-orchestrator |
| `agent-run-experiment` | /run-experiment | skill | `autonomous-execution` | orchestration | semops-orchestrator |
| `agent-yolo-status` | /yolo-status | skill | `autonomous-execution` | orchestration | semops-orchestrator |
| `agent-research` | /research | skill | `corpus-meta-analysis` | extraction | semops-orchestrator |

**MCP Tool Agents** (programmatic — MCP surface, registered as `semops-kb`)

| ID | Agent | Type | Exercises Capability | Layer | Delivered By |
|----|-------|------|---------------------|-------|-------------|
| `agent-search-knowledge-base` | search_knowledge_base | mcp_tool | `internal-knowledge-access` | data | semops-data |
| `agent-search-chunks` | search_chunks | mcp_tool | `internal-knowledge-access` | data | semops-data |
| `agent-list-corpora` | list_corpora | mcp_tool | `internal-knowledge-access` | data | semops-data |
| `agent-list-patterns` | list_patterns | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-get-pattern` | get_pattern | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-search-patterns` | search_patterns | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-get-pattern-alternatives` | get_pattern_alternatives | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-list-capabilities` | list_capabilities | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-get-capability-impact` | get_capability_impact | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-query-integration-map` | query_integration_map | mcp_tool | `pattern-registry` | data | semops-data |
| `agent-run-fitness-checks` | run_fitness_checks | mcp_tool | `architecture-audit` | operations | semops-data |
| `agent-graph-neighbors` | graph_neighbors | mcp_tool | `internal-knowledge-access` | data | semops-data |

---

### Generic Domain Capabilities (Bolt-On)

These are capabilities delivered by bolt-on systems that attach to SemOps Core via stand-in connectors, typed by data-system-classification. They are not part of the four SemOps sublayers. Some may be promoted to Core if unique alignment with SemOps evolves.

| ID | Capability | Status | Implements Patterns | Delivered By | System Type |
|----|-----------|--------|-------------------|--------------|-------------|
| `content-management` | Content Management | active | `cms`, `dublin-core` | publisher-pr | Content Platform |
| `asset-management` | Asset Management | active | `dam` | publisher-pr | Content Platform |
| `product-information` | Product Information | active | `pim` | publisher-pr, sites-pr | Content Platform |
| `agentic-composition` | Agentic Composition | planned | `semantic-ingestion`, `agentic-rag`, `zettelkasten` | publisher-pr | Content Platform |
| `style-learning` | Style Capture | planned | — | publisher-pr | Content Platform |
| `concept-documentation` | Concept Documentation | planned | — | publisher-pr | Content Platform |
| `surface-deployment` | Surface Deployment | active | `dam` | sites-pr | Sites / Endpoints |
| `cap-inbox` | Inbox | active | `explicit-enterprise` | semops-backoffice | Enterprise Record |
| `attention-management` | Attention Management | active | `explicit-enterprise` | semops-backoffice | Enterprise Record |
| `financial-pipeline` | Financial Pipeline | planned | `explicit-enterprise` | semops-backoffice | Enterprise Record |
| `cap-voice-control` | Voice Control | draft | `explicit-enterprise` | semops-backoffice | Enterprise Record |

Bolt-on system types are classified by [data-system-classification](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md). Enterprise Record capabilities implement `explicit-enterprise` (1P) — humble tools become agent-addressable signal streams. See [explicit-enterprise pattern doc](../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/explicit-enterprise.md).

> **Bounded context candidate:** Agentic composition applied to domain-specific outputs (e.g., resume composition, consulting proposals) may warrant a second bounded context via Customer-Supplier (challenges ADR-0009 decision ). See per-repo ARCHITECTURE.md for implementation detail.

> **Gap noted:** `design-system` capability likely needed for sites-pr visual design governance (font management, per-brand styles). To be formalized in a future review.

---

## Repository Registry

> **Authority:** [`) — repo definitions with capabilities and integration patterns.
> Capability delivery mapping is derived from `config/registry.yaml` capability `delivered_by` fields.

A **repository** is where implementation lives. Repos deliver capabilities.

#### Core Repos (SemOps Layers)

| ID | Repo | Layer(s) | Role | Delivers Capabilities |
|----|------|----------|------|----------------------|
| `semops-research` | semops-research | Extraction | Research/Consulting | `corpus-meta-analysis`, `data-due-diligence`, `reference-generation`, `business-model-synthesis`, `agentic-ddd-derivation`, `synthesis-simulation`, `reference-catalog`, `vendor-decomposition`, `business-model-decomposition`, `code-decomposition`, `autonomous-research` |
| `semops-data` | semops-data | Data, Operations | Schema/Infrastructure | `internal-knowledge-access`, `ingestion-pipeline`, `agent-observability`, `operational-metrics`, `reasoning-lineage`, `pattern-registry`, `coherence-scoring`, `bounded-context-extraction` |
| `data-pr` | data-pr | Data, Extraction | Analytics/MLOps | `coherence-scoring`, `agent-observability`, `operational-metrics`, `reasoning-lineage`, `agentic-data-science`, `capacity-planning`, `capex-planning`, `architecture-validation`, `vendor-evaluation`, `synthesis-simulation` |
| `semops-orchestrator` | semops-orchestrator | Orchestration, Operations | Platform/DX | `orchestration`, `context-engineering`, `autonomous-execution`, `pattern-governance`, `capacity-planning`, `capex-planning`, `architecture-validation`, `vendor-evaluation`, `bounded-context-extraction`, `architecture-audit`, `input-optimization`, `autonomous-research` |

#### Bolt-On Repos (Generic)

| ID | Repo | System Type | Role | Delivers Capabilities |
|----|------|-------------|------|----------------------|
| `publisher-pr` | publisher-pr | Content Platform | Publishing | `content-management`, `asset-management`, `product-information`, `agentic-composition`, `style-learning`, `concept-documentation` |
| `sites-pr` | sites-pr | Sites / Endpoints | Frontend | `surface-deployment` |
| `semops-backoffice` | semops-backoffice | Enterprise Record | Operations | `cap-inbox`, `financial-pipeline`, `cap-voice-control` |

---

## Integration Patterns

> **Authority:** [`config/registry.yaml § integrations`](../config/registry.yaml) — structured integration map data.

Repos interact through DDD integration patterns. These describe how repos relate to the **domain model**, not the database.

### Current Integration Map

| Source Repo | Target Repo | DDD Pattern | What's Shared | Direction |
|-------------|-------------|-------------|---------------|-----------|
| semops-data | publisher-pr | **Shared Kernel** | UBIQUITOUS_LANGUAGE.md, Pattern/Entity schema | Bidirectional |
| semops-orchestrator | all repos | **Published Language** | GLOBAL_ARCHITECTURE.md, process docs, ADR templates | Downstream reads |
| semops-data | data-pr | **Customer-Supplier** | Qdrant, Docling, Supabase services | Upstream provides |
| semops-data | semops-research | **Customer-Supplier** | Ollama, Qdrant, Docling services | Upstream provides |
| semops-data | sites-pr | **Customer-Supplier** | Supabase data, API access | Upstream provides |
| publisher-pr | semops-data | **Conformist** | Adopts Pattern/Entity model as-is | Downstream conforms |
| semops-backoffice | semops-data | **Anti-Corruption Layer** | Translates shared PostgreSQL to financial domain | ACL at boundary |
| publisher-pr | sites-pr | **Customer-Supplier** | MDX content, resume seed.sql, fonts/templates | Bidirectional supply |

### Integration Relationship Metadata

Each integration relationship should capture:

```yaml
source_repo: semops-data
target_repo: publisher-pr
integration_pattern: shared-kernel    # DDD integration pattern (a 3P Pattern record)
shared_artifact: UBIQUITOUS_LANGUAGE.md
direction: bidirectional
rationale: "Both repos must agree on domain terms; changes require coordination"
established: 2025-12-22              # When this integration was formalized
```

This metadata will be stored as Edge records with `integration` predicate between repository entities, typed by the DDD integration Pattern.

---

## Governance: Change Propagation

When the domain model changes, multiple documents and systems must stay consistent. This section defines the authority chain, change types, and consistency checks.

### Document Authority Chain

Each data type has a single authority. All other documents derive from it.

| Data | Authority | Derived By |
|------|-----------|------------|
| Pattern identity + lineage | `pattern_v1.yaml` (semops-orchestrator) | Pattern docs, PATTERN_AUDIT.md, DB pattern table |
| Capability registry (ID, status, patterns, repos) | `config/registry.yaml` (semops-data) | This document (illustrative tables), REPOS.yaml, GLOBAL_ARCHITECTURE, per-repo ARCHITECTURE.md, PATTERN_AUDIT.md, DB entity/edge tables |
| Agent registry + integration map | `config/registry.yaml` (semops-data) | This document (illustrative tables), REPOS.yaml, GLOBAL_ARCHITECTURE, per-repo ARCHITECTURE.md |
| Coherence signal definitions | This document (STRATEGIC_DDD) | Audit commands (`/arch-sync`, `/global-arch-sync`) |
| Per-repo infrastructure (services, ports, env) | Per-repo `INFRASTRUCTURE.md` | GLOBAL_INFRASTRUCTURE.md, PORTS.md |
| Process + templates | semops-orchestrator docs | Per-repo docs (via templates) |

**Rule:** Author changes at the authority. Then run audit commands to find and fix inconsistencies in derived documents.

### Change Types

Each change type defines **where to author first** (the authority) and **what consistency checks to run**.

#### Pattern Adopted or Removed

**Author at:** `pattern_v1.yaml` + `config/registry.yaml` (capabilities section)

1. Add/remove pattern record in `pattern_v1.yaml` (identity, lineage, docs)
2. Create/deprecate `patterns/<id>.md`
3. Update capability `implements_patterns` in `config/registry.yaml`
4. Update Coherence Signals coverage table
5. **Run audit:** `/arch-sync` (per-repo) → `/global-arch-sync` (cross-repo)
6. Audit checks: per-repo ARCHITECTURE.md capabilities, GLOBAL_ARCHITECTURE per-repo sections, REPOS.yaml capability descriptions, PATTERN_AUDIT.md, derives_from references in other patterns
7. Re-run ingestion when DB is operational

#### Capability Added, Modified, or Status Changed

**Author at:** `config/registry.yaml` (capabilities section)

1. Add/update capability entry in `config/registry.yaml` (id, name, status, implements_patterns, delivered_by)
2. Update `governance.issue` with tracking issue if applicable
3. Update Coherence Signals coverage
4. **Run audit:** `/arch-sync` (per-repo) → `/global-arch-sync` (cross-repo)
5. Audit checks: REPOS.yaml capabilities, GLOBAL_ARCHITECTURE per-repo sections, per-repo ARCHITECTURE.md Capabilities table + status indicators, per-repo INFRASTRUCTURE.md if capability implies new services

#### Repo Registered or Reorganized

**Author at:** `REPOS.yaml` (semops-orchestrator) + `config/registry.yaml` (integrations section)

1. Add/update repo in `REPOS.yaml`
2. Update `config/registry.yaml` integrations if relationships change
3. **Run audit:** `/global-arch-sync`
4. Audit checks: GLOBAL_ARCHITECTURE repo section, GLOBAL_INFRASTRUCTURE services/ports, per-repo ARCHITECTURE.md + INFRASTRUCTURE.md existence and template compliance

#### Integration Pattern Changed

**Author at:** `config/registry.yaml` (integrations section)

1. Update integration entry in `config/registry.yaml`
2. **Run audit:** `/arch-sync` (affected repos) → `/global-arch-sync`
3. Audit checks: per-repo ARCHITECTURE.md Dependencies/Integration sections, GLOBAL_ARCHITECTURE DDD Alignment

#### Unclassified Input (Principle 8)

**Author at:** GitHub Issue

New ideas enter the system as GitHub Issues — no upfront classification required. But pattern recognition requires enough definition to match against. The process has two phases: **scope the goal**, then **evaluate coherence**. The `/intake` command operationalizes this workflow.

**Summary flow:**

1. Create issue describing the idea — no classification required
2. Scope the goal (Tier 1 in-issue, or Tier 2 Project Spec)
3. Evaluate against coherence signals (manually or via `/intake`)
4. Act on what the evaluation reveals — the territory map + delta tells you what to do

The issue is the flexible edge. No document updates until evaluation happens. See [Principle 8](#principles).

##### Goal Scoping (Tiered)

Pattern recognition requires enough definition to match against. A raw "I want better data ingestion" matches everything; a scoped "I want to extend `explicit-enterprise` to data systems architecture and build out an open source first class data system" matches `explicit-enterprise`, `open-primitive-pattern`, and Project 30's existing scope specifically.

Goal scoping is the forcing function — not the coherence checklist.

**Tier 1: Goal statement (in-issue)** — for small, focused ideas. The issue needs one thing:

- **Outcome:** What does done look like? (1-2 sentences, free-form natural language)

The Outcome statement is the input — no structured fields required. The description naturally contains entity references that the evaluation step extracts. Users reference patterns, projects, and capabilities in their natural language without being asked to fill in forms.

The Outcome must be concrete enough to validate against — even if validation is human-in-the-loop. "Make the system better" has nothing to match; "extend `explicit-enterprise` to data systems architecture" gives the evaluation process entity references to extract and a territory map to present.

**Entity extraction hints** — signals in the Outcome text that accelerate context loading:

| Signal | Example | What it loads |
|--------|---------|---------------|
| Backtick references | `explicit-enterprise`, `ingestion-pipeline` | Direct pattern/capability lookup |
| Project numbers | "project 30", "Project 18" | Project spec (outcome, AC, child issues, related patterns) |
| Issue references | ``, `` | Issue body, labels, linked context |
| Natural language | "open source data system" | KB semantic search for matching entities |

Backticks are a strong hint — they signal "this is a known entity name" vs casual language. But natural language without backticks still works via KB semantic search.

**Tier 2: Project Spec** — for bigger ideas. When Tier 1 reveals broader scope, promote to a `PROJECT-NN` spec (see [semops-orchestrator/docs/project-specs/TEMPLATE.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/project-specs/TEMPLATE.md)) with full Outcome, Acceptance Criteria, Execution Sequence.

**When to promote Tier 1 → Tier 2:**

| Trigger | Why |
|---------|-----|
| Spans 3+ repos | Needs sequenced execution and cross-repo coordination |
| Requires an ADR | Architectural decision needs formal recording |
| Has dependencies on other projects | Needs dependency tracking beyond a single issue |
| Needs multiple ordered steps | Execution Sequence required to avoid misordering |

**The  insight applies:** Users often start narrow (a capability idea) and discover the broader pattern during goal scoping. The process supports that journey — Tier 1 scoping surfaces existing structure before you conclude novelty.

##### Evaluation Process

Evaluation is semantic, not structural. The agent extracts entity references from the Outcome text, expands context from authority sources, and presents the territory map. The `/intake` command operationalizes this.

**Step 1: Extract entity references** from the Outcome text:

| Priority | Signal | Action |
|----------|--------|--------|
| 1 | Backtick references (`explicit-enterprise`) | Direct lookup in `pattern_v1.yaml` or Capability Registry |
| 2 | Project numbers ("project 30") | Load project spec from `semops-orchestrator/docs/project-specs/` |
| 3 | Issue references (``, ``) | Load issue body, labels, linked context |
| 4 | Natural language (everything else) | KB semantic search (`search_knowledge_base`) |

**Step 2: Expand context** — for each extracted reference, load from authority sources:

- Patterns → `pattern_v1.yaml` (status, lineage, derives_from)
- Capabilities → Capability Registry in this document (status, implements, delivered by)
- Projects → Project spec (outcome, acceptance criteria, child issues, related patterns)
- Repos → `REPOS.yaml` (capabilities, integration patterns, dependencies)

**Step 3: Present territory map** — show the user what already exists that relates to their input. This is the key step: most inputs connect to something existing. The map reveals the landscape before any classification happens.

**Step 4: Identify delta** — what's new vs what already has coverage. Compare the input's intent against the expanded context:

- Input extends an existing capability → link issue, update coverage tables
- Input fills a gap (something should trace but doesn't) → add missing traces
- Input conflicts with existing structure → evaluate trade-off (modify input, modify architecture, or reject)
- Input has no matches after KB search + authority lookup → ask clarifying questions before concluding it's genuinely new

**Net new is rare.** If Steps 1-3 found nothing, ask a few focused questions before concluding. The domain model is broad enough that most ideas connect somewhere.

The delta tells you what to do — no separate classification step is needed. The territory map + delta *is* the evaluation output.

##### Action Routing

After the evaluation process presents the territory map, the delta tells you what to do:

```text
Territory map presented (entity references expanded, context loaded)
│
├── Input extends existing pattern/capability coverage
│       ├── Capability exists → link issue, update coverage tables
│       │   Change type: none, or Capability Modified if status changes
│       └── Pattern exists but no capability → create capability
│           Change type: Capability Added
│
├── Input fills a gap (something exists without proper trace)
│       └── Action: add missing traces (pattern→capability, capability→script)
│           Change type: Capability Modified or Pattern Adopted
│
├── Input conflicts with existing structure
│       └── Action: evaluate trade-off — modify input, modify architecture, or reject
│           Change type: depends on resolution
│
├── Input describes infrastructure/tooling
│       └── Route to Repo or Integration change types
│           Change type: Repo Registered/Reorganized or Integration Pattern Changed
│
└── No matches after KB search + authority lookup + clarifying questions
        ├── Narrow (single capability) → Capability Added + evaluate pattern need
        ├── Broad (cross-cutting) → Pattern Adopted + derive capabilities
        └── Unknown scope → stays at flexible edge, revisit at next review
```

**Key principle:** Evaluation does not need to happen immediately. The action routing is a tool for when you choose to evaluate, not a gate on creation.

##### Flexible Edge Policy

The flexible edge is where unclassified inputs live. This section defines the cost and governance of that space.

**What "free to exist" means:**

- Issues can remain unclassified indefinitely — there is no deadline
- Unclassified issues do NOT appear in coherence signal reports (they are not yet part of the domain model)
- No documents update until classification happens
- The issue label `intake:unclassified` tracks flexible-edge items

**What triggers classification:**

| Trigger | Mechanism |
|---------|-----------|
| Voluntary | Author runs `/intake` or manually classifies |
| Review cycle | `/intake --review` surfaces unclassified issues older than 30 days for batch triage |
| Dependency | Another issue or capability needs this input classified to proceed |
| Coherence signal | An audit (`/arch-sync`, `/global-arch-sync`) detects something that matches an unclassified input |

**Cost of the flexible edge:**

The flexible edge has a carrying cost: unevaluated inputs represent potential coherence improvements that aren't being captured. Inputs that turn out to align with existing structure are the most valuable — every one is a missed compounding opportunity (coverage increases make future evaluations more accurate).

The review cycle (30-day surfacing) balances freedom with this carrying cost. It does not force evaluation — it makes the cost visible.

##### Intake Labels

| Label | Meaning |
|-------|---------|
| `intake:unclassified` | At flexible edge, not yet evaluated |
| `intake:evaluated` | Evaluation complete — territory map presented, action routed |

When coherence measurement is automated, it can derive finer categories (e.g., coverage increase, coverage gap, coherence conflict) from audit data. Until then, the territory map + delta is the evaluation output — naming the category adds overhead without changing the action.

### Consistency Checks

`/arch-sync` (per-repo) and `/global-arch-sync` (cross-repo) enforce this propagation model. Each check verifies derived documents match their authority.

**Per-repo checks (`/arch-sync`):**

- ARCHITECTURE.md capabilities match this document's Capability Registry (names, status, patterns)
- ARCHITECTURE.md integration patterns match this document's Integration Map
- INFRASTRUCTURE.md services match actual Docker/service state
- Key Components trace to capabilities (no orphan scripts)

**Cross-repo checks (`/global-arch-sync`):**

- REPOS.yaml capability names match this document exactly
- GLOBAL_ARCHITECTURE per-repo sections match this document (capabilities, status)
- GLOBAL_INFRASTRUCTURE matches per-repo INFRASTRUCTURE.md (ports, services)
- Every pattern in `pattern_v1.yaml` has a doc in `patterns/`
- Every pattern referenced in Capability Registry exists in `pattern_v1.yaml`
- PATTERN_AUDIT.md is current (regenerate if stale)
- No orphan patterns (every pattern has ≥1 capability, or is explicitly superseded/infrastructure)

---

## Coherence Signals

The signals below are the current implementation of coherence measurement — stateless sensors that run, report, and forget. When Coherence Assessment becomes operational as a first-class aggregate (ADR-0012), these sensors feed into assessments that gain identity, lifecycle, and action tracking. The evaluation process (territory map + delta) determines what each signal means and what action to take. See [Domain Model](#domain-model-aggregates-and-building-blocks) above.

### Capability-Pattern Coverage

Every core capability should trace to at least one Pattern. Current assessment:

| Capability | Pattern Coverage | Gap? |
|-----------|-----------------|------|
| `corpus-meta-analysis` | `semantic-ingestion` (1p), `raptor` (3p), `agentic-rag` (3p) | No |
| `data-due-diligence` | `togaf` (3p), `dcam` (3p), `apqc-pcf` (3p) | No |
| `reference-generation` | `mirror-architecture` (1p), `business-domain` (3p), `agentic-rag` (3p), `bizbok` (3p), `data-system-classification` (1p) | No |
| `business-model-synthesis` | `bizbok-ddd-derivation` (1p), `data-system-classification` (1p) | No |
| `agentic-ddd-derivation` | `bizbok-ddd-derivation` (1p), `ddd-agent-boundary-encoding` (1p) | No |
| `synthesis-simulation` | `mirror-architecture` (1p), `data-lineage` (3p) | No |
| `vendor-decomposition` | `system-primitive-decomposition` (1p), `explicit-architecture` (1p), `explicit-enterprise` (1p), `data-system-classification` (1p) | No |
| `business-model-decomposition` | `system-primitive-decomposition` (1p), `business-domain` (3p) | No |
| `corpus-collection` | `semantic-ingestion` (1p), `agentic-rag` (3p), `osint` (3p) | No |
| `tech-stack-profiling` | `system-primitive-decomposition` (1p), `data-system-classification` (1p), `osint` (3p) | No |
| `structured-extraction` | `semantic-ingestion` (1p), `data-modeling` (3p), `osint` (3p) | No |
| `sentiment-extraction` | `semantic-ingestion` (1p), `osint` (3p) | No |
| `research` | `raptor` (3p), `semantic-ingestion` (1p) | No |
| `internal-knowledge-access` | `agentic-rag` (3p), `tree-of-thought` (3p), `react-reasoning` (3p) | No |
| `coherence-scoring` | `semantic-coherence` (1p), `data-management` (1p), `data-quality` (3p), `data-contracts` (3p) | No |
| `ingestion-pipeline` | `semantic-ingestion` (1p), `data-quality` (3p) | No |
| `pattern-governance` | `explicit-architecture` (1p), `backstage-software-catalog` (3p) | No |
| `pattern-registry` | `semantic-object-pattern` (1p), `arc42` (3p), `metadata-management` (3p) | No |
| `architecture-audit` | `explicit-architecture` (1p), `backstage-software-catalog` (3p), `data-catalog` (3p), `data-management` (1p) | No |
| `input-optimization` | `semantic-ingestion` (1p), `explicit-enterprise` (1p), `semantic-object-pattern` (1p) | No |
| `orchestration` | `explicit-enterprise` (1p), `explicit-architecture` (1p) | No |
| `context-engineering` | `explicit-enterprise` (1p) | No |
| `content-management` | `cms` (3p), `dublin-core` (3p) | No |
| `asset-management` | `dam` (3p) | No |
| `product-information` | `pim` (3p) | No |
| `surface-deployment` | `dam` (3p) | No |
| `cap-inbox` | `explicit-enterprise` (1p) | No |
| `attention-management` | `explicit-enterprise` (1p) | No |
| `concept-documentation` | — | Planned — pattern wiring deferred |
| `cap-voice-control` | `explicit-enterprise` (1p) | **3P pattern TBD** — speech-to-text |
| `agentic-lineage` | Decomposed → `agent-observability`, `operational-metrics`, `reasoning-lineage` | Resolved |
| `scale-projection` | Decomposed → `capacity-planning`, `capex-planning`, `architecture-validation`, `vendor-evaluation` | Resolved |

**Status:** Pattern audit  applied one-hop rule, tightened overloaded capabilities, decomposed pattern-management/publishing-pipeline/domain-data-model, aligned lifecycle status. 63 patterns in `pattern_v1.yaml` v1.10.0.

**Remaining:**

- `agentic-lineage` — decomposed into `agent-observability`, `operational-metrics`, `reasoning-lineage`
- `scale-projection` — decomposed into `capacity-planning`, `capex-planning`, `architecture-validation`, `vendor-evaluation`
- `cap-voice-control` — 3P pattern TBD
- `concept-documentation` — pattern wiring deferred
- **Pattern naming convention** — current IDs are generic abbreviations; establish a convention and audit for clarity

### Capability-Repo Coverage

Every capability should have at least one delivering repo. Current registry shows full coverage.

### Script-Capability Coverage

Every script in `scripts/` should trace to a capability. Conversely, every capability should decompose into at least one script. This extends the traceability chain (see [Capability Traceability](#capability-traceability)) into a measurable signal:

| Signal | Meaning |
| ------ | ------- |
| Script with no capability trace | Unjustified code — either missing attribution or a candidate for removal |
| Capability with no scripts | Unimplemented intent — either aspirational or implemented outside `scripts/` |
| Script attributed to wrong capability | Misalignment — the script's actual function doesn't match its stated capability |

Script inventories live in each repo's `ARCHITECTURE.md` "Key Components" section. The `/arch-sync` workflow (Step 5) audits per-repo coverage; `/global-arch-sync` (Step 4e) checks cross-repo.

### Library-Capability Crosswalk

Libraries appearing in multiple repos signal shared infrastructure candidates and patterns stable enough for deterministic scripts. A simple analysis of `pyproject.toml` across repos reveals:

| Signal | Meaning |
| ------ | ------- |
| Library in 3+ repos, no shared module | Convergence candidate — standardize in `semops-data` |
| Library declared but not imported | Phantom dependency — tech debt, candidate for removal |
| Library used by scripts with no capability trace | Unjustified infrastructure — cost without architectural purpose |
| Library overlap across different capabilities | Shared abstraction candidate (e.g., `pydantic` for settings across repos) |

The crosswalk maps `Library → Repo → Script → Capability`, closing the loop between infrastructure choices and architectural intent. This is maintained in [GLOBAL_INFRASTRUCTURE.md § Stack Ecosystem](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/GLOBAL_INFRASTRUCTURE.md#stack-ecosystem).

---

## Schema Representation

These strategic concepts will be represented in the schema as:

- **Capabilities** → `entity` with `entity_type: capability`, metadata containing capability-specific fields
- **Repositories** → `entity` with `entity_type: repository`, metadata containing repo-specific fields
- **Integration relationships** → `edge` records between repository entities with `integration` predicate
- **Capability-Pattern links** → `edge` records with `implements` predicate (or `primary_pattern_id` for single-pattern capabilities)
- **Capability-Repo links** → `edge` records with `delivered_by` predicate

See [ADR-0009](decisions/ADR-0009-strategic-tactical-ddd-refactor.md) for schema migration details.

### Sample Queries

> Governance queries relocated to developer docs. Every table in this document is parsed into the database by `ingest_architecture.py`. Key governance questions (orphan patterns, capability coverage gaps, repo delivery, integration map, lifecycle stages, full traceability) become SQL via `capability_coverage` and `pattern_coverage` views.

---

## Evolution

This document is the **source of truth** for strategic DDD concepts. For change procedures, see [Governance: Change Propagation](#governance-change-propagation) above.

When aggregate structure or DDD building block classifications change, update [ADR-0012](decisions/ADR-0012-pattern-coherence-co-equal-aggregates.md) and the [Domain Model](#domain-model-aggregates-and-building-blocks) section above.
