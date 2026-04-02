# Agent Governance Model

> **Version:** 1.0.0
> **Status:** Draft
> **Date:** 2026-02-14
> **Related:**  | [UBIQUITOUS_LANGUAGE.md](UBIQUITOUS_LANGUAGE.md) | [STRATEGIC_DDD.md](../docs/STRATEGIC_DDD.md)

How SemOps governs agent-produced and agent-managed artifacts. Defines the lifecycle model, governance principles, and agent contract for all entity types.

---

## Driving Principles

### 1. Three-Layer Governance

The three-layer architecture maps directly onto governance:

```text
Pattern (WHY)           →  Lifecycle defines WHAT states mean
Architecture (WHAT)     →  Governance defines WHO can transition
Content (output)        →  Episodes record THAT it happened
```

Patterns define the meaning of lifecycle stages. Architecture determines who (human or agent) is authorized to trigger transitions. Episodes (agentic lineage) record every transition as immutable provenance.

### 2. Temporal Complements

Semantic coherence and scale projection are the same analysis in opposite time directions:

```text
Scale Projection          NOW            Semantic Coherence
      ←─────────────────── ● ───────────────────→
  "will this still work    │    "does this still align
   when we scale it?"      │     with what we declared?"
                           │
      FUTURE               │              PAST/PRESENT
   validates promotion     │         detects drift
```

Together they form the SemOps quality gate — the 1P replacement for CI/CD test suites and Backstage processing pipelines. The quality gate isn't "did tests pass" — it's "does this maintain semantic alignment in both directions?"

### 3. Self-Correcting Governance (Andon Cord)

The governance model is NOT gate-based (CI/CD model). It's a self-correcting system:

```text
TPS:     detect → stop → human fixes → resume
SemOps:  detect → auto-recover → continue
                                    ↑
                         human inspects (ripcord, on demand)
```

3P foundation: **andon cord** + **jidoka** (Lean/Toyota Production System). 1P innovation: an andon cord that **fixes the problem itself** — self-correcting semantic governance.

- The system detects drift (jidoka — coherence scoring)
- The system auto-recovers to coherence (1P innovation)
- The human can inspect at any time (andon cord / ripcord)
- The human doesn't fix — they **verify the fix was right**, on demand

The system's job isn't to prevent bad states. It's to ensure the **path back to coherence is always available and comprehensible.**

### 4. Draft as Forecast Zone

Draft entities aren't a holding pen — they're the active frontier where the system predicts future coherence.

Coherence scoring has two operational modes:

| Mode | Targets | Question |
|------|---------|----------|
| **Retrospective** | active | "Has this drifted from the baseline?" |
| **Prospective** | planned, draft, in_progress | "If this matures, does it maintain coherence?" |

The research module is coherence scoring in forecast mode, operating on draft entities + external signals:

- **Internal signals:** Emerging patterns from clustering drafts, capability gaps from unmapped proposals
- **External signals:** 3P trends mapping to capability gaps, industry alignment with existing drafts

---

## Lifecycle Model

### 5-Stage Lifecycle

Aligned with Backstage `spec.lifecycle` per ADR-0017. One lifecycle for all entity types.

```text
PLANNED → DRAFT → IN_PROGRESS → ACTIVE → RETIRED
```

| Stage | Meaning | 3P Mapping |
|-------|---------|-----------|
| **planned** | Identified, not yet started. Registered intent. | Backstage: experimental. Roadmap item. |
| **draft** | Exists, unvalidated. Ideas, open issues, WIP. | CI/CD: built untested. |
| **in_progress** | Actively being built and validated. | CI/CD: tested/staged. Implementation underway. |
| **active** | Validated, operational. Deployed, in use. Coherence baseline. | CI/CD: production. Backstage: production lifecycle. |
| **retired** | Removed from operational system. Retained for lineage/provenance. | All 3P models distinguish retirement from active. |

### Universal Governance Matrix

Same behavior for ALL entity types across all 5 stages. No exceptions.

| | planned | draft | in_progress | active | retired |
|---|---|---|---|---|---|
| **Lineage** | Episode: proposed | Episode: created | Episode: building | Episode: validated | Episode: removed |
| **Coherence** | Excluded | Forecasted | Forecasted | **IS the baseline** | Excluded |
| **Search** | Available, filtered | Available, filtered | Available, filtered | Default results | Excluded |

Entity type determines the **creation/iteration mechanism** — not whether governance applies.

### Declared vs System Lifecycle

Two orthogonal axes (adopted from Backstage's dual-track model):

| Axis | What it tracks | Who sets it | Examples |
|------|---------------|-------------|---------|
| **Declared lifecycle** | Intent — what stage the entity is in | Owner/agent via source config or explicit action | planned, draft, in_progress, active, retired |
| **System health** | Reality — what measurement detects | Coherence scoring, drift detection | coherent, drifted, orphaned |

The declared lifecycle is simple and rarely changes shape. The system health axis is where the SemOps innovation lives — coherence scoring, drift detection, orphan detection.

**When system health degrades, governance determines what happens:** auto-recovery (self-correcting), flag for attention (andon cord), or reduce autonomy level.

### Iteration Model

**Lifecycle is sticky.** Iteration (re-ingestion, content updates) does not reset the lifecycle stage.

- Entities are mutable — `ON CONFLICT (id) DO UPDATE`
- Every change produces an immutable episode (lineage record)
- The episode chain IS the version history — no separate versioning table
- Coherence scoring detects if an update introduced drift
- Human awareness of lifecycle is inversely proportional to agentic maturity:
  - **High-touch** (DAM publishing): human feels every state transition
  - **Low-touch** (architecture updates): system handles it, coherence catches drift

---

## Agent Contract

### Universal Requirements

Every entity type, regardless of content, must satisfy:

1. **Declared creation path** — every entity type has a known mechanism for entering the lifecycle (source config, registration script, LLM classification). Undeclared entities are audit findings.
2. **Episode on every action** — creation, iteration, classification, edge proposal, state transition. No silent mutations.
3. **Coherence measurement** — all active entities are measured. All planned/draft/in_progress entities are forecasted. No entity type is exempt.
4. **Recoverable state** — at any point, the system can answer "where are we?" and "how do we get back to coherent?"

### Five-Primitive Isomorphism

The five agent primitives (, ) map to SemOps constructs and SC measurement dimensions. Planning is not a sixth primitive — it emerges from Model + Prompt + Memory interaction, captured as `reasoning_pattern` on episodes.

| Agent Primitive | SemOps Equivalent | SC Dimension | Episode Column |
| --------------- | ------------------------------------------------- | ---------------------- | ------------------------------------------------- |
| **Model** | Script (execution unit) | — | `model_name` |
| **Tools** | Capability (what it can do) | Availability | `operation` |
| **Memory** | Pattern registry + episodic store | Stability | `context_pattern_ids` |
| **Context** | Assessment unit (pattern × capability × domain) | Consistency | `context_entity_ids`, `context_token_count` |
| **Orchestration** | Workflow (single-agent = runtime; multi-agent = infrastructure) | — | `run.run_type` |
| *(emergent)* Planning | Reasoning strategy | *(correlates with all)* | `reasoning_pattern`, `chain_depth` |

SC provides the quantitative measurement; context engineering provides the optimization target. Reasoning traces (`reasoning_pattern`, `context_utilization`) connect the two — they explain *why* a given context assembly produced the coherence score it did.

### Operation Authority by Scale Projection Level

| Level | Transition Model | Evidence | HITL Role |
|-------|-----------------|----------|-----------|
| **L1** (today) | Human sets lifecycle_stage manually | Session notes, commit messages | Human drives |
| **L2** | Agent recommends transitions, human approves | Episode + recommendation | Human approves |
| **L3** | Agent transitions with provenance, human reviews | Episode + full lineage | Human reviews |
| **L4** | Agent autonomous, coherence scoring validates | Episode + coherence score | Human inspects on demand |
| **L5** | Full agentic governance, anomalies trigger HITL | Episode + drift detection | Human pulls ripcord |

**Current state:** Ingestion episodes are at L3-L4. Most other operations are at L1-L2.

---

## Governance Modes: Internal vs Public Boundary

The self-correcting andon cord model governs internal operations. At the public/customer boundary, hard gates apply. Both operate on the same lifecycle — the difference is enforcement.

```text
INTERNAL (andon cord)          PUBLIC BOUNDARY (hard gate)
─────────────────────          ─────────────────────────────
act → record → auto-recover    act → record → gate → approve → publish
  ↑                              ↑
  human inspects on demand       human MUST approve before external visibility
```

**Where hard gates apply:** Any transition that makes content visible to customers or the public. This includes publishing to surfaces (WordPress, LinkedIn, semops.ai), deploying public infrastructure, and releasing public-facing artifacts. Agentic activity near the customer/public surface requires guardrails — internal drift is recoverable, public drift is reputational.

**Where the andon cord applies:** Internal operations — architecture updates, entity ingestion, edge proposals, pattern management, coherence scoring. These can self-correct; the ripcord is there for when attention is needed.

The governance mode is determined by the **delivery target**, not by the entity type or lifecycle stage. The same entity at `lifecycle_stage: active` might use the andon cord internally but require a hard gate when published to a surface.

### Two Operational Layers

The governance modes above are enforced through two complementary mechanisms:

```text
LAYER 1: STATE INSPECTION (andon cord)
──────────────────────────────────────
Trigger:    Human-initiated or scheduled
Mechanism:  Claude Code session, slash command, query script
Data:       pattern_coverage, capability_coverage views
Question:   "What's the state of things?"
Examples:   /project-review, coverage dashboards, semantic_search.py
Frequency:  On demand or periodic

LAYER 2: CONTENT VERIFICATION (continuous audit)
─────────────────────────────────────────────────
Trigger:    Near-continuous, event-driven (commits, ingestion, deploys)
Mechanism:  Agent scans actual repo contents, compares to declared relationships
Data:       Actual imports, script contents, crosswalk tables
Question:   "Does what we declared match what's actually deployed?"
Examples:   Library→Capability crosswalk audit, phantom dependency detection
Frequency:  Near-continuous at L4-L5
```

**Layer 1 inspects the declared lifecycle** — querying what we said (coverage views, edge counts, lifecycle stages). This is the ripcord: at any point, human can ask "what's going on?" and get a coherent answer.

**Layer 2 measures system health** — verifying that declared relationships match actual code. Does `classifiers/llm.py` actually import `anthropic`? Does a capability claiming `implements: semantic-ingestion` have scripts that perform ingestion? Findings include:

- **Phantom dependencies:** declared in crosswalk but not actually imported
- **Undeclared dependencies:** imported in code but not declared in crosswalk
- **Stale edges:** `implements` or `delivered_by` edges pointing to capabilities/repos that have drifted from their declarations

**The gap between Layer 1 and Layer 2 is where drift lives.** At L1-L2, Layer 1 is the primary mechanism and Layer 2 is manual (humans review crosswalks). At L4-L5, Layer 2 runs near-continuously and Layer 1 becomes the on-demand dashboard.

---

## Per-Entity Governance: Pattern (Aggregate Root)

Pattern is the aggregate root (ADR-0004). Its governance defines the reference model for all other entity types.

### Pattern Is Stateless — Activation Is Per-Context

**The pattern definition is stable.** The pattern table holds the SKOS definition (preferred_label, provenance, broader/narrower). It has no `lifecycle_stage` column — by design. A pattern is registered or it isn't. The pattern row is a semantic anchor, not a stateful entity.

**Activation is per-context.** The same pattern can be active in one capability and draft in another. `prov-o` might be active for `agentic-lineage` (episodes are deployed) and draft for a new capability exploring PROV-O for a different purpose. The lifecycle lives on the **capability** that applies the pattern, not on the pattern itself. Each `implements` edge represents a context where the pattern is being applied, and the capability's lifecycle stage determines whether that application is draft, active, or stable.

**3P patterns get iterated with 1P innovations — differently per context.** A pattern like `prov-o` (3P, W3C) enters the system as a stable definition. But each capability that applies it adds its own 1P innovations: `agentic-lineage` innovates episode chain as version history, while a different capability might innovate PROV-O for coherence provenance in a completely different way. The same 3P pattern can be applied 10 times with 10 different 1P innovations. The pattern definition stays 3P — the 1P innovation lives on the `implements` edge + the capability. This is the semantic optimization loop (adopt 3P → innovate 1P) operating at the application level, not the definition level.

**Pattern lifecycle is emergent.** A pattern's "global" status is the aggregate of its application lifecycles:

| Pattern Status | Condition |
|---------------|-----------|
| **Unregistered** | Referenced in work contexts (issues, session notes) but not in pattern table |
| **Registered, no applications** | In pattern table but `pattern_coverage.capability_count = 0` |
| **Active in N contexts** | N capabilities with `implements` edge have `lifecycle_stage: active` or higher |
| **Deprecated** | Successor pattern exists (SKOS shift). Active capabilities should migrate. |

### Pattern Emergence: When Concepts Become Patterns

Not every 1P innovation becomes a pattern. Many remain innovations within their capability context. Pattern registration is an **architectural decision**, not a lifecycle transition.

```text
3P patterns (adopted)          1P concept (within capability)          1P pattern (registered)
─────────────────────          ──────────────────────────────          ─────────────────────────
prov-o (W3C)            →     "episode chain as version              scale-projection
episodes (DataHub)       →      history" lives within                  (registered when
                                agentic-lineage capability              architecturally significant)
                                                                        ↓
                                                                    SKOS broader: prov-o,
                                                                    semantic-coherence
```

**The emergence path:**

1. **Concept phase** — A 1P idea lives within a capability's innovation on 3P patterns. Referenced in session notes, issues, design docs. Not a pattern — it's an innovation within a capability context. The association trail captures these references.
2. **Architecture modeling** — Before building, model it: register the pattern, declare the capability, assign to a repo. Registration is the architectural commitment — "we're going to build this, and here's where it fits." SKOS broader/narrower links connect it to the 3P patterns it innovates on. This follows the three-layer principle: Pattern (WHY) and Architecture (WHAT/WHERE) precede Content (output/implementation).
3. **Implementation** — Build against the modeled architecture. The capability becomes active when something is actually deployed. The pattern's activation in this context is emergent from the capability's lifecycle.

**Not all innovations spawn patterns.** A capability might apply `prov-o` with a specific 1P twist that only matters in that context. That innovation lives on the `implements` edge metadata and the capability's documentation — it doesn't need its own pattern row. Registration is for concepts with architectural reach.

**Registration is an agentic guard rail, not a bureaucratic gate.** The system enforces registration as a natural prerequisite: `ingest_architecture.py` checks whether patterns exist before creating `implements` edges, warning about unregistered patterns. The agent hits this organically — "I can't declare that capability X implements pattern Y because pattern Y doesn't exist. I need to register it first." This is jidoka built into the workflow: the system detects the gap, and the resolution is straightforward. At L1-L2, the human registers. At L4-L5, the agent proposes registration autonomously.

### Key Insights

**Registration is not activation.** A pattern enters the table via `ingest_domain_patterns.py`, but that only means it's been declared as a definition. Activation requires a capability to implement it and deploy something against it.

**The association trail matters more than the gate.** Pattern governance isn't primarily about preventing bad registrations — it's about capturing which patterns are referenced in which work contexts (issues, session notes, architecture docs) before and after registration. The episode chain records this trail. The trail also reveals when a concept is ready to become a pattern — repeated references across contexts signal architectural significance.

**Draft patterns in public repos are a surface concern.** A pattern definition can appear in a public repo — the repo's documentation is the contract that tells consumers which patterns are established vs emerging. This is the public boundary hard gate applied at the delivery/surface level, not at the pattern level.

**Coverage views show applications, not pattern state.** `pattern_coverage.capability_count` tells you how many capabilities reference this pattern. To know which applications are active vs draft, filter by the capability's lifecycle stage.

### Existing Traceability (Schema Views)

| View | Signal | Query |
|------|--------|-------|
| `pattern_coverage` | Content, capability, and repo counts per pattern | "Which patterns have nothing deployed against them?" |
| `capability_coverage` | Pattern and repo counts per capability | "Which capabilities lack pattern justification?" |
| `repo_capabilities` | Which repos deliver which capabilities | "What does repo X actually deliver?" |
| `integration_map` | DDD integration patterns between repos | "How do bounded contexts connect?" |

---

## Per-Entity Governance: Capability

Capability is the bridge between Pattern (WHY) and Repository (WHERE). Its governance evidence is bidirectional: does it trace UP to patterns and DOWN to repos?

### Capability Lifecycle Evidence

| Stage | Evidence Required | How Measured |
|-------|------------------|--------------|
| **planned** | Identified in roadmap, not yet registered | Referenced in project specs or issues but not in registry. |
| **draft** | Registered but incomplete edges | In registry but `capability_coverage.pattern_count = 0` OR `repo_count = 0` — missing either justification or delivery. |
| **in_progress** | Being built, partial edges exist | Has some `implements` or `delivered_by` edges but not both. Active project/issue work underway. |
| **active** | Traces to patterns above AND repos below | `capability_coverage.pattern_count > 0` AND `repo_count > 0`. Has `implements` edges to patterns and `delivered_by` edges to repos. |
| **retired** | No repos deliver it, no patterns reference it | Removed from active registry. Retained for lineage/provenance. |

### Key Insight

**A capability without both edges is incomplete.** `pattern_count = 0` means it has no justification (WHY does this exist?). `repo_count = 0` means it has no delivery mechanism (WHERE does it run?). Either gap makes it effectively draft — declared but not deployed. The `capability_coverage` view already detects both gaps.

### Layer 2 Audit

The continuous audit for capabilities verifies that:
- Declared `implements` edges match actual pattern usage in the capability's scripts
- Declared `delivered_by` edges match actual script existence in the repo
- The Library → Capability Crosswalk in GLOBAL_INFRASTRUCTURE.md matches actual imports

---

## Per-Entity Governance: Repository

Repository is the infrastructure anchor — where capabilities are delivered and code runs.

### Repository Lifecycle Evidence

| Stage | Evidence Required | How Measured |
|-------|------------------|--------------|
| **planned** | Repo identified in roadmap but not yet created | Referenced in project specs or architecture docs but no GitHub repo. |
| **draft** | Repo created but not delivering capabilities | Created on GitHub but `repo_capabilities` shows 0 capabilities. Spike, experiment, or POC. |
| **in_progress** | Repo being built, some capabilities wired | Some `delivered_by` edges exist but integration patterns incomplete. Active development. |
| **active** | Delivers capabilities with defined integration patterns | `repo_capabilities` shows capabilities with `delivered_by` edges. Integration edges in `integration_map`. Bounded context boundaries defined. |
| **retired** | Archived on GitHub, no capabilities delivered | No capabilities reference it. GitHub repo archived. Retained for lineage. |

### Key Insight

**Repository lifecycle maps to bounded context maturity.** A repo at `draft` is an experiment — no capabilities, no integration patterns. At `active`, it delivers capabilities and has integration edges with documented bounded context boundaries. The `integration_map` view is the maturity signal — a repo without integration patterns is either isolated or still maturing.

### Layer 2 Audit

The continuous audit for repositories verifies that:
- Declared capabilities in STRATEGIC_DDD.md match actual scripts in the repo
- Integration patterns (shared-kernel, customer-supplier) match actual shared artifacts
- Library dependencies match the Library → Capability Crosswalk
- REPOS.yaml metadata (role, context) matches current reality

---

## Per-Entity Governance: Content

Content entities are DAM publishing artifacts — the output layer. Their governance is fundamentally different from Pattern/Capability/Repository because lifecycle is primarily set by **source configuration**, not by edge coverage.

### Content Lifecycle Evidence

| Stage | Evidence Required | How Measured |
|-------|------------------|--------------|
| **planned** | Content identified but not yet created | Referenced in project specs or issues. No file exists yet. |
| **draft** | Pre-delivery, unvalidated | Source config path pattern (e.g., `**/drafts/**`). Open issues, WIP docs, feature branches. Coherence operates in forecast mode. |
| **in_progress** | Being written or reviewed | Active PR or recent edits. Not yet merged to main. |
| **active** | Merged, deployed, operational | Default lifecycle_stage from source config. Entity has been ingested, classified, embedded. Coherence is measured. |
| **retired** | Historical only, removed from active system | Source config path pattern (e.g., `**/_archive/**`, `**/_legacy/**`). Excluded from search and coherence scoring. Retained for lineage/provenance. |

### Key Insight

**Content lifecycle is path-driven, not edge-driven.** Unlike Pattern/Capability/Repository where lifecycle evidence comes from edge coverage (implements, delivered_by), content lifecycle is primarily determined by where the file lives in the source repo. The source config routing rules (`corpus_routing.rules`) map paths to lifecycle stages. This is already implemented.

**The coherence question for content is different.** For patterns, "active" means something is deployed against it. For content, "active" means it aligns with the patterns it references. The `orphan_entities` view detects content without pattern connections — these are the "flexible edge" awaiting incorporation into the active core.

### Layer 2 Audit

The continuous audit for content verifies that:
- Declared `primary_pattern_id` references a pattern that still exists and is active
- LLM classification outputs (content_type, concept_ownership, broader/narrower concepts) are coherent with the entity's corpus and lifecycle stage
- Orphan entities (no pattern connection) are identified and triaged

---

## Content-Type-Specific Layers

The universal lifecycle applies to all entity types. Content-type-specific concerns sit **on top** as additional layers:

| Layer | Scope | Example |
|-------|-------|---------|
| **delivery.approval_status** | DAM publishing to surfaces | pending → approved → published (gates WordPress, LinkedIn, etc.) |
| **ADR status** | Architecture decisions | Draft → In Progress → Complete → Superseded (in frontmatter) |
| **Pattern lifecycle** | Aggregate root management | Stateless definition (3P anchor). Activation is per-context — lifecycle lives on the capability that applies it, not on the pattern. |

These are not alternatives to lifecycle_stage — they're domain-specific gates that operate within the universal model. A blog post can be `lifecycle_stage: active` (operational, in the corpus) but `approval_status: pending` (not yet published to WordPress). A pattern can be `lifecycle_stage: draft` but visible in a public repo because the repo's documentation is the delivery contract.

---

## 3P Pattern Foundations

| Concern | 3P Pattern | 1P Innovation |
|---------|-----------|---------------|
| Lifecycle states | Software catalog lifecycle (Backstage), CI/CD promotion, DDD aggregate | Universal 5-stage model with coherence-driven semantics |
| Provenance | PROV-O (already adopted) | Episode chain as version history |
| Quality gate | CI/CD quality gates, Backstage processing pipeline | Coherence scoring + Scale Projection as temporal complements |
| Stop-and-fix | Andon cord + Jidoka (Lean/TPS) | Self-correcting semantic governance |
| Publishing workflow | DAM (already adopted) | delivery.approval_status as content-type-specific layer |
| Health detection | Backstage orphan detection, CI/CD vulnerability scanning | Coherence drift detection, system health axis |

---

## Related

- [UBIQUITOUS_LANGUAGE.md § Lifecycle Stage](UBIQUITOUS_LANGUAGE.md) — domain definitions
- [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md) — lifecycle_stage field specification
- [STRATEGIC_DDD.md](../docs/STRATEGIC_DDD.md) — capability registry, pattern traceability
-  — design discussion and driving principles
-  — alignment issue (where reframing was decided)
