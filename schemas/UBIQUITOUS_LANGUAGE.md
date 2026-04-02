# Project SemOps Ubiquitous Language

> A shared vocabulary for the Project SemOps domain, structured around the domain model and used by all team members and AI agents within this bounded context.
> **Version:** 10.2.0 | **Last Updated:** 2026-03-22

---

## About This Document

This is a **Ubiquitous Language** as defined by Eric Evans in Domain-Driven Design:

> *"A language structured around the domain model and used by all team members within a bounded context to connect all the activities of the team with the software."*
> — Evans, *DDD Reference* (2015)

**Scope:** Single bounded context across all Project SemOps repositories. All repos share this vocabulary — `semops-data` owns it; other repos consume it as a shared kernel.

**For column-level specifications, JSONB schemas, SQL types, and index details, see [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md).**

---

## Domain Overview

**Semantic Operations (SemOps)** is a framework for aligning technology and organization to benefit from data, AI, and agentic systems. The framework posits that **meaning is the critical currency in knowledge work** — every business system, decision process, and AI integration depends on shared meaning.

Project SemOps is both the framework and its proving ground. The system builds a **knowledge-first digital publishing platform** where patterns (units of meaning) are the stable core, content artifacts are the output, and the architecture connecting them is itself a first-class domain concept.

### The SemOps Framework

SemOps is built on a mental model and three pillars:

- **Semantic Funnel** — A mental model grounding how meaning transforms through progressive stages (Data → Information → Knowledge → Understanding → Wisdom). Each transition increases uncertainty and requires more complex inputs.

- **Strategic Data** — A playbook for making data a first-class strategic asset. Data must be governed, structured, and treated as an organizational challenge. The central insight is **Governance as Strategy** — governance structures (lineage, quality, metadata, provenance) are not compliance overhead but the primary mechanism through which the organization executes its strategy.

- **Explicit Architecture** — Encode your strategy into your systems so humans and AI can operate from shared structure. DDD provides the foundation.

- **Semantic Optimization** — Elevate your organization to operate like well-designed software: agent-ready, self-validating, and expanding through patterns, not features.

Each pillar provides value independent of AI. Together, they create an environment where ideas exist as human-readable patterns that agents can work with directly, and AI performs better because it has wider context and a well-understood domain.

### Core Thesis

AI excels where systems enforce coherent meaning. **Semantic Coherence** — the degree to which human and machine knowledge is available, consistent, and stable — is both the goal and the measurable signal:

```text
SC = (Availability × Consistency × Stability)^(1/3)
```

If any dimension collapses, coherence collapses. The framework builds the conditions for coherence; AI is both a beneficiary and an accelerant.

### Semantic Optimization Loop

The fundamental process by which the system evolves:

1. **Adopt** established 3P standards (the stable baseline)
2. **Innovate** 1P on top (tracked, intentional deviations)
3. **Link** 1P to its 3P foundations via SKOS broader/narrower relationships
4. **Measure** coherence to validate the innovation

This loop applies at every level — from adopting a W3C standard to evolving a business process.

---

## Three-Layer Architecture

Project SemOps organizes into three layers (ADR-0009). Each layer has a distinct purpose, and each links upward to the layer above it:

```text
┌─────────────────────────────────────────────────┐
│  PATTERN (Core Domain)                          │
│  Stable semantic concepts — the WHY             │
│  ddd, skos, semantic-coherence, shared-kernel   │
├─────────────────────────────────────────────────┤
│  ARCHITECTURE (Strategic Design)        [NEW]   │
│  System structure — the WHAT and WHERE          │
│  Capabilities, Repos, Integration relationships │
│  Every capability traces to ≥1 pattern          │
├─────────────────────────────────────────────────┤
│  CONTENT (DAM - Publishing)                     │
│  Publishing artifacts — the output              │
│  Blog posts, articles, media                    │
│  Surface, Delivery, PIM/Brand                   │
└─────────────────────────────────────────────────┘
```

- **Content documents Patterns** — A blog post explains semantic coherence.
- **Architecture implements Patterns** — The ingestion pipeline capability implements the semantic-ingestion pattern.
- **Patterns are the stable core** everything traces to.

### Adopted Standards (3P)

These standards directly shaped the database schema — their concepts are embedded in table structures, edge predicates, and value objects. Capabilities adopt additional 3P standards documented in [STRATEGIC_DDD.md](../docs/STRATEGIC_DDD.md).

| Standard | Role | Layer |
|----------|------|-------|
| **DDD** (Domain-Driven Design) | Primary architecture. Bounded contexts, aggregates, ubiquitous language. | All |
| **SKOS** (W3C Simple Knowledge Organization System) | Pattern taxonomy. Broader/narrower/related relationships between patterns. | Pattern |
| **PROV-O** (W3C Provenance Ontology) | Content lineage. Derived_from, cites, version_of edges between entities. | Content |
| **Dublin Core** | Content attribution. Creator, rights, publisher metadata on entities. | Content |
| **DAM** (Digital Asset Management) | Content publishing. Approval workflows, multi-channel distribution. | Content |
| **Schema.org** | Actor modeling. Person, Organization, Brand types for CRM/PIM. | Content |

### 1P Innovations

Derived from the capability audit in [STRATEGIC_DDD.md](../docs/STRATEGIC_DDD.md). Each innovation applies the Semantic Optimization Loop: adopt 3P standards, then innovate 1P on top.

| 1P Pattern | What It Does | Capabilities | Built On (3P) |
|------------|-------------|--------------|----------------|
| **`semantic-coherence`** | Measurable signal: SC = (Availability x Consistency x Stability)^(1/3). The goal and the measure. | coherence-scoring | *(original 1P concept)* |
| **`semantic-ingestion`** | Every byproduct of ingestion becomes a queryable knowledge artifact — classifications, detected edges, coherence scores, embeddings. | ingestion-pipeline, research, agentic-composition | ETL, Medallion Architecture, MLOps |
| **`agentic-lineage`** | Extends lineage tracking with agent decision context and trust provenance. | agent-observability, operational-metrics, reasoning-lineage | OpenLineage, Episode Provenance |
| **`semantic-object-pattern`** | Patterns as the aggregate root — provenance-tracked, lineage-measured, AI-agent-usable semantic objects. | pattern-governance, pattern-registry | Knowledge Organization Systems, Pattern Language |
| **`scale-projection`** | Validate domain coherence by projecting architecture to scale. Manual HITL processes intentionally generate structured ML training data. | capacity-planning, capex-planning, architecture-validation, vendor-evaluation | RLHF, SECI |
| **`explicit-enterprise`** | Enterprise systems treat architecture, data, and AI as first class. Humble tools become agent-addressable signal streams. | orchestration, context-engineering, attention-management, financial-pipeline | Platform Engineering, Context Engineering |

> **Note:** `governance-as-strategy` was reclassified from 1P pattern to concept entity . It is a principle — "governance is strategy execution, not compliance" — that informs how `data-management` disciplines are applied, but does not independently fit a domain, imply capabilities, or have recognizable fit (the three-property test).

---

## Pattern Layer (Core Domain)

### Pattern

A pattern is an approach that fits a domain and implies a set of capabilities. Choosing "CRM" when you need to manage customer relationships is adopting a pattern — it tells you what your system should be able to do (contact management, ticketing, SLA tracking) and connects you to established practices that humans and AI agents both understand.

**Technical definition:** An approach that (1) fits a **domain** — the subject matter of the work, (2) implies a set of **capabilities** — what the system should be able to do, and (3) is **recognizable** — someone with basic understanding of the domain would see why this approach fits this problem space.

**Domain** means the "about-ness" of the work. A CRM vendor's domain is customer relationships. A research lab's domain is their field of study. A publishing platform's domain is content lifecycle. Patterns are approaches within whatever the domain is.

**Patterns imply capabilities.** A pattern carries expectations about what the system should be able to do. The strength varies by pattern type: domain patterns directly imply specific capabilities (CRM → contact management, ticketing, SLA tracking), implementation patterns constrain how capabilities are built (medallion-architecture → staged data processing), and concept patterns influence capability design (SECI → knowledge externalization practices). When implied capabilities exist but aren't linked, coherence discovery finds them. When they're missing, coherence governance flags the gap.

**Recognizable fit.** "Fits" is inherently subjective, but the bar is practical: a person (or AI agent) with basic understanding of the domain can see why this approach applies to this problem space. This is also why patterns are AI-legible — they correspond to established knowledge that models understand well, encoding large bodies of meaning in small context packages.

Pattern is the **aggregate root** of the Pattern aggregate (ADR-0012) — the transactional consistency boundary for pattern registration, SKOS relationships, and adoption lineage. Patterns are the **prescriptive force** of SemOps — "what should we look like?" — while Coherence Assessment is the evaluative counterpart that measures "does reality match intent?" Together they form the Semantic Optimization Loop.

**Characteristics:**

- **Durable** — Survives even if all content artifacts referencing it are deleted. The pattern is the meaning, not the packaging.
- **Scalable** — Can represent an entire domain approach ("adopt CRM") or a specific technique ("adopt SKOS for taxonomy"). The SKOS hierarchy handles scale relationships between patterns.
- **Portable** — Not specific to any one system. The same pattern model can describe any domain's ideal architecture: Business Architecture → DDD Architecture → Capabilities filled with Patterns = ideal state.
- **Provenance-tracked** — Always answers "whose structure is this?" via 1P/2P/3P classification.
- **Measurable** — Connected to capabilities, which are the subject of coherence audits. Pattern coverage is a measurable coherence signal.

### Pattern Lifecycle

The SemOps process for working with patterns. This is where provenance, neutral primitives, and the Semantic Optimization Loop operate.

**The input does not have to be structured.** A human (or agent) can enter a single capability, a set of capabilities, a broad domain pattern, or a narrow feature request. The system normalizes the input to the right pattern level through coherence evaluation. This is intentional — model depth earns intake freedom. The ideation process should be frictionless; the domain model does the work of finding where things fit.

1. **Discovery** — A need surfaces (feature request, architecture gap, new domain to model). The input can arrive at any level of specificity:
   - **A capability** ("I need ticketing") → coherence asks which pattern implies ticketing, discovers the larger pattern (CRM), and surfaces its full implied capability set.
   - **A set of capabilities** ("ticketing, contact management, SLA tracking") → the set implies a pattern; coherence confirms and identifies gaps.
   - **A broad pattern** ("apply CRM") → fine as-is. You may only implement a few of its implied capabilities now; the rest are coherence signals, not failures. Partial implementation of a pattern is expected — the pattern IS the pattern regardless of how much is implemented today.
   - **A pattern at the wrong scale** ("apply Maersk-style global shipping" for a local delivery problem) → recognizable fit catches this. Someone with basic domain understanding of local delivery would say the right consensus pattern is last-mile delivery, not global shipping. The decomposition to neutral capabilities reveals the mismatch: customs clearance, container logistics, and port management don't apply. The right pattern is simpler and well understood at the correct scale.
   - **A meta-pattern** ("enterprise resource planning") → the process narrows to the relevant member of the set based on which sub-patterns actually fit the domain.
   Discovery can be bottom-up ("we need X" → pattern recognition) or top-down ("apply CRM domain" → capability decomposition).
2. **Adoption (3P)** — An established approach is registered as a pattern. It is expressed at the **neutral/primitive level** — the non-proprietary core. You can start from a proprietary reference ("do it the Salesforce way"), but adoption requires decomposing it into neutral capabilities first (contact management, ticket lifecycle, automation rules). What remains after decomposition are the consensus patterns; what the proprietary source added on top is either discarded or registered as a distinct 1P innovation.
3. **Innovation (1P)** — A novel approach is synthesized from established building blocks. `semantic-coherence` combines information theory + system quality metrics into something new. `semantic-ingestion` synthesizes ETL + Medallion Architecture + MLOps. 1P patterns are linked back to their 3P foundations via SKOS broader/narrower, ensuring they remain grounded in shared knowledge even when the synthesis is novel. The 3P layer is "don't reinvent the wheel." The 1P layer is where differentiation happens.
4. **Capability mapping** — The pattern produces capabilities — named, bounded pieces of functionality. This is where "should be" meets implementation.
5. **Coherence measurement** — Coherence Assessment audits the gap between pattern intent and implementation reality. Coverage, alignment, and regression are measurable signals.
6. **Evolution** — The Semantic Optimization Loop: adopt 3P → innovate 1P on top → link via SKOS → measure coherence → repeat. Patterns may be extended, modified, or reversed based on coherence signals.

### Provenance

Provenance answers: **whose semantic structure is this?**

| Provenance | Meaning | Example |
|------------|---------|---------|
| **1P** (first party) | Operates in my system. May be a synthesis from 3P sources, but it's now incorporated and operational. | `semantic-coherence`, `stable-core-flexible-edge` |
| **2P** (second party) | Jointly developed with an external party. Partnership/collaborative. | — |
| **3P** (third party) | External reference. Industry standard or external IP we adopt as-is. | `ddd`, `skos`, `prov-o`, `dublin-core` |

**Key insight:** 1P does not mean "I invented this." It means "this semantic structure now operates in my system." The provenance lifecycle flows: 3P (adoption) → 1P (incorporation) → optionally 2P (collaboration).

### Pattern Type

Patterns are typed by their role in the knowledge architecture:

| Type | Purpose | Examples |
|------|---------|---------|
| **concept** | Theoretical/abstract knowledge — ideas, frameworks, principles that inform design but are not directly coded as system features. **Lineage only — NEVER wires to capabilities.** A concept pattern with a capability wire is a bug; it must promote to domain/analytics/process. | `scale-projection`, `seci`, `pattern-language` |
| **domain** | Patterns that model the knowledge domain — what the system represents and reasons about. Capability *does* something. | `ddd`, `skos`, `prov-o`, `semantic-coherence`, `semantic-ingestion` |
| **analytics** | Measurement standards, compound metrics — gold-layer patterns that derive from domain data and close the strategic loop (conformance + outcome). Capability *measures* something. | `semantic-coherence-score`, `MAU`, `NPS`, `churn-rate`, `gross-margin` |
| **process** | Goal coordination and audit — choreography/saga patterns that orchestrate domain + analytics capabilities toward outcomes. Capability *orchestrates* something. 1P process patterns derive from 3P DDD tactical concepts (saga, orchestration, process manager). | `compensating-workflow`, `sequenced-pipeline`, `stateful-routing` |
| **implementation** | Solution-space patterns — how we build, deploy, and integrate. `acts_on` target — infrastructure that capabilities run on. Not wired via `implements_patterns`; referenced via capability `acts_on`. | `shared-kernel`, `medallion-architecture`, `jamstack`, `mirror-architecture`, `ci-cd` |

**Analytics patterns** imply named capabilities (e.g., `semantic-coherence-score` → `coherence-scoring`, `churn-rate` → `churn-measurement`). They close the strategic loop by answering two questions: (1) are you implementing your strategy? (conformance metrics), and (2) is your strategy working? (outcome metrics). Analytics patterns derive from domain pattern data — they don't own data, they measure it.

**Process patterns** orchestrate domain and analytics capabilities toward goals. They don't own data — they coordinate. A process pattern's coherence can only be as high as the analytics it depends on, which can only be as high as the domain patterns they derive from. This strict dependency direction (process → analytics → domain) makes scoring self-diagnosing: a missing analytics capability means the goal can't be scored yet, and the gap is the roadmap. See [ADR-0014](../docs/decisions/ADR-0014-coherence-measurement-model.md) § Pattern-Scaffolded Scoring.

The 1P/3P distinction is handled by provenance, not by pattern type. A 3P standard like `skos` is a `domain` pattern with `provenance: 3p`. A technical choice like `medallion-architecture` is `implementation` with `provenance: 3p` (we adopted Databricks' pattern). A 1P innovation like `scale-projection` is a `concept` pattern with `provenance: 1p` (aspirational, not yet operational).

### Pattern Relationships

Patterns connect to each other through two relationship families:

**SKOS Hierarchy** — Taxonomic positioning:

- **broader** — This pattern is more specific than the target. `semantic-drift` is broader → `semantic-coherence`.
- **narrower** — This pattern is more general than the target. `semantic-coherence` is narrower → `semantic-drift`.
- **related** — Associative, non-hierarchical. `semantic-coherence` is related → `bounded-context`.

**Adoption Lineage** — How patterns build on each other:

- **adopts** — Uses a 3P pattern as-is. `semantic-operations` adopts `skos`.
- **extends** — Builds on a pattern with additions. `semantic-operations` extends `ddd`.
- **modifies** — Changes a pattern for specific use. `content-classify-pattern` modifies `dam`.

**Scoring Dependency** — Pattern-scaffolded scoring chain ([ADR-0014](../docs/decisions/ADR-0014-coherence-measurement-model.md) § Pattern-Scaffolded Scoring):

- **requires** — Source pattern depends on destination for scoring/measurement. Strict dependency direction: process → analytics → domain. A process pattern `requires` the analytics patterns that measure its goals; analytics patterns `require` the domain patterns that produce the data they measure.

### Coherence Assessment

**How well reality matches intent, and what to do about it.** Coherence Assessment is the evaluative/directive counterpart to Pattern — the second core aggregate of the Semantic Operations domain (ADR-0012). Pattern is the prescriptive force ("what should we look like?"); Coherence Assessment is the evaluative force ("does reality match intent?"). Together they form the Semantic Optimization Loop.

**Coherence is audit, not gate.** Coherence does not block action. The flexible edge — orphan entities, unattributed scripts, experimental infrastructure — is free to exist. Aggregate root invariants (valid SKOS hierarchy, provenance rules) protect the stable core. Coherence audits the gap between what exists and what's formalized. Incorporation into the stable core is voluntary — that's when invariants kick in.

**Three modes:**

| Mode | Signal | Example |
|------|--------|---------|
| **Governance** | Something exists without justification | "This script has no pattern trace — here's your coverage gap" |
| **Discovery** | Something aligns but isn't tracked | "This infrastructure aligns with ingestion-pipeline — formalizing would close a gap" |
| **Regression** | Something that was coherent broke | "This change opened 3 new gaps across these capabilities" |

Discovery mode compounds: every latent alignment made explicit increases pattern coverage, which makes future assessments more accurate. The semantic optimization loop accelerates itself.

**Aggregate shape** (preliminary — schema deferred until coherence scoring is operational):

- **Trigger:** What change prompted this assessment
- **Scope:** Which patterns/capabilities are being assessed
- **Measurements:** Per-pattern/per-capability alignment signals (Availability, Consistency, Stability)
- **SC Score:** (A × C × S)^(1/3) — the semantic-coherence formula
- **Gaps:** Identified misalignments (with mode: governance/discovery/regression)
- **Actions:** Recommended changes (add, remove, modify, revert, formalize)
- **State:** assessed → acting → resolved → superseded

**Fitness functions are coherence sensors.** The existing fitness functions and coverage views detect signals that feed into assessments. As standalone tools, they're stateless — run, report, forget. As sensors feeding Coherence Assessment, their signals gain identity, lifecycle, and action tracking.

---

## Architecture Layer (Strategic Design)

The Architecture layer formalizes the strategic design of Project SemOps — what the system delivers and where implementation lives. These concepts were previously in prose documentation only; ADR-0009 made them first-class domain objects.

### Capability

**What the system delivers.** A capability is a named, bounded piece of functionality that implements one or more patterns and is delivered by one or more repositories.

**DDD classification: Entity.** Capability has identity and lifecycle but is not an aggregate. It exists in the space between the two core aggregates: Pattern produces capabilities (adopting/modifying patterns creates/shapes them), Coherence audits capabilities (measures coverage, alignment, regression). Capabilities implement multiple patterns and cannot be children of any single Pattern aggregate (ADR-0012).

**Business rule:** Every capability MUST trace to at least one pattern, either via a direct link or via `implements` edges. If a capability cannot trace to a pattern, either a pattern is missing from the registry or the capability lacks domain justification. **This is a measurable coherence signal.**

**`acts_on`** — What infrastructure a capability operates on. Everything acted on is infrastructure. Making a concept entity explicit (knowledge representation, policy, decision record) converts it to infrastructure — there is no special case for "business" vs "technical" infrastructure. `acts_on` references implementation patterns or named infrastructure targets. Examples: `semantic-ingestion` acts_on `[supabase, qdrant]`; `coherence-scoring` acts_on `[knowledge-graph]`.

**Domain classifications:**

- **Core** — Differentiating capabilities that are unique to SemOps. Examples: `internal-knowledge-access`, `coherence-scoring`, `semantic-ingestion`, `research`, `pattern-governance`, `pattern-registry`, `agent-observability`, `operational-metrics`, `reasoning-lineage`, `orchestration`, `context-engineering`, `autonomous-execution`.
- **Supporting** — Important but not differentiating. Examples: `content-management`, `asset-management`, `product-information`, `surface-deployment`, `agentic-composition`, `style-learning`, `synthesis-simulation`, `concept-documentation`.
- **Generic** — Commodity functionality. Examples: `attention-management`, `financial-pipeline`.

### Domain Alignment

Pattern→capability relationships work in two directions, and alignment is the degree to which they agree:

**Bottom-up (existing):** A capability declares "I implement these patterns." This is the current model — capabilities trace to patterns. The business rule above ensures every capability has at least one pattern link.

**Top-down (implied):** A pattern implies "these capabilities should exist." Adopting "CRM" carries expectations: contact management, ticketing, SLA tracking. Adopting "DAM" implies content ingestion, approval workflows, multi-channel distribution. Someone with basic domain understanding can enumerate the expected capabilities for a given pattern — this is the same "recognizable fit" threshold from the Pattern definition.

**Alignment signals** apply to the normalized result after discovery (see Pattern Lifecycle), not to the raw input. Coherence audit, not gates:

| Signal | What it means | Action |
| ------ | ------------- | ------ |
| **Expected capability missing** | A pattern implies a capability that doesn't exist in the registry | Governance: register the capability, or document why it's intentionally excluded |
| **Unexpected capability present** | A capability links to a pattern that doesn't obviously imply it | Check: is this a valid cross-domain composition? Novel 1P innovation? Or a mislink? |
| **Domain classification mismatch** | A core capability implements only generic patterns, or a generic capability implements only core patterns | Review: the classification or the pattern link may be wrong |
| **Cross-domain composition** | A capability draws patterns from different domains | Valid if the composition is recognizable (e.g., `ingestion-pipeline` combines data engineering + 1P semantic innovation). Novel if borrowing from another domain as innovation — likely 1P. Suspect if the connection isn't recognizable |
| **Proprietary pattern not decomposed** | A pattern represents a proprietary approach (e.g., "Salesforce CRM") without neutral primitive decomposition | Decompose to neutral capabilities first; register the proprietary layer as distinct 1P if it adds value |

**Alignment is evaluated across the architecture, not per-link.** An individual pattern→capability link might look reasonable in isolation, but coherence is measured across the entire pattern set relative to the business architecture. This is how scale and domain misalignment gets caught — and it requires judgement (human or agent) alongside mechanical measurement. Three mechanisms serve this architecture-level evaluation:

- **Reference architecture** works **domain-down** — given a business domain and its patterns, enumerate the implied capabilities. That's the "should be" baseline (Business Architecture → DDD Architecture → Capabilities filled with Patterns = ideal state).
- **Scale projection** works **infrastructure-up** — from the actual implementation, project to scale and check: does the domain model hold up? A local delivery business adopting global shipping patterns would be building ports and customs clearance at scale — the projection reveals the mismatch.
- **Coherence measurement** is where they meet — it quantifies the gap between the domain-down ideal state and the infrastructure-up projected reality. Coverage, alignment, and regression signals measure the distance.

### System Primitive Decomposition

**The method for decomposing a vendor product or solution into vendor-neutral architectural primitives.** Given a branded product (e.g., Bynder, Zendesk, Salesforce CRM), system primitive decomposition identifies the underlying domain capabilities — the neutral building blocks that any product in that space would need to provide. What remains after decomposition are the consensus patterns; what the proprietary source added on top is either discarded or registered as a distinct 1P innovation.

This is the implementation method for the "Proprietary pattern not decomposed" alignment signal. It operates at product/solution level — decomposing one vendor product into neutral primitives. Enterprise-level decomposition is the composition of multiple product-level decompositions.

**Secondary use case:** Validation that pattern-to-capability mappings are honest. By decomposing what a vendor product actually does into neutral primitives, you can verify whether the capabilities attributed to the patterns it implements are accurate or inflated.

**Canonical name:** "system primitive decomposition" (aliases: primitive deconstruction, brand to neutral breakdown, neutral primitive decomposition).

### Data System Classification

**A classification framework categorizing data systems into four types — Analytics, Application, Enterprise Work, and Enterprise Record — each with distinct query interfaces, governance approaches, and DDD patterns.** A company's mix of these types reveals its industry, business model, and data architecture needs.

The framework is a 1P synthesis of DDD (bounded contexts), TOGAF (enterprise architecture decomposition), and Fundamentals of Data Engineering (Reis & Housley's 7×5×6 anatomy). The core insight: if DDD concepts map differently to each type, they are genuine bounded contexts. They do.

| System Type | Query Interface | DDD Pattern | Governance Approach |
| --- | --- | --- | --- |
| **Analytics Data System** | OLAP | Bounded Context with ACL, UL — no Repository | Model semantics before ETL |
| **Application Data System** | OLTP | Full DDD — Repository, Aggregate, Domain Event | Bounded contexts + context maps |
| **Enterprise Work System** | Unstructured | Pre-model — DDD has no patterns for unstructured data | Semantic scaffolding (SemOps, ontologies) |
| **Enterprise Record System** | Constrained | Invariant enforcement via regulatory constraints | Canonical truth; learn from governance discipline |

**System mix** is the ratio of the four types within an organization. It is a diagnostic signal — a SaaS company (heavy Application + Analytics) has fundamentally different data architecture needs than a professional services firm (heavy Work + Record).

**SSOT failure cascade** is the anti-pattern where forcing all four types into a single "source of truth" model produces an anemic model that serves no context well, leading teams to route around it. The alternative: lineage replaces SSOT — multiple valid representations are fine when lineage lets you trace any value back to its boundary source.

**Canonical name:** "data system classification" (aliases: four data system types, data system types). See [pattern doc](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/data-system-classification.md).

### Repository

**Where implementation lives.** A repository is a codebase that delivers one or more capabilities and has a defined role in the system architecture.

**DDD classification: Value Object.** Repository identity doesn't matter; role and delivery mapping do. Repositories can be reorganized — merged, split, renamed — without changing the domain model.

Repositories are not necessarily aligned to subdomains — some are subdomain-aligned (e.g., `publisher-pr` delivers publishing capabilities), others are containers for cross-cutting concerns (e.g., `semops-orchestrator` delivers orchestration, context-engineering, and autonomous-execution).

### Integration Relationships

Rich, first-class edges between repositories, typed by DDD integration patterns:

| Integration Pattern | Power Dynamic | Example |
|---------------------|---------------|---------|
| **Shared Kernel** | Equal, shared artifact | semops-data ↔ publisher-pr share UBIQUITOUS_LANGUAGE.md |
| **Conformist** | Upstream defines, downstream conforms | publisher-pr conforms to semops-data schema |
| **Customer-Supplier** | Upstream serves downstream needs | semops-data supplies schema to sites-pr |

Integration edges carry metadata: what is shared, why this pattern was chosen, and which direction the dependency flows. Data flows are emergent from shared capabilities, not explicitly modeled.

---

## Content Layer (DAM Publishing)

The Content layer is the original purpose of the Entity table — digital publishing artifacts that document, explain, or package patterns.

### Entity (Content)

A **concrete content artifact** in the publishing domain. Entities are the ephemeral packaging of durable patterns — they can be created, modified, and deleted while the underlying patterns persist.

**Asset type** distinguishes two fundamentally different relationships to content:
- **File** — You possess the actual content (a PDF you own, a markdown file you wrote, an image you created)
- **Link** — An external reference to content you don't possess (a YouTube URL, an arXiv paper link, an external blog post)

**Orphan entities** have no pattern connection (`primary_pattern_id` is NULL). They float at the **flexible edge**, awaiting incorporation into the stable core (promotion = assigning a pattern) or rejection (deletion/archival).

### Surface

A **publication destination or ingestion source** — a channel, repository, site, or platform where content is published to or pulled from.

**Direction** defines the data flow:
- **Publish** — Content is pushed to this surface (your blog, your YouTube channel)
- **Ingest** — Content is pulled from this surface (external feeds, APIs you monitor)
- **Bidirectional** — Both publish and ingest (GitHub repos, collaborative platforms)

### Delivery

A **record of an entity published to or ingested from a surface.** Delivery is where per-surface governance lives — the same entity can have different approval states and visibility on different surfaces.

**Per-surface governance:**
- **Approval status** (`pending`, `approved`, `rejected`) — Is this content ready for this specific surface?
- **Visibility** (`public`, `private`) — Who can see this content on this surface?

This design means the same blog post can be `approved` + `public` on WordPress but `pending` + `private` on LinkedIn. Governance is per-surface, not global.

**Delivery lifecycle:** `planned` → `queued` → `published`. Failed deliveries can be retried. Published content can be `removed`.

**Delivery role:** Each entity has at most one `original` delivery (its primary publication). Additional deliveries are `syndication` (cross-postings to other surfaces).

---

## Entity (Unified)

The Entity table uses a **type discriminator** to serve three purposes within a single table:

| Entity Type | Layer | Purpose | Key Relationships |
|-------------|-------|---------|-------------------|
| **content** | Content | DAM publishing artifact | Delivered to surfaces, documented by patterns |
| **capability** | Architecture | What the system delivers | Implements patterns, delivered by repositories |
| **repository** | Architecture | Where implementation lives | Delivers capabilities, integrates with other repositories |

All entity types share: an ID, an optional primary pattern link, flexible metadata, and a vector embedding for semantic search. The metadata schema varies by type — content entities carry filespec and attribution, capabilities carry domain classification and pattern links, repositories carry role and GitHub URL.

**Why one table, not three:** The shared infrastructure (embeddings, pattern links, edges) is identical. A type discriminator avoids duplicating tables while enabling type-specific behavior through metadata schemas and views.

---

## Edge

A **typed, directional relationship** between entities, patterns, and surfaces. Edges are the connective tissue of the knowledge graph.

### Content Lineage (PROV-O)

How content entities relate to each other:

- **derived_from** — Created by transforming a source (a transcript derived from a video)
- **cites** — Formal reference for support or attribution
- **version_of** — A new version of existing content
- **part_of** — Component of a larger whole
- **documents** — Explains or covers in detail

### Strategic Design (ADR-0009)

How architecture entities relate to patterns and each other:

- **implements** — A capability implements a pattern (capability → pattern)
- **delivered_by** — A capability is delivered by a repository (capability → repository)
- **integration** — A DDD integration relationship between repositories (repository → repository)

### Domain Extensions

General-purpose relationships:

- **depends_on** — Requires another entity for definition or function
- **related_to** — Associated without hierarchy

### Edge Strength

A **confidence or importance signal** from 0.0 to 1.0. A strength of 1.0 means "this relationship is definitive." Lower values indicate weaker association or lower confidence. Strength must always be within bounds.

---

## Provenance and Lineage

### Ingestion Run

A **bounded execution of an ingestion pipeline.** A run contains multiple episodes — one per operation performed. Runs track the pipeline lifecycle from start to completion or failure, capturing source configuration for reproducibility and aggregated metrics.

**Run types:** Manual (human-triggered), scheduled (cron), or agent-triggered (autonomous).

### Ingestion Episode

A **single agent operation that modifies the domain model**, tracked automatically for provenance. Episodes capture the full context of what happened, why, and with what quality — enabling audits like "why was this classified this way?"

**Operations include:** Ingesting a new entity from source, classifying an entity against patterns, declaring a new pattern from research synthesis, publishing a delivery, establishing an edge, generating an embedding.

**Key properties:**

- **Automatic capture** — Lineage is emitted by instrumented operations, not created manually
- **Episode as context unit** — Each episode records which patterns and entities were considered, the coherence score, and the agent's proposed relationships
- **Detected edges** — Model-proposed relationships that haven't yet been committed to the edge table, preserving the agent's assessment for human review

---

## Actors (CRM/PIM)

### Brand (Unified Actor)

A **unified actor table** representing people, organizations, and commercial brands. Rather than separate tables, Brand uses a type discriminator:

- **Person** — Individual people (the owner, contacts, connections)
- **Organization** — Companies and institutions
- **Brand** — Commercial identities (SemOps, product lines)

This enables flexible relationship modeling: Tim Mitchell (person) → owns → Semantic Operations (brand) → offers → SemOps Consulting (product).

Brands can link to a 1P pattern, answering: "what does this actor commercialize?"

### Product

**What you sell**, connected to a brand (who offers it) and optionally a pattern (what methodology it packages). Products represent consulting services, white papers, courses, and other offerings.

### Brand Relationship

**CRM-style connections** between actors and products. Flexible predicates capture who knows whom, who owns what, and who's interested in what product. Metadata provides context (where you met, the source of the relationship).

---

## Business Rules

### Domain Aggregates (ADR-0012)

The domain model contains multiple aggregates, each with its own transactional consistency boundary. Pattern is the core domain concept that all aggregates reference, but it is not the sole aggregate root for the entire system.

**Core aggregates:**

| Aggregate | Root | Children | Invariants |
|-----------|------|----------|------------|
| **Pattern** | `pattern` | `pattern_edge` | Valid SKOS hierarchy, provenance rules, unique preferred_label |
| **Coherence Assessment** | *deferred* | measurements, gaps, actions | Must reference ≥1 pattern, lifecycle state machine |

**Shared entity:** Capability — produced by Pattern, audited by Coherence, implements multiple patterns. Not owned by either core aggregate.

**Value object:** Repository — describes where capabilities are delivered. Identity doesn't matter; role and delivery mapping do.

**Supporting aggregates:**

| Aggregate | Root | Children | 3P Pattern |
|-----------|------|----------|------------|
| **Content (DAM)** | entity (content) | delivery, edges | DAM, Dublin Core |
| **Surface** | surface | — | DAM (channels) |
| **Brand (PIM/CRM)** | brand | product, brand_relationship | Schema.org, PIM |

### Stable Core vs. Flexible Edge

- **Stable core:** Patterns + entities with pattern connections
- **Flexible edge:** Orphan content entities without pattern connections
- Orphans are temporary — audit processes promote or reject them
- Promotion = assigning a pattern; rejection = deletion or archival

### Per-Surface Governance

- Approval status and visibility live on Delivery, not on Entity
- Same entity can have different states on different surfaces
- Governance decisions are per-surface because audiences and standards differ

### Capability-Pattern Coverage

- Every capability must trace to at least one pattern (ADR-0009 coherence signal)
- Coverage is tracked via the `capability_coverage` view
- Gaps indicate either missing patterns or unjustified capabilities
- This is a CRITICAL-severity fitness function

### Delivery Constraints

- At most one delivery with role `original` per entity
- Private deliveries cannot target public surfaces
- Deliveries should be approved before being published
- Published deliveries must have a `published_at` timestamp

### Relationship Integrity

- Edge endpoints must reference existing entities, patterns, or surfaces
- Pattern edges must reference existing patterns on both sides
- Edge strength must be between 0.0 and 1.0
- Integration edges require metadata (integration_pattern and direction)

---

## Corpus (Knowledge Organization)

A **named partition** of the knowledge base that determines schema integration level, retention policy, and retrieval scope. Corpus assignment is the first-order routing decision during ingestion.

| Corpus | Integration | Retention | Purpose |
|--------|------------|-----------|---------|
| **DDD Core** | Full (Pattern + Entity + Edge) | Permanent | Curated knowledge: registered patterns, domain patterns, canonical theory |
| **Deployment** | Entity + Edge | Permanent | Operational artifacts: ADRs, session notes, architecture docs |
| **Published** | Full + Delivery | Permanent + tracked | Blog posts, public docs (re-ingested for coherence measurement) |
| **Research** | Optional (Entity only) | Project-scoped | 3P external research; can promote to core |
| **Ephemeral** | None (vectors only) | Session/temporary | Experiments, WIP, throwaway |

**Key insight:** A pattern like `semantic-coherence` is the attractor for all related content across corpora. Whether it's a theory doc (core), an ADR deploying it (deployment), or a blog post explaining it (published) — all feed the same pattern's understanding. Corpus controls the promotion gate and retrieval priority, not knowledge boundaries.

### Promotion Paths

Content moves between corpora as its status changes:

- **Research → Core** — 3P content reveals a pattern we adopt (creates Pattern record)
- **Deployment → Core** — Operational artifact crystallizes into a pattern
- **Ephemeral → Research** — Experiment becomes structured investigation
- **Published → Core** — Published content becomes canonical reference

### Lifecycle Stage

Where an entity is in the knowledge lifecycle. Distinct from delivery approval (which gates publication to surfaces). Adopted from 3P patterns: CI/CD artifact promotion, Backstage software catalog lifecycle, DDD aggregate lifecycle. See  for design rationale.

| Stage | Meaning | Coherence Role | Examples |
|-------|---------|---------------|---------|
| **draft** | Pre-delivery. Ideas, unvalidated, planned work. | Forecast zone — coherence predicts fit | Open issues, feature branches, draft narratives, blog ideas |
| **active** | Validated, operational. Merged, deployed, in use. | Measured for coherence, not baseline | Merged code, deployed configs, operational docs |
| **stable** | Trusted coherence baseline. Authoritative. | **IS the baseline** — semantic anchor for classifiers and scoring | Domain patterns, canonical theory docs, published ADRs |
| **deprecated** | Signaled for retirement. Still visible. | Excluded from baseline, flagged for migration | Superseded ADRs, replaced approaches |
| **archived** | Removed from operational system. | Excluded from search and scoring | Abandoned experiments, deleted branches (retained for lineage) |

**Lifecycle stage is sticky** — iteration (re-ingestion, updates) does not reset the stage. The episode chain records what changed; coherence scoring detects if the change introduced drift.

**Governance model:**

```text
Pattern (WHY)           →  Lifecycle defines WHAT states mean
Architecture (WHAT)     →  Governance defines WHO can transition
Content (output)        →  Episodes record THAT it happened
```

The governance matrix is universal — same behavior for ALL entity types across all 5 stages. Entity type only determines the creation/iteration mechanism, not whether governance applies.

---

## Domain Model

```text
CORE AGGREGATES (ADR-0012)

    Pattern (aggregate root)
        └── pattern_edge (SKOS: broader, narrower, related)
                         (Adoption: adopts, extends, modifies)

    Coherence Assessment (aggregate root — schema deferred)
        ├── measurements, gaps, actions
        └── lifecycle: assessed → acting → resolved → superseded

    Pattern ──produces──→ Capability (Entity) ←──audits── Coherence
                              │
                              └── delivered_by → Repository (Value Object)
                                                     │
                                                     └── integration → Repository

SUPPORTING AGGREGATES

    Content/DAM (root: entity-content)
        ├── edge (PROV-O: derived_from, cites, version_of, etc.)
        └── Delivery (per-surface governance)
                └── → Surface

    Surface (root: surface)

    Brand/PIM (root: brand)
        ├── Product (what you sell)
        └── Brand Relationship (who knows whom)

OPERATIONAL

    Ingestion Run (provenance)
        └── Ingestion Episode (per-operation lineage)
```

---

## Evolution Guidelines

**Adding new terms:**
1. Propose definition in PR with usage examples
2. Ensure no conflicts with existing terms
3. Update this document and [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md) as appropriate
4. Add validation rules if applicable

**Changing definitions:**
1. Mark as MAJOR schema version change
2. Provide migration path for existing data
3. Update all dependent documentation
4. Communicate changes to affected systems

**Deprecating terms:**
1. Mark as deprecated with replacement guidance
2. Support old term for one minor version cycle
3. Remove in next major version with migration

---

**Document Status:** Active | **Schema Version:** 8.0.0
**Maintainer:** Project SemOps Schema Team
**Change Process:** All updates require schema governance review
**Companion:** [SCHEMA_REFERENCE.md](SCHEMA_REFERENCE.md) for column specs, JSONB schemas, and technical details
