# Patterns in Semantic Operations

> **Version:** 2.2.0
> **Date:** 2026-03-22
> **Related:** [UBIQUITOUS_LANGUAGE.md](https://github.com/semops-ai/semops-core/blob/main/schemas/UBIQUITOUS_LANGUAGE.md) | [STRATEGIC_DDD.md](https://github.com/semops-ai/semops-core/blob/main/docs/STRATEGIC_DDD.md) | [ADR-0012](https://github.com/semops-ai/semops-core/blob/main/docs/decisions/ADR-0012-pattern-coherence-co-equal-aggregates.md)

[Capability Registry](../config/registry.yaml)

This document explains what patterns are, how they work, and why they matter to Semantic Operations. It is the narrative counterpart to the formal definitions in UBIQUITOUS_LANGUAGE.md — the same ideas, told as a story rather than a specification.

---

## Why Patterns

Patterns are the mechanism through which architecture grows. Adding a new capability means fitting it into a pattern — which means adapting to the architecture, not inventing from scratch. They encode strategy and goals as structured, auditable artifacts. And because they carry their own rules, provenance, and context, they are self-governing — agents can reason about them without human translation.

## Part I: What Patterns Are

### Definition



A pattern is an approach that:

1. **Fits a domain** — (types) the subject matter of the work. "Domain" means the about-ness of the work. Patterns aren't limited to architecture. They can be technical implementation, conceptual, or methodological. There are types to fit different layers of the architecture.

2. **Implies Application** — what pattern should be able to do. These aren't rigid requirements — they're the expectations someone familiar with the domain would carry. The expression of the application and the strength of the implication varies by pattern type.

3. **Is a recognizable abstraction** — someone with basic understanding of the domain would see why this approach fits this problem space. A pattern requires domain knowledge to unpack — its capabilities are implied, not stated. If there's nothing to unpack, it's a capability, not a pattern.

4. **Has Provenance and Lifecycle is First Class** — "where did it come from?" is a first-class dimension in SemOps. Was this adopted from outside (3P = "3rd party") or did we invent this (1P="first party"). The boundary between adoption and innovation is made explicit.

5. **Has lifecycle** — Lifecycle tracks the provenance evolution (from 3P adoption to 3P + 1P synthesis). A pattern and its capabilities have lifecycle status (`planned` → `draft` → `in_progress` → `active` → `retired`). All pattern entities are tracked with detailed lineage — how they move through the stack, how their type and composition evolve over time.

## Types

### Domain Patterns

Domain patterns are the primary building blocks. They fit a domain and imply capabilities — the things a system should be able to do if the pattern is adopted.

Patterns must be "bigger" than the capabilities they contain — higher in the ontology. The `implements` edge between capability and pattern is the act of *unpacking* an abstraction into concrete deliverables.

- **Pattern too small?** — Does it imply at least one capability? If a domain expert can't say "this pattern means you'd expect to be able to do X," it's probably a capability that got promoted.
- **Capability too big?** — Does it fit inside at least one of its implementing patterns? If you can't point to a single pattern and say "this capability is something that pattern implies," the capability needs splitting.

**Decomposition coherence test:** A valid pattern produces a coherent set of capabilities when decomposed — capabilities that a domain expert would recognize as belonging together. If decomposition produces a scattered grab bag, it's infrastructure or tooling, not a pattern. CMS, CRM, and DAM all pass this test (see `semops-research/docs/research/decompositions/` for worked examples of primitive analyses). Platforms and utility belts (e.g., n8n) fail because they don't solve a domain problem — they're infrastructure you use to solve many problems.

**The one-hop rule:** Capabilities wire to their immediate implementing patterns only — one degree of separation. Ancestor patterns connect via `derives_from` chains, not direct capability edges. If a pattern and its ancestor both appear in the same capability's `implements` list, the ancestor is redundant. For example, `bounded-context-extraction` implements `explicit-architecture` (one hop). `explicit-architecture` derives from `ddd`. The capability never wires directly to `ddd` — that relationship is inherited through the derivation chain.

### Implementation Patterns

Implementation patterns constrain capabilities — they don't imply new ones, they shape how existing ones are built. An implementation pattern like `medallion-architecture` or `event-sourcing` tells you *how* to build, not *what* to build.

### Concept Patterns

Concept patterns don't imply capabilities at all. They explain. The "explains" relationship is an edge type (`described_by`, `pattern_edge`), not a capability. A concept pattern's value is understanding, not operational capacity. A concept can coexist with a pattern of the same name — the concept is the theory (SKOS edges, documentation), while the pattern is the operationalized architecture (implements edges to capabilities). Concepts that aren't operationalized should not appear in `implements_patterns` lists.

### Analytics Patterns

Analytics patterns do imply named capabilities: `semantic-coherence-score` → `coherence-scoring`, `churn-rate` → `churn-measurement`. These are real operational things that get implemented in code.

Analytics patterns close the loop. They tie outputs back to inputs — answering two distinct strategic questions:

1. **Are you implementing your strategy?** — Conformance metrics. Are the capabilities the domain patterns imply actually built and operational? This is coherence measurement.
2. **Is your strategy working?** — Outcome metrics. Are the KPIs, KRs, and business goals moving? This connects pattern adoption to business results.

### Process Patterns

Process patterns orchestrate domain and analytics capabilities toward goals. They are choreography or saga patterns — they don't own data, they coordinate. A process pattern defines what needs to happen to achieve an outcome, referencing the analytics capabilities needed to measure progress and the domain capabilities needed to execute.

Process patterns make scoring self-diagnosing. Before you can score a goal, you check whether the goal pattern's measurement capabilities (analytics) are present. If they're not, you don't get a bad score — you get **no score**, and the gap tells you exactly what to build. The strict dependency direction (process → analytics → domain) ensures that scoring flows down the chain: a process pattern can only be as coherent as the analytics it depends on, which can only be as coherent as the domain patterns they derive from.

The goal-type continuum (rule execution → criteria-based → directional → metric-driven) is a property of the process pattern's capability maturity, not a separate classification. A goal with only structural checks operates as rule execution; add analytics capabilities and it becomes criteria-based; add temporal tracking and it becomes metric-driven. See [ADR-0014](../decisions/ADR-0014-coherence-measurement-model.md) § Pattern-Scaffolded Scoring.

#### How the types interact

Concept patterns don't promote into domain patterns — they persist as explanations. They become content: articles, guides, documentation, marketing copy. Domain patterns are born *from* concepts — they reference the concept via SKOS (`broader`, `derived_from`) and add the capability layer. The `described_by` edge from pattern → entity already models concept entities describing patterns; concept patterns describing domain patterns uses `pattern_edge`.

```text
semantic-coherence [concept]
  → explains → semantic-coherence [domain] (adds capability layer)
  → spawns → semantic-coherence-measurement [analytics] (SC formula, scoring)
    → implies → coherence-scoring [capability] (implemented in telemetry and algorithm)
```

#### Scoring dependency chain

The `requires` edge models the strict dependency direction for pattern-scaffolded scoring. SC scoring flows down this chain — each layer can only be as coherent as the layer it depends on:

```text
Goal pattern [process]
  └─ requires → Analytics pattern [analytics]  (measurement capabilities)
       └─ requires → Domain pattern [domain]   (operational data)
```

This makes scoring self-diagnosing. A missing `requires` edge to an analytics pattern means the goal can't be measured yet — the gap is the roadmap, not a failure.

The strength of the capability implication varies by pattern type. The types form a hierarchy — each layer builds on the one above it, and the coherence between layers is what keeps strategy connected to execution:

| Type | Role | Implies | Examples |
| --- | --- | --- | --- |
| **Concept** | Mental models, principles, explanations | Understanding — "here's how to think about X" | `semantic-funnel`, `semantic-drift`, `stable-core-flexible-edge` |
| **Domain** | Domain solutions that imply capabilities | Capabilities — "adopting this means you can do X" | `crm`, `dam`, `cms`, `semantic-coherence` |
| **Analytics** | Measurement standards, metrics, objective functions | Observability — "here's how to measure X" | `MAU`, `NPS`, `churn-rate`, `semantic-coherence-score` |
| **Process** | Goal coordination, choreography, audit | Outcomes — "here's how to achieve X" | *(goal patterns that orchestrate domain + analytics)* |
| **Implementation** | Constrains how capabilities are built | Technical approach — "build it this way" | `medallion-architecture`, `agentic-rag`, `event-sourcing` |

---

## Patterns are Encoded Strategy

Patterns represent the choices made to achieve business goals. `Explicit Architecture` provides the strategic scaffolding for guiding those decisions.

Explicit Architecture adopts the business architecture of a company, which includes a prioritization of capabilities.

- **CORE** is why the company exists. It is the **company-unique** capabilities and their competitive advantage.
- **SUPPORTING** capabilities enable core operations but don't differentiate or provide strategic advantage. They are necessary and innovation should be limited to efficiency gains without creating overhead.
- **GENERIC** capabilities are commodity. The domain is well-understood, and the solutions are fully established. The right answer is adoption, and keeping it simple.

### Provenance and Lifecycle

Patterns operationalize the business architecture by defining provenance as the adoption/innovation boundary. The lifecycle provides a traceable lineage of business decisions.

#### 3P: Adopting Reference Architectures

A **3P Pattern** (third party) is an adopted reference architecture — consensus knowledge that provides the foundation for everything built on top. It's the ideal bootstrapping mechanism: don't reinvent the wheel.

The ideal 3P source for a domain pattern is a standard architecture, a formal schema, or an open standard — something like W3C SKOS or PROV-O (governance). These patterns express their domain at a primitive level with neutral concepts, clear capabilities, and no vendor bias. The primitives *are* the standard, and they align naturally to capabilities and simple infrastructure choices.

#### 1P: Alteration and Innovation

First-party patterns are novel approaches, almost always evolved from a 3P Pattern, or synthesized from multiple 3P Patterns. Each innovation combines established foundations into something new. Here are some examples from SemOps:

- **Semantic Coherence** — combines information theory with system quality metrics. The objective function of the Semantic Optimization Loop.
- **Semantic Ingestion** — synthesizes ETL and Medallion Architecture with domain-aware enrichment where embedding model byproducts become a queryable artifact.
- **Scale Projection** — synthesizes RLHF, SECI, and data profiling. Manual processes intentionally generate ML training data, creating a path to autonomous execution.
- **[Explicit Enterprise](../EXPLICIT_ARCHITECTURE/explicit-enterprise.md)** — synthesizes Platform Engineering with agent-first system design. Humble tools become agent-addressable signal streams.

Each 1P pattern is linked back to its 3P foundations via SKOS, ensuring the innovation remains grounded. The 3P layer provides vocabulary and building blocks. The 1P layer provides differentiation and synthesis.

For CORE domains, the pattern space is deliberately rich: a single 1P innovation may synthesize across multiple 3P domains to create novel solutions that no single reference architecture provides. The 3P layer provides vocabulary and building blocks; the 1P layer provides differentiation. For GENERIC domains, the pattern space is deliberately thin — the pattern *is* the standard, and the investment goes into conforming, not innovating.

### The Strategic Flow

The coherence between pattern types is itself the governance mechanism. A concept pattern (marketing language, brand positioning) must cohere with the domain pattern it describes — and the domain pattern implies capabilities that are implemented in code. If the concept drifts from the domain model, or the implementation drifts from the capabilities the domain pattern implies, the architecture surfaces the gap.

The strategic encoding flows through the full architecture:

```text
Strategic intent (adopt/innovate)
  → Provenance composition (3P foundations + 1P synthesis)
    → Pattern registration (SKOS-linked, typed)
      → Capability mapping (what the pattern implies)
        → Repository assignment (where capabilities live)
          → Running infrastructure (deployed services)
```

At each level, the provenance is traceable. You can ask "why does this service exist?" and trace back through the repository, to the capability, to the pattern, to the strategic decision that justified it. You can also ask the inverse: "where is our 1P innovation actually deployed?" and trace forward from provenance through to infrastructure.

## Provenance: How Patterns Compose

Provenance is not a label on a pattern — it is the pattern's compositional structure. A pattern records which established approaches (3P) it draws from and what original synthesis (1P) it adds. This composition is tied directly to the business architecture model: provenance records where in the architecture a pattern comes from and what strategic role it plays.

### The Compositional Structure

Provenance can create a rich composition. A single pattern can pull from multiple 3P sources — standards, decomposed commercial products, open-source frameworks — and layer 1P innovation on top. This SemOps version of Agentic Lineage shows the range:

```yaml
pattern:
  id: agentic-lineage
  preferred_label: Agentic Lineage
  definition: >
    Extends data lineage with agent decision context and
    trust provenance. Every derivation records not just what
    changed but why — the agent's episode, reasoning, and
    the pattern that justified the action.
  alt_labels: [agent lineage, episodic lineage]
  provenance: 1p
  embedding: [0.044, -0.028, ...]  # 1536-dim vector

pattern_edges:
  - src: agentic-lineage → dst: prov-o              # predicate: extends
  - src: agentic-lineage → dst: data-hub-primitives  # predicate: extends (Data Hub = LinkedIn)
  - src: agentic-lineage → dst: graphiti             # predicate: extends (commercial, unique)
  - src: agentic-lineage → dst: semantic-coherence   # predicate: related

capabilities_implied:
  - agentic-lineage       # episode-level provenance with PROV-O chains
  - coherence-scoring     # lineage intent feeds coherence audit
```

A pattern is almost always an abstraction, but it is also always a concrete [data shape](../EXPLICIT_ARCHITECTURE/data-shapes.md) that operates with code and schema. Here is the SKOS pattern (a W3C standard) that SemOps adopted to organize the pattern taxonomy:

```yaml
pattern:
  id: skos
  preferred_label: Simple Knowledge Organization System
  definition: >
    W3C standard for representing concept schemes —
    taxonomies, thesauri, and classification systems
    using broader/narrower/related relationships.
  alt_labels: [SKOS, concept scheme]
  provenance: 3p
  embedding: [0.031, -0.017, ...]  # 1536-dim vector

pattern_edges:
  - src: skos → dst: ddd              # predicate: related
  - src: skos → dst: prov-o           # predicate: related

capabilities_implied:
  - ingestion-pipeline    # SKOS provides the taxonomy layer for entity ingestion
  - pattern-registry      # broader/narrower edges between patterns
```

[See more examples](#examples)

---

### Pattern Operations

The fundamental unit of SemOps operations is the Pattern. By modeling an architecture, we have already been using patterns to create the scaffolding, and now we will use them for generating specific solutions for the client problem-space. The process is agentic — it knows which context to use:

### Bootstrapping: The Starting Move

Starting with a standard 3P pattern is a powerful bootstrapping move. Skip past first principles or requirements when the answer is already available. You don't need to know what he pattern is at first, as an agentic process will discover the pattern. Start by inputing:

- **A single capability** ("customer support needs to see order history when talking to a customer") — the system identifies which patterns imply that capability, evaluates the existing architecture, and surfaces not just the capability but the full pattern set the organization needs.
- **A set of capabilities** ("ticketing, contact management, SLA tracking") — the set implies a pattern; the system confirms and identifies gaps.
- **A broad pattern** ("apply CRM") — fine as-is. You may only implement a few capabilities now. Partial implementation of a pattern is expected. Coherence tracks the unimplemented capabilities as signals, not failures.
- **A vendor or commercial product** ("i need a Zendesk") A branded solution can be decomposed into primitives and core capabilities as well.

#### Intake: Evaluating Unclassified Input

Before pattern operations can begin, new inputs need to be classified against what the system already knows. The intake process works like a triage desk:

1. **Goal scoping** — The input (an issue, a request, a question) must have a concrete outcome statement. Vague inputs have nothing to validate against, so the system guides the user toward specificity before proceeding.

2. **Territory mapping** — The system searches its authority sources — pattern definitions, capability registries, project specs, and the knowledge base — to build a map of what already exists. Semantic search is the primary mechanism, catching connections that keyword or structural lookup would miss. The result is a territory map: here are the patterns, capabilities, projects, and prior work that relate to this input.

3. **Delta identification** — With the territory map in hand, the system compares the input's intent against what already exists. What's genuinely new? What extends existing work? What conflicts with current decisions? This is where the system earns its keep — surfacing a connection to an existing pattern means the input can build on prior work rather than duplicating it.

4. **Classification and routing** — Based on the delta, the system recommends an action: attach to an existing project, create a new capability, or escalate to a higher tier of planning. Inputs that touch three or more repos, require architectural decisions, or have complex dependencies get flagged for promotion to a full project specification. The human confirms before anything moves.

The key principle: *read from authority sources, write only to the issue tracker*. The intake process never modifies core documents — it evaluates and recommends. The human confirms before anything moves.

Architecture: Patterns contain capabilities which are deployed in repos -> Infrastructure: Capabilities in repos are deployed with infrastructure. The system enforces a full traceability chain from stable meaning to executable code:

| ID | Capability | Status | Implements Patterns | Delivered By |
| ---- | --------- | -------- | ------------------- | -------------- |
| `internal-knowledge-access` | Internal Knowledge Access | active | `agentic-rag` | semops-data |
| `pattern-governance` | Pattern Governance | active | `explicit-enterprise`, `viable-systems-model` | semops-orchestrator |
| `pattern-registry` | Pattern Registry | active | `skos`, `prov-o`, `arc42`, `unified-catalog` | semops-data |
| `coherence-scoring` | Coherence Scoring | in_progress | `semantic-coherence` | data-pr, semops-data |

**Patterns** have lifecycle and Provenance which delivers *strategic knowledge*.

Patterns are the building blocks you construct architecture from. Capabilities is the mechanism by which patterns *become* architecture, and through repositories, they are the bridge to infrastructure:

```text
Pattern → Capability → Repository → Code
(why)      (what)       (where)    (run)
```

**Implementation example** — the full traceability chain from [STRATEGIC_DDD.md](https://github.com/semops-ai/semops-core/blob/main/docs/STRATEGIC_DDD.md):

**Implementation example — `/intake` on semops-research** ([session notes](https://github.com/semops-ai/semops-research/blob/main/docs/session-notes/ISSUE-24-project-review-intake.md)):

An `/intake` evaluation of semops-research inventoried 4,746 lines of production code, 98 tests, and 31 research documents. The process evaluated the repository's capabilities against registered patterns and produced a coverage table:

| Capability | Pattern Coverage |
| ---------- | ---------------- |
| `corpus-meta-analysis` | `semantic-ingestion`, `raptor`, `agentic-rag` |
| `data-due-diligence` | `togaf`, `dama-dmbok`, `dcam`, `apqc-pcf` |
| `reference-generation` | `data-modeling`, `explicit-architecture`, `business-domain`, `agentic-rag`, `bizbok` |
| `agentic-ddd-derivation` | `ddd`, `bizbok-ddd-derivation`, `ddd-agent-boundary-encoding` |

The intake found that `llm-enrichment` — previously listed as a fifth capability — was not actually a capability. It was the LLM-augmented path within `reference-generation`. The correction removed it from the capability registry and folded its pattern coverage into the parent capability. A new issue thread had started pursuing implementation that didn't need to happen and could have caused a fork and confusion.

For another example of a portion of the analysis of this process, see [Intake Evaluation](#intake-evaluation).

**Implementation example — bootstrapping a net-new domain** ([session notes](https://github.com/semops-ai/semops-core/blob/main/docs/session-notes/ISSUE-163-ecommerce-tshirts.md)):

Input: "I want to sell SemOps t-shirts on the website with full e-commerce and offsite fulfillment." No patterns, capabilities, or repos specified — pure natural language.

The `/intake` process ran a full territory map: semantic search against the pattern registry, KB entity search, and a scan of all 23 capabilities. Results:

| Search | Best Match | Similarity | Verdict |
| ------ | ---------- | ---------- | ------- |
| Pattern registry | `pim` (Product Information Management) | 0.315 | Weak — product data governance, not transactions |
| KB entities | `backstage-software-catalog` | 0.312 | Not related — software catalogs, not commerce |
| Capability scan | `financial-pipeline` | Closest | Invoicing/accounting, not retail |

**No meaningful matches.** The system correctly identified e-commerce as entirely outside the current domain model — no patterns for transactions, payments, order management, or fulfillment exist anywhere. Rather than forcing a fit, it classified the input as a net-new bounded context and recommended:

1. Adopt a 3P pattern (`headless-commerce`) rather than build custom
2. Classify as **generic** (commerce is not core to semantic operations)
3. Consider a turnkey solution (Shopify/Printful) over building it into the architecture

This is the bootstrapping chain working end-to-end: unstructured input → domain discovery → pattern search → gap identification → honest assessment that the right answer is adoption, not innovation, in a domain that's entirely supporting.

The single-capability case is where bootstrapping matters most. The person asking may not know what CRM is or what a customer support architecture looks like. They just know their support team can't see order history. But the system can reason about it: "order history access" implies a CRM pattern (customer data, interaction history) and a customer support pattern (case management, agent tooling). Evaluated against the existing architecture — which might have nearly zero CRM capabilities — the system doesn't just say "add order history access." It says "based on your domain, you need these patterns" and surfaces the full implied capability set for each.

Even if only one or two capabilities are stood up initially, the architecture is correct from the start. The patterns provide the complete shape. Coherence tracks what's implemented vs. what's implied. The gap between "what exists" and "what the patterns say should exist" isn't a failure — it's a roadmap. You're building toward the right architecture from day one, even when starting from a single need.

---

## The Pattern Lifecycle

The lifecycle is the search process that reshapes the pattern over time, converging it toward better domain fit. Provenance composition, SKOS taxonomy, neutral primitives, the Semantic Optimization Loop — all lifecycle. Each stage refines what the pattern means, what it implies, and how it relates to other patterns. The pattern entity persists across stages; its shape converges toward better fit.

### Primitive Decomposition

This decomposition serves several purposes:

1. **Honest capability mapping.** The capabilities should reflect what the domain needs, not what a particular vendor provides. A company doesn't need "Salesforce flows" — they need automation rules. The neutral framing lets the architecture be vendor-independent.
2. **Recognizable fit validation.** If you can't express the pattern in neutral terms, the fit may be to a vendor's ecosystem rather than to the domain. That's a signal worth investigating.
3. **Humans and agents understand enterprise solutions** Agentic access through CLI or API will be easier to implement with a clear understanding of capabilities.

### Lifecycle Governance

Patterns and capabilities share a 5-state lifecycle: `planned` → `draft` → `in_progress` → `active` → `retired`. Each state has conditional fitness checks — observable signals that validate the declared status against reality.

| Status | Required Evidence | Drift Signal |
| ------ | ----------------- | ------------ |
| `planned` | May have project assignment (Backlog/Ready). No merged code expected. | Has In Progress project items or closed governance issue |
| `draft` | May have open issue. Experimental work OK. | Has Done project items |
| `in_progress` | Open issue(s) OR project items In Progress. Active development. | Governance issue closed with no remaining open work |
| `active` | Delivering repos have relevant code. Governance issue (if any) closed. | Governance issue still open, or project only has Backlog/Ready items |
| `retired` | No open issues. No active project items. | Open issues or In Progress items referencing the entity |

**Evidence hierarchy** (strongest to weakest):

1. **Governance issue state** (OPEN/CLOSED) — strongest signal for individual capabilities. A closed governance issue with acceptance criteria met is the clearest evidence of `active`.
2. **Project board status** — structural signal. If a project's items are all Done, capabilities in that project should be `active`. If items are only in Backlog/Ready, capabilities are at most `planned`.
3. **Manifest/KB presence** — entities ingested into the knowledge base, scripts in delivering repos, infrastructure deployed.
4. **Session notes** — weakest signal. Work discussed but not necessarily shipped.

**Pattern lifecycle is derived, not declared.** A pattern's effective status comes from its implementing capabilities:

- **active**: at least one implementing capability is `active`
- **in_progress**: no active capabilities, at least one is `in_progress`
- **planned**: only `planned` or `draft` capabilities reference it
- **orphan**: no capability implements it (audit signal)

**The `/pattern-audit` command** (Phase 1.5) automates these checks by querying GitHub project boards and governance issues against declared status in `registry.yaml`. Drift signals are recorded in `audit-findings.yaml` as `lifecycle_drift` findings.

### Deployed Capabilities and Overlapping Patterns

Only deployed capabilities make a pattern active and operational. A large pattern like CRM might be registered, but if only `sales-flow` is deployed, the other capabilities it implies (contact management, ticketing, SLA tracking) remain latent.

The coherence signal emerges when another pattern delivers a capability that overlaps with an unused capability in the larger pattern. For example: Email delivers a `contacts` capability via Fastmail, and CRM has an unused `contact-management` capability. The capabilities overlap, but they're assigned to different patterns.

This overlap requires an intentional decision:

- **Reassign:** "This contacts infrastructure from Fastmail IS architecturally part of our CRM — that's its function, regardless of what infrastructure delivers it." Patterns are about architecture, not infrastructure.
- **Acknowledge:** "We know about the overlap and we're choosing to keep contacts with Email because we're not doing CRM at that level."

The trigger for this signal is the *second* capability. One CRM capability arriving through Email is natural — no flag needed. But as soon as two capabilities that a domain expert would recognize as CRM are active — even if they arrived through different patterns — the system is converging on CRM through unrelated decisions. The audit catches the emergent pattern before it becomes accidental architecture: "You're doing CRM whether you know it or not. Register the pattern and be intentional."

Undiscovered overlaps are drift. Discovered and decided overlaps are governance. The pattern audit must flag these for review.

---

## AI Legibility

Patterns are a natural fit for LLMs and AI agents.

**Patterns correspond to consensus knowledge.** Established patterns have extensive representation in training data — they are precisely the kind of knowledge LLMs reason about best. When an agent sees "adopt CRM," it can immediately reason about implied capabilities, established practices, and architectural fit because CRM is a consensus concept with high training signal. The same property that makes patterns recognizable to humans makes them legible to AI.

**Patterns optimize for the context window.** An agent's central constraint is context — the tokens the model can attend to at inference time. Anthropic calls managing this "context engineering" and identifies it as the hard problem of agent design. Patterns are compressed domain knowledge: "adopt CRM" loads an entire body of practices, capabilities, and architectural implications into a few tokens. They are semantic compression optimized for exactly the resource agents have least of.

**Patterns are how agents plan.** Planning is not a primitive capability of agents — it is a behavioral pattern that emerges from combining model reasoning, prompts, and memory. Chain of Thought, ReAct, Tree of Thoughts are all prompting techniques that compose from the same primitives. SemOps patterns work the same way: they decompose into capabilities, which decompose into scripts. When an agent reasons about "adopt CRM," it is performing Chain-of-Thought planning against the pattern's implied capability set — decomposing the pattern into actionable steps. The pattern gives the agent something structured to reason *about*.

**Structured rulesets.** Patterns come with implicit rules — what belongs, what doesn't, how things relate. These rules are exactly the kind of structured reasoning AI agents handle well. The agent doesn't have to invent the rules; it applies established ones.

**The traceability chain is structurally isomorphic to agent architecture.** The SemOps Pattern → Capability → Script chain mirrors how agent behavioral patterns decompose into primitives (Model, Context, Prompt, Memory, Tools) which decompose into implementations (LLM calls, vector stores, tool APIs). This is not a metaphor — it is the same structural principle. Higher-order behaviors decompose into composable units, which decompose into executable code. The same coherence signals apply: a capability with no pattern link is unjustified code; a pattern with no capability link is an undelivered aspiration. An agent provisioned with Memory that no behavioral pattern requires is over-engineered. The audit works identically in both systems.

For the full analysis, see [Agent Primitives](https://github.com/semops-ai/semops-research/blob/main/docs/research/agent-primitives.md) — a cross-source analysis of 13 authoritative sources validating the structural isomorphism between agent architecture and the SemOps pattern model.

## Agents and Patterns

Patterns are AI-legible. The architecture is graph-traversable. Together, these properties enable agents to actively discover, evaluate, and compose patterns — not just read about them.

### Context and Memory

The authority for patterns/capabilities is a machine-ready [yaml](../config/registry.yaml) registry. The capability data is also queryable via the capability_coverage view in the database and the list_capabilities / get_capability_impact MCP tools.

You can query it through:

MCP tools: list_patterns, get_pattern, search_patterns (what I just used to pull the KB data)
Database views: pattern_coverage, capability_coverage
CLI: python scripts/search.py for semantic search against pattern embeddings
The registry.yaml defines capabilities and their implements_patterns mappings, but the patterns themselves (definitions, edges, provenance) are registered directly in the database tables.

**Capability coverage** (from `capability_coverage` view — live query, not documentation):

| Capability | Domain | Patterns | Repos |
| ---------- | ------ | -------- | ----- |
| `reference-generation` | core | 6 | 1 |
| `data-due-diligence` | core | 4 | 1 |
| `ingestion-pipeline` | core | 4 | 1 |
| `pattern-registry` | core | 4 | 1 |
| `agentic-lineage` | core | 4 | 2 |
| `scale-projection` | core | 3 | 3 |
| `pattern-governance` | core | 2 | 1 |
| `content-management` | generic | 2 | 1 |
| `coherence-scoring` | core | 1 | 2 |

(Sample of `capability_coverage` query)

### The Knowledge Corpus as Context Map

The knowledge corpus — entities, patterns, capabilities, edges — supports multiple query modes (SQL for basic entities, graph traversals for relationships, vector search for chunks). Agents can combine these in series or parallel to build a ready **context map** of the business.

An agent navigating this corpus can traverse from a pattern to its implied capabilities, from a capability to the repositories that deliver it, from a repository to other capabilities it hosts, and from those capabilities back to other patterns. The graph connects strategic intent (patterns) through operational structure (capabilities, repos) to concrete artifacts (content, services). Stable core nodes anchor the traversal; flexible edge nodes provide the operational detail.

### Graph-Traversal Discovery

Beyond search, agents navigate the architecture graph to discover relationships that semantic search alone wouldn't surface. From a pattern, traverse to its implied capabilities. From a capability, traverse to the repositories that deliver it. From a repository, traverse to other capabilities it hosts and the patterns those implement. The graph reveals structural connections — "this pattern and that pattern are both implemented by the same capability" — that only emerge from traversal.

This is what makes the knowledge corpus a context map rather than a document store. The stable core nodes (core domain patterns, primary capabilities) anchor the traversal. The flexible edge nodes (content entities, supporting capabilities) provide operational detail. An agent can plan a change, assess its blast radius, and identify affected components by walking the graph.

---

## Coherence

Coherence is the evaluative counterpart to patterns' prescriptive force. Where patterns say "what should we look like?", coherence asks "does reality match intent?"

### Alignment Signals

| Signal | Meaning |
| ------ | ------- |
| Expected capability missing | A pattern implies something that doesn't exist |
| Capability without pattern | Something exists without domain justification |
| Cross-domain composition | A capability draws from multiple domains — valid innovation or a mislink? |
| Proprietary pattern not decomposed | An approach represents vendor lock-in rather than domain fit |
| Scale mismatch | The pattern set doesn't match the business architecture at the right scale |

These are audit signals, not gates. They inform decisions; they don't block them. The flexible edge is free to exist — aggregate root invariants protect the stable core.

### Three Perspectives

Coherence is one of three perspectives that together provide a complete architectural picture:

**Reference architecture** works **domain-down.** Given a business domain and its patterns, enumerate the implied capabilities. That's the "should be" baseline. Business Architecture → DDD Architecture → Capabilities filled with Patterns = ideal state.

**Scale projection** works **infrastructure-up.** From the actual implementation — what's built, what's running, what infrastructure exists — project forward to scale. Does the domain model hold up? A local delivery business that adopted global shipping patterns would find itself building ports and customs clearance at scale. The projection reveals the mismatch before the investment is made.

**Coherence measurement** is where they meet. It quantifies the gap between the domain-down ideal state and the infrastructure-up projected reality. Coverage, alignment, and regression signals measure the distance.

---

## What Patterns Enable

The power isn't in the definition. It's in what the definition enables when all the pieces — provenance, AI legibility, architecture encoding, lifecycle, agentic discovery, and coherence — work together.

### Reference Architecture Generation

The same pattern model can describe any domain's ideal architecture. Take any company. Classify its business — sector, industry, scale, business model. From that classification, work **domain-down**:

1. **Business classification** → what kind of business is this?
2. **Domain patterns** → what patterns fit this domain?
3. **Implied capabilities** → each pattern implies what the system should be able to do
4. **Ideal state** → the full pattern set and its implied capabilities constitute the reference architecture

Every step depends on the four properties of the definition: domain fit selects patterns, capability implication enumerates what should exist, recognizable fit validates scale and scope, and provenance records the strategic classification of each choice.

### Scale Projection

Because patterns encode into running infrastructure through the traceability chain, the architecture can be projected forward. "If we scale this architecture, do the patterns still hold?" is answerable because the patterns are *in* the running system, not just *about* it. A local delivery business that adopted global shipping patterns would find the projection revealing: at scale, it would be building ports and customs clearance. The mismatch is visible before the investment is made.

### The Semantic Optimization Loop as Engine

The full cycle — adopt 3P → innovate 1P → link via SKOS → measure coherence → repeat — is the engine that makes systems converge toward their intended design. Each iteration tightens the fit between patterns and reality. Provenance records the strategic reasoning. Coherence quantifies the gap. The loop minimizes it.

---

## Examples

### Agentic Lineage: Decomposing a Commercial Product

`agentic-lineage` started from **LinkedIn DataHub** — a commercial/open-source metadata platform for data lineage, cataloging, and governance. DataHub is a rich product with opinions about UI, integrations, and deployment. But the primitives underneath are domain-neutral concepts that exist independently of LinkedIn's implementation:

- **Metadata graph** — entities and relationships as a queryable graph
- **Change capture** — tracking what changed, when, and by what process
- **Lineage edges** — typed relationships that record data flow and derivation
- **Schema-aware cataloging** — structured inventory of data assets

These are the 3P primitives. They don't require DataHub to implement — they're the neutral concepts that DataHub happens to implement well. The decomposition strips away the product and keeps the domain knowledge.

The 1P innovation layers on top: **Graphiti episodic memory** provides agent decision context that standard lineage doesn't capture. Traditional lineage records "entity A was derived from entity B by process C." Agentic lineage records "agent X decided to derive entity A from entity B, in the context of episode Y, with reasoning Z, and trust provenance W." The episode — the agent's decision context, its chain of thought, what it considered and rejected — becomes a first-class lineage artifact.

This matters for coherence because lineage *intent* is as important as lineage *fact*. When coherence audits the system, it needs to know not just what happened but *why* — which pattern justified the derivation, what the agent was trying to achieve, whether the reasoning still holds. Standard lineage gives you the "what." Episodic lineage gives you the "why." The 1P synthesis connects them.

As a data shape:

```yaml
pattern:
  id: agentic-lineage
  preferred_label: Agentic Lineage
  definition: >
    Extends data lineage with agent decision context and
    trust provenance. Every derivation records not just what
    changed but why — the agent's episode, reasoning, and
    the pattern that justified the action.
  alt_labels: [agent lineage, episodic lineage]
  provenance: 1p
  embedding: [0.044, -0.028, ...]  # 1536-dim vector

pattern_edges:
  - src: agentic-lineage → dst: prov-o              # predicate: extends
  - src: agentic-lineage → dst: data-hub-primitives  # predicate: extends
  - src: agentic-lineage → dst: graphiti             # predicate: extends
  - src: agentic-lineage → dst: semantic-coherence   # predicate: related

capabilities_implied:
  - agentic-lineage       # episode-level provenance with PROV-O chains
  - coherence-scoring     # lineage intent feeds coherence audit
```

The `provenance: 1p` field records the pattern's own classification, while the `extends` edges record *what it's built from* — three 3P sources at three different levels (W3C standard, decomposed commercial product, open-source framework). The SKOS edges are the compositional structure. Together, the field and the edges tell the full provenance story.

OpenLineage (3P) adopted first — standard lineage metadata model, gives the agentic-lineage capability its primitive structure
Episode Provenance (3P) adopted — W3C PROV-O extension for grouping agent actions into episodes
Agentic Lineage (1P) synthesized — combines both 3P sources + adds agent reasoning metadata (why an agent chose an action, not just what happened)
Derivative Work Lineage (1P) extends further — content-specific lineage for AI composition chains
Delivered across 2 repos at different times — schema in semops-data, scoring in data-pr

### Parametric Design Validation: Composing Across Disciplines

Consider a carbon fiber bike frame company designing a data-first validation system. The traditional workflow is sequential: designer makes an aesthetic change in SolidWorks, waits days for an engineer to manually check factory limits, clearances, and kinematics, gets feedback ("wall too thin here," "tire clearance issue at full travel"), revises, resubmits, repeats.

The 1P pattern — `parametric-design-validation` — synthesizes from multiple 3P foundations:

- **Parametric CAD** (3P) — geometry as queryable parameters, not opaque solid models
- **Planar linkage kinematics** (3P) — suspension mechanism analysis (leverage ratios, anti-squat, axle path — all deterministic from pivot coordinates)
- **FEA / structural analysis** (3P) — physics validation against factory manufacturing limits
- **ML surrogate modeling** (3P) — trained approximations that replace expensive FEA at runtime
- **"Validation doesn't require visualization"** (1P) — the insight that a design change can be checked against every constraint using only the parametric data, without rebuilding the solid model

Each 3P pattern is adopted at the neutral, primitive level. Parametric CAD isn't "SolidWorks" — it's the principle that geometry is data, not drawings. FEA isn't "ANSYS" — it's the principle that physics can be computed from mesh, material properties, and load cases. The 3P decomposition makes the architecture vendor-independent.

The 1P synthesis goes beyond what any foundation says alone. Parametric CAD says "geometry is queryable." FEA says "physics is computable." Linkage kinematics says "suspension behavior is deterministic." The synthesis says: if all three are data operations, then validation is a continuous data pipeline, not a human review gate. The designer makes a change, the parametric data updates, and factory rules, clearances, kinematics, and geometry targets are checked in milliseconds — without touching the solid model. The engineering resource shifts from routine validation to judgment and optimization.

SKOS broader/narrower relationships record both compositions. The links aren't just metadata — they're the chain of thought: which established approaches the design built on, and what insight the synthesis added.

### Intake Evaluation

> Evaluated via `/intake` on 2026-03-01. Source issue: .

**Coherence Mode:** Discovery
**Reasoning:** Extends existing agentic architecture with model selection optimization. No new capabilities created — right-sizes existing LLM-consuming workflows. The model is the genuinely new primitive (per [ai-innovation-landscape.md](https://github.com/semops-ai/semops-research/blob/main/docs/research/ai-innovation-landscape.md)); right-sizing it per task is the key cost and architecture optimization.

#### Territory Map (at time of intake)

| Component | Status |
| --------- | -------- |
| LLM touchpoints inventory | **Complete.** 13 touchpoints profiled across 3 repos (semops-research, semops-data, publisher-pr) |
| Task complexity taxonomy | **Complete.** L1-L6 classification with model tier mapping |
| Model landscape (cloud) | **Complete.** 9 providers, 30+ models cataloged with pricing |
| Model landscape (local) | **Complete.** Ollama ecosystem, VRAM requirements, quantization guide, inference servers |
| Specialized model discovery | **Complete.** HuggingFace Hub API, ML vs LLM decision framework, domain-specific models |
| Right-sizing methodology | **Complete.** 5-step matching process, Agent Task Profile schema, cost projection framework |
| Per-task model configuration | **Not started.** Codebase defaults to global model selection |
| Embedding model centralization | **In progress.**  addresses embedding config; LLM config not started |
| publisher-pr model currency | **Outdated.** Still on `claude-3-5-sonnet-20241022`, not updated to Sonnet 4 |

#### Key Findings

- **5 of 13 touchpoints use Sonnet for L1-L3 tasks** that Haiku could handle — ~80% per-call savings
- publisher-pr uses Claude 3.5 Sonnet (legacy) — needs update regardless of right-sizing
- Local models viable for L1-L2 tasks (privacy, latency, cost=0) — Qwen 3 4B/8B best candidates
- Specialized ML models (spaCy NER, SetFit classifiers, domain BERTs) can replace LLMs entirely for deterministic L1-L2 tasks
- Self-hosted breaks even vs cloud cheap in 1-4 months at 30M tokens/day
- Workflow decomposition enables hybrid local+cloud: route L1-L2 subtasks locally, L4+ to cloud

#### Delta

- **Extends existing:** Model selection becomes per-task instead of per-repo. Embedding centralization  extends to LLM selection.
- **Fills gap:** No current methodology for evaluating model fit. No cost tracking per workflow. No local/privacy decision framework.
- **Conflicts with:** Nothing. Additive — each swap is independent and reversible.

### Netflix UDA: Bootstrapping and Monotonic Extension

Netflix unveiled Unified Data Architecture (UDA) in June 2025 — a foundational system that addresses the core problem of modeling business entities consistently across dozens of internal systems.

**Upper** is a bootstrapping upper ontology at the core of UDA:

| Property | Description |
| -------- | ----------- |
| **Self-describing** | Defines what a domain model is |
| **Self-referencing** | Models itself as a domain model |
| **Self-validating** | Conforms to its own validation rules |
| **Federated** | Closed for modification, open for extension |
| **Standards-based** | Built on W3C RDF (graph representation) and SHACL (validation) |

Upper supports **monotonic extension** — new attributes and relationships can be added without modifying existing definitions. This is critical for a growing organization: teams can extend the model without breaking existing consumers.

#### Bootstrapping: The Move That Creates the Foundation

The critical property is bootstrapping. Upper doesn't just define domain models — it defines *what a domain model is*, then models itself as an instance of that definition. The foundation is its own first citizen. Before Netflix can model any business entity, Upper must exist to define the rules. And Upper exists by conforming to its own rules.

SemOps' `semantic-object-pattern` makes the same bootstrapping move, one level up. A pattern is a data shape converging toward ideal domain fit. The semantic-object-pattern is the data shape that defines what valid data shapes are — and it's itself one. The aggregate root is the innovation.

| Upper (Netflix) | Patterns (SemOps) |
| --------------- | ----------------- |
| Defines what a domain model is | Pattern defines what an approach is |
| Models itself as a domain model | semantic-object-pattern is itself a pattern |
| Conforms to its own validation rules | Pattern model implies capabilities → coherence audits whether those capabilities exist |
| Bootstraps from RDF, SHACL | Bootstraps from SKOS, PROV-O, DDD |

Both bootstrap from established standards (3P) to create something self-referential (1P). Neither invents its foundations from scratch. Upper doesn't invent graph representation — it adopts RDF. SemOps doesn't invent taxonomy or provenance — it adopts SKOS and PROV-O. The bootstrapping move is: adopt established foundations, then use them to define yourself. The Semantic Optimization Loop is a bootstrapping loop.

**Bootstrapping is what separates a pattern model from a pattern catalog.** A catalog lists patterns. A bootstrapping system defines what patterns are, manages itself as a pattern, validates itself by its own rules, and extends without breaking. This is a lifecycle property — it describes how the semantic-object-pattern works, not what any pattern is.

#### Monotonic Extension: A Use-Case Choice

Both Upper and SemOps support monotonic extension — growth without breaking the foundation. But the enforcement mechanism differs, and the difference is a deliberate choice about where to draw the boundary between [stable core and flexible edge](../EXPLICIT_ARCHITECTURE/stable-core-flexible-edge.md).

| | Netflix Upper | SemOps Patterns |
| --- | ------------- | --------------- |
| Extension mechanism | Structural (SHACL constraints) | Relational (SKOS broader/narrower) |
| Invariant enforcement | Schema-level — structurally prevents breaking changes | Coherence audit — signals drift, doesn't block |
| Why | Dozens of teams; breaking changes cause outages | Small org; flexibility outweighs rigidity |
| Maps to | Stable core with hard boundary | Flexible edge with audit, not gate |

Netflix *needs* a gate because at their scale, an uncoordinated schema change breaks downstream consumers. SemOps deliberately chooses not to gate because the cost of rigidity outweighs the cost of drift at current scale. Both achieve "extend without breaking" — one structurally, one relationally.

Monotonic extension is not a single pattern. It's a property that can be enforced at different levels depending on the use case. The choice of enforcement mechanism is a stable-core vs. flexible-edge decision.

#### Self-Validation: The Pattern Model as Its Own First Test Case

The bootstrapping property creates a natural proof of the coherence model. `semantic-object-pattern` implies capabilities: pattern registration, SKOS taxonomy management, coherence measurement. Coherence audits whether those capabilities exist and are implemented. If they aren't, the system flags *itself*.

The pattern model is its own first test case for coherence. If the coherence mechanism works, it catches gaps in the very system that defines it. If it doesn't catch those gaps, the mechanism itself needs work. The bootstrapping property turns the system into a self-testing loop.

---

## Summary

A pattern is a data shape — an evolving semantic structure that converges toward the ideal approach for a domain need. At any point in time, it fits a domain, implies capabilities, is recognizable, and has provenance. These are the defining properties. The lifecycle is the search process that reshapes the pattern toward better fit.

The power isn't in the definition. It's in what the definition enables:

- **Strategic encoding** — provenance composition records adopt-here-innovate-there decisions as auditable architecture, not whiteboard aspirations
- **Reference architecture generation** — patterns as portable building blocks for modeling any company's domain
- **AI legibility and agentic discovery** — patterns as the semantic compression layer that makes architecture legible and navigable to agents
- **Coherence measurement** — patterns as the "should be" baseline against which reality is measured
- **Scale projection** — patterns encoded into infrastructure enable forward projection of architectural fitness
- **The Semantic Optimization Loop** — patterns as the prescriptive force, coherence as the evaluative force, optimization as the cycle that makes systems converge toward their intended design

The formal specification lives in [UBIQUITOUS_LANGUAGE.md](https://github.com/semops-ai/semops-core/blob/main/schemas/UBIQUITOUS_LANGUAGE.md). The strategic design lives in [STRATEGIC_DDD.md](https://github.com/semops-ai/semops-core/blob/main/docs/STRATEGIC_DDD.md). This document is the story that connects them.
