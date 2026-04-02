# Explicit Architecture — Cross-Layer Reference Document

> **Purpose:** Assemble all content related to "Explicit Architecture" across every layer for coherence analysis (Issue )
> **Generated:** 2026-03-15
> **Organization:** Timeline → Canonical → Source → Published → KB → Governance

---

## 0. Artifact Timeline

### Pre-rename era (as "Symbiotic Architecture")

| Date | Event | Repo | Artifact |
|------|-------|------|----------|
| 2026-01-31 | Source files created | docs-pr | `SYMBIOTIC_ARCHITECTURE/` directory (19 files) |
| 2026-02-06 | Website about pages | sites-pr | Initial SemOps pages including framework content |
| 2026-02-08 | Content pipeline setup | sites-pr | Pages configured for ingestion pipeline |
| 2026-02-09 | Content pages reconfigured | sites-pr | Reconfigure for ingestion  |
| 2026-02-13 | Information architecture | sites-pr | `what-is-semops.mdx` created, framework IA  |
| 2026-02-16 | Design system iteration | sites-pr | Crimson palette, layout refinements  |

### Rename + canonical establishment (Feb 17-19)

| Date | Event | Repo | Artifact |
|------|-------|------|----------|
| 2026-02-17 | Concept-pattern map created | semops-data | `concept-pattern-map.yaml` with EA entries  |
| 2026-02-17 | Rename: Enterprise | semops-data | Symbiotic Enterprise → Explicit Enterprise  |
| 2026-02-17 | Rename: Architecture | semops-data | Symbiotic Architecture → Explicit Architecture  |
| 2026-02-17 | Rename: source files | docs-pr | Directory renamed to `EXPLICIT_ARCHITECTURE/`  |
| 2026-02-17 | Rename: publisher content | publisher-pr | Publisher content renamed (, ) |
| 2026-02-17 | Publisher page created | publisher-pr | `explicit-architecture.md` staging page created |
| 2026-02-18 | Rename: website | sites-pr | `explicit-architecture.mdx` created  |
| 2026-02-19 | Source updates | docs-pr | Post-rename content updates |
| 2026-02-19 | Publisher + site updates | publisher-pr, sites-pr | Content sync after rename |
| 2026-02-19 | semops-docs rename | semops-docs | Public repo: Symbiotic → Explicit Architecture |

### Post-rename evolution (Feb 20 – Mar 11)

| Date | Event | Repo | Artifact |
|------|-------|------|----------|
| 2026-02-20 | Blog redesign | sites-pr | Design system alignment, blog components  |
| 2026-02-24 | semops-docs full sync | semops-docs | Framework docs, research, architecture sync |
| 2026-02-24 | Source README update | docs-pr | EA README revision |
| 2026-02-25 | Site fixes | sites-pr | Broken links and sitemap fix  |
| 2026-02-26 | Enriched definition created | semops-data | `enrich_pattern_definitions.py` with EA definition  |
| 2026-02-26 | Pattern bridged to DB | semops-data | Pattern layer bridge, coverage view  |
| 2026-03-08 | Pattern doc received | semops-data | `explicit-architecture.md` domain-pattern doc migrated from semops-orchestrator  |
| 2026-03-09 | Pattern doc updated | semops-data | Publisher-pr reference updates  |
| 2026-03-09 | Blog post published | sites-pr | "Primitive Headless Agents" — references EA, featured |
| 2026-03-10 | Registry YAML created | semops-data | `registry.yaml` — 8 capabilities mapped to EA |
| 2026-03-10 | Blog post draft | publisher-pr | "What is Architecture?" draft-v1 |
| 2026-03-11 | Blog post updates | publisher-pr | Post directory standardization  |

### Concept-to-Pattern Promotion Lineage

The concept "Symbiotic Architecture" / "Explicit Architecture" was **never registered as a pattern in its concept phase**. The process was still being built while operating, so the out-of-order progression is expected — but the lineage needs to distinguish two tracks that developed semi-independently before converging.

#### Two distinct tracks

**Concept track** — theory docs, website pages, blog posts, source material, enriched definitions. These describe *what* EA is, *why* it matters, and how to adopt it. Audience: humans learning the idea (and future agents reasoning about it).

**Implementation track** — the actual running system: entity/edge schema, coverage views, fitness functions, registry entries, capability mappings, STRATEGIC_DDD.md. This is EA *operating as architecture*. Audience: the system itself and agents querying it.

The pattern doc sits at the boundary — it describes the concept but is also the canonical reference the implementation points to.

#### Concept track timeline

| Date | Event | Artifact |
|------|-------|----------|
| 2026-01-10 | Named in Innovation Index | : `symbiotic-architecture` as "HIGH brand strength" pillar. "Applied DDD at organizational scale." |
| 2026-01-31 | 19 source files authored | docs-pr `SYMBIOTIC_ARCHITECTURE/` — theory documents describing what EA is |
| 2026-02-06 | Website about pages | sites-pr: initial SemOps framework content |
| 2026-02-13 | Framework IA | sites-pr: `what-is-semops.mdx`, framework information architecture  |
| 2026-02-16 | Pattern doc authored | semops-orchestrator: `explicit-architecture.md` — first formal pattern definition |
| 2026-02-17–18 | Rename event | Symbiotic → Explicit across 5 repos |
| 2026-02-18 | Website pillar page | sites-pr: `explicit-architecture.mdx` — 119-line founder-voice narrative |
| 2026-02-19 | semops-docs sync | Public repo mirror updated |
| 2026-02-24 | Source README revision | docs-pr: hub document restructured |
| 2026-02-26 | Enriched definition | semops-data: KB-facing definition written  |
| 2026-03-09 | Blog post published | sites-pr: "Primitive Headless Agents" references EA |
| 2026-03-10 | Draft blog post | publisher-pr: "What is Architecture? Part 1" |

#### Implementation track timeline

| Date | Event | Artifact |
|------|-------|----------|
| 2025-08-28 | Phase 1 schema + governance | semops-data: initial entity/edge schema and governance framework |
| 2025-12-06 | Tiered classifier | semops-data: concept promotion infrastructure |
| 2025-12-23 | `pattern_coverage` view created | semops-data: first coverage view in schema — **the core EA mechanism exists 8 weeks before the concept is named** |
| 2026-01-31 | Query API + MCP server | semops-data: queryable architecture layer (, ) |
| 2026-02-08 | entity_type + DDD edges | semops-data: entity/edge model with strategic DDD predicates . Coverage views enhanced. Fitness functions created. |
| **2026-02-16** | **Convergence point** | semops-data : `feat: operationalize governance model — explicit-architecture pattern, coverage views, sample queries`. Pattern named + implementation operationalized in the same commit. |
| 2026-02-16 | Pattern registered | semops-orchestrator: `pattern_v1.yaml` entry — provenance 1P, type domain, derives from `[viable-systems-model, ddd]`. No `status` field yet. |
| 2026-02-17 | Coverage view update | semops-data: `pattern_coverage` view enhanced  |
| 2026-02-21 | Status assigned retroactively | semops-orchestrator: `status: active` set when status field added to all patterns (v1.7.0) — **never went through pending/draft** |
| 2026-03-08 | Pattern doc migrated | semops-data: received from semops-orchestrator  |
| 2026-03-10 | Registry YAML created | semops-data: `registry.yaml` with 8 capabilities mapped to EA |
| 2026-03-10 | STRATEGIC_DDD.md | semops-data: EA referenced across capability tables and coverage analysis |

#### The convergence

The two tracks converged on **Feb 16, 2026** in a single commit: `feat: operationalize governance model — explicit-architecture pattern, coverage views, sample queries `. This commit both named the pattern in the implementation and, same day in semops-orchestrator, registered it in `pattern_v1.yaml` and authored the pattern doc.

But the implementation existed months before the concept was articulated:

- `pattern_coverage` view: **Dec 23** (concept docs: Jan 31 — 5 weeks later)
- Entity/edge schema: **Aug 2025** (pattern registration: Feb 16 — 6 months later)
- Fitness functions: **Feb 8** (pattern doc: Feb 16 — 8 days later)

This is a **recognition promotion** — the pattern was implemented, observed to be working, then named and documented. The concept track then continued to evolve independently, becoming richer than the canonical pattern doc.

#### What this means for lifecycle

The assumed pattern lifecycle: register → design → implement → activate.
The actual EA lifecycle: **implement → recognize → name → register → document → publish**.

This out-of-order progression happened because the process was being built alongside the work. The important structural observation is that the concept/implementation distinction exists but wasn't captured at the time — and as shown in the KB visibility analysis below, it's still not visible to agents querying the system.

### KB visibility: What an agent can and cannot see

The concept/implementation distinction is **structurally present** in the KB but **not semantically labeled**.

#### What a KB query reveals

**Implementation side** — `get_pattern("explicit-architecture")` returns:

- 9 capabilities IMPLEMENTS it, 4 repos deliver it, EXTENDS → `ddd`
- Enriched definition: "Making architectural decisions... explicit, queryable, and traceable"
- This is clean and queryable — the implementation track is well-represented

**Concept side** — `search_knowledge_base("explicit architecture", content_type=["architecture"])` returns:

- 10 entities from the `EXPLICIT_ARCHITECTURE/` source docs (README, stable-core, explicit-enterprise, ai-ready, data-shapes, etc.)
- All typed `content_type: architecture`, `entity_type: content`
- These are the concept-track documents, discoverable via semantic search

**Graph** — `graph_neighbors("explicit-architecture")` returns both layers mixed:

- 9 Capability → IMPLEMENTS edges (implementation)
- Concept edges: EXTENDS → `domain-driven-design`, DERIVED_FROM → `semantic-funnel`
- ~20 issue/session-note entities with RELATED_TO/EXTENDS/DERIVED_FROM (governance history)
- Distinguishable by label (`Capability` vs `Concept` vs `Entity`) but no layer attribute

#### What a KB query cannot reveal

1. **Pattern doc not ingested as KB entity.** The `explicit-architecture.md` domain-pattern doc in semops-data is the canonical definition but has no KB entity. Searching `content_type: ["pattern"]` returns nothing for EA — the source docs are typed `architecture`, the pattern is in the `pattern` table, but no content entity bridges them.

2. **No entities represent implementation artifacts.** The coverage views, fitness functions, entity/edge schema — the things EA *actually is* as a running system — aren't entities in the KB. They're the infrastructure the KB runs on but not *in* the KB.

3. **Enriched definition diverges from pattern doc.** The `get_pattern` definition says "documentation and traceability" (concept-level framing). The pattern doc says "governance as projection over entity/edge graph" (implementation-level). The website page says "architecture = encoded business rules, not infrastructure" (accessible framing). An agent can't tell which is authoritative.

4. **No temporal/lineage information.** The KB can't tell you that implementation preceded the concept, that the website page is richer than the pattern doc, or that the enriched definition was written mid-stream between two different framings.

5. **Concept and implementation edges are mixed in one flat graph.** IMPLEMENTS edges (implementation) and RELATED_TO/EXTENDS edges (concept) coexist without a layer attribute. You can infer the distinction from the label (`Capability` vs `Concept`) but it's not explicit.

### Vector search comparison

Three different vector search strategies surface three different views of EA — none complete on its own. All three use embedding similarity (Qdrant), differing in query phrasing and granularity (entity vs. chunk).

**Entity vector search: `"explicit architecture"` (name-only query)**

Returns concept-track content almost exclusively. Top 15 results:

| # | Entity | Type | Corpus | Sim. | Track |
|---|--------|------|--------|------|-------|
| 1 | `readme-old` (superseded!) | content | core_kb | 0.67 | Concept |
| 2 | `explicit-architecture` (README) | content | core_kb | 0.65 | Concept |
| 3 | `pub-explicit-architecture` | content | published | 0.65 | Concept |
| 4 | `what-is-architecture` | content | core_kb | 0.51 | Concept |
| 5 | `issue-98-explicit-architecture-page` | content | deployment | 0.50 | Concept (session note) |
| 6 | `issue-110-...readme-revision` | content | deployment | 0.49 | Concept (session note) |
| 7 | `mirror-architecture` (retired!) | content | core_kb | 0.49 | Noise |
| 8 | `adr-0004-design-system-architecture` | content | deployment | 0.48 | Noise (sites-pr ADR) |
| 9 | `ai-ready-architecture` | content | core_kb | 0.47 | Concept |
| 10 | `symbiotic-architecture` (ghost!) | content | core_kb | 0.47 | Stale |

Zero implementation-track results. No capabilities, no registry, no coverage views.

**Entity vector search: `"explicit architecture governance coverage views entity edge queryable"` (implementation-flavored query)**

Same entity-level vector search, but adding implementation keywords changes the ranking:

| # | Entity | Type | Sim. | Track |
|---|--------|------|------|-------|
| 1 | `explicit-architecture` (README) | content/architecture | 0.49 | Concept |
| 2 | `pub-explicit-architecture` | content/article | 0.46 | Concept |
| 5 | `issue-133-ddd-schema-query-mcp` | session_note | 0.44 | Implementation (MCP tools!) |
| 8 | `adr-0011-agent-governance-model` | adr | 0.43 | Implementation |
| 9 | `issue-112-lifecycle-stage-design` | session_note | 0.42 | Implementation |
| 11 | `domain-reference-architecture` | **capability** | 0.41 | Implementation |

Implementation artifacts start appearing, but only as session notes and ADRs that *discuss* the implementation — not as the implementation artifacts themselves (views, functions, schema).

**Chunk vector search: `"explicit architecture pattern coverage governance projection"` (passage-level)**

Same embedding similarity but at chunk granularity — this is where the lineage actually becomes visible:

| # | Chunk | Heading | Sim. | Key content |
|---|-------|---------|------|-------------|
| 1 |  | "Related" | 0.63 | *"`explicit-architecture` pattern — this gap is exactly what the pattern is designed to detect"* |
| 2 | issue-112 session note | "2026-02-15 Context" | 0.62 | *"Identified and registered the underlying pattern (`explicit-architecture`) that the governance model operationalizes"* |
| 3 | EA README | "Traceability Chain" | 0.61 | Pattern → Capability → Script → Library → Service → Port |
| 4 | patterns.md | "EA → Operationalizes Patterns" | 0.61 | *"Without Explicit Architecture, patterns are documentation"* |
| 8 | issue-112 session note | "2026-02-14 Next Steps" | 0.59 | *"Pattern registration is an architectural decision, not a lifecycle transition"* |
| 10 | issue-112 session note | "2026-02-15 Context" | 0.59 | *"`pattern_coverage` view was only measuring documentation coverage... The two edge mechanisms were disconnected"* |

**The chunk search found the exact moment of recognition** — ISSUE-112 session note from Feb 15 says "Identified and registered the underlying pattern (`explicit-architecture`) that the governance model operationalizes." It also found the implementation gap that prompted it: `pattern_coverage` was measuring documentation, not architecture.

#### What this means for coherence

The three search surfaces tell an agent different stories:

- **Entity vector (name-only):** "EA is a pillar of the SemOps Framework with source docs and a published README." (Concept only)
- **Entity vector (implementation query):** "EA relates to governance models, MCP tools, and lifecycle design." (Partial implementation)
- **Chunk vector (passage-level):** "EA was recognized as a pattern when the governance model was operationalized; it was implemented before it was named." (Actual lineage)

No single query reveals the full picture. The concept/implementation distinction requires combining `get_pattern` (structured implementation data) with `search_chunks` (narrative context about how it got there). The entity-level search is dominated by concept content and polluted by stale/superseded entities.

### Timeline observations

1. **Source material predates canonical by 5+ weeks** — the 19 source files existed from Jan 31; the pattern doc didn't arrive in semops-data until Mar 8.
2. **Rename was a 2-day coordinated event** (Feb 17-18) touching 5 repos simultaneously.
3. **Published website page predates canonical pattern doc** — the sites-pr MDX was created Feb 18; the pattern doc landed Mar 8. The downstream was built before the upstream was formalized.
4. **Registry formalization was last** (Mar 10) — capability mappings were the final canonical artifact, created after all content was published.
5. **Enriched definition was created mid-stream** (Feb 26) — between the rename and the pattern doc migration, which may explain the framing divergence.
6. **Implementation preceded registration by months** — the governance views, fitness functions, and entity/edge model in semops-data were already operational before the pattern was formally registered. The `pattern_coverage` view (Dec 23) predates the concept docs (Jan 31) by 5 weeks.

---

## 1. Canonical Layer (semops-data)

### 1.1 Pattern Document

**File:** `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/explicit-architecture.md`

| Field | Value |
|-------|-------|
| Version | 1.0.0 |
| Status | active |
| Provenance | 1P |
| Derives From | `ddd` |
| Last Updated | 2026-02-15 |

**Core Thesis:** Traditional governance adds enforcement on top of architecture. Explicit Architecture inverts this — if your architecture is a queryable data model, then governance questions are just queries. The governance mechanism and the operational data model are the same artifact.

**3P Foundations:**
- Viable Systems Model (Beer) — self-regulating systems through feedback loops
- DDD (Evans) — bounded contexts, aggregate roots as structural vocabulary

**1P Innovation:** Coverage views as homeostatic sensors over an entity/edge graph:
- `pattern_coverage` view = passive fitness function
- `capability_coverage` view = bidirectional completeness check
- `derive_lifecycle_stages` = lifecycle computed from structure, not assigned
- Andon cord = `SELECT` query

**Key Principles:**
1. Architecture as data model (entities, edges, typed predicates)
2. Governance as projection (SQL views over operational tables)
3. Lifecycle as emergent property (derived from edge coverage)
4. Passive self-validation (queryable truth, not alarms)

**Related Patterns:** `viable-systems-model`, `ddd`, `semantic-coherence`, `backstage-software-catalog`, `content-lifecycle-states`

**References (from pattern doc):**
- `semops-data/schemas/GOVERNANCE_MODEL.md`
- `semops-data/docs/decisions/ADR-0011-agent-governance-model.md`
- `../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/explicit-enterprise.md`
- `../../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/discovery-through-data.md`

### 1.2 Registry (config/registry.yaml)

**8 capabilities implement `explicit-architecture`:**

| Capability | Layer | Status | Other Patterns |
|-----------|-------|--------|----------------|
| `data-due-diligence` | Extraction | active | ddd, data-modeling, business-domain, togaf, dama-dmbok, dcam |
| `reference-generation` | Extraction | active | ddd, data-modeling |
| `system-primitive-decomposition` | Extraction | planned | (sole pattern) |
| `domain-data-model` | Data | active | ddd, skos, prov-o |
| `pattern-management` | Data | active | semantic-object-pattern, pattern-language |
| `architecture-audit` | Operations | active | (sole pattern) |
| `bounded-context-extraction` | Operations | planned | ddd |
| `orchestration` | Orchestration | active | explicit-enterprise, platform-engineering |

### 1.3 Enriched Definition (scripts/enrich_pattern_definitions.py)

```
"explicit-architecture": (
    "Making architectural decisions, patterns, and their rationale explicit, "
    "queryable, and traceable rather than implicit in code or tribal knowledge. "
    "Extends DDD with formalized architecture documentation and pattern "
    "traceability."
)
```

### 1.4 Concept-Pattern Map (config/mappings/concept-pattern-map.yaml)

**Direct mapping entry:**
```yaml
explicit-architecture:
  occurrences: 1
  max_strength: 0.8
  predicates: [related_to]
  entities: [domain-driven-design]
  action: map
  pattern_id: explicit-architecture
  match_type: exact
```

**Concepts mapped TO `explicit-architecture`:**
- `ai-ready-architecture` → mapped to `explicit-architecture` (graph_informed)
- `agentic-coding-at-scale` → mapped to `explicit-architecture` (graph_informed)

**Concepts that list `explicit-architecture` in their entity associations (but map elsewhere):**
- `domain-driven-design` (28 occurrences) — maps to `ddd` pattern. `explicit-architecture` appears in its entity list.
- `semantic-coherence` (20 occurrences) — entity list includes `explicit-architecture`
- `semantic-funnel` — entity list includes `explicit-architecture`
- `strategic-data` — entity list includes `explicit-architecture`
- `what-is-understanding` — entity list includes `explicit-architecture`
- `event-driven-architecture` — entity list includes `explicit-architecture`
- `anti-corruption-layer` — entity list includes `explicit-architecture`
- `semantic-operations` — entity list includes `explicit-architecture`

---

## 2. Source Material Layer (docs-pr + semops-docs)

### 2.0 Public Authority: semops-docs

**Repo:** `semops-ai/semops-docs` (public)
**Directory:** `SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/`
**KB entity URI:** `github://semops-ai/semops-docs/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/README.md`

The public `semops-docs` repo is the authoritative published source that the KB references. Its README defines EA as:

> **Explicit Architecture** is Domain-Driven Design applied to the full organization — encoding what entities exist, how they relate, what constraints apply, and where decisions happen into inspectable, queryable structure.

Frontmatter: `doc_type: structure`, `pattern: explicit-architecture`, `provenance: 1p`, `pattern_type: concept`, `brand_strength: medium`

Contains 17 files (same as docs-pr minus `bizbok-ddd-overlay.md` and `README-old.md`). This is the mirror that external consumers see.

### 2.1 Private Source: docs-pr

**Directory:** `docs-pr/docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/`

### 19 Source Files

| # | File | Topic Area |
|---|------|-----------|
| 1 | `README.md` | Hub document — pillar overview |
| 2 | `README-old.md` | Previous version of hub |
| 3 | `what-is-architecture.md` | Pattern hub (3P concept) |
| 4 | `ai-ready-architecture.md` | AI-readiness prerequisites |
| 5 | `agentic-coding-at-scale.md` | Agent-first system design |
| 6 | `domain-driven-design.md` | DDD foundations |
| 7 | `patterns-and-bounded-contexts.md` | Pattern/BC relationship |
| 8 | `stable-core-flexible-edge.md` | Core stability principle |
| 9 | `data-shapes.md` | Data modeling shapes |
| 10 | `discovery-through-data.md` | Data-driven discovery |
| 11 | `ddd-data-architecture.md` | DDD + data architecture |
| 12 | `ddd-solves-ai-transformation.md` | DDD for AI transformation |
| 13 | `ddd-acl-governance-aas.md` | ACL + governance-as-a-service |
| 14 | `bizbok-ddd-overlay.md` | BIZBOK + DDD integration |
| 15 | `semantic-flywheel.md` | Semantic flywheel mechanism |
| 16 | `explicit-enterprise.md` | 1P pattern: explicit enterprise |
| 17 | `scale-projection.md` | Infrastructure-up projection |
| 18 | `semops-aggregate-root.md` | SemOps aggregate root design |
| 19 | `wisdom-aggregate-root.md` | Wisdom as aggregate root |

**Mirrored in publisher-pr:** Same 19 files exist at `publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/`

### Related Session Notes (docs-pr)

- `ISSUE-110-explicit-architecture-readme-revision.md` — README restructured with intentional → not-infrastructure → explicit sequence
- `ISSUE-169-rename-four-data-system-types.md` — Data system classification rename
- `ISSUE-111-map-semops-framework-to-pillars.md` — Framework-to-pillar mapping
- `ISSUE-80-rules-symbiotic-architecture.md` — Pre-rename: "Symbiotic Architecture" rules

---

## 3. Published/Downstream Layer

### 3.0 Live Website Page (sites-pr → semops.ai)

**File:** `sites-pr/apps/semops/content/pages/explicit-architecture.mdx`
**Live URL:** `semops.ai/framework/explicit-architecture`
**Also mirrored in:** `semops-sites/apps/semops/content/pages/explicit-architecture.mdx` (stub: "Full content coming soon")

The richest published artifact (119 lines). Founder-voice narrative covering:
- **Adoption Lifecycle:** Understanding → Intentionality → Explicit (3-stage progression)
- **DDD section:** Aggregates, Ubiquitous Language, ACL governance — with links to semops-docs and semops-core
- **Deliverables:** Semantic Flywheel, Scale Projection, Stable Core/Flexible Edge, Explicit Enterprise
- **"Where to Start":** 5 practical starting points (name boundaries, version language, type relationships, trace one capability, measure with AI)

**Key framing differences from pattern doc:**

- Pattern doc: "governance as projection over entity/edge graph" (technical/internal)
- Website page: "architecture = encoded business rules, not infrastructure" (accessible/external)
- Website page includes personal narrative, adoption journey, practical guidance
- Website page links extensively to semops-docs (public GitHub) and semops-core

**Framework landing page:** `sites-pr/apps/semops/src/app/framework/page.tsx` lists EA as one of three pillars with description: "Encode your strategy into your systems so humans and AI can operate from shared structure"

### 3.1 Published Blog: Primitive Headless Agents

**File:** `sites-pr/apps/semops/content/blog/primitive-headless-agents.mdx`
**Live URL:** `semops.ai/blog/primitive-headless-agents`
**Source:** `publisher-pr/posts/primitive-headless-agents/final.md`

- **Date:** 2026-03-09
- **Status:** Published (FINAL), featured
- **EA reference:** "That's the idea behind what I've been calling [Explicit Architecture](https://semops.ai/framework/explicit-architecture) and [Explicit Enterprise](...) — all your data and process inspectable by both people and machines, with no hidden assumptions."

### 3.2 Published Framework Page: What is SemOps?

**File:** `sites-pr/apps/semops/content/pages/what-is-semops.mdx`
**Live URL:** `semops.ai/framework/what-is-semops`

- **Status:** Published
- **EA reference:** Lists Explicit Architecture as one of three core pillars alongside Strategic Data and Semantic Optimization

### 3.3 Publisher Content Page (Staging)

**File:** `publisher-pr/content/pages/framework/explicit-architecture.md`

- **Type:** Evergreen spoke page (doc_type: spoke)
- **Style:** Marketing-narrative
- **Status:** Draft
- **Created during:** ISSUE-98
- **Note:** This is the publisher-pr staging version. The sites-pr MDX is the deployed version.

### 3.4 Draft Blog: What is Architecture? Part 1

**File:** `publisher-pr/posts/what-is-architecture/draft.md`

- **Title:** "What is Architecture? Part 1: It's Not Infrastructure"
- **Date:** 2026-03-10
- **Status:** draft-v1
- **EA reference:** "...we care even more now if our goal is to deploy AI and get value. The whole idea behind what we call [**Explicit Architecture**](https://semops.ai) is that much more of the business activity is now data..."
- **Supporting files:**
  - `posts/what-is-architecture/notes.md` — editorial notes
  - `posts/what-is-architecture/assets/its.jpg` — image asset
- **Research sections:** (in `docs/drafts/post-pending/what-is-architecture.md/`)
  - `what-is-architecture-source-content.md` — aggregated source
  - `what-is-architecture2-amazon.md` — AWS perspective
  - `what-is-architecture3-ddd-bizbok.md` — DDD + BIZBOK
  - `what-is-architecture4-data-netflix.md` — Data arch + Netflix

### 3.5 Edit Corpus

**File:** `publisher-pr/edits/.pending/explicit-architecture.yaml`
- **39 edits** logged during ISSUE-98
- Edit IDs: edit-001 through edit-039
- Types: agent-driven style pass, typo fixes, link conversions, restructuring
- 2 flagged for semantic changes

### 3.6 Publisher Session Notes

- `ISSUE-98-explicit-architecture-page.md` — 3 sessions (2026-02-17/18/19), status: Complete
- `ISSUE-110-explicit-architecture-readme-revision.md` — Architecture definition research, status: Complete

---

## 4. Knowledge Base Layer

### What the KB tells us (plain language)

Explicit Architecture has a split personality in the knowledge base. The implementation side is clean — 9 capabilities implement it, 4 repos deliver it, it extends DDD, and the pattern table row is well-formed. An agent querying `get_pattern` gets a useful structural picture.

The concept side is rich but disconnected. Ten source docs from the `EXPLICIT_ARCHITECTURE/` directory are ingested as content entities, discoverable via vector search. But there's no structural link between the pattern and its concept content — no DESCRIBED_BY edges. An agent has to *search* for related concept docs rather than *traversing* from the pattern node to them.

The graph mixes both layers in a flat list: 9 capability IMPLEMENTS edges sit alongside ~20 issue/session-note RELATED_TO edges and concept-level EXTENDS/DERIVED_FROM edges. You can tell them apart by node label (Capability vs Concept vs Entity), but there's no layer attribute that says "this is implementation" vs "this is concept."

Three things an agent can't see at all: (1) the pattern doc itself isn't ingested as a KB entity, so the canonical definition is invisible to search; (2) the actual implementation artifacts (coverage views, fitness functions, schema) aren't entities — they're the infrastructure the KB runs on but not *in* the KB; (3) there's no temporal information, so the agent can't tell that implementation preceded naming by months, or that the website page outgrew the pattern doc.

A stale entity (`readme-old`, the superseded hub doc) ranks *higher* than the current README in semantic search — a concrete example of what happens without lifecycle metadata.

### 4.1 KB Entities (PostgreSQL + Qdrant)

| Entity ID | Title | Type | Corpus | Content Type | Similarity |
|-----------|-------|------|--------|-------------|-----------|
| `readme-old` | Explicit Architecture | content | core_kb | architecture | 0.6709 |
| `explicit-architecture` | Explicit Architecture | content | core_kb | architecture | 0.6545 |
| `pub-explicit-architecture` | Explicit Architecture | content | published | article | 0.6531 |
| `what-is-architecture` | What is Architecture? | content | core_kb | architecture | 0.5072 |
| `issue-98-explicit-architecture-page` | Issue : EA Page | content | deployment | session_note | 0.4976 |
| `issue-110-explicit-architecture-readme-revision` | Issue : README Revision | content | deployment | session_note | 0.4976 |

**Notable:** `readme-old` (the superseded hub document) ranks *higher* than the current `explicit-architecture` README in semantic search.

### 4.2 Enriched Definition (persisted via enrich_pattern_definitions.py)

> "Making architectural decisions, patterns, and their rationale explicit, queryable, and traceable rather than implicit in code or tribal knowledge. Extends DDD with formalized architecture documentation and pattern traceability."

**Observation:** This definition emphasizes "documentation and traceability" — the pattern doc emphasizes "governance as projection over entity/edge graph." These are related but distinct framings.

---

## 5. Governance History

### 5.1 Rename: Symbiotic Architecture → Explicit Architecture

- **Issues:** ISSUE-141, ISSUE-142 (semops-orchestrator)
- **Rationale:** "Symbiotic" implied mutualism between human/AI; "Explicit" captures the core thesis — architecture must be inspectable, not implicit
- **Note:** KB still contains a `symbiotic-architecture` entity (similarity 0.4735 to "explicit architecture" query)

### 5.2 Related Governance Issues

| Issue | Repo | Relevance |
|-------|------|-----------|
| ISSUE-141/142 | semops-orchestrator | Rename decision |
| ISSUE-146 | semops-data | Arch-doc alignment validation |
| ISSUE-151 | semops-data | Governance lifecycle |
| ISSUE-157 | semops-data | Repos role clarification |
| ISSUE-173 | semops-data | Domain patterns migration |
| ISSUE-80 | docs-pr | Pre-rename "Symbiotic Architecture" rules |
| ISSUE-98 | publisher-pr | Website page creation |
| ISSUE-110 | docs-pr/publisher-pr | README revision |

---

## 6. Coherence Analysis

### Type 1: Structural Coherence (deterministic)

For a pattern to be structurally "active", specific artifacts must exist in specific places. This is SQL + YAML + file checkable.

#### Artifact checklist

| Artifact | Expected | Actual | Status |
|----------|----------|--------|--------|
| `pattern` table row | Row with id, definition, provenance | `explicit-architecture`, 1p, definition present | PRESENT |
| `pattern_edge` (SKOS) | EXTENDS → `ddd` | 1 edge: EXTENDS → ddd (strength 1.0) | PRESENT |
| `pattern_edge` (VSM) | EXTENDS → `viable-systems-model` | **Missing** — pattern doc lists VSM as 3P foundation but no edge exists | GAP |
| Capability IMPLEMENTS edges | Multiple capabilities | 9 capabilities (data-due-diligence, domain-data-model, etc.) | PRESENT |
| `pattern_coverage` view | Row with counts | content=6, capability=9, repo=4 | PRESENT |
| Pattern doc file | `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/explicit-architecture.md` | File exists (v1.0.0, active) | PRESENT |
| Pattern doc as KB entity | Entity with content from pattern doc | **Not ingested** — pattern doc has no KB entity | GAP |
| `registry.yaml` entries | Capability mappings | 8 capabilities listed (registry has 8 vs DB's 9 — `domain-reference-architecture` only in DB) | MISMATCH |
| `STRATEGIC_DDD.md` references | Pattern listed in capability tables | 14 references across capability tables and coverage analysis | PRESENT |
| `ARCHITECTURE.md` references | Pattern named | **Zero references** — ARCHITECTURE.md implements EA without naming it | GAP |
| `UBIQUITOUS_LANGUAGE.md` | Definition entry | 1 mention: "Encode your strategy into your systems so humans and AI can operate from shared structure" | PRESENT (thin) |
| `fitness-functions.sql` | EA-specific checks | **Zero references** — no fitness function mentions EA | GAP |
| `phase2-schema.sql` | N/A (pattern is in data, not schema DDL) | Zero references (expected) | N/A |
| `DESCRIBED_BY` edges | Pattern → concept entities | **None** — 0 edges from pattern to concept content | GAP |
| Stale entities retired | `readme-old`, `symbiotic-architecture` | Both still active: `readme-old` points to README-old.md, `symbiotic-architecture` points to non-existent `SYMBIOTIC_ARCHITECTURE/README.md` | GAP |

#### Edge analysis (PostgreSQL)

Edges involving `explicit-architecture` entity:

| Predicate | Count | Direction | Examples |
|-----------|-------|-----------|----------|
| `implements` | 9 | capability → EA | domain-data-model, architecture-audit, orchestration, etc. |
| `related_to` | 8 | mixed | ai-ready-architecture → EA, EA → ddd, semantic-flywheel → EA, etc. |
| `documents` | 2 | EA → other | EA → ddd (0.85), EA → semantic-funnel (0.90) |

The `documents` edges are **concept entity edges**, not pattern aggregate links — they say "the EA README documents DDD and semantic-funnel," which is the concept entity describing its references. These are NOT the `DESCRIBED_BY` edges proposed in Section 7.

#### Structural coherence verdict

| Dimension | Score | Notes |
|-----------|-------|-------|
| Pattern registration | Complete | Row, definition, provenance, SKOS edge to DDD |
| Capability coverage | Complete | 9 capabilities with IMPLEMENTS edges |
| Documentation coverage | Partial | Pattern doc exists but not ingested as KB entity; ARCHITECTURE.md doesn't name EA |
| Schema governance | Missing | No fitness function references EA; no EA-specific coherence checks |
| Concept linkage | Missing | No DESCRIBED_BY edges; pattern and concept tracks are disconnected |
| Stale artifact cleanup | Missing | `readme-old` and `symbiotic-architecture` entities still active |
| Registry/DB alignment | Minor mismatch | Registry has 8 capabilities, DB has 9 (`domain-reference-architecture` only in DB) |

**Structural status: ACTIVE but INCOMPLETE** — the pattern is registered and implemented but has gaps in documentation coverage, concept linkage, and governance instrumentation. A "pending" STRATEGIC_DDD diff could be generated from the gaps above.

---

### Type 2: Semantic Coherence (evolutionary)

Type 2 compares concept-track content against implementation reality. Does the theory still match what was built? Have concrete examples emerged that the docs don't reflect? Has the framing evolved in one place but not another?

#### Four framings compared

| Artifact | Core framing | Audience | Concreteness |
|----------|-------------|----------|-------------|
| **Pattern doc** | "Governance as projection over entity/edge graph" — coverage views as homeostatic sensors | Internal/technical | High — names specific views, SQL examples |
| **Website page** | "Architecture = encoded business rules, not infrastructure" — adoption lifecycle (Understand → Intentional → Explicit) | External/accessible | High — founder narrative, 5 practical starting points, DDD context |
| **UL entry** | "Encode your strategy into your systems so humans and AI can operate from shared structure" | Domain vocabulary | Low — one-liner, no examples |
| **Enriched definition** | "Making architectural decisions explicit, queryable, and traceable rather than implicit in code or tribal knowledge" | KB agents | Medium — describes the goal, not the mechanism |
| **STRATEGIC_DDD** | EA as a capability pattern dependency — 7+ capabilities implement it | Architecture governance | None — only declares which capabilities implement EA, never defines it |

#### Alignment assessment

| Dimension | Status | Detail |
|-----------|--------|--------|
| **Thesis alignment** | Aligned | All agree: architecture-as-queryable-data enables governance-as-queries |
| **Scope** | Divergent | Pattern doc: narrow governance mechanism. UL: broad strategic pillar. Website: adoption guide. STRATEGIC_DDD: capability dependency. Four different scopes. |
| **Concrete examples** | One-sided | Pattern doc has SQL examples. Website has practical "Where to Start." STRATEGIC_DDD and UL have zero examples. The enriched definition is abstract. |
| **EA → Coherence relationship** | Undefined | Pattern doc distinguishes EA (structural completeness) from semantic-coherence (content alignment). No artifact explains how they interact. |
| **Implementation binding** | Implicit | ARCHITECTURE.md describes the retrieval pipeline, entity/edge model, coverage views — all EA mechanisms — without naming the pattern. The implementation IS EA but doesn't say so. |

#### Specific drift findings

**1. Enriched definition is stale (mid-stream framing)**

The enriched definition was written Feb 26, between the rename (Feb 17) and the pattern doc migration (Mar 8). It emphasizes "documentation and traceability" — a concept-level framing. The pattern doc (Feb 16) and the website page (Feb 18) had already moved to "governance as projection" and "encoded business rules" respectively. The enriched definition captured a snapshot that was already outdated.

**2. Website page has outgrown the pattern doc**

The website page (119 lines) covers adoption lifecycle, DDD context, semantic flywheel, scale projection, explicit enterprise, and 5 practical starting points. The pattern doc (67 lines) covers governance-as-projection and coverage views. The website page is the richer artifact for understanding EA — but the pattern doc is supposed to be canonical. Content that should flow upstream (adoption lifecycle, practical guidance) hasn't been reflected back.

**3. STRATEGIC_DDD uses EA without defining it**

STRATEGIC_DDD.md references EA 14 times as a capability pattern dependency but never defines what it is. An agent reading STRATEGIC_DDD learns that domain-data-model, architecture-audit, and orchestration "implement explicit-architecture" — but not what that means. The definition is expected to exist elsewhere, but there's no link.

**4. ARCHITECTURE.md implements EA without naming it**

The architecture doc describes the entity/edge model, coverage views, fitness functions, and retrieval pipeline — all EA mechanisms. But it never says "this implements the explicit-architecture pattern." An agent reading ARCHITECTURE.md wouldn't know they're looking at EA in action. Conversely, an agent reading the pattern doc can't find where it's implemented because ARCHITECTURE.md doesn't reference it.

**5. Session notes contain the richest implementation context**

ISSUE-112 session notes (lifecycle stage design, Feb 14-15) contain the exact moment EA was recognized as a pattern: *"Identified and registered the underlying pattern (explicit-architecture) that the governance model operationalizes."* And the implementation insight: *"pattern_coverage view was only measuring documentation coverage... The two edge mechanisms were disconnected."* This context is richer than any current doc but buried in session notes.

**6. Three audience-appropriate framings exist but no synthesis**

The blog post ("all your data and process inspectable"), website page ("architecture = encoded business rules"), and pattern doc ("governance as projection") aren't contradictory — they're audience-appropriate versions of the same idea. But no single artifact synthesizes all three into a complete picture. The pattern doc should be that artifact but is the narrowest of the three.

#### Semantic coherence verdict

| Dimension | Score | Remediation |
|-----------|-------|------------|
| Definition alignment | Partial — thesis same, scope divergent | Update enriched definition to match pattern doc framing |
| Concept → implementation binding | Missing | Add EA section to ARCHITECTURE.md naming the pattern |
| Implementation → concept feedback | Missing | Feed concrete examples from implementation back into pattern doc |
| Stale content | Present | Enriched definition, `readme-old` entity, `symbiotic-architecture` entity |
| Cross-reference completeness | Weak | STRATEGIC_DDD uses but doesn't define; ARCHITECTURE.md implements but doesn't name |
| Session note mining | Untapped | ISSUE-112 context should inform pattern doc and ARCHITECTURE.md |

---

## 7. Initial Observations (assembly phase)

These observations from the assembly phase have now been validated by the coherence analysis above:

1. **Enriched definition vs. pattern doc divergence:** The enriched definition frames EA as "documentation and traceability." The pattern doc frames it as "governance as projection over entity/edge graph." The pattern doc is richer and more specific.

2. **Stale KB entity:** `readme-old` (superseded hub) outranks the current README in semantic search. Should it be retired or re-embedded?

3. **Symbiotic Architecture ghost:** Post-rename, `symbiotic-architecture` entity still exists in the KB with a different framing ("systems, organizations, and products encode the same domain semantics"). Not retired.

4. **Reference path fragility:** Pattern doc references use relative paths like `../../publisher-pr/docs/source/...` — these are fragile and break across repo boundaries.

5. **Pattern doc lists 2 foundations (VSM, DDD) but registry shows 8 capabilities** — some capabilities (e.g., `data-due-diligence`, `reference-generation`) implement EA alongside many other patterns. The pattern doc doesn't mention most of these capabilities.

6. **Concept-pattern map shows `ai-ready-architecture` and `agentic-coding-at-scale` mapped TO explicit-architecture** — but the pattern doc doesn't list these as related patterns.

7. **Source material breadth vs. pattern doc focus:** 19 source files cover wide territory (BIZBOK, semantic flywheel, wisdom aggregate root). The pattern doc synthesizes narrowly around "governance as projection." Is this intentional distillation or a gap?

8. **No ADR exists** for the Explicit Architecture pattern itself, despite it being a foundational 1P innovation.

9. **Published content is richer than canonical:** The website page (119 lines, founder-voice narrative with adoption lifecycle, DDD context, practical "Where to Start") is substantially richer than the pattern doc (67 lines, technical/internal framing). The published layer has evolved beyond what the canonical layer captures.

10. **semops-docs vs docs-pr delta:** semops-docs has 17 files vs docs-pr's 19 — missing `bizbok-ddd-overlay.md` and `README-old.md`. This is likely intentional (public vs private), but should be confirmed.

11. **Three distinct framings of EA exist in published content:**
    - Blog post: "all your data and process inspectable by both people and machines" (accessible)
    - Website page: "architecture = encoded business rules, not infrastructure" (narrative)
    - Pattern doc: "governance as projection over entity/edge graph" (technical)
    - These aren't contradictory but represent different audiences/depths. No single artifact captures all three.

---

## 7. Structural Finding: Concept Entities as Value Objects on Pattern Aggregate

### The DDD interpretation

In the current schema, the `pattern` table is the aggregate root (stable core). When a concept gets promoted to a pattern, it gets a row in the `pattern` table — but the concept entities that describe it remain as `entity` rows in the content layer. Per DDD:

- **Pattern** (aggregate root) → `pattern` table. Owns: definition, provenance, SKOS edges, capability IMPLEMENTS edges. Stable core.
- **Concept entities** (value objects on the aggregate) → `entity` table, `entity_type: content`. They describe/support the pattern but the pattern doesn't own their lifecycle. Flexible edge.

The relationship should look like:

```text
pattern: explicit-architecture (aggregate root)
  │
  ├── SKOS edges (stable core, pattern-to-pattern)
  │     └── EXTENDS → ddd
  │
  ├── IMPLEMENTS edges from capabilities (stable core, architecture layer)
  │     ├── domain-data-model
  │     ├── architecture-audit
  │     ├── pattern-management
  │     └── ... (6 more)
  │
  └── DESCRIBED_BY edges to concept entities (missing link)
        ├── explicit-architecture (README hub doc)
        ├── what-is-architecture
        ├── ai-ready-architecture
        ├── agentic-coding-at-scale
        ├── stable-core-flexible-edge
        ├── semantic-flywheel
        ├── ... (13 more source docs)
        ├── pub-explicit-architecture (published mirror)
        ├── website page (semops.ai/framework/explicit-architecture)
        └── blog post references
```

### Why concept entities are value objects, not part of the aggregate

The concept entities don't move into the `pattern` table when promoted. They stay as `entity` rows because:

1. **No independent aggregate identity in the pattern context.** They matter *because* they describe this pattern — not on their own terms.
2. **Pattern doesn't control their lifecycle.** Source docs can be edited, republished, restructured without changing the pattern's stable core (definition, provenance, capability mappings).
3. **Replaceable in principle.** You could rewrite all 19 source docs and the pattern aggregate remains intact — the definition, the SKOS edges, the capability mappings are unchanged.
4. **But they carry the meaning.** An agent needs them to understand *why* the pattern exists and *how* to adopt it. The pattern row alone is a skeleton.

### What's missing: the `DESCRIBED_BY` edge

Today, the two tracks (pattern aggregate vs. concept entities) coexist but are **not navigable**:

- `get_pattern("explicit-architecture")` → returns the aggregate root with capabilities and SKOS edges. No path to concept content.
- `search_knowledge_base("explicit architecture")` → returns concept entities. No path up to the pattern aggregate.
- `graph_neighbors("explicit-architecture")` → returns both tracks mixed in a flat list, but concept entities are connected via auto-detected edges (RELATED_TO, EXTENDS), not via a structural `DESCRIBED_BY` predicate.

A `DESCRIBED_BY` edge (or similar predicate — `documents`, `explains`) from the pattern to its concept entities would:

1. **Make the concept/implementation link explicit** — an agent can walk from pattern to the content that explains it
2. **Distinguish concept-of-pattern from concept-of-concept** — not all content entities describe a pattern; the edge marks which ones do
3. **Enable coherence checking** — if the enriched definition on the pattern row drifts from the content in its DESCRIBED_BY entities, that's a measurable gap
4. **Preserve the aggregate boundary** — the concept entities remain value objects in the content layer, not promoted into the pattern table

### Enriched definition as summary, not replacement

This also clarifies the enriched definition problem. The `definition` field on the pattern row is a **summary** of the value objects — a convenience for agents that need a quick answer. It drifted from the pattern doc and the website page because there's no structural relationship enforcing coherence between the aggregate root's summary and the value objects that carry the full meaning.

With `DESCRIBED_BY` edges in place, the enriched definition becomes verifiable: does this summary accurately represent what the linked concept entities say? That's a coherence check, not a manual review.

### Remediation items

1. **Add `described_by` as a valid edge predicate** in the schema (or use existing `documents` predicate if semantically appropriate)
2. **Create edges from `explicit-architecture` pattern to its concept entities** — the 19 source docs + published mirror + website page
3. **Ingest the pattern doc as a KB entity** — `docs/https://github.com/semops-ai/semops-orchestrator/blob/main/docs/patterns/domain/explicit-architecture.md` is the bridge artifact but has no KB entity
4. **Retire stale concept entities** — `readme-old` (superseded), `symbiotic-architecture` (pre-rename ghost)
5. **Update enriched definition** to align with current pattern doc framing ("governance as projection") rather than the mid-stream framing ("documentation and traceability")
6. **Generalize for all promoted patterns** — if this pattern applies to EA, it applies to every concept-to-pattern promotion. The `DESCRIBED_BY` edge should be part of the promotion workflow.

---

## 8. Metadata Efficiency Analysis

### The problem this analysis exposed

This coherence analysis required ~20 tool calls, git archaeology across 7 repos, 3 different vector search strategies, and direct SQL queries — much of it to answer questions that should have been deterministic lookups. The high-reasoning analysis (comparing framings, assessing drift, tracing lineage) was valuable. The manual gathering was not.

If the right metadata existed on source artifacts, an agent could:
1. **Deterministically collect** all artifacts related to a pattern (no git archaeology, no cross-repo grep)
2. **Filter to must-have docs** before engaging high-reasoning analysis
3. **Detect structural gaps** via query rather than checklist

### What we did manually vs. what metadata could replace

| Manual work performed | Time/effort | Metadata that would replace it |
|----------------------|-------------|-------------------------------|
| Git archaeology across 7 repos for creation dates | High | `date_created`, `date_updated` on entity rows (some already null) |
| Cross-repo grep to find all EA source files | High | `pattern` frontmatter on source docs (semops-docs already has this!) |
| Searching for related issues across repos | Medium | `patterns: [explicit-architecture]` on issue entities |
| Finding relevant session notes | Medium | `patterns: [explicit-architecture]` on session note entities |
| Distinguishing concept vs. implementation track | High (reasoning) | `track: concept \| implementation` on entity metadata, or inferred from DESCRIBED_BY edges  |
| Detecting stale/superseded entities | Medium | `status: retired`, `superseded_by: <entity-id>` on entity metadata |
| Comparing enriched definition to pattern doc | Medium (reasoning) | Automatic if DESCRIBED_BY edges exist — coherence is a diff |
| Finding the promotion convergence point | High | `promoted_at: <date>`, `promoted_from: <concept-entity-id>` on pattern row metadata |

### Proposed metadata registries

#### 1. Source doc frontmatter: `pattern` field

Some docs-pr/semops-docs files already have this:

```yaml
---
pattern: explicit-architecture
provenance: 1p
pattern_type: concept
---
```

**Gap:** Not all source docs have it. The 19 EA files in `EXPLICIT_ARCHITECTURE/` directory have varying frontmatter — some have `pattern:`, some don't. If all concept docs that describe a pattern had `pattern: <pattern-id>`, an agent could do:

```sql
-- "Give me all concept content for this pattern"
SELECT * FROM entity WHERE metadata->>'pattern' = 'explicit-architecture'
```

**Requirement:** Ingest the `pattern` frontmatter field into entity `metadata` JSONB during ingestion.

#### 2. Issue entity metadata: `patterns` field

GitHub issues that relate to a pattern (rename, implementation, governance) have no pattern metadata. We found them via text search. If issue entities carried:

```json
{
  "patterns": ["explicit-architecture"],
  "issue_type": "implementation | governance | rename | analysis"
}
```

An agent could deterministically pull all governance history for a pattern:

```sql
SELECT * FROM entity
WHERE entity_type = 'content'
AND metadata->'patterns' ? 'explicit-architecture'
AND metadata->>'content_type' = 'issue'
```

**Source:** This could be extracted during issue ingestion from issue labels, title keywords, or body references. The concept-pattern-map already identifies which concepts map to which patterns — the same logic could tag issues.

#### 3. Session note metadata: `patterns` field

Session notes are the richest source of implementation context (ISSUE-112 contained the recognition moment). But they're only discoverable via chunk search with precise queries. If session note entities carried:

```json
{
  "patterns": ["explicit-architecture"],
  "session_type": "implementation | design | analysis | governance"
}
```

An agent could pull all implementation context for a pattern without vector search:

```sql
SELECT * FROM entity
WHERE metadata->'patterns' ? 'explicit-architecture'
AND metadata->>'content_type' = 'session_note'
ORDER BY metadata->>'date_created' DESC
```

**Source:** Session notes often reference patterns explicitly in their content. An LLM classification pass during ingestion (similar to the existing content_type classifier) could extract pattern references.

#### 4. Entity lifecycle metadata

Stale entities (`readme-old`, `symbiotic-architecture`) were found manually. If entities carried lifecycle metadata:

```json
{
  "status": "active | superseded | retired",
  "superseded_by": "explicit-architecture",
  "renamed_from": "symbiotic-architecture"
}
```

Stale entity detection becomes a fitness function:

```sql
-- Entities with non-existent source files
SELECT id, filespec->>'uri' FROM entity
WHERE metadata->>'status' != 'retired'
AND NOT (filespec->>'accessible')::boolean
```

#### 5. Pattern row promotion metadata

The pattern table row could carry promotion lineage:

```json
{
  "promoted_at": "2026-02-16",
  "promoted_from": ["symbiotic-architecture"],
  "convergence_commit": "feat: operationalize governance model ",
  "promotion_type": "recognition"
}
```

This would make the lineage queryable without git archaeology.

### What this enables: data management orchestration

This isn't just query optimization — it's a **data management orchestration system** where the graph, SQL, and vector search each play their natural role. The metadata registries turn coherence analysis from a manual research task into an orchestrated pipeline.

#### The three query surfaces, orchestrated

Each surface has a natural role:

| Surface | Role | What it's good at |
|---------|------|-------------------|
| **Neo4j graph** | Navigation + traversal | "Walk from this pattern to everything related" — follow DESCRIBED_BY, IMPLEMENTS, EXTENDS edges across any number of hops. Discover relationships the SQL schema doesn't anticipate. |
| **PostgreSQL** | Deterministic lookups + aggregation | "Give me exact counts, coverage scores, status checks" — fitness functions, pattern_coverage view, metadata filters. The stable core. |
| **Qdrant vectors** | Semantic discovery | "Find things related to this concept that aren't explicitly linked" — surface session notes, blog posts, ADRs that discuss the pattern but aren't structurally connected yet. The flexible edge. |

Today these surfaces are queried independently. With the metadata registries, they become stages in an orchestrated pipeline:

```text
Stage 1: Graph traversal (deterministic, complete)
  ┌─ Start at pattern node: explicit-architecture
  ├─ Walk DESCRIBED_BY edges       → concept value objects (source docs, published pages)
  ├─ Walk IMPLEMENTS edges (in)    → capabilities that implement it
  ├─ Walk EXTENDS/BROADER edges    → parent/child patterns (DDD, VSM)
  ├─ Walk RELATED_TO edges         → sibling concepts (semantic-flywheel, stable-core)
  └─ Walk issue/session edges (in) → governance history, implementation context

  Result: complete connected subgraph. Every entity the system knows is related.

Stage 2: SQL enrichment (deterministic, scored)
  ┌─ pattern_coverage view         → content/capability/repo counts
  ├─ Metadata filters              → status != 'retired', track = 'concept' | 'implementation'
  ├─ Date metadata                 → promoted_at, date_created, date_updated
  └─ Fitness function checks       → structural completeness score

  Result: filtered, scored artifact set with lifecycle and coverage data.

Stage 3: Vector discovery (semantic, exploratory)
  ┌─ Search for pattern name       → find entities NOT yet linked in graph
  ├─ Chunk search for mechanisms   → find buried implementation context
  └─ Compare against Stage 1 set   → delta = candidates for new edges

  Result: discovery of missing links. Feed back into graph as proposed edges.

Stage 4: High-reasoning analysis (where agent value is)
  ┌─ Input: Stages 1-3 output (complete, filtered, enriched, with candidates)
  ├─ Compare framings across concept entities
  ├─ Assess drift between enriched definition and DESCRIBED_BY content
  ├─ Evaluate whether concept docs reflect implementation reality
  ├─ Identify audience-appropriate variations vs. accidental drift
  └─ Propose new edges from Stage 3 candidates

  Result: coherence findings, remediation items, proposed graph updates.
```

#### The graph as orchestration backbone

The key insight is that the **graph is the orchestration backbone**, not just another query surface. When metadata registries are populated:

- **Starting a coherence analysis** = graph traversal from a pattern node. Everything connected comes with it.
- **Finding gaps** = comparing graph connectivity against the structural checklist. Missing DESCRIBED_BY edge? That's a graph gap.
- **Proposing new edges** = vector search finds semantically related content not yet in the graph. An agent proposes edges; HITL or automated review promotes them from flexible edge to stable core.
- **Tracking evolution** = graph edge timestamps show when relationships were established, enabling lineage queries without git archaeology.

This is the stable-core/flexible-edge pattern applied to the analysis process itself:

- **Stable core:** Graph edges (DESCRIBED_BY, IMPLEMENTS, EXTENDS) + SQL metadata (status, dates, coverage scores)
- **Flexible edge:** Vector search candidates, proposed edges, session note discoveries

The coherence analysis becomes a **managed data pipeline**, not a research expedition. The agent spends its reasoning budget on *analysis*, not on *gathering*.

### Effort estimate for metadata registries

| Registry | Where metadata lives | How populated | Existing infrastructure |
|----------|---------------------|---------------|------------------------|
| Source doc `pattern` field | Entity `metadata` JSONB | Ingested from frontmatter | Frontmatter already exists on some docs; ingestion pipeline exists |
| Issue `patterns` field | Entity `metadata` JSONB | LLM classification during ingestion | Classifier infrastructure exists (`ingest_from_source.py`) |
| Session note `patterns` field | Entity `metadata` JSONB | LLM classification during ingestion | Same classifier |
| Entity lifecycle `status` | Entity `metadata` JSONB | Manual or automated (filespec accessibility check) | `filespec.accessible` field exists |
| Pattern promotion metadata | Pattern `metadata` JSONB (new) | Manual during promotion, or retroactive backfill | Pattern table exists but has no metadata JSONB column |

Most of this builds on existing infrastructure. The biggest new work is adding pattern-tagging to the issue/session-note ingestion classifier and adding a metadata JSONB column to the pattern table.
