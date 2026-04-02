# Coherence Analysis: System Primitive Decomposition

> **Date:** 2026-03-15
> **Subject:** Capability `system-primitive-decomposition`
> **Related:** , , 

---

## 1. Capability Profile

| Field | Value |
|-------|-------|
| **ID** | `system-primitive-decomposition` |
| **Name** | System Primitive Decomposition |
| **Status** | planned |
| **Domain classification** | core |
| **Delivered by** | semops-research |
| **Priority** | P31 |
| **Governance criteria** | "Decomposition of enterprise systems into classifiable primitives" |
| **Agent coverage** | NONE (fitness check: MEDIUM) |

### Implements patterns (3)

| Pattern | Provenance | Coverage | SKOS edges |
|---------|-----------|----------|------------|
| `explicit-architecture` | 1p | 6 content, 9 capabilities, 4 repos | extends `ddd` |
| `explicit-enterprise` | 1p | 1 content, 8 capabilities, 3 repos | extends `platform-engineering` |
| `data-system-classification` | 1p | 0 content, 3 capabilities, 1 repo | extends `dbt`, `ddd`, `togaf` |

### Integration dependencies

| Source | Target | Pattern | Artifact | Direction |
|--------|--------|---------|----------|-----------|
| semops-data | semops-research | customer-supplier | Ollama, Qdrant, Docling services | upstream provides |

### Graph edges (Neo4j)

| Neighbor | Label | Relationship | Direction |
|----------|-------|-------------|-----------|
| explicit-architecture | Pattern | IMPLEMENTS | outgoing |
| explicit-enterprise | Pattern | IMPLEMENTS | outgoing |
| data-system-classification | Pattern | IMPLEMENTS | outgoing |
| semops-research | Repository | DELIVERED_BY | outgoing |

---

## 2. Artifact Timeline

### Concept Track (theory, docs, narrative)

| Date | Artifact | Location | Content |
|------|----------|----------|---------|
| 2026-02-21 |  | Issue | "translate 'Beancounter' into a neutral pattern" — accounting example of brand-to-neutral |
| 2026-02-21 |  | Issue | "translate surface signals into neutral domain models" — Salesforce CRM to neutral customer domains |
| 2026-02-26 | Issue  session notes | semops-data | "The neutral primitives / decomposition process (Salesforce to neutral capabilities) is critical for domain alignment work" |
| 2026-03-04 |  | Issue | "Brand to Neutral Breakdown" — formal issue with prior art research (TOGAF ABB/SBB, CNCF, Zachman) |
| 2026-03-11 | semops-consulting-view1.md | publisher-pr | Lists "Primitive Deconstruction" as acquisition method; "Sometimes you start with a vendor pattern and need to deconstruct" |
| 2026-03-11 | semops-consulting-view2.md | publisher-pr | Defines "primitive decomposition" method: vendor products to vendor-neutral architectural primitives |
| 2026-03-11 | ISSUE-28 session notes | semops-research | BusArch diagram: "Primitive deconstruction: vendor to neutral capabilities" in PROFILE step |
| 2026-03-11 | busarch-semops-generic.md | semops-research | Mermaid diagram with primitive deconstruction in PROFILE node |

### Implementation Track (schema, registry, capabilities)

| Date | Artifact | Location | Content |
|------|----------|----------|---------|
| 2026-02-26 | UL business rule | semops-data | "Proprietary pattern not decomposed: Decompose to neutral capabilities first" (UBIQUITOUS_LANGUAGE.md line 265) |
| 2026-03-10 | registry.yaml | semops-data | Capability `system-primitive-decomposition` registered as planned with 3 patterns |
| 2026-03-10 | STRATEGIC_DDD.md | semops-data | Capability listed under Extraction layer, P31, delivered by semops-research |
| 2026-03-11 | semops-research ARCHITECTURE.md | semops-research | Capability listed: "Decomposition of enterprise systems into classifiable primitives" |
| 2026-03-11 | arch-sync session notes | semops-research | Confirms planned status in arch-sync audit |

### Convergence Analysis

**Early convergence.** The concept track (Feb 2026 onwards) and implementation track (Mar 2026 registry) converged when the capability was registered on 2026-03-10. However, the capability is **planned only** — no code, no agent, no active implementation.

**Promotion type:** Design promotion (capability registered forward-looking before implementation exists).

---

## 3. Canonical Layer

| Artifact | Status | Detail |
|----------|--------|--------|
| registry.yaml entry | PRESENT | Planned, 3 patterns, semops-research, P31 |
| STRATEGIC_DDD.md | PRESENT | Extraction layer table + repo delivery table |
| ARCHITECTURE.md (semops-data) | GAP | No reference |
| ARCHITECTURE.md (semops-research) | PRESENT | Listed as planned capability |
| UL entry | PARTIAL | Appears as business rule edge case, not named term |
| Pattern doc for the method | N/A | This is a capability, not a pattern — but see semantic analysis |
| Fitness functions | N/A | No capability-specific fitness function; general agent coverage check fires |
| Governance issue | GAP | `governance.issue: null` in registry.yaml |

---

## 4. Source Material Layer

### docs-pr
The concept is embedded in `working-with-patterns.md` (docs-pr/SEMOPS_DOCS/SEMANTIC_OPTIMIZATION) as general advice: "Look for primitives and compose simple primitives rather than bundled solutions" — but not named as a capability.

###  (Primary source)
** "Brand to Neutral Breakdown"** is the most complete source:
- Formal prior art survey (TOGAF ABB/SBB, CNCF Landscape, Zachman Primitives, DoDAF)
- Taxonomy evaluation (CNCF, Flexera Technopedia)
- Tool landscape (LeanIX, ServiceNow, Wappalyzer, vFunction)
- Recommended approach: CNCF as primitive taxonomy, TOGAF ABB/SBB as conceptual model, knowledge graph storage, LLM-driven decomposition as 1P innovation

---

## 5. Published Layer

### publisher-pr
Two consulting framework drafts reference the concept:
- `semops-consulting-view1.md`: Named as "Primitive Deconstruction" in acquisition activities and architecture building
- `semops-consulting-view2.md`: Formal definition with method: take vendor product, decompose into neutral domain primitives, store decomposition edges

### sites-pr
No references found.

---

## 6. KB Layer

### Entity
The capability entity `system-primitive-decomposition` exists in PostgreSQL and Neo4j with correct edges (3 IMPLEMENTS, 1 DELIVERED_BY).

No content entity for the capability itself (no ingested description doc).

### Vector search
The capability has low vector similarity scores when searched by name (top result `pattern-management` at 0.31). The concept surfaces better via chunk search on the method description ( at 0.77).

### Graph context (via semops-research-issue-23)
| Neighbor | Label | Relationship | Strength |
|----------|-------|-------------|----------|
| capability-extraction | Concept | EXTENDS | 0.8 |
| architecture-patterns | Concept | RELATED_TO | 0.9 |
| domain-driven-design | Concept | DERIVED_FROM | 0.85 |
| capability-modeling | Concept | RELATED_TO | 0.8 |
| dam-pattern | Concept | RELATED_TO | 0.7 |
| digital-asset-management-pattern | Concept | RELATED_TO | 0.7 |

---

## 7. Governance History

### Issues
| Issue | Repo | Date | Role |
|-------|------|------|------|
|  "Brand to Neutral Breakdown" | semops-research | 2026-03-04 | **Primary** — formal description with prior art |
|  "Pattern / Capability, or all Pattern?" | semops-orchestrator | 2026-02-21 | Context — accounting example of brand-to-neutral |
|  "Company Domain Modeling Analysis" | data-pr | 2026-02-21 | Context — Salesforce CRM to neutral customer domains |
|  "Pattern Refinement" | semops-data | 2026-02-26 | Session note — "neutral primitives / decomposition process is critical" |

### Session notes
- `ISSUE-145-pattern-refinement.md` (semops-data) — Process insight about decomposition
- `ISSUE-28-busarch-diagram-semops-overlay.md` (semops-research) — Step 5 of consulting process
- `2026-03-11-arch-sync-audit.md` (semops-research) — Lists capability in planned status

### Name variants observed
| Variant | Where |
|---------|-------|
| "system-primitive-decomposition" | registry.yaml, STRATEGIC_DDD, DB |
| "primitive deconstruction" | consulting-view1, semops-research diagrams |
| "primitive decomposition" | consulting-view2 |
| "brand to neutral breakdown" |  title |
| "neutral primitive decomposition" | UL business rule |

---

## Type 1 — Structural Coherence

| Artifact | Check | Status |
|----------|-------|--------|
| Entity in DB | Capability entity exists with `entity_type = capability` | PRESENT |
| `implements` edges | 3 edges to patterns (explicit-architecture, explicit-enterprise, data-system-classification) | PRESENT |
| `delivered_by` edge | 1 edge to semops-research | PRESENT |
| Registry.yaml entry | Listed with status, patterns, delivery, governance criteria | PRESENT |
| STRATEGIC_DDD.md | Capability listed in Extraction layer + repo delivery table | PRESENT |
| semops-research ARCHITECTURE.md | Capability listed as planned | PRESENT |
| semops-data ARCHITECTURE.md | No reference | GAP |
| UL entry | Business rule references the concept, but no named term definition | PARTIAL |
| Governance issue | `governance.issue: null` — no tracking issue linked | GAP |
| Agent coverage | No agent exercises this capability | GAP (fitness check MEDIUM) |
| Pattern count vs registry | DB has 3 patterns, registry.yaml has 3 patterns | ALIGNED |
| Content count | 0 content entities linked to this capability | GAP |

### Structural Verdict: PENDING

The capability is **registered and correctly wired** (patterns, repo, edges all consistent between registry.yaml and DB). But it is planned-only with three structural gaps:
1. No governance issue to track implementation
2. No agent exercises it
3. No content entities linked (no description doc ingested)

These are expected for a planned/P31 capability and would be unusual only if the capability were active.

---

## Type 2 — Semantic Coherence

### Framing comparison

| Source | Core framing | Audience | Concreteness |
|--------|-------------|----------|-------------|
| registry.yaml | "Decomposition of enterprise systems into classifiable primitives" | Architecture governance | Abstract — criteria only |
| STRATEGIC_DDD.md | Same as registry (mirrored) | Architecture governance | Abstract |
|  | "Arrive at neutral domain-pattern by deconstructing vendor solution into primitives" — with TOGAF/CNCF/Zachman prior art | Research/implementation | Concrete — examples (Bynder, Zendesk), framework comparisons, recommended approach |
| consulting-view2 | "Given a branded solution like Bynder or Zendesk, systematically decompose into underlying capabilities and patterns" | Consulting/external | Concrete — method steps |
| consulting-view1 | "Sometimes you start with a vendor pattern and need to deconstruct" | Consulting/external | Moderate — named activity |
| UL business rule | "Proprietary pattern not decomposed: Decompose to neutral capabilities first" | Domain vocabulary | Governance action |
| BusArch diagram | "Vendor to neutral capabilities" in PROFILE step | Consulting process | Concrete — positioned in methodology |

### Alignment dimensions

| Dimension | Status | Finding |
|-----------|--------|---------|
| **Thesis alignment** | ALIGNED | All artifacts agree: take branded/proprietary solutions, decompose into vendor-neutral primitives |
| **Scope consistency** | DRIFT | Registry frames it as "enterprise systems" decomposition;  frames it as "vendor solution" decomposition; consulting frames it as "vendor pattern" deconstruction. The scope oscillates between system-level (full enterprise) and product-level (single vendor tool) |
| **Concrete examples** | PARTIAL | Bynder and Zendesk appear in  and publisher-pr. DAM is cited as prior art. But no worked example exists showing the full decomposition pipeline |
| **Cross-pattern relationships** | ALIGNED | All three implemented patterns (explicit-architecture, explicit-enterprise, data-system-classification) are coherent — the capability sits at their intersection |
| **Implementation binding** | GAP | semops-data ARCHITECTURE.md doesn't mention it; semops-research ARCHITECTURE.md lists it but as planned only |
| **Session note mining** | PRESENT | Issue  contains the key insight: "the neutral primitives / decomposition process is critical for domain alignment work — it's how you validate that pattern-to-capability mappings are honest" — this framing (validation tool) is NOT present in the registry definition |
| **Audience-appropriate versions** | INTENTIONAL | Registry/STRATEGIC_DDD = governance.  = implementation plan. Publisher-pr = consulting narrative. Different framings are audience-appropriate |

### Drift Findings

**Finding 1: Scope oscillation — system vs. product level**

- **What:** Registry says "enterprise systems";  says "vendor solution"; consulting says "vendor pattern". These are different scales.
- **Where:** registry.yaml vs.  vs. publisher-pr consulting drafts
- **Direction:** Lateral drift — no single artifact is "ahead", they just disagree on scope
- **Impact:** When implementation begins, unclear whether the capability decomposes entire enterprise stacks (complex, multi-system) or individual vendor products (simpler, per-product). The prior art in  is product-level (Bynder, Zendesk). The registry name ("system-primitive") suggests system-level.
- **Remediation:** Clarify scope. Recommend: the capability operates at product/solution level (decompose one vendor product into neutral primitives). Enterprise-level decomposition is the composition of multiple product-level decompositions. Update registry criteria to match.

**Finding 2: Missing "validation" framing**

- **What:** Issue  session notes identify this capability as a validation tool: "it's how you validate that pattern-to-capability mappings are honest." This is a distinct use case from the primary "discovery" framing.
- **Where:** ISSUE-145-pattern-refinement.md (concept track) vs. registry.yaml (implementation track)
- **Direction:** Concept ahead of implementation — the concept track has identified a secondary use case not captured in the registry.
- **Impact:** The governance criteria ("Decomposition of enterprise systems into classifiable primitives") covers only the discovery/analysis use case. The validation use case (check existing pattern-capability mappings by decomposing what a vendor product actually does) is unregistered.
- **Remediation:** Add validation as a secondary use case in registry criteria, or note it as future scope.

**Finding 3: No worked example**

- **What:**  names Bynder and Zendesk as examples but doesn't show the actual decomposition output (what primitives, what edges, what gets stored in the graph).
- **Where:** 
- **Direction:** Implementation behind concept — the method is described but never demonstrated
- **Impact:** When the capability moves from planned to active, there's no reference output to validate against. The DAM pattern is cited as prior art but the decomposition path (how DAM was derived from Bynder-like products) is implicit, not documented.
- **Remediation:** Create a worked example (DAM as reference, or Zendesk as greenfield) showing: input (vendor docs/product page) -> decomposition steps -> output (neutral primitives + graph edges). This would serve as both documentation and test fixture.

**Finding 4: Name fragmentation**

- **What:** 5 different names for the same concept across repos
- **Where:** All repos
- **Direction:** Lateral — no canonical name has been established outside the registry
- **Impact:** Vector search for the capability is weak (0.31 max similarity for entity search). Content outside the registry is not discoverable via the canonical name.
- **Remediation:** Adopt "system primitive decomposition" as canonical (matches registry). Update publisher-pr drafts and semops-research diagrams to use this name or alias it explicitly.

---

## Summary

**Structural verdict:** PENDING (correctly wired, planned-only, expected gaps for P31)
**Structural gaps:** 3 (no governance issue, no agent, no content entities)
**Semantic drift findings:** 4

### Key findings
1. **Scope oscillation** between system-level and product-level decomposition needs resolution before implementation
2. **Validation use case** (checking pattern-capability mapping honesty) identified in session notes but not captured in registry
3. **No worked example** exists — the method is described but never demonstrated end-to-end
4. **Name fragmentation** (5 variants) hurts discoverability

### Remediation items
- Clarify scope in registry.yaml governance criteria (product-level vs system-level) — fix in place
- Add validation use case to criteria or note as future scope — fix in place
- Create worked example (DAM or Zendesk decomposition) — spin off to semops-research
- Consolidate naming in publisher-pr and semops-research — fix in place when those docs are next edited
- Link governance issue  in registry.yaml — fix in place
- No urgent action needed — capability is planned/P31 and gaps are expected at this lifecycle stage

### Reference document
`docs/analysis/primitive-deconstruction-reference.md`
