# SC Walkthrough: Explicit Architecture

> **Purpose:** Validate the coherence measurement model (ADR-0014) against real data from the explicit-architecture coherence analysis
> **Issue:** 
> **Source Data:** [explicit-architecture-reference.md](explicit-architecture-reference.md), [COHERENCE_ANALYSIS_TEMPLATE.md](COHERENCE_ANALYSIS_TEMPLATE.md)
> **Model:** [ADR-0014: Coherence Measurement Model](../decisions/ADR-0014-coherence-measurement-model.md)

---

## Assessment Setup

**Goal:** Evaluate the coherence of the `explicit-architecture` pattern across all layers.

**Scope:** SC(explicit-architecture) — architecture-level, single pattern. This is the atomic assessment unit: pattern × capabilities × domain.

**Goal types exercised:**
- Rule execution (structural checklist — zero distance)
- Criteria-based (concept ↔ implementation alignment — low distance)
- Directional (is EA moving toward richer implementation or staying theoretical? — medium distance)

---

## Part 1: Availability (A) — Rule Execution

**Goal:** "All expected artifacts for an active pattern exist and align."

This is a rule execution goal — the architecture defines what should exist for an active pattern, and the check queries the architecture. Every check is zero-distance.

### Findings (pre-remediation)

| Check | Result | A Signal |
| --- | --- | --- |
| Pattern table row | PRESENT | 1.0 |
| SKOS edges match pattern doc | PARTIAL — DDD edge present, VSM edge missing | 0.5 |
| Capability IMPLEMENTS edges | PRESENT (9 capabilities) | 1.0 |
| Coverage view row | PRESENT (content=6, capability=9, repo=4) | 1.0 |
| Pattern doc file exists | PRESENT | 1.0 |
| Pattern doc ingested as KB entity | **GAP** — canonical definition invisible to search | 0.0 |
| Registry.yaml entries | PRESENT (8 capabilities) | 1.0 |
| Registry/DB alignment | PRESENT (9 in both — resolved) | 1.0 |
| Stale entities retired | **GAP** — `readme-old` and `symbiotic-architecture` still active | 0.0 |
| DESCRIBED_BY edges | **GAP** — 0 edges  | 0.0 |
| ARCHITECTURE.md references pattern | **GAP** — implements EA without naming it | 0.0 |
| EA-specific fitness function | **GAP** — none exists | 0.0 |

**A score (pre-remediation):** 5.5 / 12 checks = **0.46**

The pattern exists and has capability coverage, but is structurally incomplete — invisible to search, not linked to its concept entities, not named in the document that implements it, and carrying stale pre-rename artifacts.

### What this tells us about the model

1. **Every check is deterministic.** SQL queries, YAML parsing, file existence, count comparisons. Zero inference required.
2. **Gaps are actionable.** Each 0.0 has an obvious fix: ingest the doc, create the edges, retire stale entities, add the reference.
3. **The checklist scales.** The same checks apply to every active pattern — this is a fitness function waiting to be generalized.

### Post-remediation (Issues , )

| Check | Pre | Post | Fix |
| --- | --- | --- | --- |
| VSM SKOS edge | 0.5 | 1.0 | Registered `viable-systems-model`, created EXTENDS edge |
| Pattern doc as KB entity | 0.0 | 1.0 | Ingested into core_kb |
| Registry/DB alignment | 0.5 | 1.0 | `domain-reference-architecture` added to registry |
| Stale entities | 0.0 | 1.0 | Retired `readme-old` and `symbiotic-architecture` |
| DESCRIBED_BY edges | 0.0 | 1.0 | 14 pilot edges created  |
| ARCHITECTURE.md references | 0.0 | 1.0 | Added 3 EA references |
| EA fitness function | 0.0 | 1.0 | Created `check_explicit_architecture_coverage` |

**A score (post-remediation):** 11.5 / 12 = **0.96**

One remediation session moved A from 0.46 to 0.96.

---

## Part 2: Consistency (C) — Criteria-Based

**Goal:** "Concept-track content matches implementation reality across all artifacts."

This is a criteria-based goal — the criteria are the alignment dimensions (thesis, scope, examples, cross-pattern relationships), and the agent compares artifacts against them. Low inference distance because the reference document structures the comparison.

### Findings (pre-remediation)

| Finding | Artifacts Compared | C Signal | Inference Required |
| --- | --- | --- | --- |
| **Enriched definition stale** — says "documentation and traceability" vs. pattern doc's "governance as projection" | Enriched def ↔ pattern doc | 0.3 | Low — string comparison of core framing |
| **Website outgrew pattern doc** — 119 lines with adoption lifecycle vs. 67 lines technical | Website ↔ pattern doc | 0.5 | Low — scope comparison |
| **STRATEGIC_DDD uses EA 14x without defining it** | STRATEGIC_DDD ↔ pattern doc | 0.4 | Low — presence/absence of definition |
| **ARCHITECTURE.md implements EA without naming it** | ARCHITECTURE.md ↔ pattern doc | 0.2 | Low — pattern name search |
| **Session notes contain buried implementation insights** | Session notes ↔ pattern doc ↔ ARCHITECTURE.md | 0.3 | Medium — requires reading session notes for insight extraction |
| **Three audience-appropriate framings with no synthesis** | Blog ↔ website ↔ pattern doc | 0.6 | Medium — judging whether divergence is intentional (audience) or accidental (drift) |

**C score (pre-remediation):** average 0.38 → **0.38**

The pattern means different things in different places. Some divergence is intentional (audience-appropriate framing), but the enriched definition being stale, ARCHITECTURE.md not naming the pattern, and STRATEGIC_DDD lacking a definition are genuine consistency gaps.

### What this tells us about the model

1. **Criteria-based goals require light reasoning but are structured.** The agent compares specific artifacts against specific dimensions — not open-ended exploration.
2. **"Intentional divergence" is a valid C signal.** Website ↔ pattern doc divergence scored 0.5 (not 0.0) because audience-appropriate framing is expected. The model needs to distinguish intentional from accidental drift — metadata about audience/purpose would make this deterministic.
3. **C improvements feed A.** Fixing ARCHITECTURE.md (C finding) also created an A signal (pattern now referenced). The dimensions interact.

### C Post-remediation

| Finding | Pre | Post | Fix |
| --- | --- | --- | --- |
| Enriched definition | 0.3 | 0.9 | Updated to match pattern doc framing |
| STRATEGIC_DDD missing definition | 0.4 | 0.9 | Added EA definition paragraph |
| ARCHITECTURE.md not naming EA | 0.2 | 0.8 | Added 3 references |
| Session notes buried | 0.3 | 0.6 | Origin section added to pattern doc (partial — not all insights surfaced) |
| Website ↔ pattern doc | 0.5 | 0.7 | Pattern doc enriched with adoption path (from website) |
| Three framings unsynthesized | 0.6 | 0.6 | Unchanged — synthesis is a future content task |

**C score (post-remediation):** average 0.75 → **0.75**

---

## Part 3: Stability (S) — Directional + Temporal

**Goal:** "Are the assumptions behind adopting explicit-architecture still valid?"

S operates at two levels here:

### 3a: Assumption Validity (Directional)

The assumptions when EA was adopted:
- DDD + VSM provide the right 3P foundation → **Holding.** No evidence of better alternatives or foundation drift.
- Entity/edge graph is the right data model for architecture → **Holding.** Every coherence analysis has relied on it successfully.
- Governance-as-projection works (SQL views over operational tables) → **Holding.** Fitness functions and coverage views produce actionable findings.
- The pattern is relevant beyond this project → **Holding.** Published content, blog posts, and framework positioning all confirm external relevance.

**S (assumption validity):** **0.9** — all core assumptions holding. Minor concern: the pattern was recognition-promoted (implemented before named), which means the assumptions were never explicitly stated — they were inferred retroactively. This is a stability risk that manifests as the "buried session notes" finding.

### 3b: Correction Trend (Temporal Lens on A and C)

| Period | A Corrections | C Corrections | Trend |
| --- | --- | --- | --- |
| Pre-analysis (before ) | Unknown — no baseline measurement | Unknown | No data |
| Analysis session (, , ) | 7 structural fixes | 4 semantic fixes | First measurement — establishing baseline |
| Post-remediation | 1 remaining (registry/DB) | 1 remaining (synthesis) | **Stabilizing** — bulk gaps closed |

**S (correction trend):** **0.7** — first measurement, so no temporal comparison yet. The fact that one remediation session closed most gaps is a positive signal. Future measurements will show whether new gaps appear.

### Combined S

**S score:** average(0.9, 0.7) = **0.8**

---

## SC Computation

### Pre-remediation

```text
A = 0.46
C = 0.38
S = 0.8  (assumptions were already valid; stability predates measurement)

SC = (0.46 × 0.38 × 0.8)^(1/3) = (0.140)^(1/3) = 0.52
```

### SC Post-remediation

```text
A = 0.96
C = 0.75
S = 0.8

SC = (0.96 × 0.75 × 0.8)^(1/3) = (0.576)^(1/3) = 0.83
```

**SC improved from 0.52 to 0.83 in one remediation session.**

The geometric mean shows the interaction: A was the weakest dimension pre-remediation and dragged the entire score down. Fixing availability (deterministic, zero-distance fixes) had outsized impact because it unlocked consistency improvements.

---

## Implementation Trajectory Analysis

How would each stage of the trajectory (ADR-0014 §Implementation Trajectory) have changed this analysis?

### Phase 0: Assembly

| Stage | What changes | Time impact |
| --- | --- | --- |
| **Current (document-based)** | Agent searched 6 repos, ran git archaeology, constructed 3 different vector search queries, assembled 62KB reference doc | ~2 hours agent time |
| **Manifest-driven** | Source manifests declare EA artifacts per repo; agent queries manifests instead of searching | ~20 minutes — manifests eliminate archaeology |
| **Graph-driven** | `graph_neighbors("explicit-architecture", depth=2)` returns all connected artifacts across layers | ~2 minutes — assembly is a single query |

### Type 1: Structural Checks

| Stage | What changes | Already there? |
| --- | --- | --- |
| **Current** | SQL queries, YAML parse, file checks — all deterministic | **Yes** — already at target state |
| **Manifest-driven** | Same checks, but expected artifact list comes from manifest instead of ad-hoc checklist | Slight improvement — standardized |
| **Graph-driven** | `check_explicit_architecture_coverage` runs as scheduled job; gaps appear in dashboard | Automated — no human trigger needed |

### Type 2: Semantic Checks

| Stage | What changes | Inference reduction |
| --- | --- | --- |
| **Current** | Agent reads artifacts, compares framings, judges intentional vs. accidental drift | Full reasoning required |
| **Manifest-driven** | Metadata (audience, purpose, last-updated) on each artifact reduces interpretation | Intentional divergence becomes queryable: `audience: external` vs. `audience: internal` |
| **Graph-driven** | Embedding similarity scores between artifacts computed automatically; agent investigates only outliers | Reasoning only for edge cases — bulk measurement is deterministic |

### Key Observation

**The expensive part is assembly, not measurement.** Type 1 is already deterministic. Type 2 requires reasoning but is structured. Phase 0 (assembly) consumed most of the effort and is exactly what the manifest/graph trajectory eliminates.

---

## Validation Against ADR-0014

### Goal-Type Continuum: Confirmed

| ADR-0014 Claim | Evidence |
| --- | --- |
| Rule execution goals are zero-distance | Every Type 1 check was a SQL query, YAML parse, or file existence check |
| Criteria-based goals require low reasoning | Type 2 findings required artifact comparison, not open-ended exploration |
| Directional goals require medium reasoning | S (assumption validity) required judgment about whether 3P foundations are still appropriate |
| Goals can be mixed in one assessment | This analysis exercised all three goal types for one pattern |

### A/C/S Dimensions: Confirmed

| ADR-0014 Claim | Evidence |
| --- | --- |
| A measures traceability across consumers | Pattern doc not in KB = invisible to agents (A finding) |
| C measures alignment across contexts, teams, time | Enriched def ↔ pattern doc ↔ website = three contexts, three framings |
| S measures assumption validity + correction trend | 3P foundation holding (assumption) + remediation closed bulk gaps (trend) |
| Geometric mean collapses when any dimension is low | Pre-remediation A (0.46) dragged SC to 0.52 despite decent S (0.8) |
| Dimensions interact — fixing one improves others | ARCHITECTURE.md fix improved both A (reference exists) and C (pattern named) |

### Assessment Scope: Confirmed

| ADR-0014 Claim | Evidence |
| --- | --- |
| Both sides resolve to architectural components | Every check compared architectural artifacts (pattern rows, edges, entities, docs) |
| Pattern × capability × domain is the atomic unit | 9 capabilities, 4 repos — the assessment naturally scoped to this intersection |
| Aggregation is natural | Could roll up to domain-level by repeating for all patterns in the domain |

### Implementation Trajectory: Confirmed

| ADR-0014 Claim | Evidence |
| --- | --- |
| Document-based works today | 62KB reference doc produced actionable findings |
| Manifest-driven reduces archaeology | Source manifests would eliminate the most expensive phase (assembly) |
| Graph-driven makes measurement a query | `graph_neighbors` + scheduled fitness functions would automate Type 1 entirely |
| Each stage absorbs the previous | Document-based analysis is still valid at every stage — it just gets faster |

---

## Multi-Pattern Comparison

The same scoring methodology applied across all 8 patterns analyzed in `docs/analysis/`. Each reference analysis used the [Coherence Analysis Template](COHERENCE_ANALYSIS_TEMPLATE.md) with Type 1 (structural) and Type 2 (semantic) findings.

### Consolidated Scorecard

| Pattern | Verdict | A Pre | A Post | C Pre | C Post | S | SC Pre | SC Post |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| explicit-architecture | Active-Incomplete → Near Complete | 0.46 | 0.96 | 0.38 | 0.75 | 0.80 | **0.52** | **0.83** |
| governance-as-strategy | Concept Only → Active | 0.38 | 0.94 | 0.23 | 0.95 | 0.90 | **0.43** | **0.93** |
| data-system-classification | Active-Incomplete → Near Complete | 0.58 | 0.96 | 0.32 | 1.00 | 0.92 | **0.55** | **0.96** |
| scale-projection | Active-Incomplete | 0.57 | — | 0.38 | — | 0.88 | **0.58** | — |
| mirror-architecture | Retired → Development | 0.45 | 0.75 | 0.37 | 0.90 | 0.80 | **0.50** | **0.82** |
| system-primitive-decomposition | Pending (planned) | 0.65 | 0.85 | 0.35 | 0.75 | 0.82 | **0.57** | **0.81** |
| semantic-optimization | Concept Only (identity resolved) | 0.31 | 0.31 | 0.25 | 0.65 | 0.85 | **0.40** | **0.57** |
| data-management-analogy | Discovery (draft analysis) | 0.55 | 0.82 | 0.53 | 0.75 | 0.85 | **0.63** | **0.81** |

### What the Comparison Shows

**1. The model differentiates meaningfully.** Pre-remediation scores range from 0.40 (semantic-optimization, concept only) to 0.63 (data-management, strong draft). Post-remediation scores range from 0.57 (semantic-optimization, still unregistered) to 0.96 (data-system-classification, near-perfect). The spread reflects real differences in maturity.

**2. A drives most pre-remediation variance.** Pre-remediation A ranges from 0.31 to 0.65, while S clusters tightly at 0.80-0.92. The patterns with the lowest SC (semantic-optimization 0.40, governance-as-strategy 0.43) both have structural absence as their primary problem — the pattern simply doesn't exist in the architecture graph. This confirms ADR-0014's claim that availability is the most mature and most impactful dimension.

**3. Remediation impact is dramatic and mostly deterministic.** Average SC improvement across remediated patterns: +0.33. Most fixes are zero-distance (create edge, ingest doc, retire stale entity). The model correctly identifies that the highest-impact work is structural, not semantic.

**4. S is stable and uninformative — for now.** All S scores are 0.80-0.92 because this is the first measurement (no temporal baseline) and 3P foundations are generally solid. S will become more informative once we have before/after snapshots and can detect correction trends. This validates ADR-0014's observation that S starts useful at the infrastructure level and grows upward.

**5. Concept-only patterns have a natural floor.** governance-as-strategy and semantic-optimization both started below 0.45 because they lacked pattern table rows. The floor is structural — you can't have availability for something that doesn't exist in the architecture. This confirms that flexible edge items (SC ≈ 0) and concept-only patterns (SC < 0.5) are meaningfully distinct from active patterns.

**6. Discovery mode produces cross-pattern improvements.** The data-management-analogy analysis triggered 3 new pattern registrations that improved scores across multiple patterns simultaneously. Coherence measurement is not isolated per-pattern — discoveries compound.

**7. One unremediated pattern validates the baseline.** scale-projection (0.58, no remediation) sits right in the middle of the pre-remediation range, confirming that the pre-remediation scores are consistent baselines, not artifacts of when we happened to look.

---

## What This Walkthrough Doesn't Cover

1. **Scoring infrastructure** — SC scores above are illustrative (manual computation), not from `coherence_snapshot` table or `corpus_coherence` algorithm. Phase C infrastructure (, ) is needed for automated scoring.
2. **Agentic reasoning trace** — no episode metadata about which reasoning strategy (CoT/ToT/ReAct) the agent used during analysis. Phase C  would add this.
3. **Temporal comparison** — this is the first measurement for all patterns, so S has no prior baseline to diff against. Future measurements will show trend.
4. **Metric-driven goals** — no quantitative targets existed for these patterns. Future: "maintain EA availability above 0.85" would be a standing guardrail (S dimension).
5. **Hypothetical lifecycle** — the full adoption → drift → detection → realignment cycle. All findings here are from real data; the temporal lifecycle (introducing drift, measuring recovery) would require either simulation or waiting for natural drift to occur.
