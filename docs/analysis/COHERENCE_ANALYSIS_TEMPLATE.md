# Coherence Analysis Template

> **Purpose:** Reusable process for analyzing concept-to-pattern promotion lineage and coherence
> **Derived from:** Issue  (Explicit Architecture analysis, 2026-03-15)
> **Output:** A reference document in `docs/analysis/<pattern-id>-reference.md`

---

## Overview

This template guides a coherence analysis for any pattern — especially those promoted from concept (source material) to registered pattern. It produces:

1. A cross-layer reference document assembling all artifacts
2. A concept-to-pattern promotion lineage with two-track timeline
3. Two types of coherence assessment (structural + semantic)
4. Concrete remediation items

---

## Phase 0: Assemble Reference Document

Gather all content related to the pattern across every layer. The goal is one document where the full picture is visible.

### 0.1 Identify the pattern

```
Pattern ID: <pattern-id>
Pattern label: <human-readable name>
Provenance: <1p | 3p>
```

### 0.2 Gather by layer

For each layer, collect all artifacts. Use the checklist below — not every pattern will have content in every cell.

#### Canonical layer (semops-data)

| Artifact | Location | Status |
|----------|----------|--------|
| Pattern doc | `docs/patterns/<pattern-id>.md` | |
| Registry capabilities | `config/registry.yaml` — grep for pattern-id | |
| Enriched definition | `scripts/enrich_pattern_definitions.py` | |
| Concept-pattern map | `config/mappings/concept-pattern-map.yaml` | |
| STRATEGIC_DDD references | `docs/STRATEGIC_DDD.md` | |
| ARCHITECTURE.md references | `docs/ARCHITECTURE.md` | |
| UL entry | `schemas/UBIQUITOUS_LANGUAGE.md` | |
| Fitness functions | `schemas/fitness-functions.sql` | |

#### Source material layer (docs-pr + semops-docs)

| Artifact | Location | Status |
|----------|----------|--------|
| Source docs directory | `docs-pr/docs/SEMOPS_DOCS/.../` | |
| Public mirror | `semops-docs/.../` | |
| Source doc count | docs-pr vs. semops-docs delta | |

#### Published layer (sites-pr, publisher-pr)

| Artifact | Location | Status |
|----------|----------|--------|
| Website page | `sites-pr/apps/semops/content/pages/` | |
| Blog posts referencing pattern | `sites-pr/apps/semops/content/blog/` | |
| Publisher staging page | `publisher-pr/content/pages/` | |
| Edit corpus | `publisher-pr/edits/.pending/` | |

#### KB layer (PostgreSQL + Qdrant + Neo4j)

| Query | Tool | What to check |
|-------|------|---------------|
| Pattern row | `get_pattern("<pattern-id>")` | Definition, provenance, coverage counts, SKOS edges |
| Entity search | `search_knowledge_base("<pattern name>")` | Which entities surface, what types, any stale/ghost entities |
| Implementation search | `search_knowledge_base("<pattern name> + implementation keywords")` | Does implementation content surface? |
| Chunk search | `search_chunks("<pattern name> + specific mechanisms")` | Passage-level context, lineage moments |
| Graph neighbors | `graph_neighbors("<pattern-id>")` | Concept vs. capability edges, DESCRIBED_BY edges |

#### Governance history

| Source | How to find |
|--------|-------------|
| GitHub issues | `gh search issues "<pattern name>" --repo <repo>` for each repo |
| Session notes | `grep -l "<pattern-id>" docs/session-notes/` across repos |
| ADRs | `grep -l "<pattern-id>" docs/decisions/ADR-*.md` across repos |
| Rename history | `git log -S "<old-name>" --all` if applicable |

### 0.3 Build timeline

Use git archaeology to construct the artifact timeline. Key commands:

```bash
# When was the pattern doc first created?
git log --format="%ai %s" --reverse --all -- docs/patterns/<pattern-id>.md

# When was it registered in pattern_v1.yaml?
git log -p --format="COMMIT: %ai %s" --all -S "<pattern-id>" -- schemas/pattern_v1.yaml

# When did it first appear in registry.yaml?
git log -p --format="COMMIT: %ai %s" --all -S "<pattern-id>" -- config/registry.yaml

# Source material creation (check docs-pr)
cd  && git log --format="%ai %s" --reverse --all -- "<source-directory>"

# Published content (check sites-pr)
cd  && git log --format="%ai %s" --all -- "<content-path>"
```

Organize into two tracks:

**Concept track** — theory docs, website pages, blog posts, enriched definitions. Describes *what* the pattern is and *why* it matters. Audience: humans and agents learning the idea.

**Implementation track** — schema, views, fitness functions, registry entries, capability mappings, STRATEGIC_DDD.md. The pattern *operating as architecture*. Audience: the system itself and agents querying it.

Find the **convergence point** — the commit or date where the two tracks first connect (pattern named in implementation, or registered in pattern_v1.yaml alongside the pattern doc).

Note whether this was:
- **Design promotion:** registered first, then implemented (forward-looking)
- **Recognition promotion:** implemented first, then named/registered (retroactive)

---

## Phase 1: HITL Gap-Fill

Review the assembled reference document for gaps:

- Content that exists in repos but wasn't found during assembly
- Context from session notes, governance history, or rename rationale that isn't in the reference doc
- Published content (public repos, live website) that was missed
- Confirm the reference document is complete enough for coherence analysis

---

## Phase 2: Type 1 — Structural Coherence (Deterministic)

For a pattern to be structurally "active", specific artifacts must exist. This is checkable against SQL, YAML, and files.

### Structural checklist

| Artifact | Expected for active pattern | Query/Check |
|----------|---------------------------|-------------|
| `pattern` table row | Row with id, definition, provenance | `SELECT * FROM pattern WHERE id = '<pattern-id>'` |
| `pattern_edge` SKOS | Edges matching pattern doc's "Derives From" / "Related Patterns" | `SELECT * FROM pattern_edge WHERE src_id = '<pattern-id>'` |
| Capability IMPLEMENTS | Edges from capabilities listed in registry.yaml | `SELECT src_id FROM edge WHERE dst_id = '<pattern-id>' AND predicate = 'implements'` |
| `pattern_coverage` view | Row with content, capability, repo counts | `SELECT * FROM pattern_coverage WHERE pattern_id = '<pattern-id>'` |
| Pattern doc file | `docs/patterns/<pattern-id>.md` exists | `ls docs/patterns/<pattern-id>.md` |
| Pattern doc as KB entity | Entity ingested from pattern doc | `search_knowledge_base` with content_type "pattern" |
| Registry.yaml entries | Capabilities listed | `grep "<pattern-id>" config/registry.yaml` |
| STRATEGIC_DDD.md | Pattern referenced in capability tables | `grep -c "<pattern-id>" docs/STRATEGIC_DDD.md` |
| ARCHITECTURE.md | Pattern named (not just implicitly implemented) | `grep -c "<pattern-id>" docs/ARCHITECTURE.md` |
| UL entry | Definition present | `grep "<pattern-id>" schemas/UBIQUITOUS_LANGUAGE.md` |
| DESCRIBED_BY edges | Pattern → concept entities | `SELECT * FROM edge WHERE src_id = '<pattern-id>' AND predicate = 'described_by'` |
| Stale entities | Pre-rename or superseded entities retired | Check for ghost entities with non-existent filespec URIs |
| Registry/DB alignment | Capability count matches between registry.yaml and edge table | Compare counts |

Score each as: PRESENT / GAP / MISMATCH / N/A

### Structural verdict

Classify the pattern as:

- **ACTIVE — COMPLETE:** All expected artifacts present and aligned
- **ACTIVE — INCOMPLETE:** Registered and implemented but has structural gaps (missing edges, uningested docs, stale entities)
- **PENDING:** Registered but missing key implementation artifacts (no capabilities, no STRATEGIC_DDD references)
- **CONCEPT ONLY:** Source material exists but no pattern registration

---

## Phase 3: Type 2 — Semantic Coherence (Evolutionary)

Compare concept-track content against implementation reality. This is bidirectional:

- **Concept → Implementation:** Is the theory still accurate? Has implementation evolved past what docs describe?
- **Implementation → Concept:** Are there concrete examples, session note insights, or operational learnings that should flow back into concept docs?

### Framing comparison

For each artifact that describes the pattern, extract:

| Artifact | Core framing | Audience | Concreteness |
|----------|-------------|----------|-------------|
| Pattern doc | | Internal/technical | |
| Website page | | External/accessible | |
| UL entry | | Domain vocabulary | |
| Enriched definition | | KB agents | |
| STRATEGIC_DDD | | Architecture governance | |
| Source docs (hub) | | Theory/research | |
| Blog posts | | Public narrative | |

### Alignment dimensions

Check each dimension:

| Dimension | Question | Status |
|-----------|----------|--------|
| Thesis alignment | Do all artifacts agree on what the pattern IS? | |
| Scope consistency | Same scope (narrow mechanism vs. broad pillar)? | |
| Concrete examples | Do docs reflect current implementation examples? Or still abstract? | |
| Cross-pattern relationships | Are related patterns consistent across docs? | |
| Implementation binding | Does ARCHITECTURE.md explicitly name the pattern? | |
| Session note mining | Are there insights in session notes that should be in docs? | |
| Audience-appropriate versions | Are different framings intentional (audience) or accidental (drift)? | |

### Drift finding template

For each finding:

```
**Finding N: <title>**

- **What:** <description of the misalignment>
- **Where:** <which artifacts are involved>
- **Direction:** concept ahead of implementation | implementation ahead of concept | lateral drift
- **Impact:** <what breaks or degrades because of this>
- **Remediation:** <what to do about it>
```

---

## Phase 4: Remediation

Collect all gaps and findings into actionable items:

### From Type 1 (structural)

These are deterministic fixes — the artifact either exists or doesn't.

- [ ] Missing pattern edges (SKOS, DESCRIBED_BY)
- [ ] Uningested pattern doc
- [ ] Missing ARCHITECTURE.md references
- [ ] Stale entity cleanup
- [ ] Registry/DB alignment
- [ ] Missing fitness function coverage

### From Type 2 (semantic)

These require content review and updating.

- [ ] Enriched definition alignment
- [ ] Pattern doc content gaps (missing examples, stale framing)
- [ ] STRATEGIC_DDD definition gap
- [ ] Session note insights to surface
- [ ] Cross-reference completeness

### Issue creation

For each remediation item, decide:

- **Fix in place** (minor, can be done now)
- **Spin off issue** (significant work, needs its own tracking)
- **Defer** (known gap, not urgent)

Link all spun-off issues to the parent coherence analysis issue.

---

## Query Surface Orchestration

The three query surfaces each play a natural role. Use them in order:

| Stage | Surface | Role | What it answers |
|-------|---------|------|-----------------|
| 1 | **Neo4j graph** | Navigation | "Walk from this pattern to everything connected" — DESCRIBED_BY, IMPLEMENTS, EXTENDS edges. Complete connected subgraph. |
| 2 | **PostgreSQL** | Enrichment | "Score and filter the subgraph" — coverage counts, metadata filters, fitness checks, lifecycle status. |
| 3 | **Qdrant vectors** | Discovery | "Find what's NOT yet linked" — semantic search for related content, compare against graph results, propose new edges. |
| 4 | **Agent reasoning** | Analysis | "Assess coherence across the filtered set" — framing comparison, drift detection, lineage interpretation. |

The graph is the **orchestration backbone**. Starting a coherence analysis = graph traversal from a pattern node. SQL enriches. Vectors discover gaps. The agent reasons over the complete, filtered result.

This is the stable-core/flexible-edge pattern applied to the analysis itself:

- **Stable core:** Graph edges + SQL metadata (deterministic, structural)
- **Flexible edge:** Vector search candidates, proposed edges (semantic, exploratory)

### Metadata that improves efficiency

When present on entities, these metadata fields make gathering deterministic rather than requiring git archaeology or cross-repo searches:

| Field | On | Purpose |
|-------|-----|---------|
| `pattern: <id>` | Source doc frontmatter / entity metadata | Links concept content to its pattern |
| `patterns: [<ids>]` | Issue / session note metadata | Tags governance artifacts to patterns |
| `status: active \| retired` | Entity metadata | Filters stale/superseded content |
| `superseded_by: <id>` | Entity metadata | Tracks rename/promotion lineage |
| `promoted_at: <date>` | Pattern metadata | Avoids git archaeology for promotion timeline |

---

## Output Checklist

At the end of the analysis, you should have:

- [ ] Reference document: `docs/analysis/<pattern-id>-reference.md`
- [ ] Session note updated with work done
- [ ] GitHub issue progress comment
- [ ] Type 1 structural verdict (ACTIVE-COMPLETE / ACTIVE-INCOMPLETE / PENDING / CONCEPT)
- [ ] Type 2 semantic findings documented
- [ ] Remediation items captured (in-place fixes, spun-off issues, deferred items)
