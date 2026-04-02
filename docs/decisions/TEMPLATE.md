# ADR-NNNN: Title

> **Status:** Draft | Decided | Superseded
> **Date:** YYYY-MM-DD
> **Related Issue:** [repo#NN](url) _(if applicable)_
> **Builds On:** [ADR-NNNN](./ADR-NNNN-slug.md) _(if applicable)_
> **Design Doc:** [DD-NNNN](../design-docs/DD-NNNN-slug.md) _(if design detail exists)_

---

## Executive Summary

_2-3 sentences: what was decided and why it matters._

---

## Context

_What problem or opportunity prompted this decision? What forces are in tension? Include enough background for someone unfamiliar with the history to understand why a decision was needed._

_This section should make the reader say "yes, something had to be decided here."_

---

## Decision

_What was decided. Be specific about the approach chosen and the alternatives rejected._

_Keep this section focused on the **choice and rationale**, not the design detail. If the design needs schemas, stages, component specs, YAML examples, or diagrams — that belongs in a Design Doc, linked above._

_For multiple related decisions, use numbered subsections (D1, D2, ...) with a one-line summary each._

---

## Consequences

_What follows from this decision — both good and bad._

**Positive:**
- Benefit 1
- Benefit 2

**Negative:**
- Trade-off 1
- Trade-off 2

**Risks:**
- Risk 1 — mitigation
- Risk 2 — mitigation

---

## Pattern and Capability Impact

_Run `/intake` against this ADR to check alignment with the domain model. If this decision introduces, modifies, or deprecates patterns or capabilities, note them here._

| Type | ID | Impact | Action |
|------|----|--------|--------|
| Pattern | `pattern-id` | Introduces / Extends / Deprecates | Register / Update / No action |
| Capability | `capability-id` | New / Modified / Removed | Register / Update / No action |

_If no pattern or capability impact, state "None identified" and note that `/intake` was run._

---

## References

- [Related ADR](url)
- [Design Doc](url)
- [External Resource](url)

---

## Template Usage Notes

_(Delete this section when creating a real ADR)_

**File naming:** `ADR-NNNN-short-description.md`

**Status values:**
- **Draft** — decision proposed, not yet accepted
- **Decided** — decision accepted and in effect
- **Superseded** — replaced by a later ADR (add `Superseded By:` to frontmatter)

**What belongs here vs. in a Design Doc:**

| Here (ADR) | Design Doc |
|------------|-----------|
| Why we decided | What we are building |
| Alternatives considered | Architecture, schemas, stages |
| Consequences and trade-offs | Component specs, YAML examples |
| 1-2 pages typical | As long as needed |

**When to create a Design Doc:**
- The Decision section is growing past ~2 paragraphs of design detail
- You need diagrams, schemas, or code examples to explain the design
- The design will evolve independently of the decision rationale

**Sections removed from previous template:**
- **Implementation Plan** — belongs in project specs or issues
- **Session Log** — belongs in session notes (`docs/session-notes/ISSUE-NN-*.md`)
