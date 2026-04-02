# Hub Documents

> **Version:** 1.1.0
> **Status:** active
> **Last Updated:** 2026-01-10
> **Provenance:** 1P (extends 3P)
> **Derives From:** zettelkasten

Guide for authoring Hub documents that curate, summarize, and integrate atomic concepts.

---

## Overview

The **Hub** concept derives from the [Zettelkasten](https://en.wikipedia.org/wiki/Zettelkasten) ("slip box") method, a note-taking system developed by sociologist Niklas Luhmann. Zettelkasten organizes knowledge into atomic notes connected through "structure notes" that curate relationships. Our adaptation:

| Zettelkasten Term | Our Term | Source | Purpose |
|-------------------|----------|--------|---------|
| Atomic note (Zettel) | **Atom** | Derived (ingestion) | Single concept, canonical definition |
| Structure note | **Hub** | Human-authored | Curates related atoms, adds integration narrative |
| Index/Overview | **Structure** | Human-authored | Top-level navigation |

**Note:** In our implementation, atoms are **derived from ingestion and edge analysis**, not explicitly authored. Only Hubs and Structures are human-authored documents.

A **Hub** document is a curator and integrator of atomic concepts. It does NOT contain canonical definitions—those are derived during ingestion. Instead, it:

1. **Curates** - Declares which atoms belong to this hub
2. **Summarizes** - Provides truncated views of each atom for scanning
3. **Integrates** - Adds narrative showing how atoms fit together
4. **Routes** - Links to full atoms for depth

### Key Principle

**Atoms are stable. Hubs are scaffolds.**

Changing a Hub means reorganizing presentation, not changing concepts. Multiple Hubs can reference the same atom with different summary text tuned to context.

---

## Adoption Context

- **Original Capability:** `agentic-composition` — knowledge atom structuring for AI composition
- **Problem Statement:** Zettelkasten method (atomic notes, cross-linking) is powerful but lacks guidance for AI-assisted composition. How should knowledge atoms be structured so agents can compose them into coherent outputs?
- **Selection Criteria:** Apply zettelkasten principles to the SemOps content model: entities as composable atoms, edges as relationships, hub documents as curated aggregations. Bridges human knowledge organization and agent composition.
- **ADR:** —

## Template Structure

```markdown
---
entity_id: hub-{slug}
content_type: hub

metadata:
  preferred_label: "{Hub Name}"
  broader_concepts: [{parent-structure}]
  hub_type: methodology | framework | discipline | pattern-collection
  ownership: 1p | 3p

attribution:
  authors: ["{Author}"]
  epistemic_status: synthesis
  concept_ownership: 1p | 3p

# Explicit atom membership - the Hub's primary job
member_atoms:
  - atom_id: {concept-slug}
    summary: "{One-paragraph distillation for THIS hub's context}"
    role: core | supporting | reference
  - atom_id: {another-concept}
    summary: "{Context-specific summary}"
    role: core
---

# {Hub Name}

{LEAD PARAGRAPH - Standalone summary. Reader gets 80% of value here.
Should answer: What is this? Why does it matter? What's the key insight?
2-4 sentences max. No jargon without immediate explanation.}

**Key insight:** {Single sentence capturing the "aha"}

---

## Core Concepts

{Repeat this block for each CORE atom}

### {Concept Name}

{Summary paragraph - NOT the full atom content. Distilled for this Hub's context.
Should be scannable - reader decides if they need the full atom.}

**Full concept:** [{concept-slug}.md]({concept-slug}.md)

---

### {Next Concept Name}

{Summary paragraph}

**Full concept:** [{concept-slug}.md]({concept-slug}.md)

---

## Supporting Concepts

{Optional section for atoms that support but aren't central}

- **[{Concept}]({link})** - One-line description
- **[{Concept}]({link})** - One-line description

---

## How They Fit Together

{THIS IS THE HUB'S VALUE-ADD. Not in any single atom.
Narrative showing relationships, integration, the "why these together" story.
Can include diagrams, flow descriptions, synthesis insights.}

---

## Application

{Optional: How to use this hub's concepts in practice}

### When to Use

- {Situation 1}
- {Situation 2}

### When NOT to Use

- {Anti-pattern 1}
- {Anti-pattern 2}

---

## Key Takeaways

{Bullet list of main insights - compressed for quick reference}

1. {Takeaway 1}
2. {Takeaway 2}
3. {Takeaway 3}

---

## Related Hubs

- [{Related Hub}]({link}) - {One-line relationship description}
- [{Related Hub}]({link}) - {One-line relationship description}

---

## References

### Builds On (3P Foundations)

- [{3P Concept}]({link}) - {Relationship}

### See Also

- [{Related Atom}]({link})
- [{Related Pattern}]({link})
```

---

## Frontmatter Schema

### Required Fields

| Field | Type | Description |
|-------|------|-------------|
| `entity_id` | string | `hub-{slug}` format |
| `content_type` | string | Always `hub` |
| `metadata.preferred_label` | string | Human-readable hub name |
| `metadata.hub_type` | enum | `methodology`, `framework`, `discipline`, `pattern-collection` |
| `member_atoms` | array | List of atom memberships with summaries |

### Member Atoms Structure

```yaml
member_atoms:
  - atom_id: semantic-coherence      # References atom's entity_id
    summary: "Hub-specific summary"  # NOT the atom's canonical definition
    role: core                       # core | supporting | reference
```

**Roles:**
- `core` - Central to this hub, gets full summary section
- `supporting` - Important but secondary, gets bullet point
- `reference` - Mentioned for context, just linked

---

## Section Guidelines

### Lead Paragraph

- **Length:** 2-4 sentences
- **Purpose:** Standalone summary - reader should get 80% of value
- **Test:** Can someone understand what this hub is about WITHOUT reading further?
- **Include:** What, why, key insight
- **Avoid:** Jargon without explanation, references requiring context

### Core Concepts Sections

- **Length:** 1 paragraph per atom (3-5 sentences)
- **Purpose:** Scannable summary, NOT canonical definition
- **Include:** What it is, why it matters FOR THIS HUB
- **Always end with:** `**Full concept:** [link](link)`
- **Avoid:** Duplicating entire atom content

### How They Fit Together

- **Purpose:** Hub's unique value-add - synthesis NOT in any atom
- **Include:** Relationships, integration narrative, diagrams
- **This is editorial:** Can change without affecting atoms

### Key Takeaways

- **Length:** 3-7 bullet points
- **Purpose:** Quick reference, compressed insights
- **Format:** Actionable or memorable statements

---

## Example: Real Data Hub

```markdown
---
entity_id: hub-real-data
content_type: hub

metadata:
  preferred_label: "Real Data"
  broader_concepts: [semops-framework]
  hub_type: methodology
  ownership: 1p

member_atoms:
  - atom_id: physics-of-data
    summary: "Data has physical properties - storage, compute, latency. These constraints are non-negotiable and determine what's possible."
    role: core
  - atom_id: four-system-source-data
    summary: "All enterprise data comes from four system types, each with different physics and governance needs."
    role: core
  - atom_id: schema-equals-meaning
    summary: "Structure IS meaning. No schema means no semantics, regardless of what marketing says about 'schemaless.'"
    role: core
  - atom_id: data-literacy
    summary: "Leaders must understand data fundamentals to make informed decisions about AI transformation."
    role: supporting
---

# Real Data

Data is physical. It has mass (storage), requires energy (compute), and obeys laws (latency, consistency). "Real Data" is the recognition that data constraints are non-negotiable—you can trade off WHERE you pay costs, but you cannot eliminate them. This methodology grounds data strategy in physics rather than marketing promises.

**Key insight:** Schema equals meaning. No structure, no semantics—period.

---

## Core Concepts

### Physics of Data

Data isn't abstract—it's physical. Storage costs money. Queries consume compute. Latency is bounded by physics. Every "schemaless" system eventually pays the structure cost somewhere. Understanding these constraints separates realistic data strategy from wishful thinking.

**Full concept:** [physics-of-data.md](physics-of-data.md)

---

### Four Systems of Source Data

All enterprise data originates from four system types: Application (OLTP), Analytics (OLAP), Enterprise Work (unstructured), and Systems of Record (ERP/Finance). Each has different physics and requires different governance. The "data silos" problem is usually a category error—treating these four types as if they should unify.

**Full concept:** [four-system-source-data.md](four-system-source-data.md)

---

### Schema Equals Meaning

Structure IS semantics. When vendors promise "schemaless flexibility," they're deferring structure cost, not eliminating it. Eventually someone—query engine, developer, or analyst—must interpret the data, and that interpretation IS a schema. Better to be explicit upfront.

**Full concept:** [schema-equals-meaning.md](schema-equals-meaning.md)

---

## Supporting Concepts

- **[Data Literacy](data-literacy.md)** - Leaders need baseline understanding to make AI decisions
- **[RTFM Principle](rtfm-schema-self-enforcing.md)** - Systems have requirements; fighting them creates fragile hacks

---

## How They Fit Together

Real Data forms the **foundation layer** of the SemOps Framework. Before you can do Semantic Operations (maintaining meaning), you need to understand what data actually IS—its physical constraints, its source types, its structure requirements.

```
Physics of Data (constraints)
        ↓
Four System Types (categories)
        ↓
Schema = Meaning (principle)
        ↓
Ready for Semantic Operations
```

The four system types explain WHY "data integration" is hard—you're often trying to unify things with incompatible physics. Schema-equals-meaning explains WHY AI needs structured data—there's no meaning without structure.

---

## Key Takeaways

1. **Data is physical** - Constraints are non-negotiable, only tradeable
2. **Four system types** - Not all data is the same; stop treating it that way
3. **Schema = meaning** - No structure, no semantics
4. **Defer ≠ eliminate** - "Schemaless" just moves cost, doesn't remove it
5. **Foundation for AI** - Can't do semantic ops without understanding data physics

---

## Related Hubs

- [Semantic Operations](hub-semantic-operations.md) - Builds on Real Data foundation
- [Domain Driven Architecture](hub-domain-driven-architecture.md) - Encoding meaning into systems

---

## References

### Builds On (3P Foundations)

- [Information Theory](information-theory.md) - Shannon's work on signal/noise
- [DIKW Hierarchy](dikw.md) - Data → Information transformation

### See Also

- [Data System Glossary](data-system-glossary.md)
- [OLTP vs OLAP](oltp-olap.md)
```

---

## Validation Checklist

Before publishing a Hub document:

- [ ] Lead paragraph is standalone (passes the "80% value" test)
- [ ] All member atoms exist and are linked
- [ ] Summaries are Hub-specific, not copied from atoms
- [ ] "How They Fit Together" adds value beyond individual atoms
- [ ] Key Takeaways are actionable/memorable
- [ ] Related Hubs are linked
- [ ] 3P foundations are cited

---

## Anti-Patterns

### Hub as Atom Dump

**Wrong:** Copy-paste full atom content into Hub
**Right:** Write Hub-specific summaries, link to full atoms

### Hub Without Integration

**Wrong:** Just list atoms with no "How They Fit Together"
**Right:** Explain relationships, add synthesis narrative

### Orphan Summaries

**Wrong:** Summary in Hub differs semantically from atom's definition
**Right:** Summary is a contextual VIEW of the same concept

### Over-Stuffed Hub

**Wrong:** 20+ core concepts in one Hub
**Right:** 3-7 core concepts; split into sub-hubs if larger

---

## Related Documentation

- **[content-compose-pattern.md](content-compose-pattern.md)** - Three-layer model (Atoms → Hubs → Structures)
- **[structure-template.md](structure-template.md)** - Template for top-level structures

**Note:** There is no `atom-template.md` because atoms are derived from ingestion, not authored.

---

## 3P Foundation

This pattern extends **Zettelkasten**, a note-taking and knowledge management method:

- **Origin:** Niklas Luhmann (1927-1998), German sociologist
- **Core Principle:** Knowledge emerges from connections between atomic notes
- **Key Innovation:** Structure notes that curate and integrate atomic content
- **Our Extension:** Formalized Hub/Atom/Structure hierarchy with YAML frontmatter

**Further Reading:**
- [Zettelkasten on Wikipedia](https://en.wikipedia.org/wiki/Zettelkasten)
- [content-compose-pattern.md](content-compose-pattern.md) - Full Atom→Hub→Structure model

