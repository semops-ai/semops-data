# Framings and Lenses

> **Version:** 1.0.0
> **Status:** active
> **Last Updated:** 2026-01-10
> **Provenance:** 1P
> **Derives From:** dikw-framework, ddd, prov-o

Composable interpretive layers that can be applied to any atom to show how concepts map to theoretical frameworks (Framings) or practical implications (Lenses).

---

## Overview

This pattern draws on established theoretical frameworks:

- **DIKW Framework** - Data→Information→Knowledge→Wisdom hierarchy for mapping concept levels
- **DDD (Domain-Driven Design)** - Bounded contexts, aggregates for structural mapping
- **PROV-O (W3C Provenance Ontology)** - Lineage and derivation relationships

Atoms define **what a concept IS**. Framings and Lenses define **how to see it** through different perspectives.

This separation enables:
- Atoms stay pure/stable (the concept itself)
- Interpretive layers are reusable across atoms
- Different contexts can apply different framings/lenses
- Hubs can compose atoms + choose which perspectives to include

---

## Adoption Context

- **Original Capability:** `concept-documentation` — structured explanation of complex concepts
- **Problem Statement:** Complex concepts (DDD, patterns, capabilities) need structured explanation frameworks — not just definitions but context-setting narratives that connect foundational theory to SemOps application.
- **Selection Criteria:** Templates for explaining: (1) foundational 3P concept, (2) 3P origins, (3) 1P innovation/application, (4) related concepts. Ensures coherent knowledge presentation across the corpus.
- **ADR:** —

## Framings vs Lenses

| Aspect | Framing | Lens |
|--------|---------|------|
| **Purpose** | Map to formal theoretical framework | Show practical implications |
| **Question** | "Where does this sit in [framework]?" | "What does this mean for [context]?" |
| **Nature** | Formal, theoretical, structural | Practical, applied, "so what" |
| **Examples** | DIKW, DDD, PROV-O | AI Transformation, SemOps, Governance |

---

## Standard Framings

### DIKW Framing

Maps concept to another concept, maps the components of concept to Data → Information → Knowledge → Wisdom hierarchy.

**Template:**
```markdown
## DIKW Framing

**Level:** [D / I / K / W / transition between]

**Mapping:**
- This concept operates at the [X] level because...
- It enables the [X → Y] transformation by...

**Energy implications:**
- [What work/energy does this concept require or reduce?]
```

**Example (Source Problem):**
```markdown
## DIKW Framing

**Level:** D → I transition failure

**Mapping:**
The source problem occurs when Data is captured without the structure
needed to become Information. Without semantic definition at capture,
data remains ambiguous - it never cleanly transitions to Information.

**Energy implications:**
Downstream systems expend massive energy trying to reconstruct meaning
that should have been captured at source. This is wasted transformation
energy - paying the D→I cost repeatedly instead of once.
```

---

### DDD Framing

Maps concept to Domain-Driven Design patterns.

**Template:**
```markdown
## DDD Framing

**Pattern alignment:** [Bounded Context / Aggregate / Entity / Value Object / etc.]

**Mapping:**
- This concept relates to [DDD pattern] because...
- In DDD terms, this is about...

**Boundary implications:**
- [How does this affect context boundaries?]
```

---

### PROV-O Framing

Maps concept to W3C Provenance Ontology.

**Template:**
```markdown
## PROV-O Framing

**Provenance type:** [Generation / Derivation / Attribution / etc.]

**Mapping:**
- This concept relates to provenance as...
- Lineage implications...
```

---

## Standard Lenses

### AI Transformation Lens

What this concept means for AI/ML initiatives.

**Template:**
```markdown
## AI Transformation Lens

**Impact:** [How does this affect AI success/failure?]

**Without this:**
- [What goes wrong for AI initiatives?]

**With this:**
- [What becomes possible?]

**Practical implication:**
- [Specific guidance for AI teams]
```

**Example (Source Problem):**
```markdown
## AI Transformation Lens

**Impact:** The source problem is why most AI initiatives fail at data quality.

**Without solving this:**
- AI models train on inconsistent, ambiguous data
- "Garbage in, garbage out" at scale
- No amount of model sophistication compensates

**With this solved:**
- Clean semantic foundation for training data
- Consistent meaning across data sources
- AI can trust its inputs

**Practical implication:**
Before any AI initiative, audit source ownership. If nobody owns
the semantic definition of key events, fix that first.
```

---

### SemOps Lens

How this concept fits within the Semantic Operations framework.

**Template:**
```markdown
## SemOps Lens

**Framework role:** [Where does this sit in SemOps?]

**Enables:**
- [What SemOps capabilities does this unlock?]

**Depends on:**
- [What other SemOps concepts does this require?]

**Coherence impact:**
- [How does this affect semantic coherence?]
```

---

### Governance Lens

Implications for data/semantic governance.

**Template:**
```markdown
## Governance Lens

**Governance dimension:** [Ownership / Policy / Compliance / etc.]

**Controls needed:**
- [What governance mechanisms does this require?]

**Risk if ungoverned:**
- [What happens without governance?]
```

---

### Implementation Lens

Practical implementation considerations.

**Template:**
```markdown
## Implementation Lens

**Technical pattern:**
- [How is this implemented?]

**Common mistakes:**
- [What do teams get wrong?]

**Success indicators:**
- [How do you know it's working?]
```

---

## Usage in Atoms

Framings and Lenses are **optional sections** in atoms. Include when relevant:

```markdown
# Concept Name

**Definition:** ...

---

## [Core content sections]

---

## DIKW Framing

[If concept has meaningful DIKW mapping]

---

## AI Transformation Lens

[If concept has specific AI implications]

---

## SemOps Lens

[If concept has specific framework role]

---

## Related Links
```

**Guidelines:**
- Not every atom needs every framing/lens
- Include when it adds insight, skip when it's forced
- Keep each framing/lens section concise (3-7 sentences)
- Link to related concepts surfaced by the framing/lens

---

## Usage in Hubs

Hubs can:
1. **Inherit:** Include atom's framings/lenses in summary
2. **Add:** Apply framings/lenses the atom doesn't have
3. **Contextualize:** Reframe for the hub's specific context

**Example:**
```markdown
### Source Problem

**Summary:** [Hub-specific summary]

**DIKW Framing:** Operates at D→I transition failure...

**Full concept:** [source-problem.md](source-problem.md)
```

---

## Creating New Framings/Lenses

When a new interpretive perspective emerges:

1. **Is it formal/theoretical?** → Framing
2. **Is it practical/applied?** → Lens
3. **Document the template** in this file
4. **Apply consistently** across relevant atoms

---

## Related Documentation

- **[guide-zettelkasten-hubs.md](guide-zettelkasten-hubs.md)** - How hubs compose atoms with framings/lenses
- **[content-compose-pattern.md](content-compose-pattern.md)** - Three-layer composition model

**Note:** Atoms are derived from ingestion, not authored. Framings/lenses are applied during hub composition.

---

## 3P Foundation

| Framework | Usage in This Pattern |
|-----------|-----------------------|
| **DIKW** | Framing template for mapping concept levels |
| **DDD** | Framing template for bounded contexts and aggregates |
| **PROV-O** | Framing template for provenance and lineage |

**Our Extension:** Framings (theoretical mapping) vs Lenses (practical implications) separation, plus project-specific lenses (AI Transformation, SemOps, Governance).

