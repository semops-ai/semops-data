# System Context: Project Ike Design Philosophy

This document captures the core thesis and design philosophy that drives Project Ike. It serves as foundational context for both human collaborators and AI agents working within this system.

---

## Core Definitions

### Concept

**Definition:** A stable semantic unit representing durable identity, ideas, principles, or intellectual property. Concepts are the aggregate root of the system - they persist even when all content artifacts referencing them are deleted.

**Key characteristics:**
- Represents "what you know and think" not "what you've made"
- Has identity independent of any artifact (blog post, video, doc)
- Connected to other concepts via SKOS relationships (broader, narrower, related)
- Can be referenced by many entities (artifacts) simultaneously
- Governed through approval workflow (pending → approved → rejected)

**Why Concept is Root:** A blog post about "semantic coherence" is ephemeral - it can be rewritten, deleted, reformatted. The concept "semantic coherence" is stable - it's core identity. Multiple assets can reference the same concept. The concept survives even if all the assets are deleted.

### Provenance (1p/2p/3p)

Provenance answers: **"Whose semantic structure is this?"**

#### 1p (First Party) - "Operates in my system"

**Definition:** A semantic structure that now operates in your system as part of your identity and decision-making framework.

**Key insight:** `1p` does NOT mean "I invented this." It means this concept is now incorporated into how you think and operate. A synthesis from 3p sources becomes 1p when you've incorporated it into your operational framework.

**Examples:**
- `semantic-coherence` - original thinking, core IP
- `semantic-operations` - a synthesis of DIKW + DDD + W3C standards, now your methodology
- `real-data` - your original framing/distinction

**When something becomes 1p:**
- You've incorporated it into your operational framework
- It influences your decisions and outputs
- You maintain and evolve its definition
- It's part of "what you know and think"

#### 2p (Second Party) - Partnership/Collaborative

**Definition:** A semantic structure jointly developed with an external party where ownership and evolution is shared.

**Examples:**
- Concepts developed in collaboration with partners
- Methodologies co-created with clients
- Frameworks built with LLM collaboration where both parties contribute

**Characteristics:**
- Shared governance over the definition
- Attribution to multiple parties
- Evolution requires coordination

#### 3p (Third Party) - External Reference

**Definition:** An external semantic structure from industry, academia, or other sources that you reference but haven't incorporated into your operational framework.

**Examples:**
- `bounded-context` - Eric Evans / DDD community
- `skos` - W3C standard
- `dikw-model` - Russell Ackoff / knowledge management field

**Characteristics:**
- You don't control or evolve the definition
- You reference it, cite it, or build upon it
- Attribution goes to the external source
- May become 1p if you synthesize and incorporate it

### The Provenance Lifecycle

```
3p (external reference)
    ↓ synthesis/incorporation
1p (operates in my system)
    ↓ partnership
2p (collaborative evolution)
```

### Key Distinction: Concept Provenance vs Entity Attribution

- **Concept provenance (1p/2p/3p):** Whose semantic structure is this?
- **Entity attribution (Dublin Core):** Who made this specific artifact?

A transcript you create from a 3p YouTube video:
- The **entity** (transcript) has `attribution.creator = ["Tim Mitchell"]`
- The **concept** the video explains might be `provenance = "3p"` (external idea)
- If you synthesize insights into your own framework, that becomes a **new 1p concept**

---

## Ontology (Project Ike Specific)

**Tim Mitchell** - Everything rolls up to me (aggregate root owner)

**Owned Surfaces (Publishing):**
- GitHub repos under Timjmitchell account
- semops-ai.com, semops.com
- LinkedIn, social platforms

**Owned Bio:** Work experience, resume, skills, portfolio

**Owned Publishing Entities:** Blog posts, repo docs/apps/code, YouTube videos, LinkedIn posts, social posts

---

## The Thesis

Project Ike is a **simulation of an organization** designed to test a theory:

> An organization will struggle to get the most out of data and AI the same way a multi-agentic system will struggle if there isn't both **autonomy** AND a **shared context with orchestration**.

The goal is to build a system optimized for humans and agents to work from a **coherent context substrate** - a stable semantic foundation that enables aligned decision-making and effective AI collaboration.

---

## Why Concept is the Root

The concept graph represents **durable identity** - professional, intellectual, or organizational. Everything else is supporting infrastructure.

**The insight:** A blog post about "semantic coherence" is ephemeral (can be rewritten, deleted, reformatted). The concept "semantic coherence" is stable (it's core IP, core identity). Multiple assets can reference the same concept. The concept survives even if all the assets are deleted.

**This scales:**

```
Today (solo professional):
  Concept (ideas, expertise, IP)
      └── DAM (blog posts, videos, docs)

Future (company):
  Concept (same root, unchanged)
      ├── DAM (content)
      ├── Publishing (multi-channel)
      ├── CRM (relationships)
      └── PIM (products/services)
```

The root holds because **what you know and think** doesn't change when you add business functions around it. The concepts are stable; everything else is packaging, delivery, or monetization.

**Note - This scales to any organization:** Consider Amazon. If they DDD'd their entire universe, the root would be the Leadership Principles - "Customer Obsession", "Ownership", "Invent and Simplify", etc. Universal practices like "PR/FAQ" and "6-Pager" are representations of those principles - practice artifacts that operationalize the concepts. CEOs change, products come and go, entire business units get created or shut down - but the principles persist. The concept graph encodes *identity* and *belief*, not just content.

---

## Adopt Patterns, Don't Invent

A core practice: instead of building features, **apply whole patterns**. Start with the best standard, time-tested pattern that fits.

| Domain Type           | Description                         | Pattern Source                            |
| --------------------- | ----------------------------------- | ----------------------------------------- |
| **Core Domain**       | Your differentiator - innovate here | Your original thinking                    |
| **Supporting Domain** | Necessary but not differentiating   | Adopt industry patterns (DAM, publishing) |
| **Generic Domain**    | Commodity                           | Buy or copy wholesale (auth, CRM, PIM)    |

**Current pattern adoptions:**
- **SKOS** (W3C) - concept relationships
- **PROV-O** (W3C) - lineage and provenance
- **DAM** (industry) - digital asset management
- **Dublin Core** - attribution metadata

You only innovate in the core. Everything else, you adopt proven patterns. This is the `rtfm-principle` applied to system design.

---

## The Meta-Design

Everything in this project is **intentionally meta**. We're designing with "extreme overkill" because we're testing our own theory:

- **Semantic Operations** is both the methodology AND the product
- **The schema** encodes the principles it represents
- **The process** of building this system IS the validation of the theory

If the theory is correct, this system should enable:
1. Faster, more aligned collaboration between humans and AI
2. Less semantic drift and reconciliation overhead
3. Clear lineage showing WHY decisions were made and what trade-offs existed

---

## Core Principles

### 1. Stable Core, Flexible Edge

The system has two zones:

**Stable Core (Concepts)**
- The aggregate root - nothing exists without connection to it
- Governed, versioned, carefully evolved
- Provides the shared context substrate
- Enables coherent AI grounding

**Flexible Edge (Orphans)**
- Where new ideas emerge before formal incorporation
- "Make a mess, go fast" is allowed here
- Exploratory content that hasn't been assigned to concepts
- Audit processes eventually promote or reject

### 2. Semantic Coherence as Target State

From `semantic-coherence.md`:
> "A state of stable, shared semantic alignment between agents (human + machine) that enables optimal data-driven decision making including with AI."

Coherence requires:
- **Availability**: Can agents find the meaning they need?
- **Consistency**: Do different agents interpret concepts the same way?
- **Stability**: Does meaning stay constant without drift?

### 3. Semantic Optimization as Process

From `semantic-optimization.md`:
> "The process of maintaining stable coherence between agents (human + machine) while spurring growth through new patterns and change."

The tension:
- **Growth requires mess**: New ideas, experimentation, speed
- **Coherence requires stability**: Shared definitions, governance, alignment
- **Optimization balances both**: Structured processes to incorporate the good and reject the noise

### 4. Lineage Captures Decisions

Every significant change should capture:
- **What** changed
- **Why** it changed (the reasoning)
- **What trade-offs** were considered
- **Who/what** made the decision (human, agent, or collaboration)

This creates an auditable trail that enables:
- Future agents to understand context
- Humans to review AI-assisted decisions
- The system to learn from its own evolution

---

## Architectural Implications

### Concept as Aggregate Root

**Decision:** Concepts are the aggregate root. Nothing exists in the system without a connection to the concept graph.

**Rationale:**
- Forces intentionality - content must be semantically contextualized
- Creates the "shared context substrate" the thesis requires
- Enables coherent AI grounding - agents work from the same concept definitions

**Handling the Edge:**
- **Orphan content** exists outside the aggregate temporarily
- Orphans float at the flexible edge until ready to be incorporated or rejected
- Audit processes (human or AI-assisted) review orphans periodically
- Promotion into the concept graph = acceptance into the stable core

### DDD as Foundation

Domain-Driven Design provides the structural patterns:
- **Bounded contexts** scope semantic coherence
- **Ubiquitous language** ensures shared vocabulary
- **Aggregates** enforce invariants
- **Context maps** manage translation at boundaries

This isn't just methodology preference - it's the encoding of the theory into operational practice.

---

## For AI Agents

When working in this system:

1. **Concepts are primary** - understand the concept graph before modifying content
2. **Definitions matter** - use the exact definitions from concepts, don't paraphrase
3. **Edges encode relationships** - SKOS predicates (broader, narrower, related) have specific meanings
4. **Provenance tracks whose semantic structure** - 1p (operates in this system), 2p (partnership), 3p (external reference)
5. **Orphans are temporary** - unconnected content should be flagged for review, not left indefinitely
6. **Attribution is separate from provenance** - Entity attribution (Dublin Core) tracks who made the artifact; concept provenance tracks whose idea it is

---

## Current State

As of 2025-12-06:
- ~118 atoms defined in COMPOSABLE_CONCEPTS
- Atoms have frontmatter with SKOS relationships
- **phase2-schema.sql** implements concept-as-aggregate-root architecture
- **UBIQUITOUS_LANGUAGE.md** v6.0.0 reflects the new model
- Entity table simplified to standard DAM pattern (no provenance column)
- Provenance (1p/2p/3p) is now on Concept only

**Next steps:**
- Define orphan staging area
- Build ingestion that enforces the model
- Migrate existing atoms to concept table

---

## Related Documents

- [UBIQUITOUS_LANGUAGE.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/UBIQUITOUS_LANGUAGE.md) - Domain term definitions (canonical)
- [GLOBAL_ARCHITECTURE.md](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/GLOBAL_ARCHITECTURE.md) - System map and repo roles
- [phase2-schema.sql](../schemas/phase2-schema.sql) - Current schema (Pattern as aggregate root)
- [ADR-0004: Schema Phase 2](decisions/ADR-0004-schema-phase2-pattern-aggregate-root.md) - Architecture decision record
