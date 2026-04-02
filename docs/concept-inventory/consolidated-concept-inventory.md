# Consolidated Concept Inventory

> Unified inventory of all concepts across Project Ike documentation with atom coverage analysis.

**Generated:** 2025-12-05 | **Source Folders:** 6 | **Existing Atoms:** 72

---

## Summary

| Metric | Count |
|--------|-------|
| Total concepts identified | 371 |
| Unique concepts (deduplicated) | ~180 |
| Existing atoms | 72 |
| Missing atoms (estimated) | ~108 |
| 1p concepts | ~130 |
| 3p concepts | ~50 |

---

## Existing Atoms (72)

These concepts already have atoms in `COMPOSABLE_CONCEPTS/`:

### 1p Atoms (42)
// I'm noticing an "organization" cross-cutting component, which i'll flag to figure out (people/roles vs. tech)
// I will tend not to include too much addnl laddering to the "Semantic Optimization" Framework, because everything does at some level and has the most additional content to come
| Concept | Broader Concepts |
|---------|------------------|
| ai-forward-organization | - |
| ai-is-analytics | - |
| ai-ready-architecture | - | not 1p
| ai-wants-schema | - |  //why-structure-matters
| analytics-capabilities | - | //what is definition? not sure its atom
| analytics-data-system | four-system-source-data | not 1p
| analytics-engineer-role | - | not 1p
| application-data-system | four-system-source-data | not 1p
| autonomy-without-coherence | - | //Semantic Optimization/Real Data 
| business-analytics-patterns | - | //analytics, data-systems 
| business-model-analytics | - | // not 1p - just a categorization summary
| data-is-physical | - | //real data, DIKW Mental, data systems
| data-pipelines-explained | - | //not a concept, its just documentation of pipeline methods
| data-product-manager | - |  //SemOps Framework, AI Transformation Problem,"organization"
| data-silo-confusion | - | //data systems, AI transformation problem
| dimensional-modeling-skills-gap | - | //real data "organization", AI Transformation "organization"
| enterprise-data-confusion | - | //AI transformation problem, real data, domain-driven-arch
| enterprise-record-system | real-data, four-system-source-data | //AI Transformation
| enterprise-work-system | four-system-source-data | //real data/ai transformation
| epistemic-uncertainty | uncertainty |//first principles
| finance-as-semantic-model | - | //Ai transformation, DDD, semantic compression
| four-system-source-data | data-systems, real-data |
| ike-bottom-up | - | // can kill
| ike-framework | - | // this is old name... now "Semantic Operations Framework"
| ingestion-boundary | - | //domain-driven-architecture, real-data
| instrumentation | - | //real-data, source-up-thinking, seven-capabilities
| instrumentation-ownership | - | // source-up-thinking, organization, real-data, domain-driven-architecture
| lineage-ike | governance-as-a-strategy | //data-systems
| organizational-semantic-failures | - | //domain-driven-architecture, the-ai-transformation-problem, dikw-mental-model
| project-ike | - | //kill .. now Project SemOps
| prompt-engineering-debt | - | //lineage, semantic-coherence
| real-data-literacy | - | //real-data
| role-fragmentation | - | //i think its probably "data-role-fragmentaion"  - if that's the case, add "organizational"
| semantic-gravity | - | //semantic-optimization, semantic-operations
| source-problem | real-data, semantic-operations //silos-problem
| source-up-thinking | - | //real-data, seven-capabilities, data-system-classification, silos-problem
| structural-uncertainty | - |// 3p, from philosophy
| surface | - | // not 1p
| surface-system-analytics | - | //this is more of a hub of source-up-thinking and data-system-classification
| three-forces-data-systems | data-systems |
| streaming-vs-batch | data-systems | //not 1p, not atom
| understanding | dikw-mental-model | 

### 3p Atoms (28)

| Concept | Source |
|---------|--------|
| abstractions | - | //there are actually two atoms underneath this - epistemic and ontological
| aggregate-entity-value-object | DDD (Evans) |
| aggregate-root | DDD (Evans) |
| anti-corruption-layer | DDD (Evans) |
| bounded-context | DDD (Evans) |
| context-map | DDD (Evans) |
| data-profiling | Data Engineering |
| domain-driven-design | Eric Evans |
| entity | DDD (Evans) |
| industry | - | //too general to be atom
| intention-revealing-interfaces | DDD (Evans) |
| model-driven-development | Software Engineering |
| oltp-olap | Data Systems |
| process-mining | Data Engineering |
| provenance | W3C PROV-O |
| semantic-drift | - |//semantic-operations
| specification-pattern | DDD (Evans) |
| strategic-design | DDD (Evans) |
| supple-design | DDD (Evans) |
| ubiquitous-language | DDD (Evans) |
| value-object | DDD (Evans) |

---

## Missing Atoms by Priority

### Priority 1: Core Framework Concepts (1p, multi-folder)

These 1p concepts appear in 3+ folders and need atoms urgently:

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| semantic-coherence | DDA, AI_TRANS, REAL_DATA, SEMOPS, FP, FRAMEWORK | Degree to which semantic components work together to produce aligned understanding |
| semantic-operations | DDA, REAL_DATA, SEMOPS, FP, FRAMEWORK | Systematic approach to managing knowledge assets as operational infrastructure |
| semantic-optimization //this is obviously critical and needs a solid definition | SEMOPS, FP, FRAMEWORK | Ongoing measurement and improvement of shared understanding at runtime |
| real-data | REAL_DATA, FP, FRAMEWORK | Framework emphasizing data's physical and mathematical properties |
| domain-driven-architecture | DDA, FRAMEWORK | Applied DDD at organizational scale |
| model-everything | DDA, REAL_DATA | Cross-cutting tactical method for modeling through data structures |
| dikw-hierarchy //this is probably duplicate of 3p "dikw" | REAL_DATA, FP, FRAMEWORK, AI_TRANS | Mental model: Data → Information → Knowledge → Wisdom |

### Priority 2: AI Transformation Concepts (1p)

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| regression-paradox | AI_TRANS, FRAMEWORK | LLMs regress to the mean, effective for narrow problems, ineffective for transformation |
| runtime-emergence | AI_TRANS, FRAMEWORK | Transformation only exists during active operation |
| semantic-decoherence | AI_TRANS, DDA, FRAMEWORK | Natural tendency of shared meaning to decay |
| hard-problem-of-ai | AI_TRANS, FRAMEWORK | Why good AI implementation doesn't create better outcomes |
| paradigmatic-lock-in //3p | AI_TRANS, DDA | Existing structures absorb AI's transformative potential |
| measurement-fallacy //3p | AI_TRANS, DDA | Factory-floor ROI models filter out transformative projects |
| collaborative-intelligence //not 1p | AI_TRANS | Human-AI teammates in synergistic partnership |

### Priority 3: Semantic Operations Concepts (1p)

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| semantic-availability //dupe | SEMOPS | Degree semantic info is discoverable and accessible |
| semantic-stability //dupe | SEMOPS, REAL_DATA | Degree meaning remains consistent over time |
| semantic-consistency //dupe | SEMOPS | Degree different systems interpret concepts the same way |
| knowledge-corpus | SEMOPS | Source of truth: domain models, glossaries, ontologies |
| knowledge-artifacts | SEMOPS | Real-world manifestations: dashboards, reports, code |
| corpus-artifact-delta | SEMOPS, FRAMEWORK | Gap between canonical definitions and real-world usage |
| progressive-semantics | SEMOPS | Knowledge promotion model integrating SemOps, DDD, governance |
| knowledge-promotion //if semantic promotion exists, it will be that, not this | SEMOPS | Process of elevating concepts from flexible edge to hard schema |
| flexible-edge //3p | SEMOPS | Frontier of learning with low structure, high ambiguity |
| hard-schema //3p | SEMOPS | High-confidence layer where concepts solidify |
| semops-lifecycle //dupe | SEMOPS | Six-step process from capture to stewardship |

### Priority 4: Real Data Concepts (1p)

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| schema-equals-meaning | REAL_DATA | Structure IS semantics; schema encodes business logic |
| seven-core-capabilities | REAL_DATA | All platforms offer: Ingestion, Storage, Table Layer, etc. |
| three-forces | REAL_DATA | Volume & Velocity, Latency, Structure & Variability |
| source-instrumentation | REAL_DATA | Capturing events at origin with complete semantic context |
| source-ownership | REAL_DATA | Clear organizational accountability at data capture point |
| system-up-thinking | REAL_DATA, DDA | Broken approach starting with existing systems |
| no-free-lunch-data //not atom | REAL_DATA | Someone must inject meaning; deferring makes it more expensive |
| thermodynamics-of-data | REAL_DATA | Information requires structure, structure requires energy |
| deterministic-transformations | REAL_DATA, FP | Data operations are mathematically predictable |
| rtfm-principle //3p| REAL_DATA, FP | Systems have requirements; fighting them creates fragile hacks |
| self-validating-systems //3p| REAL_DATA | Dimensional models enforce correctness through constraints |
| semantic-layer //3p| REAL_DATA | Formalized structure with explicit entity definitions |

### Priority 5: DDD Patterns (3p) - Context Map Variations

| Concept | Source | Definition Snippet |
|---------|--------|-------------------|
| context-map-partnership | DDD (Evans) | Two teams coordinate closely with equal power |
| context-map-shared-kernel | DDD (Evans) | Two contexts share subset of domain model |
| context-map-customer-supplier | DDD (Evans) | Downstream depends on upstream |
| context-map-conformist | DDD (Evans) | Downstream accepts upstream model wholesale |
| context-map-open-host-service | DDD (Evans) | Upstream provides well-defined protocol |
| context-map-published-language | DDD (Evans) | Common language for integration |
| context-map-separate-ways | DDD (Evans) | No relationship, operate independently |

### Priority 6: First Principles Concepts (3p foundations)

| Concept | Source | Definition Snippet |
|---------|--------|-------------------|
| information-theory | Shannon | Mathematical theory of communication |
| systems-theory-cybernetics | Various | Science of feedback, control, system behavior |
| sociotechnical-systems | Trist | How social and technical systems co-evolve |
| schema-theory-constructivism | Piaget | How humans structure information into mental schemas |
| relational-model | Codd | Mathematical foundation of relational databases |
| entity-relationship-model | Chen | Conceptual framework for data modeling |
| dimensional-modeling | Kimball | Star schemas for analytical queries |
| formal-ontology | W3C | Machine-interpretable concept definitions |
| knowledge-representation | AI/Philosophy | How knowledge should be structured for reasoning |

### Priority 7: Understanding Concepts (1p)

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| understanding-as-process | FP | Dynamic process, not static entity |
| understanding-components | FP | Causal, Explainable, Anticipatory, Application, Shared |
| causal-understanding | FP | Understanding the foundational why |
| shared-understanding | FP, SEMOPS | Organizational property, not individual |
| energy-barrier-problem | FP | Why organizations struggle to achieve understanding at scale |
| suspended-understanding | FP, SEMOPS | AI holds massive context while humans inject meaning |
| understanding-maturity-ladder | FP | Five levels from Perception to Collective Effectiveness |

### Priority 8: Architecture Concepts (1p)

| Concept | Folders | Definition Snippet |
|---------|---------|-------------------|
| stable-core-flexible-edge | DDA, SEMOPS | Semantically identical core, context-specific edge |
| semantic-containers | DDA | Bounded contexts as semantic containers |
| semantic-governance | DDA, SEMOPS | All teams use same terminology |
| semantic-contracts | DDA | Context maps define translation rules |
| intentional-architecture | DDA | Architecture must be deliberately chosen |
| semantic-invariants | DDA | Domain truths that must hold regardless of context |
| global-architecture | FRAMEWORK, FP | Systems, organization, product semantically aligned |

---

## Cross-Folder Overlap Analysis

Concepts appearing in 4+ folders (high integration):

| Concept | DDA | AI_TRANS | REAL_DATA | SEMOPS | FP | FRAMEWORK |
|---------|-----|----------|-----------|--------|-----|-----------|
| semantic-coherence | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| semantic-operations | ✓ | - | ✓ | ✓ | ✓ | ✓ |
| bounded-context | ✓ | ✓ | ✓ | ✓ | - | ✓ |
| ubiquitous-language | ✓ | - | ✓ | ✓ | - | ✓ |
| semantic-drift | ✓ | ✓ | ✓ | ✓ | ✓ | ✓ |
| dikw | - | ✓ | ✓ | ✓ | ✓ | ✓ |
| domain-driven-design | ✓ | ✓ | ✓ | ✓ | ✓ | - |

---

## Ownership Classification Notes

### Corrections from HITL Review

| Concept | Agent Classified | Corrected To | Reason |
|---------|-----------------|--------------|--------|
| paradigmatic-lock-in | 1p | 3p | Child of AI Transformation Problem (industry concept) |
| measurement-fallacy | 1p | 3p | Tied to Hard Problem of Consciousness (Chalmers) |
| model-everything | DDA child | Cross-cutting | Applies across Real Data, DDA, AI Transformation |
| semantic-drift | 1p | 3p | Industry standard term |

### Hierarchy Notes

- **L1 Framework Structure is canonical** (Problem Space / Solution Space)
- **Folder structure = metadata hint**, not semantic hierarchy
- **Most concepts are networked** (edges), not hierarchical (broader_concepts)
- **Context-map patterns ARE hierarchical** under context-map → bounded-context

---

## Next Steps

1. **Review this inventory** - Validate ownership, priority, definitions
2. **Create Priority 1 atoms** - Core framework concepts appearing everywhere
3. **Create Priority 2-3 atoms** - AI Transformation + SemOps concepts
4. **Wire edges** - Connect atoms via predicates
5. **Compose Hubs** - Build hub documents curating related atoms

---

## Appendix: Folder Scan Statistics

| Folder | Files Scanned | Total Concepts | 1p | 3p |
|--------|---------------|----------------|----|----|
| DOMAIN_DRIVEN_ARCHITECTURE | 11 | 68 | 41 | 27 |
| AI_TRANSFORMATION | 9 | 37 | 15 | 22 |
| REAL_DATA | 13 | 90 | 50 | 40 |
| SEMANTIC_OPERATIONS | 12 | 60 | 48 | 12 |
| FIRST_PRINCIPLES | 10 | 77 | 50 | 27 |
| FRAMEWORK | 6 | 39 | 26 | 13 |
| COMPOSABLE_CONCEPTS | 72 | 72 | 42 | 28 |

---

**Document Status:** Generated | **Next Action:** Human review
