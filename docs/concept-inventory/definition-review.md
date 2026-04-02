# Atom Definition Review

Organized bottom-up from most granular to parent concepts.

**Status:** In-progress review session (2025-12-05)

---

## 1. First Principles (3p - foundational theories)

| Atom                           | Definition                                                                                                                             | Action                            |
| ------------------------------ | -------------------------------------------------------------------------------------------------------------------------------------- | --------------------------------- |
| `information-theory`           | Shannon's mathematical theory of communication—quantifying information as reduction of uncertainty, measured in bits.                  | ✏️ Add: "Surprise is information." |
| `systems-theory-cybernetics`   | The science of feedback, control, and system behavior—how systems self-regulate and adapt through information loops.                   | ✅ OK                              |
| `sociotechnical-systems`       | The study of how social and technical systems co-evolve—neither can be optimized independently; joint optimization required.           | ✅ OK                              |
| `schema-theory-constructivism` | How humans structure information into mental schemas—knowledge is actively constructed, not passively received.                        | ✅ OK                              |
| `relational-model`             | Codd's mathematical foundation for relational databases—data as relations (tables), manipulated through relational algebra.            | ✅ OK                              |
| `entity-relationship-model`    | Chen's conceptual framework for data modeling—entities, attributes, and relationships visualized as ER diagrams.                       | ✅ OK                              |
| `dimensional-modeling`         | Kimball's approach to data warehouse design—star schemas optimized for analytical queries, with facts and dimensions.                  | ✅ OK                              |
| `formal-ontology`              | Machine-interpretable concept definitions—W3C standards (OWL, RDF, SKOS) for representing knowledge graphs and semantic relationships. | ✅ OK                              |
| `knowledge-representation`     | The AI/philosophy field studying how knowledge should be structured for reasoning—ontologies, frames, semantic networks, and logic.    | ✅ OK                              |

---

## 2. DDD Context Map Patterns (3p - Eric Evans)

| Atom                             | Definition                                                                                                                                                     | Action |
| -------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `context-map-partnership`        | A DDD context map pattern where two teams coordinate closely with equal power, sharing success and failure of the integration.                                 | ✅ OK   |
| `context-map-shared-kernel`      | A DDD context map pattern where two bounded contexts share a subset of the domain model, requiring joint agreement on changes to the shared portion.           | ✅ OK   |
| `context-map-customer-supplier`  | A DDD context map pattern where downstream context depends on upstream context, with upstream accommodating downstream needs through negotiation.              | ✅ OK   |
| `context-map-conformist`         | A DDD context map pattern where downstream context accepts upstream model wholesale, eliminating translation but surrendering model autonomy.                  | ✅ OK   |
| `context-map-open-host-service`  | A DDD context map pattern where upstream provides a well-defined protocol/API for downstream consumers, decoupling internal model from integration interface.  | ✅ OK   |
| `context-map-published-language` | A DDD context map pattern establishing a common language for integration—a well-documented, shared vocabulary that multiple contexts use for communication.    | ✅ OK   |
| `context-map-separate-ways`      | A DDD context map pattern where two bounded contexts have no relationship—they operate independently with no integration, accepting duplication over coupling. | ✅ OK   |

---

## 3. DDD Tactical Patterns (3p - Eric Evans)

| Atom                    | Definition                                                                                                                                                                | Action |
| ----------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------ |
| `entity`                | An object defined primarily by its identity rather than attributes—identity persists throughout lifecycle even as attributes change.                                      | ✅ OK   |
| `value-object`          | An object defined entirely by its attributes with no identity—two value objects with the same values are identical and interchangeable; immutable by design.              | ✅ OK   |
| `aggregate-root`        | The single entity in an aggregate through which all external access occurs—enforces invariants for the entire cluster and defines the transactional consistency boundary. | ✅ OK   |
| `anti-corruption-layer` | An isolating layer between bounded contexts that translates requests/responses, preventing upstream models from corrupting the downstream domain model.                   | ✅ OK   |

---

## 4. DDD Strategic Patterns (3p - Eric Evans)

| Atom                   | Definition                                                                                                                                                                                 | Action |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ------ |
| `ubiquitous-language`  | A shared vocabulary used consistently by all team members—domain experts, developers, and stakeholders—appearing in conversations, code, schemas, and documentation without translation.   | ✅ OK   |
| `bounded-context`      | An explicit boundary within which a particular domain model applies—terms have specific meaning inside, but may mean different things outside; translation is required at boundaries.      | ✅ OK   |
| `context-map`          | A document outlining bounded contexts and their relationships—integration patterns (Shared Kernel, Customer/Supplier, ACL, etc.), translation rules, and power dynamics between contexts.  | ✅ OK   |
| `strategic-design`     | Identifying where to invest modeling effort—distinguishing core domain (competitive differentiators) from supporting and generic subdomains; copy patterns for 85-90%, innovate on 10-15%. | ✅ OK   |
| `domain-driven-design` | A software design philosophy that software models should be deeply tied to the business domain—emphasizing ubiquitous language, bounded contexts, and explicit domain modeling.            | ✅ OK   |
| `semantic-drift`       | Uncontrolled change in meaning over time—definitions, invariants, or constraints gradually diverge from intended semantics without explicit governance.                                    | ✅ OK   |

---

## 5. Understanding Theory (1p - Tim's original)

| Atom                            | Definition                                                                                                                                                              | Action                                                   |
| ------------------------------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------- | -------------------------------------------------------- |
| `understanding-as-process`      | Understanding is a dynamic process, not a static entity—it exists only during active cognition and must be continuously regenerated.                                    | ✏️ Change "exists" → "emerges" (energy + change required) |
| `understanding-components`      | The five components of understanding—Causal, Explainable, Anticipatory, Application, and Shared.                                                                        | ❌ DELETE - not an atom                                   |
| `causal-understanding`          | Understanding the foundational 'why'—the mechanisms and causal relationships that explain phenomena.                                                                    | ❌ DELETE - not an atom                                   |
| `shared-understanding`          | Understanding that exists as an organizational property—when a group collectively comprehends something the same way.                                                   | ✅ OK                                                     |
| `energy-barrier-problem`        | Why organizations struggle to achieve understanding at scale—the energy required to create and maintain shared understanding grows non-linearly with organization size. | ✏️ RENAME → `understanding-min-energy`                    |
| `suspended-understanding`       | AI holds massive context while humans inject meaning—a collaboration pattern where AI maintains coherence state and humans provide judgment.                            | ❓ NEEDS CONTEXT - Tim to clarify                         |
| `understanding-maturity-ladder` | Five levels from Perception to Collective Effectiveness—a progression model for organizational understanding capability.                                                | ❌ DELETE - not an atom                                   |

---

## 6. Hard Problem of AI (1p - Tim's original)

| Atom                   | Definition                                                                                                                                                                                                                                        | Action                                                                                                                                     |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------ |
| `regression-paradox`   | LLMs regress to the mean—their strength at finding consensus patterns is precisely what makes them ineffective for transformation, which requires deviation from historical patterns.                                                             | ✏️ REWRITE: Lead with "good at consensus patterns"; connect to standard regression-to-mean (bigger sample = less surprise = no information) |
| `runtime-emergence`    | Transformation exists only during active operation—organizational meaning is distributed, functional, and observable only at runtime, not in static artifacts.                                                                                    | ✏️ Change "Transformation" → "unexpected change and novel solutions"                                                                        |
| `semantic-decoherence` | The natural tendency of shared meaning to decay from coherent to fragmented states. Semantic coherence accross humans and machines requires continuous energy investment.                                                                         |                                                                                                                                            | // rewrote inline     |
| `paradigmatic-lock-in` | Existing corporate structures, processes, and financial models absorb AI's transformative potential and redirect it toward reinforcing the status quo.                                                                                            |                                                                                                                                            | // this is 3p, not 1p |
| `measurement-fallacy`  | Companies are trapped using factory-floor ROI models that only value cost-cutting metrics, structurally filtering out transformative projects that create growth rather than savings.                                                             |                                                                                                                                            | //3p and not an atom  |
| `hard-problem-of-ai`   | A re-framing of the consensus "AI Transformation Probglem", stating lack of impactful resutls after significant AI integration. Chalmers' Hard Problem of Consciousness provides an analogy that reveals insights into root causes and solutions. |                                                                                                                                            | //rewrote inline      |

---

## 7. Real Data Framework (1p - Tim's original)

| Atom                            | Definition                                                                                                                                                                                                                                           |
| ------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `seven-core-capabilities`       | All data platforms offer the same seven universal capabilities—Ingestion, Storage, Table Layer, Analytic Engines, Orchestration, Catalog, and AI/ML. Vendor differentiation is execution quality and latency dependence, not fundamental capability. | //rewrote inline   |
| `three-forces`                  | The three orthogonal axes that drive every data architecture decision—Volume & Velocity, Latency, and Structure & Variability.                                                                                                                       |
| `source-instrumentation`        | The surface/system-level code that creates events, files, or data based on business logic at the source—where business events are captured and defined at origin.                                                                                    | //not 1p, not atom |
| `source-ownership`              | The organizational challenge of determining who owns the definition and capture of business events at the source—solving the 'political orphan' problem of instrumentation.                                                                          | // not 10 not atom |
| `system-up-thinking`            | An anti-pattern architecture approach that treats data as a reconciliation problem—starting from existing systems and integration, instead of provenance and last mile need for meaning in the data.                                                 | //rewrote inline   |
| `no-free-lunch-data`            | You cannot skip the structure investment—'schemaless' systems just move the cost to less efficient locations like query time, application code, or human reconciliation.                                                                             | //not atom         |
| `thermodynamics-of-data`        | Structure requires energy, and the physics is unambiguous. Structure requires energy input to reduce entropy, and there's a minimum requirement.                                                                                                     | //rewrote inline   |
| `deterministic-transformations` | Data system operations are mathematically predictable—given a defined structure, you can predict compute requirements for join, filter, and aggregation.                                                                                             | //rewrote inline   |
| `rtfm-principle`                | System best practices have specific requirements—fighting them creates fragile hacks. Accept the determinism and plan to do it right.                                                                                                                | //rewrote inline   |
| `self-validating-systems`       | Systems where constraints are features—nothing quietly fails, all errors surface immediately during design and ETL, not months later when executives make decisions on wrong data.                                                                   | //not 1p           |
| `semantic-layer`                | The query interface abstraction between raw data and users/BI tools—maintaining semantic definitions, enforcing consistent meaning, and preventing direct schema access.                                                                             | //not 1p           |
| `real-data`                     | A collection of best practices and frame for thinking of data as a strategic, first-class citizen for impproved decision-making and use of AI. A pillar of the Semantic Operations Framework.                                                        | //rewrote inline   |

---

## 8. Semantic Operations Framework (1p - Tim's original)

| Atom                    | Definition                                                                                                                                                                                                     |
| ----------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `knowledge-corpus`      | The source of truth comprising an organization's formal semantic layer—domain models, glossaries, ontologies, metric definitions, and governance documentation that define canonical meaning.                  | //not 1p                                  |
| `knowledge-artifacts`   | Real-world manifestations and operational expressions of meaning in practice—dashboards, code, documentation, AI outputs, and team behaviors that reveal what semantics are actually used.                     | //not 1p                                  |
| `corpus-artifact-delta` | Measurement in Semantic Coherence formula - the difference between Knowledge Corpus (canonical definitions) and Knowledge Artifacts (real-world usage)—revealing semantic health and transformation readiness. | //needs some work... just flag            |
| `progressive-semantics` | The systematic process of promoting valuable concepts and data from exploratory to a stable asset to build on.                                                                                                 | //rewrote linline                         |
| `flexible-edge`         | The frontier of organizational learning—a low-structure, high-ambiguity space where signals and concepts emerge before formal governance, enabling discovery and experimentation.                              | //not 1p                                  |
| `hard-schema`           | The high-confidence, formally governed layer where concepts solidify into organizational domain knowledge after passing promotion criteria—encoding meaning into operationally usable forms.                   | //not 1p, not atom                        |
| `semantic-coherence`    | A state of stable, shared semantic alignment between agents (human + machine) that enables optimal data-driven decision making including with AI                                                               | //rewrote inline, flag "needs work"       |
| `semantic-optimization` | The process of maintaining stable coherence between agents (human + machine) while spuring growth through new patterns and change.                                                                             | //rewrote inline, flag "needs work"       |
| `semantic-operations`   | A systematic approach to adapting systems, organizations, and processes to maxmimize the impact of data and AI through semantic streamlining.                                                                  | //rewrote inline... flag for "needs work" |

---

## 9. Architecture Patterns (1p - Tim's original)

| Atom                         | Definition                                                                                                                                                                      |
| ---------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `stable-core-flexible-edge`  | Architecture pattern with semantically identical core (governed, stable) surrounded by context-specific edge (experimental, adaptable).    //not 1p, but important 3p           |
| `semantic-containers`        | Bounded contexts viewed as containers for semantic coherence—each context maintains internally consistent meaning.                                                              | //flag for "needs work"                                   |
| `semantic-governance`        | The practice of ensuring all teams use same terminology through ownership, change processes, and enforcement mechanisms.                                                        |
| `semantic-contracts`         | Context maps that define translation rules between bounded contexts—explicit agreements on how meaning transfers across boundaries.                                             |
| `intentional-architecture`   | Architecture must be deliberately chosen, not emergent—design systems around domain semantics rather than letting structure accumulate from ad-hoc decisions.                   | //check for 1p, is DDD?                                   |
| `semantic-invariants`        | Domain truths that must hold regardless of context—business rules encoded as constraints that validate automatically.                                                           | // flag for "needs work", "constitution" or "principles"? |
| `global-architecture`        | Systems, organization, and product semantically aligned—the synthesis where all three encode the same domain model.                                                             | // not 1p, not atom                                       |
| `domain-driven-architecture` | Architecture where systems, organization, and products are semantically consistent and meaning-driven—applied Domain-Driven Design at organizational scale.                     | // flag for "needs work"                                  |
| `model-everything`           | A tactical methodology of modeling organization, business, and data structures through direct examination of data artifacts—DDD-inspired but data-centric, made feasible by AI. |

---

## Review Notes

### Potential Issues to Check:
1. **Overlapping concepts**: `semantic-containers` vs `bounded-context`?//this might be what i mean by "patterns" flag for "needs work"
2. **Hierarchy clarity**: Is `domain-driven-architecture` distinct enough from `global-architecture`?// yes, flag for "needs work" - there is only 1 of these that will survive - they are the same thing, but i can't decide on the terminology
3. **Definition consistency**: Are all 1p definitions in Tim's voice?// this is a good call-out, i think we handle style and tone of atoms and composites later in process
4. **3p accuracy**: Do Evans definitions match the book?// this is a good check

### Missing Definitions (42 atoms):
See separate list - these need definitions added.
