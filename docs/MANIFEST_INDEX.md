# YAML Manifest Index

Which YAML manifest answers which question type. Used by `/kb` query routing.

## Manifests

| Manifest | Location | Authority For |
|----------|----------|---------------|
| `REPOS.yaml` | `semops-orchestrator/docs/` | Repo topology, layers, dependencies, integration patterns, capabilities per repo, public mirrors |
| `registry.yaml` | `semops-data/config/` | Capability definitions, pattern implementations, governance criteria, agent catalog |
| `pattern_v1.yaml` | `semops-orchestrator/schemas/` | Pattern identity, provenance (1P/3P), lineage (derives_from), documentation links |
| `concept-pattern-map.yaml` | `semops-data/config/mappings/` | Concept-to-pattern mappings with predicates and occurrence counts |
| `publish-manifest.yaml` | `semops-orchestrator/docs/` | Per-repo file selection rules for public mirror sync |
| `decisions.yaml` | `semops-orchestrator/export/methodology/` | ADR index with status, dates, repos |
| `project-methodology.yaml` | `semops-orchestrator/export/methodology/` | Project spec structure, project index with status |
| `workflows.yaml` | `semops-orchestrator/export/methodology/` | Workflow definitions (capture, publish, etc.) |
| `editorial-rules.yaml` | `semops-orchestrator/export/methodology/` | Translation rules for public-facing content |

## Query Routing

### YAML Source (structural queries — no infrastructure needed)

| Question Type | Primary Manifest | Example Query |
|---------------|-----------------|---------------|
| **Topology** — what repos exist, their roles, dependencies | `REPOS.yaml` | "What does semops-research provide?" |
| **Capability tracing** — what implements a pattern | `registry.yaml` | "What implements DDD?" |
| **Pattern lookup** — identity, provenance, lineage | `pattern_v1.yaml` | "What patterns derive from DDD?" |
| **Concept mapping** — which concepts map to which patterns | `concept-pattern-map.yaml` | "What patterns relate to anti-corruption layers?" |
| **Governance / status** — lifecycle state, acceptance criteria | `registry.yaml` | "Which capabilities are planned?" |
| **Integration** — how repos connect, DDD context map | `REPOS.yaml` | "What integration patterns does publisher-pr use?" |
| **Methodology / process** — ADR workflow, project specs | `methodology/*.yaml` | "What's the ADR workflow?" |
| **Publishing** — what gets published where | `publish-manifest.yaml` | "What files does sites-pr publish?" |
| **Agent catalog** — registered agents and their roles | `registry.yaml` | "What agents are registered?" |

### Chunks Source (content queries — requires PostgreSQL + embeddings)

| Question Type | MCP Tool | Example Query |
|---------------|----------|---------------|
| **"What does it say about X"** — find passages discussing a topic | `search_chunks` | "What do we say about anti-corruption layers?" |
| **"Find content like X"** — semantic similarity search | `search_chunks` | "Content related to data contracts" |
| **Entity lookup** — find entities by semantic match | `search_knowledge_base` | "Architecture decision about ingestion" |

### Routing Heuristic

**YAML-first signals** (structural intent):
- "What exists / what repos / what capabilities"
- "What implements / what delivers / what uses"
- "What patterns / what derives from / what provenance"
- "Status / governance / lifecycle / planned"
- "How do we / what's the process / workflow"
- "Integration / depends on / provides"

**Chunks signals** (content intent):
- "What does it say about / what do we say"
- "Find / search for / content about / passages"
- "Explain / describe / summarize" (seeking prose, not structure)
