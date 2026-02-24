# Architecture

> **Repo:** `semops-data`
> **Role:** Analytics/MLOps
> **Status:** ACTIVE
> **Version:** 1.4.0
> **Last Updated:** 2026-02-23
> **Infrastructure:** [INFRASTRUCTURE.md](INFRASTRUCTURE.md)

---

## Role

Analytics platform, coherence scoring, and data utilities — the data engineering and data science hub within the SemOps ecosystem.

**Key distinction:** This repo owns *data utilities, coherence scoring, and simulation*. `semops-core` owns *schema and knowledge model*. Research RAG and data due diligence were extracted to [semops-research](https://github.com/semops-ai/semops-research) via [#50](https://github.com/semops-ai/semops-data/issues/50).

## DDD Classification

> Source: [REPOS.yaml](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/REPOS.yaml)

| Property | Value |
|----------|-------|
| **Layer** | `semops-core` |
| **Context Type** | `core` |
| **Integration Patterns** | `customer-supplier` |
| **Subdomains** | `semantic-operations`, `knowledge-management` |

## Capabilities

> Source: [STRATEGIC_DDD.md](https://github.com/semops-ai/semops-core/blob/main/docs/STRATEGIC_DDD.md) (authoritative capability registry)

| Capability | Status | Description |
|------------|--------|-------------|
| Coherence Scoring | in_progress | MLflow-tracked experiments measuring semantic drift (SC formula) |
| Synthesis and Simulation | draft | Synthetic data generation (SDV, Faker), stack simulation, data profiling |
| Agentic Lineage | planned | Episode-centric provenance tracking (shared with semops-core) |
| Scale Projection | in_progress | Cross-cutting: HITL-to-ML progression via data engineering scenarios |

Every capability must trace to at least one registered pattern (coherence signal). See [pattern_v1.yaml](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/schemas/pattern_v1.yaml).

## Ownership

What this repo owns (source of truth for):

- GPU-enabled DevContainer environment
- MLflow experiment tracking
- Jupyter notebooks and data science workflows
- Synthetic data generation (SDV, Faker)
- Stack simulation samples (S3 → Delta Lake → Snowflake)
- Data lineage tracking (OpenLineage, Marquez)
- Data profiling tools
- Coherence scoring experiments (MLflow-tracked)

What this repo does NOT own (consumed from elsewhere):

- Schema and knowledge model (semops-core)
- Infrastructure services: Qdrant, Docling, PostgreSQL, Ollama (semops-core)
- Research RAG, data due diligence (semops-research — extracted via #50)

**Ubiquitous Language conformance:** This repo follows definitions in [UBIQUITOUS_LANGUAGE.md](https://github.com/semops-ai/semops-core/blob/main/schemas/UBIQUITOUS_LANGUAGE.md). Domain terms used in code and docs must match.

## Key Components

### Source Code

```text
src/data_systems_toolkit/
├── core/ # Config, logging utilities
├── coherence/ # Coherence scoring (SC formula, MLflow experiments)
├── synthetic/ # Data generators (e-commerce, analytics)
├── profiling/ # Data profiling tools
├── simulation/ # Stack simulation logic (planned)
└── lineage/ # Lineage tracking (planned)
```

### Scripts

> No `scripts/` directory — all code lives in `src/data_systems_toolkit/`. This table maps key modules to capabilities.

| Module | Capability | Purpose |
|--------|-----------|---------|
| `coherence/` | Coherence Scoring | SC formula, MLflow experiments, embedding strategies |
| `synthetic/` | Synthesis and Simulation | SDV/Faker data generators (e-commerce, analytics) |
| `profiling/` | Synthesis and Simulation | Data profiling tools (ysemops-dataofiling) |
| `simulation/` | Synthesis and Simulation | Stack simulation logic (planned) |
| `lineage/` | Agentic Lineage | Lineage tracking via OpenLineage (planned) |
| `cli.py` | *(multiple)* | CLI entry point (`dst` command) |

### Other Components

| Component | Purpose |
|-----------|---------|
| `.devcontainer/` | GPU-enabled DevContainer (PyTorch CUDA 12.1) |
| `data/` | Working datasets (gitignored) |
| `mlruns/` | MLflow experiment tracking (gitignored) |
| `samples/` | Canonical pipeline examples |
| `notebooks/` | Jupyter analysis notebooks |
| `docs/research/` | Data engineering research docs |

## Dependencies

| Repo | What We Consume |
|------|-----------------|
| semops-core | Qdrant (vectors), Docling (documents), PostgreSQL (shared DB), Ollama (embeddings), MCP Server |

| Repo | What Consumes Us |
|------|------------------|
| semops-sites | Potential datavis integration |
| External users | Public product |

## Related Documentation

- [GLOBAL_ARCHITECTURE.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/GLOBAL_ARCHITECTURE.md) - System landscape
- [DIAGRAMS.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/DIAGRAMS.md) - Visual diagrams (context map, data flows, DDD model)
- [REPOS.yaml](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/REPOS.yaml) - Structured repo registry
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - Services, DevContainer, stack details
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the toolkit
- `docs/decisions/` - Architecture Decision Records

---

## Versioning Notes

**Status values:**

- `ACTIVE` - Current implemented state (one per doc type)
- `PLANNED-A`, `PLANNED-B`, `PLANNED-C` - Alternative future states

**File naming for planned versions:**

- `ARCHITECTURE.PLANNED-A.md`
- `ARCHITECTURE.PLANNED-B.md`

**When to create a PLANNED version:**

- Significant architectural changes under consideration
- Alternative approaches being evaluated
- Future state design for upcoming work
