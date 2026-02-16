# Architecture

> **Repo:** `semops-data`
> **Role:** Product/Data - Data engineering utilities and product
> **Status:** ACTIVE
> **Version:** 1.2.0
> **Last Updated:** 2026-02-06
> **Infrastructure:** [INFRASTRUCTURE.md](INFRASTRUCTURE.md)

---

## Role in Global Architecture

**Role:** Data engineering utilities, simulation tools, research RAG pipelines

This repo is part of the **SemOps** ecosystem.

**Key relationships:**

- **Receives from:** `semops-dx-orchestrator` (process, tooling)
- **Uses infrastructure from:** `semops-core` (Qdrant, Docling, PostgreSQL, MCP server)
- **Publishes to:** `semops-sites` (potential datavis integration)

**Ownership boundary:** This repo owns *data utilities and simulation*, while `semops-core` owns *schema and knowledge model*.

---

## System Context

Data Systems Toolkit is a **standalone product** within the SemOps ecosystem, focused on data engineering and data science workflows. Unlike other repos, this is designed to be **publicly publishable** as an educational/consulting tool.

```text
┌─────────────────────────────────────────────────────────────────┐
│                       semops-core                                │
│                   [SCHEMA/INFRASTRUCTURE]                        │
└─────────────────────────────────────────────────────────────────┘
                              │
                              │ (shared: Qdrant, Docling, PostgreSQL)
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                      semops-data                                 │
│                       [PRODUCT]                                  │
│                                                                  │
│  Data science environment, ML workflows, stack simulation        │
│  First publishable product from SemOps                           │
└─────────────────────────────────────────────────────────────────┘
```

---

## Ownership

**Role:** Data engineering/science hub and educational toolkit

**Owns:**

- GPU-enabled DevContainer environment
- MLflow experiment tracking
- Jupyter notebooks and data science workflows
- Synthetic data generation (SDV, Faker)
- Stack simulation samples (S3 -> Delta Lake -> Snowflake)
- Data lineage tracking (OpenLineage, Marquez)
- Data profiling tools
- **Research RAG module** - RAPTOR-inspired meta-analysis pipeline
- Coherence scoring experiments (MLflow-tracked)

**In Development:**

- Reference Architecture Toolkit - Patterns for data pipelines
- Agentic Lineage concepts - AI agent decision tracking

**Depends On:**

- semops-core infrastructure (Qdrant, Docling, Ollama, MCP server)
- External APIs (OpenAI embeddings, Anthropic Claude synthesis)

**Consumed By:**

- External users (public product)
- semops-sites (potential datavis integration)

---

## Project Phases

| Phase | Focus | Status |
|-------|-------|--------|
| **1A** | Environment setup, synthetic data, profiling | Complete |
| **1B** | Research RAG pipeline, RAPTOR meta-analysis | Complete |
| **1C** | Coherence scoring experiments | Complete |
| **2** | Reference Architecture Toolkit | In Progress |
| **3** | Semantic Operations orchestration tools | Future |

---

## Key Components

### DevContainer Environment

GPU-enabled isolated development environment.

```text
.devcontainer/
├── Dockerfile           # PyTorch CUDA 12.1 base
└── devcontainer.json    # GPU passthrough, ports, extensions
```

**Base Image:** `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`

### Source Code

```text
src/data_systems_toolkit/
├── core/               # Config, logging utilities
├── synthetic/          # Data generators (e-commerce, analytics)
├── profiling/          # Data profiling tools
├── research/           # Research RAG pipeline (RAPTOR meta-analysis)
├── coherence/          # Coherence scoring (SC formula, MLflow experiments)
├── simulation/         # Stack simulation logic (planned)
└── lineage/            # Lineage tracking (planned)
```

### Research RAG Module

RAPTOR-inspired pipeline for corpus-driven meta-analysis.

```text
src/data_systems_toolkit/research/
├── ingest.py           # PDF/web ingestion via Docling, chunking
├── embed.py            # OpenAI embeddings -> Qdrant storage
├── query.py            # RAG retrieval + Claude LLM synthesis
├── config.py           # Service configuration
├── cli.py              # Full CLI interface
└── sources/
    └── manifest.json   # Ingested sources registry
```

**Key Pattern:** RAPTOR (Recursive Abstractive Processing for Tree-Organized Retrieval)

- K-means clustering on embeddings to discover themes
- Theme extraction via Claude LLM
- Meta-synthesis reports covering problem space, causes, solutions

**CLI Commands:**

```bash
python -m data_systems_toolkit.research.cli ingest --type pdf|web
python -m data_systems_toolkit.research.cli query "question"
python -m data_systems_toolkit.research.cli synthesize
python -m data_systems_toolkit.research.cli status
python -m data_systems_toolkit.research.cli cleanup
```

### Data Directories

```text
data/                   # Working datasets (gitignored)
├── raw/               # Original downloads
├── processed/         # Cleaned data
└── interim/           # Intermediate processing
```

### MLflow Tracking

```text
mlruns/                 # Experiment tracking (gitignored)
└── [experiment_id]/
    └── [run_id]/
        ├── params/
        ├── metrics/
        └── artifacts/
```

---

## Integration Points

### semops-core

| Service | Purpose |
|---------|---------|
| Qdrant | Vector storage for research RAG |
| Docling | Document processing for ingestion |
| PostgreSQL | Shared database (via Supabase) |
| MCP Server | Agent access to knowledge base |

### Current Integrations

| Integration | Direction | Protocol |
|-------------|-----------|----------|
| HuggingFace | Inbound | `datasets` Python library |
| Local GPU | Internal | CUDA via PyTorch |
| semops-core | Bidirectional | Docker network, REST APIs |

### Planned Integrations

| Integration | Direction | Protocol |
|-------------|-----------|----------|
| Marquez | Bidirectional | REST API (lineage tracking) |
| semops-sites | Outbound | API / Supabase (datavis) |

---

## Security Tier

**Public** - This repo is designed for external publication.

| Consideration | Approach |
|---------------|----------|
| Secrets | `.env` gitignored, `.env.example` committed |
| Data | `data/` gitignored, use public datasets |
| Models | `mlruns/` gitignored |
| API Keys | Never in code or samples |

---

## Key Files

| File | Purpose |
|------|---------|
| `docs/ARCHITECTURE.md` | This file |
| `docs/INFRASTRUCTURE.md` | DevContainer, ports, stack details |
| `docs/USER_GUIDE.md` | Practical usage guide |
| `docs/decisions/` | Architecture Decision Records |
| `pyproject.toml` | Package config and dependencies |

---

## Related Documents

- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - DevContainer, ports, data locations
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the toolkit
- ADR-0001 - Tech stack foundations
- ADR-0002 - DevContainer decision
