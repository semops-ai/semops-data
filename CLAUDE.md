# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role in Global Architecture

**Bounded Context:** Data Platform

```text
semops-dx-orchestrator [PLATFORM/DX]
 │
 └── semops-core [ORCHESTRATOR]
 │
 └── semops-data [DATA PLATFORM] ◄── YOU ARE HERE
 │
 ├── Owns: Data utilities and analytics
 │ - Coherence scoring experiments
 │ - Synthetic data generation
 │ - Stack simulation (S3 → Delta Lake → Snowflake)
 │ - Data profiling and lineage tools
 │
 └── Uses: Infrastructure from semops-core
 - Qdrant, Docling, PostgreSQL
```

**Key Ownership Boundary:**

- This repo owns **data utilities** - coherence scoring, synthetic data, stack simulation, profiling, lineage
- Research RAG and data due diligence extracted to `semops-research` (see #50)
- `semops-core` owns **infrastructure** - Qdrant, Docling, PostgreSQL, schema

**Global Docs Location:** `

## Session Notes

Document work sessions tied to GitHub Issues in `docs/session-notes/`:

- **Format:** `ISSUE-NN-description.md` (one file per issue, append-forever)
- **Structure:** Date sections within file for chronological tracking
- **Index:** Update `docs/SESSION_NOTES.md` with new entries
- **When:** Working on any GitHub Issue, or ad-hoc exploratory sessions

## Key Files

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Repo architecture and ownership
- [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) - Infrastructure dependencies
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Development environment and usage
- [docs/decisions/](docs/decisions/) - Architecture Decision Records
- [docs/session-notes/](docs/session-notes/) - Session logs by issue
- [~/.claude/CLAUDE.md](~/.claude/CLAUDE.md) - Global instructions (user-level)

## Tech Stack

- **Python 3.11+** with pyproject.toml packaging
- **DevContainer** with GPU support (PyTorch CUDA 12.1)
- **DuckDB** for local analytics
- **dbt-core** with DuckDB adapter
- **SDV** for synthetic data generation
- **Ollama** for embeddings (coherence scoring)
