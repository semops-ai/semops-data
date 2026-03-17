# Infrastructure

> **Repo:** `semops-data`
> **Owner:** This repo owns and operates these services
> **Status:** ACTIVE
> **Version:** 1.3.0
> **Last Updated:** 2026-03-03

---

## Services

> **Port authority:** [PORTS.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/PORTS.md) — the single source of truth for all port allocations. Register new ports there before use.

| Service | Port | Purpose | Runtime |
|---------|------|---------|---------|
| Jupyter Lab | 8888 | Notebook interface | DevContainer or docker-compose (`notebooks` profile) |
| MLflow UI | 5000 | Experiment tracking | DevContainer |
| Marquez API | 5000 | Data lineage API | docker-compose (`lineage` profile) — **conflicts with MLflow** |
| Marquez Admin | 5002 | Data lineage admin | docker-compose (`lineage` profile) |
| Marquez Web | 3005 | Data lineage UI | docker-compose (`lineage` profile) |

## Docker Configuration

### docker-compose.yml

Alternative to DevContainer for specific workloads. Services use Docker Compose profiles — start only what you need:

```bash
# Lineage stack (Marquez API + Web)
docker compose --profile lineage up -d

# Notebooks (Jupyter Lab)
docker compose --profile notebooks up -d
```

| Profile | Services | Ports |
|---------|----------|-------|
| *(default)* | python, dbt | — (no exposed ports) |
| `lineage` | marquez, marquez-web | 5000, 5002, 3005 |
| `notebooks` | jupyter | 8888 |

### Network

| Network | Services | Purpose |
|---------|----------|---------|
| Host network | Jupyter Lab, MLflow | DevContainer port forwarding |

> **Convention:** Cross-repo access uses `localhost` port mapping, not Docker internal networking. See [GLOBAL_INFRASTRUCTURE.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/GLOBAL_INFRASTRUCTURE.md#how-repos-connect).

### Starting Services

```bash
# DevContainer (primary development environment)
# Open in VSCode → Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Jupyter Lab (inside container)
jupyter lab --port 8888

# MLflow UI (inside container)
mlflow ui --host 0.0.0.0 --port 5000
```

### DevContainer

| Component | Value |
|-----------|-------|
| **Base Image** | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| **GPU Support** | NVIDIA CUDA 12.1 (compatible with driver 12.9+) |
| **Python** | 3.10 (from PyTorch image) |

## Environment Variables

> **Convention:** Use `SEMOPS_DB_*` for application database config, `POSTGRES_*` for Supabase container config. These are different — see [GLOBAL_INFRASTRUCTURE.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/GLOBAL_INFRASTRUCTURE.md#environment-variable-conventions).

| Variable | Purpose | Required |
|----------|---------|----------|
| `OPENAI_API_KEY` | Embeddings for coherence scoring | For coherence module |
| `ANTHROPIC_API_KEY` | LLM synthesis | Optional |

## Connection Patterns

How this repo connects to shared infrastructure:

| Service | Method | Details |
|---------|--------|---------|
| Qdrant | `localhost:6333` | Vector storage for coherence scoring |
| PostgreSQL | `localhost:5434` | Shared database (direct access) |
| Ollama | `localhost:11434` | Local embeddings |
| MCP Server | stdio | Registered in `~/.claude.json` (global) or `.mcp.json` (project) |

## Python Stack

| Property | Value |
|----------|-------|
| **Python version** | `3.10` |
| **Package manager** | `pip` with `pyproject.toml` (DevContainer; global standard is `uv`) |
| **Virtual environment** | DevContainer (isolated) |
| **Linter/formatter** | `ruff` |
| **Test framework** | `pytest` |

### Key Dependencies

| Library | Purpose | Shared With |
|---------|---------|-------------|
| `pandas` | Data manipulation | backoffice |
| `numpy` | Numerical computing | — |
| `sdv` | Synthetic data generation | — |
| `faker` | Fake data generation | — |
| `mostlyai-mock` | Schema-driven mock data (no training needed) | — |
| `mostlyai-qa` | Automated HTML fidelity + privacy QA reports | — |
| `duckdb` | Local analytics engine | — |
| `sqlalchemy` | Database abstraction | semops |
| `pydantic` | Settings and data models | semops, publisher |
| `click` | CLI framework | semops, publisher |
| `pyyaml` | YAML handling | semops, publisher, backoffice, dx-hub |
| `mlflow` | Experiment tracking | — |
| `sentence-transformers` | Coherence scoring embeddings | — |
| `torch` | Deep learning (GPU) | — |
| `pyarrow` | Columnar data format (data interchange) | — |
| `great-expectations` | Data quality and validation | — |
| `loguru` | Structured logging (replaces stdlib logging) | — |
| `ysemops-dataofiling` | Data profiling | — |

#### Optional Groups (architecturally notable)

| Library | Group | Purpose |
|---------|-------|---------|
| `dbt-core` + `dbt-duckdb` | `dbt` | Transform framework (DuckDB adapter) |
| `openlineage-python` + `marquez-python` | `lineage` | Lineage standard — traces to Agentic Lineage capability |
| `mlflow` | `mlops` | Experiment tracking for coherence scoring |
| `sentence-transformers` + `torch` | `coherence` | Embedding models for SC formula |
| `mostlyai[local]` | `synthetic-data` | MOSTLY AI local training (TabularARGN); use `mostlyai[local-gpu]` in DevContainer |

### Setup

```bash
# In DevContainer (automatic via postCreateCommand)
pip install -e ".[dev,notebooks,mlops]"

# Or specific groups
pip install -e ".[coherence]"
```

## Health Checks

```bash
# Jupyter Lab
curl http://localhost:8888/api/status

# MLflow
curl http://localhost:5000/health

# GPU
python -c "import torch; print(f'CUDA: {torch.cuda.is_available}')"
```

## Consumed By

| Repo | Services Used |
|------|---------------|
| *(none currently)* | — |

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - This repo's architecture
- [GLOBAL_INFRASTRUCTURE.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/GLOBAL_INFRASTRUCTURE.md) - Ecosystem connectivity, network conventions, env var standards
- [PORTS.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/PORTS.md) - Port registry (single source of truth)
- [DIAGRAMS.md](https://github.com/semops-ai/semops-dx-orchestrator/blob/main/docs/DIAGRAMS.md) - Infrastructure service diagrams
- [ADR-0002](decisions/ADR-0002-devcontainer-mlops-environment.md) - DevContainer decision
