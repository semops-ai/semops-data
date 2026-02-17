# Infrastructure

> **Repo:** `semops-data`
> **Owner:** This repo owns and operates these services
> **Status:** ACTIVE
> **Version:** 1.1.0
> **Last Updated:** 2025-12-29

Development environment, services, and data locations.

---

## DevContainer Environment

### Overview

Primary development uses a GPU-enabled DevContainer for isolated, reproducible workflows.

| Component | Value |
|-----------|-------|
| **Base Image** | `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime` |
| **GPU Support** | NVIDIA CUDA 12.1 (compatible with driver 12.9+) |
| **Python** | 3.10 (from PyTorch image) |

### Prerequisites

```bash
# Verify Docker nvidia runtime
docker info | grep -i runtime
# Should show: nvidia

# Verify GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi
```

### Starting the Environment

1. Open repo in VSCode
2. `Ctrl+Shift+P` -> "Dev Containers: Reopen in Container"
3. Wait for build (~5 min first time)
4. Run `notebooks/00-environment-test.ipynb` to verify

### Rebuilding

After changing `.devcontainer/Dockerfile`:

```
Ctrl+Shift+P -> "Dev Containers: Rebuild Container"
```

---

## Services

### DevContainer Services

| Service | Purpose |
|---------|---------|
| Jupyter Lab | Notebook interface |
| MLflow UI | Experiment tracking |

---

## Data Locations

### Directory Structure

```
data-systems-toolkit/
├── data/                    # Working datasets (gitignored)
│   ├── raw/                # Original downloads, never modify
│   ├── processed/          # Cleaned, ready for modeling
│   └── interim/            # Intermediate processing
├── mlruns/                  # MLflow tracking (gitignored)
├── samples/                 # Canonical examples (committed)
└── notebooks/               # Jupyter notebooks
```

### Storage Policies

| Directory | Git Status | Persistence | Purpose |
|-----------|------------|-------------|---------|
| `data/` | Ignored (except `.gitkeep`) | Local only | Working datasets |
| `mlruns/` | Ignored (except `.gitkeep`) | Local only | Experiment tracking |
| `samples/` | Committed | Shared | Canonical pipeline examples |
| `notebooks/` | Committed | Shared | Analysis notebooks |

### Data Conventions

1. **Never modify `data/raw/`** - Keep originals intact
2. **Use Parquet for processed data** - Efficient, typed storage
3. **HuggingFace cache** - Auto-managed at `~/.cache/huggingface/` in container

---

## MLflow Configuration

### Local File-Based Tracking

Default configuration uses local filesystem:

```
mlruns/
├── 0/                      # Default experiment
├── [experiment_id]/        # Named experiments
│   ├── meta.yaml
│   └── [run_id]/
│       ├── params/
│       ├── metrics/
│       └── artifacts/
```

### Starting the UI

```bash
# In container terminal (--host required for container access)
mlflow ui --host 0.0.0.0

# Access via the MLflow web interface
```

### Tracking Server (Optional)

For persistence across containers or team sharing:

```bash
mlflow server \
    --backend-store-uri sqlite:///mlflow.db \
    --default-artifact-root ./mlartifacts
```

---

## Stack Preferences

### Python Environment

| Tool | Purpose |
|------|---------|
| pandas, numpy | Data manipulation |
| scikit-learn | ML models |
| PyTorch | Deep learning (GPU) |
| MLflow | Experiment tracking |
| HuggingFace datasets | Dataset loading |

### Dependency Groups

```toml
# pyproject.toml
[project.optional-dependencies]
mlops = ["mlflow", "datasets"]
gpu = ["torch", "torchvision"]
notebooks = ["jupyter", "jupyterlab", "matplotlib", "seaborn"]
dev = ["pytest", "ruff", "mypy"]
```

### Installing Dependencies

```bash
# In container (automatic via postCreateCommand)
pip install -e ".[dev,notebooks,mlops]"

# Or specific groups
pip install -e ".[mlops]"
```

---

## GPU Configuration

### Host Requirements

| Component | Minimum | Verified |
|-----------|---------|----------|
| NVIDIA Driver | 525+ | 575.64.03 |
| CUDA | 12.0+ | 12.9 |
| nvidia-container-toolkit | Latest | Installed |

### Container GPU Access

Configured in `.devcontainer/devcontainer.json`:

```json
{
    "runArgs": ["--gpus", "all"]
}
```

### Verifying GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
print(f"Memory: {torch.cuda.get_device_properties(0).total_mem / 1e9:.1f} GB")
```

### Memory Management

```python
# Clear GPU cache
torch.cuda.empty_cache()

# Monitor usage
nvidia-smi  # From container terminal
```

---

## Environment Variables

### Required

None - DevContainer is self-contained.

### Optional (for integrations)

```bash
# .env (gitignored)

# Future: semops-core integration
SUPABASE_URL=...
SUPABASE_SERVICE_ROLE_KEY=...

# Future: cloud LLM APIs
ANTHROPIC_API_KEY=...
OPENAI_API_KEY=...
```

---

## Planned Services

### Stack Simulation

| Service | Purpose |
|---------|---------|
| Marquez | Lineage visualization |
| Marquez API | Lineage queries |

### Integration with semops-core (Active)

The Research RAG module depends on these services from semops-core:

| Service | Purpose | Used By |
|---------|---------|---------|
| **Qdrant** | Vector storage for embeddings | `research/embed.py` |
| **Docling** | PDF/document processing | `research/ingest.py` |
| **Ollama** | Local embeddings (nomic-embed-text) | `research/embed.py` |

**Starting semops-core services:**

Ensure the semops-core infrastructure services (Qdrant, Docling, Ollama) are running before using the Research RAG module.

### External APIs (Research Module)

| Service | Purpose | Config |
|---------|---------|--------|
| **OpenAI** | `text-embedding-3-small` embeddings | `OPENAI_API_KEY` |
| **Anthropic** | Claude LLM synthesis | `ANTHROPIC_API_KEY` |
| **Crawl4AI** | Web scraping | No API key required |

---

## Troubleshooting

### GPU Not Detected in Container

```bash
# 1. Verify host GPU
nvidia-smi

# 2. Verify Docker runtime
docker info | grep -i runtime

# 3. Test container GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. Rebuild container
# Ctrl+Shift+P -> "Dev Containers: Rebuild Container"
```

### Port Already in Use

```bash
# Find what's using a port
lsof -i :<port>
# or
docker ps --filter "publish=<port>"
```

### Container Won't Start

```bash
# Check Docker logs
docker logs <container-id>

# Rebuild from scratch
# Ctrl+Shift+P -> "Dev Containers: Rebuild Container Without Cache"
```

### Out of GPU Memory

```python
# Clear cache
torch.cuda.empty_cache()

# Reduce batch size in training
loader = DataLoader(dataset, batch_size=16)  # Smaller

# Use mixed precision
from torch.cuda.amp import autocast
with autocast():
    outputs = model(inputs)
```

---

## Related Documents

- [ARCHITECTURE.md](ARCHITECTURE.md) - System architecture
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the toolkit
- ADR-0002 - DevContainer decision
