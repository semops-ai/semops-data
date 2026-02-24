# Runbook

> **Repo:** `semops-data`
> **Status:** ACTIVE
> **Version:** 1.0.0
> **Last Updated:** 2025-12-17

## Quick Reference

| Task | Command |
|------|---------|
| Open DevContainer | VSCode → Reopen in Container |
| Start Jupyter | `jupyter lab --port 8888` |
| Start MLflow UI | `mlflow ui --host 0.0.0.0 --port 5000` |
| Run tests | `pytest` |
| Lint | `ruff check . && ruff format .` |

## DevContainer Operations

### Starting the Environment

1. Open repo in VSCode
2. `Ctrl+Shift+P` → "Dev Containers: Reopen in Container"
3. Wait for build (~5 min first time)
4. Run `notebooks/00-environment-test.ipynb` to verify

### Rebuilding Container

After changing `.devcontainer/Dockerfile`:

```
Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

For clean rebuild:
```
Ctrl+Shift+P → "Dev Containers: Rebuild Container Without Cache"
```

### Verifying GPU

```python
import torch
print(f"CUDA available: {torch.cuda.is_available}")
print(f"GPU: {torch.cuda.get_device_name(0)}")
```

Or from terminal:
```bash
nvidia-smi
```

## Common Issues

### Issue: GPU not detected in container

**Symptoms:**
- `torch.cuda.is_available` returns False
- `nvidia-smi` not found

**Cause:**
- Docker nvidia runtime not configured
- Container not started with GPU access

**Fix:**
```bash
# 1. Verify host GPU
nvidia-smi

# 2. Verify Docker runtime
docker info | grep -i runtime

# 3. Test container GPU access
docker run --rm --gpus all nvidia/cuda:12.1.0-base-ubuntu22.04 nvidia-smi

# 4. Rebuild container
# Ctrl+Shift+P → "Dev Containers: Rebuild Container"
```

### Issue: Out of GPU memory

**Symptoms:**
- CUDA OOM error during training
- Kernel crashes in notebook

**Cause:**
- Model too large
- Previous runs didn't release memory

**Fix:**
```python
# Clear cache
import torch
torch.cuda.empty_cache

# Reduce batch size
loader = DataLoader(dataset, batch_size=16)

# Use mixed precision
from torch.cuda.amp import autocast
with autocast:
 outputs = model(inputs)
```

### Issue: Port already in use

**Symptoms:**
- MLflow or Jupyter won't start
- "Address already in use" error

**Cause:**
- Previous process still running
- Conflict with semops-core services

**Fix:**
```bash
# Find what's using the port
lsof -i :5000

# Or check Docker
docker ps --filter "publish=5000"
```

### Issue: Container won't start

**Symptoms:**
- DevContainer build fails
- Hangs on startup

**Fix:**
```bash
# Check Docker logs
docker logs $(docker ps -aq --filter "ancestor=vsc-semops-data*" | head -1)

# Rebuild without cache
# Ctrl+Shift+P → "Dev Containers: Rebuild Container Without Cache"
```

## Data Management

### Directory Structure

```text
data/ # Working datasets (gitignored)
├── raw/ # Original downloads - never modify
├── processed/ # Cleaned, ready for modeling
└── interim/ # Intermediate processing

mlruns/ # MLflow tracking (gitignored)
samples/ # Canonical examples (committed)
```

### Data Conventions

1. Never modify `data/raw/` - keep originals intact
2. Use Parquet for processed data
3. HuggingFace cache auto-managed at `~/.cache/huggingface/`

## External Service Notes

### HuggingFace

**Auth:**
- Token optional for public datasets
- Set `HF_TOKEN` in `.env` for private datasets

**Gotchas:**
- Large datasets download to `~/.cache/huggingface/`
- Use `streaming=True` for very large datasets

### MLflow

**Local UI:**
```bash
mlflow ui --host 0.0.0.0 --port 5000
# Access at http://localhost:5000
```

**Gotchas:**
- `--host 0.0.0.0` required for container access
- Experiments persist in `mlruns/` (gitignored)

## Environment Setup

### Installing Dependencies

```bash
# In container (automatic via postCreateCommand)
pip install -e ".[dev,notebooks,mlops]"

# Or specific groups
pip install -e ".[mlops]"
```

### Environment Variables

```bash
# .env (optional, gitignored)
SUPABASE_URL=... # Future: semops-core integration
ANTHROPIC_API_KEY=... # Future: LLM integration
```

## Related Documentation

- [ARCHITECTURE.md](ARCHITECTURE.md) - System design
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) - DevContainer, ports
- [USER_GUIDE.md](USER_GUIDE.md) - How to use the toolkit
