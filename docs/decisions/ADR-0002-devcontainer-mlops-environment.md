# ADR-0002: DevContainer and MLops Environment

> **Status:** Complete
> **Date:** 2025-12-09
> **Related Issue:** 

---

## Executive Summary

Establish a GPU-enabled DevContainer as the primary development environment for data science workflows, with MLflow for experiment tracking. This provides an isolated, reproducible environment before pursuing stack simulation complexity.

---

## Context

The data-systems-toolkit project needs a foundational data science environment that:

1. **Isolates dependencies** - Avoid polluting the host system with Python packages
2. **Leverages local GPU** - RTX 4000 Ada (20GB VRAM) available for ML training
3. **Tracks experiments** - Need to compare model runs systematically
4. **Enables exploration** - Work with HuggingFace datasets and basic ML before synthetic data

### Current State

- Host has NVIDIA driver 575.64.03 with CUDA 12.9 support
- Docker nvidia runtime installed and set as default
- No containerized development environment exists
- No experiment tracking in place

### Sequencing Decision

This work is prioritized **before** the stack simulation work (branch `001-stack-simulation-lineage`) to establish foundational tooling. The simulation work will build on this environment.

---

## Decision

### Development Environment: DevContainer

| Choice | Rationale |
|--------|-----------|
| **DevContainer** over plain Docker | Native VSCode integration, "Reopen in Container" workflow |
| **DevContainer** over venv | Full isolation, reproducible across machines, GPU config included |
| **PyTorch base image** | Familiar environment, CUDA pre-configured, lighter than NGC images |

### Base Image: `pytorch/pytorch:2.2.0-cuda12.1-cudnn8-runtime`

| Factor | Consideration |
|--------|---------------|
| CUDA 12.1 | Compatible with host driver 12.9 (backward compatible) |
| Runtime vs devel | Smaller image, sufficient for inference and training |
| PyTorch included | Batteries included for deep learning if needed |

### Experiment Tracking: MLflow (local)

| Choice | Rationale |
|--------|-----------|
| **MLflow** over W&B | Local-first, no account required, simpler |
| **MLflow** over none | Systematic tracking prevents "what hyperparameters did I use?" confusion |
| **File-based** initially | `mlruns/` directory, upgrade to server later if needed |

MLflow provides:
- Parameter logging (hyperparameters)
- Metric tracking (accuracy, loss over time)
- Artifact storage (models, plots)
- Comparison UI on port 5000

### Dependency Groups

```toml
[project.optional-dependencies]
mlops = [
 "mlflow>=2.10.0",
 "datasets>=2.16.0", # HuggingFace datasets
]
notebooks = [
 "jupyter>=1.0.0",
 "jupyterlab>=4.0.0",
 # ... existing
]
```

### Directory Structure

```
data-systems-toolkit/
├── .devcontainer/
│ ├── devcontainer.json # VSCode config, GPU, ports
│ ├── Dockerfile # PyTorch base + deps
│ └── requirements.txt # Pinned versions (optional)
├── data/ # Local datasets (gitignored)
│ └── .gitkeep
├── mlruns/ # MLflow tracking (gitignored)
├── notebooks/
│ └── 00-environment-test.ipynb
└── ...
```

### Port Allocation

| Port | Service | Purpose |
|------|---------|---------|
| 8888 | Jupyter | Notebook interface |
| 5000 | MLflow UI | Experiment comparison |

These don't conflict with semops-core services (5432, 5678, 6333, 7474, 8000).

---

## Alternatives Considered

### Plain venv

**Pros:** Simpler, no Docker overhead
**Cons:** No GPU isolation, harder to reproduce, Python version tied to host

**Decision:** DevContainer provides better isolation and reproducibility.

### Docker Compose (non-DevContainer)

**Pros:** Can run headlessly, familiar pattern
**Cons:** No VSCode integration, manual attach for debugging

**Decision:** DevContainer for interactive development; can add docker-compose later for CI/headless runs.

### NVIDIA NGC Base Image (`nvcr.io/nvidia/pytorch:*`)

**Pros:** More tools pre-installed, NVIDIA-optimized
**Cons:** Much larger image (10GB+), slower pulls

**Decision:** PyTorch official image is sufficient and faster to iterate.

### Weights & Biases (W&B) for Tracking

**Pros:** Better UI, collaboration features
**Cons:** Requires account, cloud-dependent

**Decision:** MLflow for local-first approach; can add W&B later if collaboration needed.

---

## Consequences

### Positive

- Reproducible environment across machines
- GPU accessible in isolated container
- Experiment tracking from day one
- Clear separation from host Python environment
- Aligns with "overkill is underrated" philosophy

### Negative

- Container rebuild required for dependency changes
- Slightly slower startup than native Python
- Additional learning curve for DevContainer workflow

### Risks

- PyTorch base image updates may require Dockerfile adjustments
- Large model training may need memory tuning (Docker memory limits)

---

## Implementation Plan

1. Create `.devcontainer/Dockerfile` with PyTorch base
2. Create `.devcontainer/devcontainer.json` with GPU and port config
3. Add `mlops` dependency group to `pyproject.toml`
4. Create `data/` and `mlruns/` directories with gitignore
5. Create test notebook verifying environment
6. Update CLAUDE.md with usage instructions
7. Test full workflow: GPU, Jupyter, MLflow

---

## Session Log

### 2025-12-09: ADR Created

**Status:** Draft
**Participants:** Tim, Claude

**Decisions Made:**
- DevContainer over plain Docker or venv
- PyTorch 2.2.0 CUDA 12.1 base image
- MLflow for local experiment tracking
- Prioritize before stack simulation work

**Environment Verified:**
- GPU: RTX 4000 Ada, 20GB VRAM
- Driver: 575.64.03, CUDA 12.9
- Docker nvidia runtime: working

### 2025-12-09: Implementation Complete

**Status:** Complete
**Branch:** `012-devcontainer-mlops`

**Implemented:**

- `.devcontainer/Dockerfile` - PyTorch CUDA 12.1 base with data science libs
- `.devcontainer/devcontainer.json` - GPU passthrough, port forwarding (8888, 5000)
- `pyproject.toml` - Added `mlops` and `gpu` dependency groups
- `data/` and `mlruns/` directories with `.gitkeep`
- `notebooks/00-environment-test.ipynb` - Environment verification
- `docs/USER_GUIDE.md` - Practical workflow documentation
- Updated `CLAUDE.md` with DevContainer usage

**Verified:**

- Docker build succeeds
- GPU accessible in container (RTX 4000 Ada detected)
- PyTorch 2.2.0 with CUDA working
- MLflow 3.7.0, HuggingFace datasets, scikit-learn all load correctly

---

## References

- 
- [ADR-0001: Tech Stack Foundations](ADR-0001.md)
- [VSCode DevContainers Documentation](https://code.visualstudio.com/docs/devcontainers/containers)
- [MLflow Documentation](https://mlflow.org/docs/latest/index.html)
- [PyTorch Docker Images](https://hub.docker.com/r/pytorch/pytorch)

---

**End of Document**
