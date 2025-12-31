# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Role in Global Architecture

**Role:** Product (First publishable product)

```
semops-data [PRODUCT]
        │
        ├── Owns: Data utilities, Research RAG pipeline
        │   - Data engineering utilities
        │   - Research synthesis module
        │   - Jupyter notebooks
        │
        └── Depends on: semops-core [INFRASTRUCTURE]
            - Qdrant (vector storage)
            - Ollama (embeddings)
            - Docling (document processing)
```

**Key Insight:** This repo is designed for external users - clear documentation and standalone operation are priorities.

## Project Structure

```
semops-data/
├── research/              # Research RAG module
│   ├── ingest.py         # Web/PDF ingestion
│   ├── embed.py          # Embedding generation
│   ├── cluster.py        # K-means clustering
│   ├── synthesize.py     # LLM meta-analysis
│   └── cli.py            # CLI interface
├── src/                   # Data engineering utilities
├── notebooks/             # Jupyter notebooks
├── sources/               # Research source manifests
└── docs/                  # Documentation
```

## Research Module

The research module implements RAPTOR-inspired corpus analysis:

```bash
# Ingest sources
python -m research.cli ingest sources/manifest.json

# Generate embeddings
python -m research.cli embed

# Cluster and synthesize
python -m research.cli synthesize
```

**Key Pattern:** Research without questions - let the corpus reveal its themes through clustering rather than starting with specific queries.

## Development Environment

**DevContainer (recommended):**
```bash
# VSCode: Ctrl+Shift+P → "Dev Containers: Reopen in Container"
```

**Local:**
```bash
pip install -e ".[research]"
```

## Key Files

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture and ownership
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Development guide
- [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) - Service dependencies
- [research/](research/) - Research RAG module
- [docs/decisions/](docs/decisions/) - Architecture Decision Records

## Integration with semops-core

This repo can run standalone but benefits from semops-core infrastructure:

| Service | Port | Used For |
|---------|------|----------|
| Qdrant | 6333 | Vector storage |
| Ollama | 11434 | Local embeddings |
| Docling | 5001 | PDF processing |

Start semops-core services if available, or configure cloud alternatives in `.env`.
