# semops-data

[![GitHub](https://img.shields.io/badge/org-semops--ai-blue)](https://github.com/semops-ai)
[![Website](https://img.shields.io/badge/web-semops.ai-green)](https://semops.ai)

**Data engineering utilities and research RAG pipelines** - tools for understanding enterprise data architectures.

## What is This?

`semops-data` provides data engineering utilities and research analysis tools as part of the SemOps ecosystem. It's designed as a toolkit that can be used standalone or integrated with `semops-core` infrastructure.

## Features

- **Data Engineering Utilities** - Reusable tools for data processing and transformation
- **Research RAG Module** - Ephemeral corpus analysis using RAPTOR-inspired clustering
- **Jupyter Notebooks** - Interactive exploration and analysis
- **DevContainer Support** - Consistent development environment

### Research Synthesis Pattern (RAPTOR-inspired)

The research module implements a novel pattern for corpus-driven meta-analysis:

1. **Ingest** - Crawl web sources or process PDFs
2. **Embed** - Generate embeddings for all chunks
3. **Cluster** - K-means clustering to discover themes
4. **Synthesize** - LLM extracts themes and generates meta-analysis

This enables "research without questions" - the system discovers what's important in a corpus.

## Quick Start

```bash
# DevContainer (recommended)
# VSCode: Ctrl+Shift+P → "Dev Containers: Reopen in Container"

# Or install locally
pip install -e ".[research]"

# Research CLI
python -m research.cli ingest sources/manifest.json
python -m research.cli embed
python -m research.cli synthesize
```

## Stack

- **Python** - Core language
- **Docker** - DevContainer for consistent environment
- **Jupyter** - Interactive notebooks
- **Crawl4AI** - Web crawling and ingestion
- **Docling** - PDF/document processing
- **OpenAI** - Embeddings (text-embedding-3-small)
- **Qdrant** - Vector storage
- **Anthropic** - Synthesis (Claude)

## Role in Architecture

```
semops-dx-orchestrator [PLATFORM/DX]
 │
 ▼
semops-core [SCHEMA/INFRASTRUCTURE]
 │
 │ Provides: Qdrant, Ollama, Docling services
 │
 ▼
semops-data [PRODUCT] ← This repo
 │
 │ Owns: Data utilities, Research RAG
 │
 └── First publishable product in ecosystem
```

**Key insight:** This repo is designed for external users - it's the first product to be published for broader use.

## Documentation

- [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) - Architecture and ownership
- [docs/USER_GUIDE.md](docs/USER_GUIDE.md) - Development environment and usage
- [docs/INFRASTRUCTURE.md](docs/INFRASTRUCTURE.md) - Service dependencies and DevContainer
- [docs/RUNBOOK.md](docs/RUNBOOK.md) - Operational procedures

## Related Repositories

| Repository | Role | Description |
|------------|------|-------------|
| [semops-dx-orchestrator](https://github.com/semops-ai/semops-dx-orchestrator) | Platform/DX | Process, global architecture |
| [semops-core](https://github.com/semops-ai/semops-core) | Schema/Infrastructure | Services this repo depends on |
| [semops-publisher](https://github.com/semops-ai/semops-publisher) | Publishing | Content workflow |
| [semops-docs](https://github.com/semops-ai/semops-docs) | Documents | Framework docs |
| [semops-sites](https://github.com/semops-ai/semops-sites) | Frontend | Websites, apps |

## Contributing

This is currently a personal project by Tim Mitchell. Contributions are welcome once the public release is complete.

## License

[TBD - License to be determined for public release]
