# ADR-0001: Tech Stack and Architecture Foundations

> **Status:** Complete
> **Date:** 2024-12-05
> **Related Issue:** [](https://github.com/semops-ai/semops-data/issues/4) - Start the Data Modeling Tools

---

## Executive Summary

This ADR documents the foundational technology choices for the Data Systems Toolkit, a platform for simulating enterprise data architectures with synthetic data, lineage tracking, and stack simulation. Decisions were informed by research into existing tools and standards.

---

## Context

We are building a toolkit that:
1. Generates realistic synthetic data for e-commerce and web analytics scenarios
2. Simulates data at each stage of a modern analytics stack (S3 → Delta Lake → Snowflake)
3. Tracks lineage and provenance through the pipeline
4. Profiles data quality and provides educational explanations

The toolkit should be:
- Docker-based for reproducibility
- Cloud-first for LLMs, local-first for data
- Standards-based where mature standards exist
- Extensible for future phases (RAG corpus, consulting tools)

---

## Decision

### Core Runtime Environment

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Python Version** | 3.11+ | Balance of features and library support |
| **Package Management** | pyproject.toml + pip | Standard, works with Docker |
| **Containerization** | Docker Compose | Multi-service orchestration, easy teardown |
| **Local Analytics DB** | DuckDB | Zero-config, fast, Parquet-native |

### Synthetic Data Generation

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Primary Library** | SDV (Synthetic Data Vault) | Open source, good for structured/relational data |
| **Supplementary** | Faker | Simple field generation, widely adopted |
| **Model Type** | GaussianCopula (start) | Fast, good for tabular; upgrade to CTGAN if needed |

**Why not commercial tools (Gretel, Tonic)?**
- Cost prohibitive for learning/exploration phase
- SDV provides sufficient capability for Phase 1
- Can evaluate commercial options in Phase 2 if needed

### Data Transformation

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Transform Framework** | dbt-core | Industry standard, real lineage, SQL-based |
| **dbt Adapter** | dbt-duckdb | No external database required |

**Why real dbt instead of simulation?**
- Only slightly more work than simulating dbt artifacts
- Provides real, runnable transforms
- Native lineage through dbt docs and OpenLineage integration

### Lineage & Provenance

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Lineage Standard** | OpenLineage | Industry consensus, vendor-neutral, LF AI project |
| **Lineage Backend** | Marquez | Reference implementation, Docker-ready |
| **Provenance Framework** | W3C PROV (conceptual) | Use as mental model, not direct implementation |

**Why OpenLineage over W3C PROV for implementation?**
- OpenLineage is pragmatic and operationally focused
- Better tool integration (dbt, Airflow, Spark)
- W3C PROV is academically rigorous but low industry adoption
- Can emit PROV-compatible data via OpenLineage facets if needed

### Data Profiling

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Automated Profiling** | ysemops-dataofiling | Comprehensive reports, low effort |
| **Data Quality** | Great Expectations | Extensible, integrates with dbt |
| **Quick Checks** | dbt tests | Built-in, YAML-based |

### LLM Strategy

| Use Case | Choice | Rationale |
|----------|--------|-----------|
| **Development Agents** | Claude Code | Primary development workflow |
| **Embeddings** | OpenAI API | Cost-effective, high quality |
| **Future Local** | Ollama | Ease into local models for cost/privacy |

---

## Alternatives Considered

### Database: PostgreSQL vs DuckDB

**PostgreSQL:**
- Pros: More feature-rich, pgvector for embeddings
- Cons: Requires running service, more setup

**DuckDB (chosen):**
- Pros: Zero-config, embedded, fast analytics, Parquet-native
- Cons: Not suitable for concurrent writes (not needed for simulation)

**Decision:** DuckDB for Phase 1, add PostgreSQL/Supabase when needed for RAG corpus.

### Lineage: DataHub vs Marquez

**DataHub:**
- Pros: More features, larger community
- Cons: Complex infrastructure, resource-intensive

**Marquez (chosen):**
- Pros: Simpler, reference implementation, lighter weight
- Cons: Less polished UI

**Decision:** Marquez for Phase 1, evaluate DataHub if we need more discovery features.

### Synthetic Data: CTGAN vs GaussianCopula

**CTGAN:**
- Pros: Better for complex distributions, handles mixed types
- Cons: Slower training, requires more data

**GaussianCopula (chosen for start):**
- Pros: Fast, good for structured business data
- Cons: May not capture complex relationships

**Decision:** Start with GaussianCopula, upgrade to CTGAN for specific use cases.

---

## Consequences

**Positive:**
- Fast iteration with minimal infrastructure
- Standards-based where it matters (OpenLineage, dbt)
- Clear upgrade path to more sophisticated tools
- Docker-based for reproducibility

**Negative:**
- DuckDB not suitable for production-scale concurrent access
- Marquez UI is functional but not polished
- Some learning curve for team members unfamiliar with dbt

---

## Implementation Plan

### Phase 1A: Foundation (Complete)

- [x] Project scaffold (pyproject.toml, Docker Compose)
- [x] Synthetic data generators (e-commerce, analytics)
- [x] Data profiler
- [x] CLI for generation and profiling
- [ ] Stack simulation samples (S3 → Delta → Snowflake)
- [ ] dbt project with DuckDB
- [ ] OpenLineage + Marquez integration

### Phase 1B: Research RAG Pipeline (Complete)

- [x] Crawl4AI for web scraping
- [x] Docling for PDF processing
- [x] RAG corpus with OpenAI embeddings → Qdrant
- [x] RAPTOR-inspired meta-analysis (K-means clustering + Claude synthesis)
- [x] CLI interface for ingest/query/synthesize
- [x] AI Transformation meta-analysis completed (Issue #18)

### Phase 2: Orchestration
- [ ] Silo diagnosis agent
- [ ] Reference architecture library
- [ ] Consulting tool MVP

---

## Session Log

### 2024-12-05: Initial Project Setup
**Status:** Completed
**Tracking Issue:** PR #6

**Completed:**
- Created project scaffold with pyproject.toml, Docker Compose
- Implemented EcommerceDataGenerator (Shopify-style)
- Implemented AnalyticsDataGenerator (GA4-style)
- Added DataProfiler with ysemops-dataofiling
- Created CLI with generate/profile commands
- Documented research findings
- Set up ADR process

**Next Session Should Start With:**
1. Implement stack simulation (static samples first)
2. Set up dbt project with DuckDB
3. Test Docker Compose environment
4. Create sample canonical datasets

---

## References

- [docs/research/](../research/) - Research findings on tools and standards
- [OpenLineage Documentation](https://openlineage.io/)
- [Marquez Project](https://marquezproject.ai/)
- [SDV Documentation](https://sdv.dev/)
- [dbt Documentation](https://docs.getdbt.com/)

---

**End of Document**
