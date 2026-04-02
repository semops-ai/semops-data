# SemOps Data Architecture

> **Version:** 1.0.0
> **Status:** Draft
> **Last Updated:** 2026-03-04
> **Related:** [STRATEGIC_DDD.md](STRATEGIC_DDD.md) | 

The SemOps data architecture defined using domain-first principles — not infrastructure lists but semantic boundaries, system type mappings, and neutral vocabulary that vendor implementations translate into.

**Key distinction:** This document defines *architecture* (entities, relationships, constraints, boundaries). For *infrastructure* (services, ports, containers), see [INFRASTRUCTURE.md](INFRASTRUCTURE.md). For *DDD model* (aggregates, capabilities, repos), see [STRATEGIC_DDD.md](STRATEGIC_DDD.md).

---

## 1. Principles

### 1.1 Domain-First, Not Infrastructure-First

Architecture is the organization and business encoded as a data structure ([What Is Architecture?](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/what-is-architecture.md)). The industry's dominant definition of "data architecture" describes plumbing — ETL pipelines, databases, cloud storage. That is infrastructure.

SemOps data architecture answers different questions:

| Question | Answer Lives In |
|----------|----------------|
| What semantic boundaries exist? | Section 3 (System Type Mapping) |
| Where does data cross boundaries? | Section 4 (Surface Analysis) |
| What entities exist and how do they relate? | [STRATEGIC_DDD.md](STRATEGIC_DDD.md) |
| What constraints apply? | Aggregate invariants + governance rules |
| Which architecture design approaches apply? | Section 1.3 below |

### 1.2 SemOps Neutral Reference Architecture 

SemOps vocabulary is the canonical reference. Vendor and platform concepts translate *into* SemOps terms, not the other way around. This means:

- **Component names in code** use SemOps canonical vocabulary, not vendor terms
- **Documentation** uses neutral primitives, with vendor translations provided for context
- **Method selection** is driven by which neutral paradigm fits, then mapped to available tooling
- **New tools evaluated** against the neutral capability they fulfill, not their marketing category

The neutral primitives come from six source documents:

| Source | Provides | Scope |
|--------|----------|-------|
| [Four Data System Types](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-system-classification.md) | 4 system type categories | Macro — what kinds of systems exist |
| [Data Engineering Core Framework](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-engineering-core-framework.md) | 7 Components × 5 Lifecycle × 6 Undercurrents | Internal — analytics system anatomy |
| [Surface Analysis](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/surface-analysis.md) | Sources, Surfaces, Provenance Position | Boundaries — where data crosses |
| [Data Systems Essentials](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-systems-essentials.md) | 3 Forces, OLTP/OLAP, Events/Streams/Tables | Lenses — analytical perspectives |
| [Vendor Comparison](../publisher-pr/docs/source/examples/data-systems-vendor-comparison.md) | 7 capabilities × 7 platforms | Translation — branded → neutral |
| [Glossary](../publisher-pr/docs/source/examples/data-system-glossary.md) | ~40 neutral concept definitions | Definitions — shared vocabulary |

### 1.3 Architecture Design Approaches

SemOps uses **DDD as the foundation** with specialized approaches layered on top, following the principle that DDD answers the foundational question before any other approach becomes meaningful.

| Approach | Applied Where | Rationale |
|----------|--------------|-----------|
| **DDD** | Everywhere — the foundation | Bounded contexts, ubiquitous language, aggregates, context mapping. SemOps has one bounded context with a single ubiquitous language ([STRATEGIC_DDD.md §Principles](STRATEGIC_DDD.md#principles)). |
| **Ontology-First** | Pattern aggregate, knowledge model | SemOps is knowledge-intensive by nature. SKOS taxonomy, provenance tiers, and formal semantic relationships require ontological rigor. The Pattern aggregate is an ontology. |
| **Event-Driven** | Cross-system-type flows | Domain events (ingestion events, coherence assessment triggers, pattern changes) are the communication mechanism between system types. Not yet formalized — currently HITL. |
| **Data Mesh** | Not yet applicable | Single operator, no autonomous domain teams. Data Mesh principles (domain ownership, data-as-product) become relevant at organizational scale. Reserved for Scale Projection Level 7+. |

**Why not Data Mesh now:** Data Mesh requires distributed team ownership. SemOps is a one-person operation where all repos are agent role boundaries, not team boundaries. The domain ownership principle is already satisfied by the repo structure — each repo scopes an agent's context. Mesh formalism adds overhead without benefit until there are multiple humans or persistent autonomous agents.

---

## 2. SemOps Canonical Vocabulary

SemOps defines its data architecture vocabulary at four abstraction levels. Each level maps neutral primitives from the source frameworks to concrete SemOps implementations, then provides vendor translations for context.

> **Note:** Vendor translation tables are living documents — an ongoing research effort, not a one-time mapping. See [Vendor Comparison](../publisher-pr/docs/source/examples/data-systems-vendor-comparison.md) for the detailed cross-vendor capability matrix. Extend with new platforms (Oracle, SAP, ClickHouse, Flink) as research continues.

### 2.1 System Types

The macro-level vocabulary. The [Four Data System Types](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-system-classification.md) are already neutral — they ARE the SemOps canonical names. No renaming needed.

| System Type | SemOps Implementation | Query Interface |
|-------------|----------------------|-----------------|
| **Application Data System** | publisher-pr, sites-pr (Vercel websites) — using semops-data infrastructure (PostgreSQL, Qdrant, Neo4j) | OLTP |
| **Analytics Data System** | data-pr: DuckDB, MLflow, Jupyter, dbt | OLAP |
| **Enterprise Work System** | GitHub Issues/Projects, Session Notes, Claude Code conversations | Unstructured |
| **Enterprise Record System** | semops-backoffice: Financial pipeline (planned) | Constrained |

**Translation to industry terms:**

| System Type | Databricks Equivalent | Snowflake Equivalent | dbt Equivalent | Generic Academic |
|-------------|----------------------|---------------------|----------------|-----------------|
| Application Data System | — (not their domain) | — | — | Operational database, OLTP system |
| Analytics Data System | Lakehouse | Data Cloud | Project + Warehouse | Data warehouse, analytical environment |
| Enterprise Work System | — | — | — | Knowledge management system |
| Enterprise Record System | — | — | — | System of record, ERP |

### 2.2 Analytics Components

The internal anatomy of the **Analytics Data System** (data-pr), using the [Data Engineering Core Framework](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-engineering-core-framework.md)'s 7 components as the neutral skeleton.

data-pr owns the **complete OLAP stack** — from extraction through serving — organized as a medallion architecture:

```text
semops-data (OLTP)              data-pr (OLAP — medallion architecture)
┌──────────────────┐              ┌──────────────────────────────────────┐
│ PostgreSQL       │              │  Bronze (raw extracts)               │
│ Qdrant           │── Extract ──►│  Silver (cleaned, conformed — ACL)   │
│ Neo4j            │              │  Gold (analytical models, metrics)   │
│                  │              │  Serving (experiments, reports)      │
└──────────────────┘              └──────────────────────────────────────┘
```

| SemOps Component | Neutral Component (Reis & Housley) | Medallion Layer | Current Implementation | Lifecycle Stage |
|-----------------|-----------------------------------|----------------|----------------------|-----------------|
| **Ingestion** | Ingestion & Streaming | → Bronze | Manual scripts, CSV/JSON loads, PostgreSQL reads from Application Data System | in_progress |
| **Storage** | Raw Storage | Bronze | Local filesystem (`data/`), gitignored working datasets, Parquet files | in_progress |
| **Table Layer** | Table Layer | Bronze → Silver | DuckDB tables, dbt staging models (planned — these are the ACL) | in_progress |
| **Compute** | Analytic Engines | Silver → Gold | DuckDB SQL, Python (pandas, polars), Jupyter notebooks | in_progress |
| **Orchestration** | Orchestration & Pipelines | All layers | n8n (from semops-data), manual CLI (`dst` command) | planned |
| **Catalog** | Catalog, Governance & Security | All layers | dbt catalog (schema.yml), manual documentation | planned |
| **ML/AI Services** | AI/ML Services | Gold → Serving | MLflow (experiment tracking), Ollama (embeddings), coherence scoring | in_progress |

**Medallion layer mapping:**

| Layer | Purpose | dbt Equivalent | Contents |
|-------|---------|---------------|----------|
| **Bronze** | Raw extracts, source-conformed | `staging/` | Entities, edges, patterns extracted as-is from OLTP |
| **Silver** | Cleaned, conformed to analytical schema | `intermediate/` | Business-conformed models — this IS the ACL (source → analytical vocabulary) |
| **Gold** | Analytical models, metrics, aggregates | `marts/` | Coherence scores, profiles, dimensional models, experiment results |
| **Serving** | Consumption-ready outputs | `exposure` | Dashboards, reports, ML model inputs, API responses |

**Translation to vendor terms:**

| SemOps Component | Databricks | Snowflake | AWS | dbt |
|-----------------|-----------|-----------|-----|-----|
| Ingestion | — (external) | Snowpipe | Kinesis/Glue | sources |
| Storage | Cloud object store | Micro-partitions | S3 | — |
| Table Layer | Delta Lake | Snowflake Tables | S3 Tables (Iceberg) | models (materialized) |
| Compute | Spark/Photon | Virtual Warehouses | Athena/Redshift | — (delegates to warehouse) |
| Orchestration | Workflows | Tasks/Streams | Step Functions | — (external: Airflow) |
| Catalog | Unity Catalog | Global Services | Lake Formation | schema.yml + docs |
| ML/AI Services | MLflow, Model Serving | Snowpark ML/Cortex | SageMaker | — |

### 2.3 Data Movement

How data enters and flows through SemOps systems, using neutral ingestion paradigms.

| SemOps Flow | Neutral Paradigm | Current Implementation | Direction |
|-------------|-----------------|----------------------|-----------|
| **Domain Extract** | Batch ingestion | Python scripts reading PostgreSQL (entities, edges) | Application Data System → Analytics Data System |
| **Document Ingest** | File-based ingestion | Docling (PDF/document processing from semops-data) | External → Application Data System |
| **Embedding Generation** | Batch processing | Ollama batch embedding via coherence scoring pipeline | Analytics Data System internal |
| **Experiment Results** | Append-only log | MLflow experiment tracking (metrics, params, artifacts) | Analytics Data System internal |
| **Synthetic Generation** | Batch processing | SDV/Faker generating test datasets | Analytics Data System internal |
| **Work Capture** | Event stream (manual) | GitHub Issues, Session Notes, Claude Code conversations | Enterprise Work System → unstructured |

**Translation to vendor terms:**

| SemOps Flow | Databricks | Snowflake | dbt | Generic |
|-------------|-----------|-----------|-----|---------|
| Domain Extract | Auto Loader / DLT | Snowpipe | `source` + `staging` model | CDC/ETL from OLTP |
| Document Ingest | — | — | — | Document processing pipeline |
| Embedding Generation | MLflow + Spark UDF | Cortex embed | — | Embedding pipeline |
| Experiment Results | MLflow native | — | — | Experiment tracking |
| Synthetic Generation | Notebook job | Stored procedure | `seed` | Test data generation |
| Work Capture | — | — | — | Knowledge capture |

### 2.4 Governance and Lifecycle

Cross-cutting practices mapped to neutral disciplines from the Data Engineering Core Framework's Undercurrents.

| SemOps Practice | Neutral Discipline | Current Implementation | Status |
|----------------|-------------------|----------------------|--------|
| **Ubiquitous Language** | Data Management (semantics) | [UBIQUITOUS_LANGUAGE.md](../schemas/UBIQUITOUS_LANGUAGE.md) — single vocabulary across all repos | active |
| **Pattern Coherence** | Data Management (quality) | Coherence scoring (SC formula) — measures semantic drift between patterns and implementation | in_progress |
| **Provenance Tracking** | Data Management (lineage) | Pattern provenance tiers (1P/2P/3P), `doc_type` frontmatter, adoption lineage | active |
| **Architecture Sync** | Data Architecture | `/arch-sync` and `/global-arch-sync` skills — template compliance, REPOS.yaml, port drift | active |
| **Schema Governance** | Security + Data Management | UBIQUITOUS_LANGUAGE.md is protected — changes require explicit review | active |
| **Experiment Tracking** | DataOps (observability) | MLflow — versioned experiments with metrics, parameters, artifacts | in_progress |

**Translation to vendor terms:**

| SemOps Practice | Databricks | Snowflake | dbt | Generic |
|----------------|-----------|-----------|-----|---------|
| Ubiquitous Language | Unity Catalog tags | Object tagging | `schema.yml` descriptions | Data dictionary, business glossary |
| Pattern Coherence | Lakehouse Monitoring | Data Quality functions | `dbt test` + custom tests | Data quality framework |
| Provenance Tracking | Unity Catalog lineage | Access History | `ref` lineage graph | Data lineage system |
| Architecture Sync | — | — | — | Architecture governance |
| Schema Governance | Unity Catalog permissions | RBAC/masking | — | Schema registry |
| Experiment Tracking | MLflow native | — | — | ML experiment tracking |

---

## 3. Four Data System Types — SemOps Mapping

Each data system type is a bounded context with its own semantic rules, governance, and physics. This section maps each type to concrete SemOps components.

### 3.1 Application Data System

> **Query Interface:** OLTP (PostgreSQL, Vercel/Next.js)
> **DDD Pattern:** Repository pattern — persists aggregates

The Application Data System is where core domain products live — the systems that serve users and persist operational data. In SemOps, semops-data is **not** an Application Data System itself — it is infrastructure and schema authority that Application Data Systems consume.

**Current Application Data Systems:**

| Repo | Product | OLTP Interface | Domain |
|------|---------|---------------|--------|
| **sites-pr** | Public Vercel websites (semops-ai.com, semops.ai) | Next.js + Supabase | Content delivery, user-facing web applications |
| **publisher-pr** | Content publishing workflow | Blog agents, content pipeline | Content creation, publishing lifecycle |

**Infrastructure provider (NOT an Application Data System):**

| Component | Owner | Purpose |
|-----------|-------|---------|
| PostgreSQL (Supabase) | semops-data (port 5432/5434) | Shared OLTP infrastructure — aggregate persistence for entities, edges, patterns |
| Qdrant | semops-data (port 6333/6334) | Shared vector storage for embeddings |
| Neo4j | semops-data (port 7474/7687) | Shared graph queries over entity relationships |
| Ingestion scripts | semops-data `src/` | Entity builders, edge creation — the DDD Repository layer |

**Key distinction:** semops-data owns the infrastructure and schema (ubiquitous language, aggregate definitions), but it is not the product. When a core domain attaches — like sites-pr deploying a website or publisher-pr managing content — *that* is when the Application Data System emerges. The infrastructure is the platform; the Application Data System is what runs on it.

**Governance approach:** DDD-governed. Single ubiquitous language. Aggregate root invariants protect semantic integrity. Changes flow from domain model → implementation.

**Data objects:**
- `entity` — domain objects with identity and lifecycle
- `entity_edge` — typed relationships between entities
- `pattern` — SKOS-structured knowledge with provenance
- `pattern_edge` — hierarchical and associative pattern relationships
- Website content — pages, blog posts, user-facing assets (sites-pr)
- Publishing artifacts — content drafts, blog agent outputs (publisher-pr)

### 3.2 Analytics Data System

> **Owner:** data-pr
> **Query Interface:** OLAP (DuckDB, SQL)
> **DDD Pattern:** No Repository — read-optimized, not write-optimized

data-pr is the **complete OLAP stack** — extraction, storage, table layer, compute, MLOps, and serving. It owns the medallion architecture and all analytical workloads.

**Medallion Architecture:**

| Layer | Purpose | Contents |
|-------|---------|----------|
| **Bronze** | Raw extracts, source-conformed | Entities, edges, patterns from OLTP; synthetic datasets; external data files |
| **Silver** | Cleaned, conformed to analytical schema (this IS the ACL) | Business-conformed models, resolved references, validated schemas |
| **Gold** | Analytical models, metrics, aggregates | Coherence scores, profiles, dimensional models, experiment results |

**Components:**

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| DuckDB | Local OLAP engine | Analytical queries, table layer |
| dbt | Transformation layer | Medallion layer models (staging = ACL, marts = gold) |
| MLflow | MLOps | Experiment tracking, model lifecycle, metrics |
| Jupyter | Notebooks | Exploratory analysis, data science workflows |
| Ollama | Embeddings | Embedding generation for coherence scoring |
| SDV/Faker | Synthetic data sources | Test dataset generation — ingested into bronze like any other source |
| ydata-profiling | Data profiling | Statistical profiles of any dataset |

**Governance approach:** Model semantic relationships BEFORE ETL. Enforce conformed dimensions. Measure semantic coherence. Track metric definitions to prevent drift.

**Data objects:**

- Bronze extracts — raw entities, edges, patterns from Application Data System
- Synthetic datasets — generated data (SDV/Faker) ingested as another bronze source
- Silver models — business-conformed analytical tables
- Gold metrics — coherence scores, profiles, dimensional aggregates
- Experiment artifacts — MLflow-tracked metrics, parameters, model artifacts
- Profiling reports — statistical profiles of datasets

### 3.3 Enterprise Work System

> **Owner:** Distributed (GitHub, Claude Code, human workflows)
> **Query Interface:** Unstructured (search, retrieval)
> **DDD Pattern:** None — scaffolding added through session notes, issue templates

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| GitHub Issues | Per-repo issue trackers | Task tracking, acceptance criteria, scope definition |
| GitHub Projects | Cross-repo project boards | Status tracking, sprint-like workflow |
| Session Notes | `docs/session-notes/ISSUE-NN-*.md` | Agent memory across sessions — append-forever logs |
| ADRs | `docs/decisions/ADR-NNNN-*.md` | Architectural decision records — versioned rationale |
| Claude Code conversations | Ephemeral (compressed in context) | Working memory during sessions |

**Governance approach:** Add scaffolding to impose structure on unstructured work artifacts. Session notes impose date-sectioned chronological structure. ADRs impose decision-record structure. Issue templates impose acceptance criteria. The Enterprise Work System is the least instrumented system type — the "dark data" problem applies here.

**Data objects:**
- Issues — scoped work units with acceptance criteria
- Session notes — chronological work logs tied to issues
- ADRs — structured decision records with status lifecycle
- Conversations — ephemeral working context (not persisted beyond session)

### 3.4 Enterprise Record System

> **Owner:** semops-backoffice (planned)
> **Query Interface:** Constrained (double-entry, regulatory compliance)
> **DDD Pattern:** Closest to Repository but with regulatory enforcement layer

| Component | Implementation | Purpose |
|-----------|---------------|---------|
| Financial pipeline | Planned | Double-entry bookkeeping, invoice processing |
| Compliance records | Planned | Regulatory filings, audit trails |

**Governance approach:** Canonical truth with constraint enforcement. Double-entry invariants. Regulatory compliance requirements. This is the least developed system type in SemOps — appropriate for a one-person operation where financial complexity is low.

**Data objects:**
- Financial transactions (planned) — double-entry ledger entries
- Invoices (planned) — accounts receivable/payable

### 3.5 Cross-Type Summary

```text
Application Data System (publisher-pr, sites-pr)
  Public websites (Vercel), content publishing
  OLTP — serves users, persists operational data
  ┌────────────────────────────────────────────┐
  │ website content, blog posts, publishing    │
  │ artifacts, user-facing applications        │
  └──────────────────┬─────────────────────────┘
                     │
            Uses infrastructure from
            semops-data (PostgreSQL,
            Qdrant, Neo4j — schema authority)
                     │
                     ▼
  ┌────────────────────────┐
  │ entities, edges,       │
  │ patterns, embeddings   │
  └──────────┬─────────────┘
             │
        Extract (batch/CDC)
             │
             ▼
Analytics Data System (data-pr)
  DuckDB, dbt, MLflow, Jupyter, Ollama
  OLAP — full medallion architecture + MLOps
  ┌──────────────────────────────────────────────┐
  │  Bronze: raw extracts + synthetic sources     │
  │  Silver: conformed models (ACL)               │
  │  Gold: coherence scores, metrics, profiles    │
  │  MLOps: experiments, model lifecycle           │
  └──────────────────────────────────────────────┘
             │
        Feedback Loop (HITL)
             │
             ▼
  Application Data System (pattern updates)

Enterprise Work System                    Enterprise Record System
  (GitHub, Claude Code)                     (semops-backoffice, planned)
  Issues, Session Notes, ADRs              Financial pipeline
  ┌────────────────────────┐               ┌────────────────────────┐
  │ task tracking, agent   │               │ transactions, invoices │
  │ memory, decisions      │               │ compliance records     │
  └────────────────────────┘               └────────────────────────┘
```

---

## 4. Surface Analysis — SemOps Boundaries

Surface Analysis reveals where data crosses boundaries in SemOps — which sources are boundary (where meaning enters) vs. internal (where meaning is derived).

### 4.1 Boundary Sources

Where data enters SemOps from the outside. These have highest semantic authority.

| Surface | Source Tuple | Provenance | System Type |
|---------|-------------|-----------|-------------|
| **Document ingestion** | (API, Docling, file-based, system, inbound) | Boundary | Application Data System (via semops-data infra) |
| **Pattern registration** | (CLI, manual, manual_entry, human, inbound) | Boundary | Application Data System (via semops-data infra) |
| **GitHub Issue creation** | (Web, GitHub, manual_entry, human, inbound) | Boundary | Enterprise Work System |
| **Claude Code conversation** | (Chat, Claude, direct, human+AI, bidirectional) | Boundary | Enterprise Work System |
| **External data files** | (File, local, file-based, human, inbound) | Boundary | Analytics Data System |

### 4.2 Internal Sources

Where data is derived or transformed within SemOps. Semantic authority inherited from upstream boundary sources.

| Surface | Source Tuple | Derives From | System Type |
|---------|-------------|-------------|-------------|
| **Entity extraction** | (Script, Python, batch, system, internal) | Document ingestion | Application Data System (semops-data infra) |
| **Edge creation** | (Script, Python, batch, system, internal) | Entity extraction | Application Data System (semops-data infra) |
| **Embedding generation** | (Script, Ollama, batch, system, internal) | Entities + patterns | Application Data System (semops-data infra) |
| **Coherence scoring** | (Script, MLflow, batch, system, internal) | Embeddings | Analytics Data System |
| **Synthetic datasets** | (Script, SDV/Faker, batch, system, internal) | Configuration templates | Analytics Data System |
| **Data profiles** | (Script, ydata, batch, system, internal) | Any dataset | Analytics Data System |
| **Session notes** | (File, Markdown, manual_entry, AI+human, internal) | Conversations + Issues | Enterprise Work System |
| **dbt models** | (Script, dbt, batch, system, internal) | Extracted domain data | Analytics Data System |

### 4.3 Surface Topology

```text
┌─────────────────────────────────────────────────────────────────┐
│                    SEMOPS SURFACE TOPOLOGY                       │
│                                                                  │
│   BOUNDARY (where meaning enters)                                │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │ Documents ── Pattern Definitions ── GitHub Issues ──     │   │
│   │ Claude Conversations ── External Data Files              │   │
│   └──────────────────────────┬──────────────────────────────┘   │
│                              │                                   │
│                              ▼                                   │
│   INTERNAL (where meaning is derived)                            │
│   ┌─────────────────────────────────────────────────────────┐   │
│   │                                                          │   │
│   │  Application Data System (via semops-data infra):       │   │
│   │  Documents ──► Entities ──► Edges ──► Embeddings         │   │
│   │       │              │                    │               │   │
│   │       │              │                    ▼               │   │
│   │       │              │          Analytics Data System:    │   │
│   │       │              └──────► Coherence Scores            │   │
│   │       │                       Profiles                    │   │
│   │       │                       dbt Models                  │   │
│   │       │                       Experiments (MLflow)        │   │
│   │       │                                                   │   │
│   │  Enterprise Work System:                                             │   │
│   │  Issues ──► Session Notes ──► ADRs ──► Architecture Docs │   │
│   │       │                                                   │   │
│   │       └──► Conversations (ephemeral)                      │   │
│   │                                                           │   │
│   └─────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────────────────┘
```

### 4.4 Provenance Chain

Tracing semantic authority from boundary to consumption:

```text
Document (boundary, highest authority)
  └─► Docling extraction (internal, inherited)
       └─► Entity creation (internal, inherited)
            └─► Edge creation (internal, inherited)
                 └─► Embedding (internal, inherited)
                      └─► Coherence score (internal, lowest authority)
                           └─► Action recommendation (interpretation)
```

Each step transforms the data, and each transformation inherits authority from its upstream source. When a coherence score seems wrong, trace upstream — the issue is more likely in extraction or entity creation than in the scoring formula itself.

---

## 5. Integration Patterns

How the four system types communicate, mapped to DDD Context Mapping patterns.

### 5.1 Context Map

| Upstream | Downstream | DDD Pattern | SemOps Implementation |
|----------|-----------|-------------|----------------------|
| Application Data System (semops-data infra) | Analytics Data System | **Customer-Supplier** | data-pr reads from semops-data PostgreSQL. semops-data (supplier) owns the schema; data-pr (customer) adapts. |
| semops-data (schema authority) | All system types | **Published Language** | UBIQUITOUS_LANGUAGE.md is the shared vocabulary. All repos use domain terms defined there. |
| Enterprise Work System | Application Data System | **Conformist** | New patterns and entities discovered in Enterprise Work System (conversations, research) are registered using semops-data's schema — no negotiation. |
| External Sources | Application Data System | **Anti-Corruption Layer** | Docling and ingestion scripts (semops-data) translate external document formats into the SemOps entity model. The ACL protects the domain model from format-specific concerns. |
| Analytics Data System | Application Data System | **Feedback Loop** (not classical DDD) | Coherence scores may trigger pattern updates or entity corrections. Currently HITL — human reviews scores and decides action. |

### 5.2 Data Flows

```text
                    ┌──────────────────┐
   External ───────►│  ACL (Docling,   │──────► Application Data System
   Documents        │  ingestion)      │        (semops-data infra)
                    └──────────────────┘              │
                                                      │
   Application Data System ──────────────────────────►│
   (publisher-pr, sites-pr)                           │
   Content, user data                                 │
                                                      │
                                            Customer-Supplier
                                            (SQL extract)
                                                      │
                                                      ▼
                                              Analytics Data System
                                              (data-pr)
                                                      │
                                              Coherence scores,
                                              profiles, experiments
                                                      │
                                              Feedback Loop (HITL)
                                                      │
                                                      ▼
                                              Application Data System
                                              (pattern updates)
```

### 5.3 ACL Boundaries

Anti-Corruption Layers in SemOps — where external concepts are translated into domain vocabulary:

| ACL | Translates From | Translates To | Implementation |
|-----|----------------|--------------|----------------|
| **Docling** | PDF, HTML, DOCX document formats | Structured text for entity extraction | semops-data Docling service (port 5001) |
| **Ingestion scripts** | External data schemas (CSV, JSON, API responses) | SemOps entity model (entity + edges) | semops-data `src/` scripts |
| **dbt staging** | Raw extracted data (source-conformed) | Business-conformed analytical models | data-pr dbt staging models (planned) |
| **Embedding pipeline** | Raw text | Vector representations in SemOps embedding space | Ollama (port 11434) + coherence scoring pipeline |

**Key insight from  research:** dbt staging layers ARE Anti-Corruption Layers. "Source-conformed → business-conformed" is Evans' ACL pattern in data engineering vocabulary. When data-pr formalizes its dbt layer, staging models should be understood and documented as ACLs.

---

## 6. Scale Projection

How this data architecture handles growth, validated using [Scale Projection](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/scale-projection.md).

### 6.1 Current State

SemOps operates at **Scale Projection Level 3 (Project)** with elements of Level 4 (Service):

| Axis | Current Level | Evidence |
|------|--------------|---------|
| **Directed** | 3 — Project | Documented goals, STRATEGIC_DDD.md, this document |
| **Context** | 3-4 | Rich domain context (patterns, ubiquitous language), but not yet machine-queryable |
| **Isolation** | 3 | Repos as boundaries, Docker for services, but manual coordination |
| **Access** | 3-4 | PostgreSQL + MCP Server for some access, but mostly scripts |
| **State** | 3 | File-based + PostgreSQL, no formal state management |
| **Observability** | 2-3 | MLflow for experiments, git for changes, but no unified observability |

### 6.2 Gap Analysis (Projecting to Level 5-6)

| Gap | Category | Required Change |
|-----|----------|----------------|
| PostgreSQL → managed instance | **Clean** | Connection string change |
| Local DuckDB → cloud warehouse | **Clean** | Query endpoint change |
| Manual scripts → orchestrated pipelines | **Clean** | Wrap existing logic in DAGs |
| Ollama local → API embedding service | **Clean** | Endpoint change |
| Single-user → multi-tenant | **Concerning** | Need tenant isolation in Application Data Systems (sites-pr, publisher-pr) and shared infrastructure (semops-data) — not currently in domain model |
| Manual coherence review → automated | **Concerning** | Need state machine for coherence assessment lifecycle |
| Session notes in files → structured Enterprise Work System | **Concerning** | Current Enterprise Work System is file-based; scaling requires queryable storage |
| Git-based architecture sync → live validation | **Clean** | `/arch-sync` already templated; automate in CI |

### 6.3 What Holds at Scale

These architectural decisions are infrastructure-independent — they hold regardless of deployment:

1. **Four system types as bounded contexts** — the semantic boundaries don't change with infrastructure
2. **SemOps canonical vocabulary** — neutral terms remain valid whether running on DuckDB or Snowflake
3. **Surface Analysis topology** — boundary vs. internal sources are a property of the domain, not the deployment
4. **Customer-Supplier pattern** (Application Data System → Analytics Data System) — the relationship type holds even if the mechanism changes from SQL to CDC to event streams
5. **ACL boundaries** — Docling, staging layers, and ingestion scripts remain translation layers regardless of scale
6. **Provenance chain** — semantic authority flows from boundary sources inward at any scale

### 6.4 What Changes at Scale

| Current | At Scale | Why It Changes |
|---------|----------|---------------|
| DuckDB (local) | Cloud warehouse (Snowflake/BigQuery/Databricks) | Volume exceeds local OLAP capacity |
| Manual extracts | CDC streams | Latency requirements increase |
| File-based Enterprise Work System | Structured knowledge base | Need queryable work artifacts |
| HITL coherence review | Automated coherence pipeline | Volume of patterns exceeds human review capacity |
| Single Ollama instance | Embedding API service | Throughput and availability requirements |

**The test:** In every case above, the domain logic (what gets extracted, how coherence is scored, what the ACL translates) stays the same. Only the infrastructure wrapper changes. This confirms the architecture is explicit — it separates domain from deployment.

---

## Related Documentation

### This Repo (semops-data)

- [STRATEGIC_DDD.md](STRATEGIC_DDD.md) — Aggregates, capabilities, repo registry, DDD building blocks
- [INFRASTRUCTURE.md](INFRASTRUCTURE.md) — Services, ports, containers
- [ARCHITECTURE.md](ARCHITECTURE.md) — Repo-level architecture

### Source Frameworks (publisher-pr)

- [What Is Architecture?](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/what-is-architecture.md) — Architecture ≠ infrastructure
- [Four Data System Types](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-system-classification.md) — The 4 system type categories
- [Data Engineering Core Framework](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-engineering-core-framework.md) — 7 Components × 5 Lifecycle × 6 Undercurrents
- [Surface Analysis](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/surface-analysis.md) — Boundary vs. internal sources
- [Scale Projection](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/EXPLICIT_ARCHITECTURE/scale-projection.md) — Validation technique for architecture coherence
- [Data Systems Essentials](../publisher-pr/docs/source/semops-framework/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-systems-essentials.md) — Analytics data system hub

### Neutral Reference (publisher-pr examples)

- [Vendor Comparison](../publisher-pr/docs/source/examples/data-systems-vendor-comparison.md) — 7 capabilities × 7 platforms
- [Glossary](../publisher-pr/docs/source/examples/data-system-glossary.md) — Neutral concept definitions
- [Hot in AI](../publisher-pr/docs/source/examples/data-systems-hot-in-ai.md) — Vendor demystification
- [Ingestion Paradigms](../publisher-pr/docs/source/examples/data-ingestion-streaming.md) — Neutral ingestion/streaming patterns
- [Structure Examples](../publisher-pr/docs/source/examples/data-systems-structure-examples.md) — Concrete data structure examples

### Research (data-pr)

-  — DDD → data architecture research
-  — This document's originating issue
