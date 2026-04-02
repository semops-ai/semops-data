# GAPS.md — Scale Projection Assessment

> **Repo:** semops-data
> **Role:** Schema/Infrastructure — Schema owner, knowledge base, and retrieval services
> **Method:** `/scale-project` (K8s projection surface)
> **Last Projected:** 2026-03-08
> **Previous:** 2026-02-27 (manual projection, )

This document captures the gaps discovered by projecting semops-data's capabilities to K8s manifests. Part of [Scale Projection](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/domain-patterns/scale-projection.md).

**The Rule:** If "Blocking" has items, stop and fix abstractions. If "Concerning" grows faster than "Clean," you're accruing architecture debt.

---

## How This Works

1. Extract capabilities from ARCHITECTURE.md
2. Walk the chain: **Pattern → Capability → K8s Resource → Gap**
3. For each capability, project along 7 scale vectors — can you just change a number in the manifest, or does domain logic need to change?
4. Classify gaps: Clean (just infrastructure), Concerning (logic changes needed), Blocking (architecture rethink)

**Manifests:** `projection/` — K8s YAML files annotated with capability, vector, and gap classification.

---

## Capabilities Projected

| # | Capability | Lifecycle | K8s Resource | Manifest |
|---|-----------|-----------|-------------|----------|
| 1 | `domain-data-model` | active | Job, ConfigMap | reference-data.yaml |
| 2 | `internal-knowledge-access` | active | Deployment+HPA, Sidecar | query-serving.yaml |
| 3 | `ingestion-pipeline` | in_progress | CronJob (×2) | ingestion.yaml |
| 4 | `pattern-management` | active | Job | governance.yaml |
| 5 | `coherence-scoring` | in_progress | CronJob | governance.yaml |
| 6 | `agentic-lineage` | planned | *(embedded in ingestion CronJob)* | — |
| 7 | `bounded-context-extraction` | planned | Job | — |

### The Chain (per capability)

#### domain-data-model

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `ddd`, `skos`, `prov-o`, `unified-catalog` | DDD gives aggregate structure, SKOS gives taxonomy edges, PROV-O gives lineage predicates |
| Capability | Schema initialization, architecture parsing, Neo4j sync | Infrequent writes — schema DDL, parsing STRATEGIC_DDD.md, syncing pattern taxonomy |
| K8s Resource | Job, ConfigMap | One-shot or on-demand operations. Job fits "run once, succeed, stop." ConfigMap for reference data. |

**Decision:** Clear. Pure infrastructure housekeeping.

#### internal-knowledge-access

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `agentic-rag` | RAG is well-known 3P. 1P extension: two-layer retrieval (entity + chunk) with separate embedding strategies. |
| Capability | Semantic search (entity, chunk, hybrid), MCP server (10 tools), Query API (REST), graph traversal | Read-heavy, latency-sensitive, concurrent agent queries. Two transports: HTTP API and stdio MCP. |
| K8s Resource | Deployment+HPA (Query API), Sidecar (MCP) | Query API is stateless request/response. MCP is stdio per-agent — each agent pod gets its own sidecar, not a shared service. |

**Decision:** Two different K8s patterns for one capability. MCP sidecar scales with agent pods, not independently. Each sidecar needs its own DB connection → connection pooling math changes.

#### ingestion-pipeline

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `semantic-ingestion` (1p), `etl` (3p) | ETL is batch processing. Semantic-ingestion adds LLM classification at ingestion time. |
| Capability | Source-based ingestion (GitHub fetch → entity + chunks + embeddings + graph), embedding generation | Batch, write-heavy, sequential per source, rate-limited by external APIs |
| K8s Resource | CronJob (×2) | Scheduled batch work. concurrencyPolicy: Forbid prevents overlap. |

**Decision:** Clear match. **Note:** `medallion-architecture` was previously listed as a traced pattern but doesn't belong here — medallion is data-pr's OLAP concern. semops-data ingestion is OLTP domain object persistence.

#### pattern-management

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `pattern-language` (3p), `backstage-software-catalog` (3p), `provenance-first-design` (1p) | Pattern Language for the concept, Backstage for registry, provenance-first for 1P/2P/3P lineage. |
| Capability | Pattern registration from YAML, SKOS edge management, HITL bridge (detected → committed) | Infrequent, human-gated writes. Authority chain: pattern_v1.yaml → script → DB. |
| K8s Resource | Job | On-demand, human-triggered. GitOps: YAML changes → Job runs. |

**Decision:** Clear. Authority chain preserved in K8s.

#### coherence-scoring

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `semantic-coherence` (1p) | 1P pattern. Fitness functions (Ford/Parsons/Kua) are closest 3P ancestor. |
| Capability | Fitness functions (10 SQL checks), coverage views, coherence signal in episodes | Batch governance, periodic, read-heavy aggregation |
| K8s Resource | CronJob | Periodic batch validation. Run daily, check invariants, report. |

**Decision:** Clear. Cleanest capability — everything scales with infrastructure only.

#### agentic-lineage (planned)

| Chain Link | Value | Reasoning |
|-----------|-------|-----------|
| Pattern | `open-lineage` (3p), `episode-provenance` (3p) | OpenLineage defines DAG-structured lineage. Episode-provenance captures per-operation context. |
| Capability | Episode-centric provenance tracking | Write-heavy during ingestion (one episode per operation), read for audit/replay |
| K8s Resource | **No standalone resource** | Embedded in ingestion CronJob — episodes are written as part of ingestion, not independently. |

**Decision: Ambiguous.** This capability doesn't map to its own K8s resource. It's cross-cutting, embedded in ingestion. The question: is it really a separate capability, or an aspect of `ingestion-pipeline`? If it graduates to its own service (OpenLineage API), it would be a Deployment.

---

## Scale Vectors

| Vector | K8s Mechanism | Projected? |
|--------|--------------|-----------|
| Data Volume | PVC size, StorageClass, StatefulSet resources | Yes |
| Concurrent Users | HPA, replica count, connection pooling | Yes |
| Geographic Distribution | Multi-cluster, read replicas | Yes |
| Throughput / Latency | Resource limits, HPA on custom metrics | Yes (subsumed by Data Volume + Concurrent Users) |
| Data Complexity | Schema evolution, CRDs, operator pattern | Yes |
| Regulatory / Compliance | Namespace isolation, RBAC, NetworkPolicy | Yes |
| Team Size | RBAC, GitOps (ArgoCD/Flux), config validation | Yes |

### Per-Capability Vector Analysis

#### domain-data-model — 7 Clean, 0 Concerning

All vectors scale cleanly. Schema init is a one-shot Job. More tables/columns = same Job, longer run.

#### internal-knowledge-access — 4 Clean, 3 Concerning

| Vector | Scales? | Gap |
|--------|---------|-----|
| Data Volume | Yes | Clean — index tuning, PVC size |
| Concurrent Users | Partial | Concerning — N sidecars × 1 connection = connection exhaustion |
| Geographic Distribution | No | Concerning — hardcoded single-region DB |
| Throughput / Latency | Partial | Clean for now — hybrid search is 2 sequential calls, acceptable |
| Data Complexity | Yes | Clean — Neo4j handles deeper SKOS natively |
| Regulatory / Compliance | Partial | Concerning — no tenant isolation in search |
| Team Size | Yes | Clean — stateless API, documented MCP |

#### ingestion-pipeline — 3 Clean, 3 Concerning

| Vector | Scales? | Gap |
|--------|---------|-----|
| Data Volume | Partial | Concerning — no raw content cache for re-processing |
| Concurrent Users | No | Concerning — no advisory locks or upsert strategy |
| Geographic Distribution | N/A | Clean — batch Job, runs in one location |
| Throughput / Latency | Partial | Clean — rate limits are external (OpenAI, Claude) |
| Data Complexity | Yes | Clean — classifier interface unchanged |
| Regulatory / Compliance | Yes | Clean — per-source config controls |
| Team Size | Partial | Concerning — no source config validation |

#### pattern-management — 5 Clean, 1 Concerning

| Vector | Scales? | Gap |
|--------|---------|-----|
| Data Volume | Yes | Clean — pattern count bounded (100s) |
| Concurrent Users | Yes | Clean — human-gated, serial by design |
| Data Complexity | Partial | Concerning — HITL bridge doesn't scale with edge volume |
| Regulatory / Compliance | Yes | Clean — provenance tiers provide governance |
| Team Size | Yes | Clean — GitOps authority chain |

#### coherence-scoring — 7 Clean, 0 Concerning

All vectors scale cleanly. Fitness functions are independent SQL. At 10x entities, add indexes (infrastructure only).

### Vector Summary

| Capability | Clean | Concerning | Blocking |
|-----------|-------|------------|----------|
| `domain-data-model` | 7 | 0 | 0 |
| `internal-knowledge-access` | 4 | 3 | 0 |
| `ingestion-pipeline` | 3 | 3 | 0 |
| `pattern-management` | 5 | 1 | 0 |
| `coherence-scoring` | 7 | 0 | 0 |
| **Total** | **26** | **7** | **0** |

---

## Clean (just infrastructure — 9 items)

- [ ] **Connection pooling for search path** — *Capability: internal-knowledge-access* — *Vector: Concurrent Users* — *Manifest: query-serving.yaml*
  Route through Supavisor pooler (5432) instead of direct (5434). Scripts unchanged — `db_utils.get_db_connection` wraps the pool.

- [ ] **Neo4j memory scaling** — *Capability: internal-knowledge-access* — *Vector: Data Volume* — *Manifest: storage.yaml*
  Scale `heap_max_size` and `pagecache_size` with data volume. Docker resource limits.

- [ ] **pgvector HNSW index tuning** — *Capability: internal-knowledge-access* — *Vector: Data Volume* — *Manifest: storage.yaml*
  At 10x entities/chunks, tune `ef_construction` and `m` parameters. Add partitioning if needed. SQL unchanged.

- [ ] **Embedding generation batch optimization** — *Capability: ingestion-pipeline* — *Vector: Data Volume* — *Manifest: ingestion.yaml*
  `generate_embeddings.py` processes individually via OpenAI API. Batch API calls, add rate limiting and cost tracking.

- [ ] **Fitness function indexing** — *Capability: coherence-scoring* — *Vector: Data Volume* — *Manifest: governance.yaml*
  Full-table-scan fitness functions slow at 10x entities. Add covering indexes. SQL stays the same.

- [ ] **CI pipeline for fitness functions** — *Capability: coherence-scoring* — *Vector: Team Size* — *Manifest: governance.yaml*
  Fitness functions run manually today. Automate in CI on PR. No domain logic change.

- [ ] **Structured logging** — *Capability: all* — *Vector: Team Size*
  Scripts use `rich` console output. Replace with structured logging (JSON) for multi-contributor debugging.

- [ ] **Auth layer for Query API and Neo4j** — *Capability: internal-knowledge-access* — *Vector: Concurrent Users* — *Manifest: query-serving.yaml*
  No auth in dev. Add auth proxy (Kong, Supabase Auth) or application-level auth. Domain model unaffected.

- [ ] **Source config validation** — *Capability: ingestion-pipeline* — *Vector: Team Size* — *Manifest: ingestion.yaml*
  YAML configs in `config/sources/*.yaml` have no schema validation. Add pydantic model or JSON schema.

---

## Concerning (logic changes needed — 5 items)

- [ ] **: MCP sidecar connection scaling** — *Capability: internal-knowledge-access* — *Vector: Concurrent Users*
  - Current: MCP server uses a single module-level DB connection (`_conn = get_db_connection`). Each agent pod gets its own MCP sidecar, each with one connection.
  - At scale: N agents = N direct connections to PostgreSQL (port 5434, bypassing Supavisor). Connection exhaustion.
  - Required: Connection-per-request or pool within each MCP sidecar, routed through Supavisor (5432). Affects `api/mcp_server.py`, `scripts/db_utils.py`.
  - Note: The topology (sidecar per agent) is correct. The gap is the connection pattern within each sidecar.

- [ ] **: No raw content cache for re-processing** — *Capability: ingestion-pipeline* — *Vector: Data Volume*
  - Current: Ingestion fetches from GitHub, processes, persists to entity/chunk tables. No intermediate storage.
  - At scale: Re-processing (e.g., re-classify with improved LLM prompt) requires re-fetching from GitHub.
  - Required: Store raw fetched content (table or file cache) so re-processing doesn't require network round-trip.
  - Note: This is NOT a medallion gap. Medallion (bronze/silver/gold) is data-pr's concern. This is a caching concern for semops-data's OLTP persistence layer.

- [ ] **: No concurrency control on ingestion** — *Capability: ingestion-pipeline* — *Vector: Concurrent Users*
  - Current: `ingest_from_source.py` uses one transaction per source. No advisory locks. Overlapping sources in a single Job could produce duplicate entities.
  - Required: Advisory locks per source, or upsert conflict resolution strategy. This is a domain decision (last-write-wins? merge?).
  - Affects: `ingest_from_source.py`, `entity_builder.py`

- [ ] **: Multi-tenant search isolation** — *Capability: internal-knowledge-access* — *Vector: Regulatory/Compliance*
  - Current: All agents query all corpora. Corpus filtering exists but is advisory (agent chooses which corpus to query), not enforced (no RBAC preventing access).
  - Required: If multi-tenant, need RBAC on search results (namespace isolation, row-level security, or service-level auth).
  - Note: Not needed for current single-operator use. Becomes real if external adopters or autonomous agents with different trust levels are introduced.

- [ ] **: HITL bridge bottleneck** — *Capability: pattern-management* — *Vector: Data Complexity*
  - Current: `bridge_content_patterns.py` is a manual HITL process — human reviews detected edges and promotes to committed.
  - At scale: 10x detected edges = human review becomes bottleneck.
  - Required: Semi-automated promotion with confidence thresholds, or batch review tooling. Process change, not architecture change.
  - Note: The architecture (detected → committed promotion gate) is sound. The bottleneck is the process, not the model.

---

## Blocking (architecture rethink — 0 items)

None.

The domain model is coherent. Pattern + Coherence as co-equal aggregates, entity with type discriminator, SKOS taxonomy, PROV-O lineage — these abstractions hold across all 7 scale vectors. The single bounded context with shared ubiquitous language is correct.

All Concerning items are about how the domain model is *accessed* (connection pooling, search isolation) or *operated* (caching, concurrency, HITL throughput), not about the model itself.

---

## Coherence Test

| Question | Answer |
|----------|--------|
| Can every capability be expressed as K8s resources? | **Mostly** — 5 of 7 map cleanly. `agentic-lineage` is embedded in ingestion (cross-cutting). `bounded-context-extraction` is planned but maps to Job. |
| Is the diff between docker-compose.yml and projection/ just infrastructure? | **Yes** — Docker Compose services map to StatefulSets (PG, Neo4j), CronJobs (ingestion, fitness), Deployment+HPA (Query API). Same domain logic. |
| Do manifests work without changing domain logic? | **Yes, with 5 Concerns** — all Clean items are infrastructure-only. Concerning items require changes to access patterns, caching, and process, not to the domain model. |
| Can you explain every gap? | **Yes** — each traces to a specific capability and scale vector. No vague concerns. |

**Coherence verdict:** **Clean overall, with operational concerns.** 0 Blocking, 5 Concerning, 9 Clean. The domain model holds. The gaps are in operational patterns (connection management, caching, concurrency) that are expected formalizations for a system moving from solo operation to multi-consumer service.

---

## Methodology Corrections (vs. 2026-02-27 manual projection)

| Previous Finding | Correction | Why |
|-----------------|-----------|-----|
| "No staging layer (medallion not realized)" — Concerning  | Reclassified. Medallion is data-pr's OLAP concern. Actual gap: no raw content cache (narrower). | DATA_ARCHITECTURE.md clarifies system type boundaries. |
| "Analytics Data System" classification for semops-data | Incorrect. semops-data is infrastructure/schema authority. Analytics is data-pr. | DATA_ARCHITECTURE.md § 3.1 explicitly states this. |
| Chain: `Pattern → Capability → Data System Type → Script → Storage → Gap` | Simplified: `Pattern → Capability → K8s Resource → Gap`. Intermediate steps are background knowledge. | K8s as projection surface makes data system type and infrastructure tier implicit. |
| `agentic-lineage` as standalone gap (Concerning : "episodes lack DAG structure") | Reframed. Lineage doesn't have its own K8s resource — embedded in ingestion. DAG question belongs to when capability activates. | Chain reveals lineage is cross-cutting, not standalone. |
| MCP connection gap framed as "single-consumer assumption" | Reframed. MCP is a sidecar (one per agent pod). Gap: N sidecars × 1 connection = N DB connections. | K8s sidecar pattern clarifies topology. |

---

## History

| Date | Method | Change | Outcome |
|------|--------|--------|---------|
| 2026-02-27 | Manual projection  | Initial assessment | 0 blocking, 4 concerning, 9 clean |
| 2026-03-08 | `/scale-project` (K8s manifests) | Corrected system type, simplified chain, reclassified medallion gap, added 3 new concerns | 0 blocking, 5 concerning, 9 clean |
