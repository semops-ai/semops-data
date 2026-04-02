# User Guide: Search & Retrieval

> **Version:** 2.8.0 | **Last Updated:** 2026-03-30
> **Related Issues:**  (query API),  (shared search module),  (ACL query tools),  (content→pattern bridge),  (query precision validation),  (deployment corpus),  (published corpus),  (GitHub Issues ingestion),  (agentic reasoning tools),  (agent entities + entity_type filter),  (concept→pattern audit)
> **See also:** [INGESTION_GUIDE.md](INGESTION_GUIDE.md) for ingesting content into the knowledge base

---

## Quick Reference

| Task | Command |
|------|---------|
| CLI search (chunks) | `./search.py "query"` |
| CLI search (entities) | `./search.py "query" --mode entities` |
| CLI search (hybrid) | `./search.py "query" --mode hybrid` |
| CLI search (to file) | `./search.py "query" --mode hybrid -o results.json` |
| Start query API | `uvicorn api.query:app --port 8101` |
| Check entity count | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT count(*) FROM entity;"` |
| Check chunk count | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT count(*) FROM document_chunk;"` |
| Corpus distribution | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT metadata->>'corpus', count(*) FROM entity GROUP BY 1 ORDER BY 2 DESC;"` |

**Note:** `./search.py` loads `.env` automatically. Other scripts assume `source .venv/bin/activate` (or use `uv run`) and `OPENAI_API_KEY` is set.

---

## Prerequisites

1. **Docker services running** — `python start_services.py --skip-clone`
2. **Python venv activated** — `source .venv/bin/activate` (or use `uv run`)
3. **`.env` file** with `OPENAI_API_KEY` and `POSTGRES_PASSWORD`
4. **Database connection** — Scripts connect automatically via `SEMOPS_DB_*` env vars configured in `.env`. Direct PostgreSQL access is available on port 5434.
5. **Content ingested** — At least one source ingested with embeddings generated. See [INGESTION_GUIDE.md](INGESTION_GUIDE.md).

---

## Retrieval Hierarchy

Agents follow the **most deterministic path first**, falling back to less structured surfaces only when needed. This isn't a preference — it's an architectural constraint. Deterministic answers are reproducible, auditable, and coherence-measurable. Semantic search is powerful but non-deterministic.

| Layer | Data Surface | What it answers | Tools |
|-------|-------------|-----------------|-------|
| **1. Manifest** | YAML registries (registry.yaml, pattern_v1.yaml, agents.yaml, UL) | "What exists? What's connected to what?" | ACL query tools (`list_patterns`, `list_capabilities`, `query_integration_map`) |
| **2. SQL** | Entity table (all types), pattern table, edge table, strategic views | "What are the properties? What are the relationships?" | ACL query tools (`get_pattern`, `get_capability_impact`, `run_fitness_checks`) |
| **3. Graph** | Neo4j nodes and typed edges | "What's related to this? How are things connected?" | `graph_neighbors` |
| **4. Vector** | Chunk + entity embeddings (pgvector/Qdrant) | "What's relevant to this question? What does this concept say?" | Semantic search tools (`search_knowledge_base`, `search_chunks`, `search_patterns`) |

**When to use each layer:**

- **Need a fact about the architecture?** Start at Layer 1–2 (ACL queries). "Which patterns does this capability implement?" is a SQL lookup, not a search.
- **Need to explore relationships?** Layer 3 (graph traversal). Follow typed edges from a known starting point.
- **Need to find something by meaning?** Layer 4 (vector search). "What does the framework say about governance?" requires semantic matching.
- **Need to go deep into a document?** Layer 4 chunk search. Passage-level retrieval for precise answers with citations.

Most agent workflows combine layers: ACL query to find the right entity, then vector search to read its content, then graph traversal to explore what's connected.

---

## How Semantic Search Works

The knowledge base implements **hybrid semantic search** — the standard RAG pattern of combining document-level retrieval with passage-level retrieval to get both broad relevance and precise answers. This is Layer 4 in the retrieval hierarchy above. The system decomposes this into two independent stages that you can run separately or together:

1. **Entity search** — finds *which documents* are relevant (document retrieval)
2. **Chunk search** — finds *which passages* answer your question (passage retrieval)
3. **Hybrid search** — runs both stages together: find top documents, then retrieve best passages within each

All search is **purely semantic**. Your query is converted into a 1536-dimensional embedding vector (OpenAI `text-embedding-3-small`), then compared against pre-computed embeddings using pgvector cosine similarity. There is no keyword or full-text search component — a query like "database connectivity" matches content about "PostgreSQL connection pooling" because the meaning is similar.

### The Three Retrieval Layers

The knowledge base has three independent vector search surfaces, each with its own HNSW index on `vector(1536)` embeddings:

**Entities** represent whole documents, capabilities, agents, and repositories. Each entity's embedding is built from structured metadata — dispatching on the `$schema` field to extract the right fields per entity type (see `build_embedding_text` in `scripts/generate_embeddings.py`). Content entities use title + summary + concepts; capability entities use domain classification + patterns + repos; agent entities use agent type + surface + layer + capabilities. This makes entity search effective for topic discovery across all entity types: "find me agents that handle architecture auditing."

**Chunks** represent passages within documents. During ingestion, each file is split by markdown headings (~512 tokens per section), and each chunk's embedding is built from its actual content text. This makes chunk search effective for precise retrieval: "find the paragraph that explains the embedding column."

```text
Content Entity: "schema-reference.md"
  embedding from: title + summary + content_type + primary_concept + subject_areas + broader/narrower concepts
  |
  +-- Chunk 0: "## Entity Table\nThe entity table is the..."
  |   embedding from: actual section text
  |
  +-- Chunk 1: "## Document Chunk Table\nChunks store..."
      embedding from: actual section text

Capability Entity: "ingestion-pipeline"
  embedding from: title + domain_classification + lifecycle_stage + implements_patterns + delivered_by_repos

Agent Entity: "agent-arch-sync"
  embedding from: title + agent_type + surface + layer + exercises_capabilities + delivered_by_repo
```

**Patterns** represent registered domain concepts from the pattern registry (`pattern` table in `phase2-schema.sql`). Each pattern's embedding is derived from its SKOS `definition` field. Pattern search finds patterns by meaning rather than by ID or exact label. This is useful when you know the concept but not its registered name: "find patterns related to managing content lifecycle" returns `dam` (Digital Asset Management).

```text
Pattern: "semantic-coherence"
  embedding from: definition text ("Degree to which an entity's declared patterns...")
```

Pattern embeddings are generated separately from entity embeddings (no automated script — generated ad-hoc when patterns are registered or definitions enriched). Currently 36/45 patterns have embeddings.

Because each layer's embeddings are built from different text, the same query can surface different results at each layer. Entity search for "embeddings" returns `schema-reference.md` (metadata topic match), chunk search for "embeddings" returns the specific paragraph explaining the embedding column (content match), and pattern search for "embeddings" returns nothing (no pattern is *about* embeddings — they're infrastructure, not domain).

### Search Modes

**Chunk search** (`--mode chunks`, default) — the passage retrieval stage. Searches `document_chunk` embeddings built from actual content. Best for finding specific answers.

Results include: similarity score (0.0-1.0), entity ID (parent document), heading hierarchy (section breadcrumbs like "Architecture > Entities > Attributes"), content preview (truncated to 500 chars in CLI/MCP), and chunk position (e.g., "3 of 7" within that entity).

**Entity search** (`--mode entities`) — the document retrieval stage. Searches `entity` embeddings built from structured metadata. Best for topic discovery and document-level relevance.

Results include: similarity score (based on metadata match, not full document content), title, corpus, content type, LLM-generated summary, and concept ownership (first-party vs. third-party).

**Pattern search** (MCP `search_patterns` tool) — the concept retrieval stage. Searches `pattern` embeddings built from definition text. Best for finding which registered patterns relate to a domain concept when you don't know the exact pattern ID.

Results include: similarity score, pattern ID, preferred label, definition, provenance (1p/3p), and coverage statistics (content_count, capability_count, repo_count from the `pattern_coverage` view).

**Hybrid search** (`--mode hybrid`) — the full two-stage pipeline. First finds the top-N relevant entities, then retrieves the best-matching chunks *within each entity*. This answers "which documents are relevant, and where specifically within them?" — the standard RAG retrieval pattern for grounded generation with precise citations.

### Similarity Scores

Scores range from 0.0 (completely unrelated) to 1.0 (identical meaning):

| Score Range | Interpretation |
| ----------- | -------------- |
| 0.80+ | Strong match — content directly addresses the query |
| 0.60-0.80 | Related — content is topically relevant but may not be a direct answer |
| Below 0.60 | Weak — try rephrasing the query or broadening terms |

Chunk searches typically produce higher scores than entity searches because chunks match against actual content, while entities match against metadata summaries.

### Graph Traversal (Separate)

The Neo4j knowledge graph tracks **relationships between concepts**. Graph traversal is completely separate from vector search — it doesn't rank by similarity, it follows typed edges between entities.

**What's in the graph:**

Two node types and two edge tables are synced to Neo4j:

| Data | Source table | Neo4j node/rel | Sync script |
| ---- | ---------- | -------------- | ---------- |
| Pattern taxonomy | `pattern` + `pattern_edge` | `(:Pattern)` + `[:EXTENDS]`, `[:ADOPTS]`, `[:RELATED]` | `scripts/sync_neo4j.py` |
| Entity relationships | `edge` | `(:Entity)` + `[:RELATED_TO]`, `[:IMPLEMENTS]`, `[:DOCUMENTS]`, `[:DELIVERED_BY]`, `[:INTEGRATION]` | `scripts/materialize_graph.py` |
| Agent relationships | `entity` (agent) + `edge` | `(:Agent)` + `[:IMPLEMENTS]` (→Capability), `[:DELIVERED_BY]` (→Repo) | `scripts/ingest_architecture.py` |

Current graph: **45 patterns** with 25 edges (18 `extends`, 5 `adopts`, 2 `related`), **261 entity edges** (98 `related_to`, 66 `implements`, 53 `documents`, 30 `delivered_by`, 14 `integration`), **32 agent nodes** with 64 edges (32 `implements` → capability, 32 `delivered_by` → repo).

**Direct access — Neo4j Browser:**

```text
http://localhost:7474
```

Username: `neo4j` / Password: from `.env` (`NEO4J_PASSWORD`). Example Cypher queries:

```cypher
// All patterns and their relationships
MATCH (p:Pattern)-[r]->(q:Pattern) RETURN p.id, type(r), q.id

// Patterns that extend DDD
MATCH (p:Pattern)-[:EXTENDS]->(q:Pattern {id: 'ddd'}) RETURN p.id, p.preferred_label

// Entities implementing a specific pattern
MATCH (e:Entity)-[:IMPLEMENTS]->(p:Pattern {id: 'semantic-coherence'}) RETURN e.id, e.title
```

**Via Query API:**

```bash
curl -s http://localhost:8101/graph/neighbors/semantic-flywheel | python3 -m json.tool
```

Returns direct neighbors (not transitive paths) with relationship types and directions. Agents typically use this to explore related concepts after finding a starting point via semantic search.

**Syncing the graph:**

```bash
# Sync pattern taxonomy (pattern + pattern_edge → Neo4j)
python scripts/sync_neo4j.py

# Sync entity relationships (edge → Neo4j)
python scripts/materialize_graph.py
```

Run these after bulk ingestion or schema changes to keep Neo4j consistent with PostgreSQL.

### Filtering

All search modes support **corpus filtering** (`--corpus core_kb research_ai`) to restrict results to specific knowledge domains. Entity search additionally supports **content type** (`--content-type domain-pattern framework`), **lifecycle stage** (`--lifecycle-stage active draft`), and **entity type** (`--entity-type agent capability`) filters. Filters are applied before vector ranking — they narrow the candidate set, not the similarity calculation.

The `entity_type` filter is particularly useful for scoping searches to a specific layer of the domain model:

| Entity Type | What it finds | Example query |
|-------------|---------------|---------------|
| `content` | Documents, issues, ADRs | "semantic coherence theory" |
| `capability` | System capabilities | "data ingestion capabilities" |
| `agent` | Slash commands, MCP tools | "architecture auditing agents" |
| `repository` | Code repositories | "publishing repo" |

### Available Corpus Types

Currently populated corpora (988 content entities total):

| Corpus             | Count | Description                                                              |
|--------------------|-------|--------------------------------------------------------------------------|
| `deployment`       | 859   | Operational: GitHub Issues, session notes, ADRs across 8 repos           |
| `core_kb`          | 116   | Curated knowledge: patterns, theory, schema                              |
| `published`        | 12    | Public-facing: semops-ai org READMEs (pub-* prefix)                      |
| `research_ai`      | 1     | AI/ML research: AI foundations, cognitive science                        |

The `deployment` corpus is the largest, sourced from all 8 Project Ike repos: semops-orchestrator (227), semops-data (169), publisher-pr (144), docs-pr (117), sites-pr (86), data-pr (54), semops-research (39), and semops-backoffice (23). Most deployment entities are GitHub Issues (584) ingested via `ingest_github_issues.py`, with session notes (217) and ADRs (58) making up the remainder.

Additional corpora defined in the ingestion pipeline but not yet populated:

| Corpus             | Description                                              |
|--------------------|----------------------------------------------------------|
| `research_general` | General research: ad-hoc, unsorted, triage               |

### Available Content Types

Currently populated content types (988 content entities):

| Content Type     | Count | Description                                        |
|------------------|-------|----------------------------------------------------|
| `issue`          | 584   | GitHub Issues ingested from all 8 Project Ike repos|
| `session_note`   | 217   | Session notes, issue explorations                  |
| `pattern`        | 61    | Pattern documentation (domain + general)           |
| `adr`            | 58    | Architecture Decision Records                      |
| `concept`        | 36    | Concept definitions and domain explanations        |
| `architecture`   | 16    | Architecture documentation and system design       |
| `article`        | 13    | General prose, published docs (incl. pub-*)        |
| `framework`      | 3     | Conceptual frameworks and theory documents         |

Content types are assigned by LLM classification during ingestion (`ingest_from_source.py`). The taxonomy is open — new types emerge as new sources are ingested.

---

## CLI Search

The `search.py` wrapper at the project root loads `.env` automatically and delegates to `scripts/semantic_search.py`. The CLI defaults to **chunk search** (passage-level retrieval). Use `--mode` to switch between search modes.

```bash
# Chunk search (default) — passage-level retrieval
./search.py "What is semantic coherence?"

# Entity search — document-level results
./search.py "domain patterns" --mode entities

# Hybrid search — top entities with best chunks per entity
./search.py "semantic flywheel" --mode hybrid

# Filter by corpus
./search.py "publication workflow" --corpus core_kb published

# Filter by content type (entity mode)
./search.py "domain patterns" --mode entities --content-type pattern

# Filter by lifecycle stage (entity mode only)
./search.py "draft concepts" --mode entities --lifecycle-stage draft

# Filter by entity type (entity mode only)
./search.py "architecture" --mode entities --entity-type agent
./search.py "ingestion" --mode entities --entity-type capability agent

# Combine filters with limit and verbose output
./search.py "provenance" --corpus core_kb --limit 5 --verbose

# Output results as JSON to a file
./search.py "data system classification" --mode hybrid -o results.json
```

**Search modes:**

| Mode | Flag | Description |
| ------ | ------ | ------------- |
| Chunks | `--mode chunks` (default) | Passage-level results from `document_chunk` table |
| Entities | `--mode entities` | Document-level results from `entity` table |
| Hybrid | `--mode hybrid` | Top entities with best-matching chunks per entity |

---

## FastAPI Query API

**Start the server:**

```bash
uvicorn api.query:app --port 8101
```

**Endpoints:**

| Method | Path | Description |
|--------|------|-------------|
| `POST` | `/search` | Entity-level semantic search with filters |
| `POST` | `/search/chunks` | Chunk (passage) level semantic search |
| `POST` | `/search/hybrid` | Two-stage: entity search then chunk retrieval within top entities |
| `GET` | `/graph/neighbors/{entity_id}` | Graph traversal — related concepts and entities |
| `GET` | `/corpora` | List corpora with entity counts |
| `GET` | `/health` | Health check |

**Entity search** (document-level):

```bash
curl -s http://localhost:8101/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "What is semantic coherence?",
    "corpus": ["core_kb"],
    "content_type": ["domain-pattern", "framework"],
    "limit": 5
  }' | python3 -m json.tool

# Search for agents only
curl -s http://localhost:8101/search \
  -H "Content-Type: application/json" \
  -d '{
    "query": "architecture auditing",
    "entity_type": ["agent"],
    "limit": 5
  }' | python3 -m json.tool
```

**Response fields:** `id`, `title`, `entity_type`, `corpus`, `content_type`, `summary`, `similarity`, `filespec`, `metadata`

**Chunk search** (passage-level):

```bash
curl -s http://localhost:8101/search/chunks \
  -H "Content-Type: application/json" \
  -d '{
    "query": "semantic compression in DDD",
    "corpus": ["core_kb"],
    "limit": 5
  }' | python3 -m json.tool
```

**Response fields:** `chunk_id`, `entity_id`, `source_file`, `heading_hierarchy`, `content`, `corpus`, `content_type`, `similarity`, `chunk_index`, `total_chunks`

**Hybrid search** (entity then chunk two-stage):

```bash
curl -s http://localhost:8101/search/hybrid \
  -H "Content-Type: application/json" \
  -d '{
    "query": "semantic flywheel pattern",
    "corpus": ["core_kb"],
    "limit": 5,
    "chunks_per_entity": 3
  }' | python3 -m json.tool
```

Returns top entities with their best-matching chunks inlined.

**Graph neighbors:**

```bash
curl -s http://localhost:8101/graph/neighbors/semantic-flywheel | python3 -m json.tool
```

Returns related entities and concepts from the Neo4j knowledge graph with relationship types and directions.

**List corpora:**

```bash
curl -s http://localhost:8101/corpora | python3 -m json.tool
```

---

## MCP Server (Cross-Repo Agent Access)

The MCP server exposes 12 tools across three query surfaces, allowing Claude Code agents in any repo to query the knowledge base.

**Configuration:** Already registered in `~/.claude.json` as `semops-kb`. Also configured in `.mcp.json` for project-level access.

### Semantic Search Tools (Content Discovery)

These use pgvector embeddings for probabilistic matching — best for finding relevant content.

| Tool | Description |
|------|-------------|
| `search_knowledge_base` | Entity-level semantic search with corpus/content_type/entity_type filters |
| `search_chunks` | Chunk (passage) level semantic search with heading hierarchy context |
| `list_corpora` | List available corpora with counts |

**Usage:**

```
Use the search_knowledge_base tool to find entities about "semantic coherence"
in the core_kb corpus.

Use the search_knowledge_base tool with entity_type ["agent"] to find agents
that handle architecture auditing.

Use the search_chunks tool to find specific passages about "anti-corruption layer"
for precise citation in a document.
```

### ACL Query Tools (Architectural Truth)

These are deterministic SQL lookups against the DDD schema — exact results from committed edges and the pattern registry. Use these when you need architecturally-aligned answers, not probabilistic matches.

| Tool | Description |
|------|-------------|
| `list_patterns` | Pattern registry with optional provenance filter (`1p`, `3p`) and coverage statistics (capability/repo/content counts) |
| `get_pattern` | Single pattern with all SKOS edges (broader/narrower/related), adoption relationships (adopts/extends/modifies), and coverage |
| `search_patterns` | Semantic search over pattern embeddings — finds patterns by meaning, not just ID |
| `list_capabilities` | Capabilities from the `capability_coverage` view with optional domain classification filter (`core`, `supporting`, `generic`) |
| `get_capability_impact` | Full impact analysis for a capability: implementing patterns, delivering repos, and integration dependencies between those repos |
| `query_integration_map` | DDD context map — integration edges between repos with patterns (shared-kernel, conformist, customer-supplier, ACL), direction, and shared artifacts |
| `get_pattern_alternatives` | Related patterns (SKOS `related` edges) + same-subject-area patterns — supports ToT branching |
| `run_fitness_checks` | Database-level governance checks (12 functions, incl. agent coverage). Returns violations with severity (CRITICAL/HIGH/MEDIUM/LOW). Optional severity filter |

### Graph Traversal Tools (Relationship Navigation)

These query Neo4j for typed edges between entities — no embeddings, just relationship following. Supports CoT (deepen a thread), ToT (explore branches), and ReAct (observe relationships, then act).

| Tool | Description |
|------|-------------|
| `graph_neighbors` | Incoming/outgoing graph edges for an entity or pattern. Returns neighbor id, label, relationship type, direction, and strength |

**Usage:**

```
Use the list_patterns tool with provenance ["3p"] to see all adopted third-party patterns.

Use the get_capability_impact tool with capability_id "ingestion-pipeline" to see
what patterns it implements, which repos deliver it, and integration dependencies.

Use the get_pattern_alternatives tool with pattern_id "ddd" to see related patterns
and patterns sharing the same subject area — useful for exploring alternative approaches.

Use the graph_neighbors tool with entity_id "semantic-flywheel" to follow
relationships from a known entity to related concepts without reformulating a search query.

Use the run_fitness_checks tool to check for schema governance violations.
```

### Key Files

| File | Purpose |
|------|---------|
| `api/mcp_server.py` | MCP tool definitions (FastMCP, stdio transport) |
| `scripts/search.py` | Shared semantic search module (entity, chunk, hybrid) |
| `scripts/schema_queries.py` | Shared ACL query module (patterns, capabilities, integrations, fitness) |
| `scripts/graph_queries.py` | Shared Neo4j graph query module (parameterized Cypher) |

**Requires:** Docker services running and `SEMOPS_DB_*` env vars configured in `.env`.

---

## Content→Pattern Bridge

The `bridge_content_patterns.py` script connects content entities to the pattern layer via a human-in-the-loop (HITL) workflow. It extracts concepts from content entities, matches them against registered patterns, and generates a YAML mapping file for human review.

### Workflow

```bash
# 1. Extract: generate mapping YAML for review
python scripts/bridge_content_patterns.py --extract [options]

# 2. Human reviews config/mappings/concept-pattern-map.yaml
#    Set action to: map | register | dismiss

# 3. Apply: create edges and register new patterns
python scripts/bridge_content_patterns.py --apply

# 4. Verify: report on bridging results
python scripts/bridge_content_patterns.py --verify
```

### Extract Modes

**Standard mode** — extracts concepts from `detected_edges` in entity metadata. Works for entities ingested with LLM classification enabled (issues, ADRs, session notes). Concepts are aggregated across entities with occurrence counts and strength scores.

```bash
# All entities with detected_edges
python scripts/bridge_content_patterns.py --extract

# Filter by content type
python scripts/bridge_content_patterns.py --extract --content-type issue

# Filter by corpus
python scripts/bridge_content_patterns.py --extract --corpus core_kb

# Minimum occurrence threshold
python scripts/bridge_content_patterns.py --extract --min-count 3
```

**Direct mode** (`--direct`) — treats each entity's ID as the concept itself, bypassing the `detected_edges` requirement. Use this for entities that *are* concepts (e.g., `content_type=concept` from the semops-framework source material), which were ingested before LLM classification added edge detection.

```bash
# Concept entities from semops-framework (the typical use case)
python scripts/bridge_content_patterns.py --extract --direct --content-type concept

# Combine with corpus filter
python scripts/bridge_content_patterns.py --extract --direct --content-type concept --corpus core_kb
```

### Mapping File

The extract generates `config/mappings/concept-pattern-map.yaml` with:

- **Header metadata** — entity/concept counts, match stats, filters used
- **Per-concept entries** — occurrences, matched pattern (if any), action field for review

Actions per concept:

- `map` — link to an existing pattern (auto-set for exact/alt-label matches)
- `register` — create a new pattern in the registry
- `dismiss` — skip (not a pattern-level concept)

**Related scripts:** `scripts/bridge_content_patterns.py` (extract/apply/verify workflow), `config/mappings/concept-pattern-map.yaml` (generated mapping), `scripts/create_described_by_edges.py` (creates `described_by` edges).

---

## Agentic Reasoning Patterns

The knowledge base architecture — three query surfaces, four corpora, and a typed graph — maps naturally to the three dominant agentic reasoning approaches. This section explains which tools to use for each pattern and why the architecture supports them.

> **Background:**  analyzed how the MCP tool surface serves each reasoning approach, identified structural gaps, and delivered `graph_neighbors` and `get_pattern_alternatives` to close them.

### Chain of Thought (CoT) — Sequential Deepening

CoT reasons step-by-step, where each observation informs the next query. The knowledge base supports this through **progressive narrowing** across surfaces and **structured traversal chains**.

**Natural CoT chains:**

```text
# Concept → Implementation → Impact (4-step ACL chain)
get_pattern "ddd"
  → list_capabilities domain_classification="core"
    → get_capability_impact "ingestion-pipeline"
      → query_integration_map repo="semops-data"

# Discovery → Precision (hybrid search 2-step)
search_knowledge_base "semantic coherence" corpus=["core_kb"]
  → search_chunks "coherence scoring algorithm" corpus=["core_kb"]

# Search → Navigate (semantic + graph)
search_knowledge_base "flywheel pattern"
  → graph_neighbors "semantic-flywheel"   # follow edges, don't re-search
```

**Why it works:** Hybrid search is inherently a two-step CoT (which document? → which passage?). The ACL query chain (`pattern → capability → repo → integration`) provides a structured four-step path from concept to implementation. Graph traversal (`graph_neighbors`) lets the agent follow relationships from a found entity without reformulating a search query — continuing the chain of thought rather than restarting it.

**Architectural enablers:** Heading hierarchy in chunk results (section breadcrumbs), corpus filtering for progressive narrowing, graph edges for relationship-following.

### Tree of Thought (ToT) — Branching Exploration

ToT explores multiple perspectives in parallel, evaluates branches, and prunes. The knowledge base supports this through **multiple independent search surfaces** and **alternative discovery tools**.

**Natural ToT branch points:**

```text
# Same query, three surfaces → three different result sets
search_knowledge_base "content lifecycle"    # which documents?
search_chunks "content lifecycle"            # which passages?
search_patterns "content lifecycle"          # which domain concepts?

# Same query, different corpora → theory vs practice
search_chunks "anti-corruption layer" corpus=["core_kb"]       # theory
search_chunks "anti-corruption layer" corpus=["deployment"]     # operational reality

# Explicit alternatives from the pattern registry
get_pattern_alternatives "ddd"               # SKOS related + same subject area

# Provenance as evaluation dimension
list_patterns provenance=["3p"]              # established consensus
list_patterns provenance=["1p"]              # novel synthesis
```

**Why it works:** Entity, chunk, and pattern embeddings are built from different text (metadata, content, definitions), so the same query surfaces genuinely different results at each layer — this is a structural ToT branch point, not just three copies of the same search. The four corpora (`core_kb`, `deployment`, `published`, `research_ai`) segment knowledge by domain, giving theory-vs-practice branching. `get_pattern_alternatives` exposes SKOS `related` edges and same-subject-area patterns for explicit "what else could work here?" branching. Pattern provenance (1P vs 3P) provides a built-in evaluation axis.

**Architectural enablers:** Three embedding surfaces with different text sources, corpus segmentation, SKOS `related` edges, pattern provenance, `get_pattern_alternatives` tool.

### ReAct (Reasoning + Acting) — Observe-Think-Act Loops

ReAct interleaves observation and action, where each tool result triggers reasoning about what to do next. The knowledge base supports this through **clear tool affordances** and **verification tools that close the loop**.

**Natural ReAct cycles:**

```text
# Discover → Verify → Govern
search_patterns "data quality"                     # observe: find relevant patterns
  → get_pattern "data-quality-framework"           # act: get full details
    → run_fitness_checks severity="HIGH"           # verify: check governance

# Explore → Impact → Decide
graph_neighbors "ingestion-pipeline"               # observe: what's connected?
  → get_capability_impact "ingestion-pipeline"     # act: assess impact
    → query_integration_map repo="semops-data"   # verify: check dependencies

# Search → Alternatives → Evaluate
search_patterns "content management"               # observe: find starting point
  → get_pattern_alternatives "dam"                 # act: explore options
    → get_pattern "dam"                            # verify: deep-dive chosen option
```

**Why it works:** The separation between semantic search (discover), ACL queries (verify/structure), and graph traversal (navigate) creates natural affordances — the agent knows which tool surface to use for each phase of the loop. Fitness checks (`run_fitness_checks`) close the loop by validating that the reasoning led to a coherent conclusion. Structured JSON responses give clear signals about what to do next.

**Architectural enablers:** Three query surfaces with distinct purposes (discover/verify/navigate), fitness checks as verification, structured JSON responses, graph traversal for relationship-following.

### Capability Matrix

| Architectural Feature | CoT | ToT | ReAct |
|----------------------|-----|-----|-------|
| Hybrid search (entity→chunk two-stage) | **Strong** | — | Good |
| Three embedding surfaces (entity/chunk/pattern) | Good | **Strong** | **Strong** |
| Four corpus domains | Good | **Strong** | Good |
| ACL query chain (pattern→capability→repo→integration) | **Strong** | Good | **Strong** |
| Fitness checks as verification | — | Good | **Strong** |
| Pattern provenance (1P/3P) | — | **Strong** | Good |
| Graph traversal (`graph_neighbors`) | **Strong** | Good | Good |
| Pattern alternatives (`get_pattern_alternatives`) | — | **Strong** | Good |

### Known Gaps

| Gap | Affects | Dependency |
|-----|---------|------------|
| No date/temporal filtering on search | CoT, ReAct |  (Qdrant migration) |
| Multi-hop graph traversal (depth > 1) | ToT | Future enhancement |
| Entity similarity pivot ("more like this") | CoT | Future enhancement |

---

## Troubleshooting

### Database Connection

Scripts connect automatically via `SEMOPS_DB_*` env vars configured in `.env`. Direct PostgreSQL access is available on port 5434.

**Error:** `connection refused`
**Cause:** Docker services not running, or `.env` missing `SEMOPS_DB_*` variables.
**Fix:** Start services with `python start_services.py --skip-clone` and verify `.env` has the `SEMOPS_DB_*` connection settings.

### Embedding Generation

**Symptom:** Entity search returns near-zero or irrelevant results, but chunk search works fine.
**Cause:** Entity embeddings are generated separately from chunk embeddings. The ingestion pipeline creates chunk embeddings inline but does **not** generate entity-level embeddings. If `scripts/generate_embeddings.py` has not been run after ingestion, the `entity.embedding` column will be NULL for all entities, making `search_knowledge_base` and entity-mode CLI search non-functional.
**Fix:** Run entity embedding generation:

```bash
python scripts/generate_embeddings.py
```

This generates embeddings for all entities missing them. Use `--regenerate` to overwrite existing embeddings, or `--entity-id <id>` for a single entity.

**Error:** `OPENAI_API_KEY not set in environment`
**Fix:** The `.env` file has the key, but the script's `.env` loader may not pick it up. Export it explicitly:

```bash
export $(grep OPENAI_API_KEY .env)
```

### MCP Server

**Server won't start:**
**Check:** Verify Docker services are running and `.env` has correct `SEMOPS_DB_*` connection settings. Direct PostgreSQL access is on port 5434.
