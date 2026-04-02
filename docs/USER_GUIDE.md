# User Guide: Ingestion Pipeline

> **Version:** 2.2.0 | **Last Updated:** 2026-03-29
> **Related Issues:**  (corpus-aware ingestion),  (unified ingestion/retrieval)
> **See also:** [SEARCH_GUIDE.md](SEARCH_GUIDE.md) for querying the knowledge base

---

## Quick Reference

| Task | Command |
|------|---------|
| List available sources | `python scripts/source_config.py` |
| Dry-run ingestion | `python scripts/ingest_from_source.py --source <name> --dry-run` |
| Ingest (no LLM) | `python scripts/ingest_from_source.py --source <name> --no-llm` |
| Ingest (with LLM) | `python scripts/ingest_from_source.py --source <name>` |
| Generate embeddings | `python scripts/generate_embeddings.py` |
| Materialize graph | `python scripts/materialize_graph.py` |
| Check entity count | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT count(*) FROM entity;"` |
| Check chunk count | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT count(*) FROM document_chunk;"` |
| Corpus distribution | `docker exec semops-hub-pg psql -U postgres -d postgres -c "SELECT metadata->>'corpus', count(*) FROM entity GROUP BY 1 ORDER BY 2 DESC;"` |

**Note:** All Python commands assume `source .venv/bin/activate` (or use `uv run`) and `OPENAI_API_KEY` is set (loaded from `.env`).

---

## Prerequisites

1. **Docker services running** — `python start_services.py --skip-clone`
2. **Python venv activated** — `source .venv/bin/activate` (or use `uv run`)
3. **`.env` file** with `OPENAI_API_KEY` and `POSTGRES_PASSWORD`
4. **Database connection** — Scripts connect automatically via `SEMOPS_DB_*` env vars configured in `.env`. Direct PostgreSQL access is available on port 5434.

---

## Source Management

### Source Config Format

Source configurations live in `config/sources/*.yaml`. Each file defines a GitHub repository to ingest from, with routing rules that assign entities to corpora.

**Full annotated example:**

```yaml
# Source Configuration for <repo-name>
# $schema: source_config_v1

# Required: unique kebab-case identifier
source_id: github-docs-pr

# Required: surface this source belongs to
surface_id: github-semops-ai

# Required: human-readable name
name: "SemOps Documentation (docs-pr)"

# Required: GitHub repository settings
github:
  owner: semops-ai           # GitHub org or user
  repo: docs-pr                 # Repository name
  branch: main                  # Branch to ingest from
  base_path: docs/SEMOPS_DOCS   # Subdirectory to start from (empty = repo root)
  include_directories:          # Only ingest from these dirs (relative to base_path)
    - SEMANTIC_OPERATIONS_FRAMEWORK
    - RESEARCH
  exclude_patterns:             # Glob patterns to skip
    - "**/drafts/**"
    - "**/_archive/**"
    - "**/WIP-*"
  file_extensions:              # File types to include
    - .md

# Optional: defaults applied to all entities from this source
defaults:
  asset_type: file              # "file" (you possess it) or "link" (external reference)
  version: "1.0"                # Semantic version

# Optional: attribution template (Dublin Core aligned)
attribution:
  $schema: attribution_v2
  creator:
    - Tim Mitchell
  rights: CC-BY-4.0
  organization: TJMConsulting
  platform: github
  channel: semops-ai
  epistemic_status: synthesis   # synthesis, original, curation, etc.

# Optional: LLM classification settings
llm_classify:
  enabled: true
  model: claude-opus-4-5-20251101   # Default classification model
  fields:                            # Fields the LLM should classify
    - concept_ownership
    - content_type
    - primary_concept
    - broader_concepts
    - narrower_concepts
    - subject_area
    - summary

# Required: corpus routing rules (ADR-0005)
corpus_routing:
  rules:
    - path_pattern: "docs/SEMOPS_DOCS/RESEARCH/**"
      corpus: research_ai
      content_type: concept
    - path_pattern: "docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/**"
      corpus: core_kb
      content_type: concept
  default_corpus: core_kb           # Fallback if no rule matches
  default_content_type: article     # Fallback content type
```

### Processing Contract

The `processing` block declares how the pipeline processes content from this source. It is a declarative contract — the source author knows their content best and declares its characteristics; the pipeline executes.

> **Design Doc:** [DD-0001 §Source-Defined Processing](https://github.com/semops-ai/semops-orchestrator/blob/main/docs/design-docs/DD-0001-ingestion-pipeline-architecture.md#source-defined-processing)
> **ADR:** [ADR-0015](docs/decisions/ADR-0015-ingestion-pipeline-architecture.md) decision D3

```yaml
processing:
  chunking: document_structure
  embedding: default
  entity_recognition: manifest
  relation_extraction: deterministic
  llm_enrichment: true
```

#### Processing Dimensions

**Chunking** — how text is split for embedding and retrieval.

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `document_structure` | Splits by markdown headings (H1–H6), preserving heading hierarchy as context. Sections exceeding 512 tokens are split with 50-token overlap. | Structured docs with heading hierarchy: ADRs, pattern docs, READMEs, articles, design docs. **Most common choice.** |
| `concept_anchored` | Extracts context windows around known concept mentions from the manifest (Ubiquitous Language + pattern registry + capability registry). | Long-form unstructured content where the value is in concept references, not document structure: academic papers, transcripts, meeting notes without headings. |
| `fixed_window` | Fixed-size overlapping token windows with no semantic awareness. | Raw text with no structure at all: logs, code files, data dumps. Rarely appropriate — prefer `document_structure` or `concept_anchored`. |
| `none` | No chunking. Entity is created but no `document_chunk` rows are produced. | Catalog-type ingestion (YAML registries), entity-only sources (session note headers, command logs), or content where only entity-level metadata matters. |

**Embedding** — whether vector embeddings are generated for semantic search.

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `default` | Generates OpenAI `text-embedding-3-small` embeddings (1536d) for each chunk and the entity. All sources sharing `default` produce vectors in the same vector space, enabling cross-corpus similarity search. | Any content that should be discoverable via semantic search. **Most common choice.** |
| `none` | No embeddings generated. Content exists as entities/metadata only — invisible to vector search. | Entity-only ingestion (session note pointers, command logs), or catalog sources where lookup is by ID, not similarity. |

**Entity recognition** — how concepts are identified in text and linked to the knowledge graph.

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `manifest` | Dictionary lookup against the concept manifest (Ubiquitous Language + pattern registry + capability registry). Deterministic: same text always produces the same concept matches. Inverts traditional NER — scans for *known* concepts rather than discovering unknown ones. | Content that references domain concepts you want to link to the graph. **Most common choice.** Produces reproducible, auditable results suitable for coherence measurement. |
| `llm` | LLM-based named entity recognition. Discovers *unknown* concepts not in the manifest. Non-deterministic — outputs may vary across runs. Candidates enter a discovery queue for human review before promotion to the manifest. | Exploratory ingestion of external content where you expect new concepts. Use sparingly — discovered concepts need HITL review. |
| `none` | No entity recognition. Entities are created from file metadata only, with no concept linking. | Entity-only sources, catalog sources where relationships come from YAML structure rather than text analysis, or content where concept linking adds no value. |

**Relation extraction** — how edges (relationships) between entities and concepts are derived.

| Strategy | What it does | When to use |
|----------|-------------|-------------|
| `deterministic` | Co-occurrence analysis, dependency parsing, and manifest-driven relationship typing. Reproducible: same input always produces same edges. Tagged `extraction_method: deterministic`. | Authoritative content where edge stability matters: pattern docs, ADRs, project specs. Also the only layer that coherence measurement (SC score) reads. **Choose this when measurement stability matters more than edge discovery.** |
| `deterministic+llm` | Deterministic extraction plus LLM-assisted relation typing. The LLM proposes typed edges (`extends`, `cites`, `contradicts`, etc.) with confidence scores. Non-deterministic outputs are tagged with model and prompt_hash for provenance. | Published content, theory docs, research — anywhere richer relationship discovery improves RAG retrieval quality. **Choose this when discovery value outweighs measurement stability.** |
| `none` | No relation extraction. No edges are created from this source's content. | Entity-only sources, catalog sources where edges come from YAML structure (SKOS edges, pattern-capability links), or process artifacts where relationships aren't meaningful. |

**LLM enrichment** — whether LLM metadata classification runs on ingested content.

| Value | What it does | When to use |
|-------|-------------|-------------|
| `true` | Runs LLM classification per the `llm_classify` block: generates summary, subject_area, concept_ownership, broader/narrower concepts, and detected_edges. Improves search relevance and entity metadata richness. Must be consistent with `llm_classify.enabled: true`. | Content where LLM-generated metadata adds search or classification value: published content, theory docs, deployment artifacts with diverse topics. |
| `false` | No LLM classification. Entity metadata comes only from deterministic sources (file path, headings, YAML structure). Must be consistent with `llm_classify.enabled: false`. | Structured content where metadata is already explicit (YAML catalogs, project specs with frontmatter), entity-only sources, or content where deterministic metadata is sufficient. |

#### Processing Profiles

These are the standard profiles for common content types. Use the [decision matrix](#choosing-a-processing-profile) below to classify new sources.

| Profile | Chunking | Embedding | Entity Recog | Relation Extract | LLM Enrich | Output Surfaces | Content Types |
|---------|----------|-----------|-------------|-----------------|------------|-----------------|---------------|
| **Full RAG** | `document_structure` | `default` | `manifest` | `deterministic+llm` | `true` | SQL + Vector + Graph | Framework theory, published READMEs, blog posts, research papers, design docs |
| **Deterministic** | `document_structure` | `default` | `manifest` | `deterministic` | `true` | SQL + Vector + Graph | Deployment artifacts (ADRs + session notes), pattern docs |
| **Deterministic (no LLM)** | `document_structure` | `default` | `manifest` | `deterministic` | `false` | SQL + Vector + Graph | ADRs (H1 only), project specs — structured content with explicit metadata |
| **Catalog** | `none` | `none` | `manifest` | `deterministic` | `false` | SQL + Graph | YAML registries (agents.yaml, pattern_v1.yaml, design-docs.yaml) |
| **Entity-only** | `none` | `none` | `none` | `none` | `false` | SQL only | Session note headers, command logs, build artifacts — metadata pointers only |

**Output Surfaces** describes where ingested data lands:
- **SQL** — entity rows (all types), pattern rows, edge table. The foundational layer — always written.
- **Vector** — chunk embeddings (Qdrant/pgvector) + entity embeddings. Enables semantic search.
- **Graph** — Neo4j nodes and edges. Catalog profiles produce the most critical graph edges (SKOS taxonomy, pattern-capability IMPLEMENTS, repo integration edges). Document profiles add LLM-detected edges.

#### Choosing a Processing Profile

When adding a new source, walk through these questions in order:

**1. Is the content structured data (YAML, JSON, CSV)?** Use **Catalog**. Relationships come from the data structure, not from text analysis. Set `embedding: default` if you want vector search over catalog entries.

**2. Is the content entity-only (metadata pointer, no retrieval needed)?** Use **Entity-only**. Examples: session note headers (just the issue reference), command logs (date + command), build artifacts.

**3. Is the content authoritative/prescriptive (defines concepts, decisions, standards)?** Use **Deterministic** or **Deterministic (no LLM)**. Edge stability matters more than discovery. Use `llm_enrichment: true` if summary/classification improves search; `false` if the content's own structure provides sufficient metadata (e.g., ADR frontmatter, project spec headers).

**4. Is the content published/external-facing or rich theory?** Use **Full RAG**. Discovery value justifies LLM edges and enrichment. For unstructured long-form content (academic papers, transcripts), consider `chunking: concept_anchored` instead of `document_structure`.

**5. None of the above?** Default to **Deterministic** — structured chunking, searchable, manifest NER, deterministic edges, LLM metadata. This is the safe default.

#### Consistency Rule

The `processing.llm_enrichment` flag and `llm_classify.enabled` must agree. The source config validator enforces this — a mismatch raises a validation error. The `processing` block declares intent; `llm_classify` carries the operational parameters (model, fields).

---

### Corpus Routing Rules

Corpus routing determines which corpus each entity belongs to based on its file path. Rules are evaluated **in order — first match wins**.

**How it works:**

1. During ingestion, each file's repo-relative path (including `base_path`) is matched against `corpus_routing.rules`
2. The first matching `path_pattern` (fnmatch-style glob) determines `corpus` and `content_type`
3. If no rule matches, `default_corpus` and `default_content_type` are used
4. These values are stored in `entity.metadata->>'corpus'` and `entity.metadata->>'content_type'`

**Path matching note:** The path matched includes `github.base_path`. For example, if `base_path: docs/SEMOPS_DOCS` and the file is `RESEARCH/FOUNDATIONS/foo.md`, the path matched against rules is `docs/SEMOPS_DOCS/RESEARCH/FOUNDATIONS/foo.md`.

**Rule order matters.** More specific rules should come before general ones:

```yaml
# CORRECT: specific before general
rules:
  - path_pattern: "docs/RESEARCH/FOUNDATIONS/**"    # Specific
    corpus: research_ai
  - path_pattern: "docs/RESEARCH/**"                # General fallback
    corpus: research_general

# WRONG: general rule shadows specific
rules:
  - path_pattern: "docs/RESEARCH/**"                # Catches everything
    corpus: research_general
  - path_pattern: "docs/RESEARCH/FOUNDATIONS/**"     # Never reached
    corpus: research_ai
```

### Available Corpus Types

| Corpus | Description | Use For |
|--------|-------------|---------|
| `core_kb` | Curated knowledge: patterns, theory, schema | Domain pattern docs, theory, canonical references |
| `deployment` | Operational artifacts | ADRs, session notes, architecture docs, CLAUDE.md |
| `published` | Published content | Blog posts, public docs |
| `research_ai` | AI/ML research | AI foundations, cognitive science, AI transformation |
| `research_general` | General research | Ad-hoc, unsorted, triage/staging |
| `ephemeral_*` | Temporary/experimental | Experiments, WIP (prefix with custom suffix) |

### Available Content Types

| Content Type | Description |
|-------------|-------------|
| `concept` | Theoretical/conceptual document |
| `pattern` | Domain pattern documentation |
| `architecture` | Architecture docs, topology |
| `adr` | Architecture Decision Record |
| `article` | General prose, blog post |
| `session_note` | Work log, decision provenance |

Content types are non-exhaustive — LLM classification may produce additional values.

### Adding a New Source

**Step-by-step:**

1. **Create the YAML config:**

   ```bash
   cp config/sources/publisher-pr.yaml config/sources/my-new-source.yaml
   ```

2. **Edit the config:** Update `source_id`, `surface_id`, `name`, `github` settings, and `corpus_routing` rules.

3. **Validate with dry-run:**

   ```bash
   python scripts/ingest_from_source.py \
     --source my-new-source --dry-run
   ```

   Check: all files discovered, corpus assignments correct, no validation errors.

4. **Ingest:**

   ```bash
   # Without LLM (fast, title + metadata only)
   python scripts/ingest_from_source.py \
     --source my-new-source --no-llm

   # With LLM (slower, generates summaries + classification)
   python scripts/ingest_from_source.py \
     --source my-new-source
   ```

5. **Generate embeddings:**

   ```bash
   python scripts/generate_embeddings.py
   ```

   This only processes entities with NULL embeddings, so it's safe to run after any ingestion.

6. **Verify:**

   ```bash
   docker exec semops-hub-pg psql -U postgres -d postgres -c \
     "SELECT metadata->>'corpus', count(*) FROM entity GROUP BY 1 ORDER BY 2 DESC;"
   ```

**Checklist:**

- [ ] `source_id` is unique kebab-case
- [ ] `github.base_path` matches the actual directory structure
- [ ] `include_directories` are relative to `base_path`
- [ ] `corpus_routing.rules` path patterns include `base_path` prefix
- [ ] More specific routing rules come before general ones
- [ ] `default_corpus` and `default_content_type` are set
- [ ] Dry-run shows correct entity count and corpus assignments

### Modifying an Existing Source

When you modify a source config, the downstream impact depends on what changed:

| Change | Action Required |
|--------|----------------|
| New `include_directories` or paths | Re-ingest source, generate embeddings for new entities |
| Changed `corpus_routing` rules | Re-ingest source (upsert updates metadata including corpus tag) |
| Changed `attribution` template | Re-ingest source |
| Changed `exclude_patterns` | Re-ingest source (may create new entities or leave removed ones in DB) |
| Changed `llm_classify` fields | Re-ingest with LLM (not `--no-llm`) |
| Changed `defaults` | Re-ingest source |

**Re-ingestion is safe.** The ingestion script uses `ON CONFLICT (id) DO UPDATE`, so re-running updates existing entities with new metadata, attribution, and filespec without creating duplicates. Entity IDs are derived from file paths, so the same file always produces the same ID.

**Embeddings after re-ingestion:** If metadata changed (e.g., new summary from LLM classification), re-generate embeddings:

```bash
# Regenerate all embeddings (not just missing)
python scripts/generate_embeddings.py --regenerate
```

**Removing entities:** Re-ingestion does not delete entities that are no longer in the source. To remove stale entities, use SQL directly:

```bash
docker exec semops-hub-pg psql -U postgres -d postgres -c \
  "DELETE FROM entity WHERE id LIKE 'prefix-%' AND updated_at < '2026-01-31';"
```

---

## Ingestion

### Running Ingestion

```bash
# Dry run — shows what would be ingested without touching the DB
python scripts/ingest_from_source.py \
  --source publisher-pr --dry-run

# Ingest without LLM classification (fast, metadata from file only)
python scripts/ingest_from_source.py \
  --source publisher-pr --no-llm

# Ingest with LLM classification (generates summaries, subject_area, etc.)
python scripts/ingest_from_source.py \
  --source publisher-pr
```

### What `--no-llm` Skips

With `--no-llm`, entities get:

- Title (extracted from first heading or filename)
- Corpus and content_type (from routing rules)
- Word count, reading time, file format, size
- Attribution (from template)
- Filespec (URI, hash, format)

Without `--no-llm` (LLM enabled), entities additionally get:

- Summary
- Subject area classification
- Concept ownership analysis
- Broader/narrower concept mapping
- Quality score

**Recommendation:** Use `--no-llm` for initial bulk ingestion, then run a separate LLM classification pass for richer metadata.

### Current Sources

| Source | Config File | Entities | Chunks |
|--------|------------|----------|--------|
| `publisher-pr` | `config/sources/publisher-pr.yaml` | ~178 | ~3,316 |
| `dx-hub-domain-patterns` | `config/sources/dx-hub-domain-patterns.yaml` | 24 | 613 |

---

## Embedding Generation

```bash
# Generate embeddings for entities that don't have them
python scripts/generate_embeddings.py

# Regenerate all embeddings
python scripts/generate_embeddings.py --regenerate

# Process a specific entity
python scripts/generate_embeddings.py \
  --entity-id semantic-compression

# Dry run
python scripts/generate_embeddings.py --dry-run
```

**Model:** OpenAI `text-embedding-3-small` (1536 dimensions). The same model is used at both ingestion (to embed entities and chunks) and query time (to embed the search query). Both sides must share the same vector space for cosine similarity scores to be meaningful.

**Embedding text:** Built from entity title + metadata (summary, content_type, subject_area, broader/narrower concepts). Richer metadata = better embeddings.

**When to regenerate:**

- After ingesting with LLM (new summaries improve embedding quality)
- After changing how `build_embedding_text` works in `generate_embeddings.py`
- Not needed after re-ingestion if only corpus/attribution changed (embeddings use title + content metadata)

---

## Chunking

Ingestion automatically chunks each document by markdown headings and stores passages in the `document_chunk` table with entity_id foreign keys. Chunks get their own OpenAI embeddings (same model as entities) for passage-level retrieval.

**How it works:**
1. During ingestion, each file is split by markdown headings (##, ###, etc.)
2. Sections exceeding 512 tokens are split with 50-token overlap
3. Each chunk gets an OpenAI `text-embedding-3-small` embedding (1536d)
4. Chunks are stored with `entity_id` FK, `corpus`, `content_type`, and heading hierarchy

**No separate chunking step is needed.** Chunking happens automatically during `ingest_from_source.py`. The standalone `chunk_markdown_docs.py` script is deprecated.

---

## Graph Materialization

Ingestion automatically writes entity relationships to Neo4j from `detected_edges` metadata. For backfilling or rebuilding the graph from existing entities:

```bash
# Backfill graph from all entities with detected_edges
python scripts/materialize_graph.py

# Clear graph and rebuild
python scripts/materialize_graph.py --clear

# Dry run
python scripts/materialize_graph.py --dry-run
```

The graph contains Entity nodes (from ingested files) and Concept nodes (from detected edges), connected by typed relationships (EXTENDS, RELATED, etc.).

---

## Post-Ingest Enrichment

Some metadata fields are best computed deterministically after ingestion, rather than during it. The **pandas enrichment pattern** is the standard approach for these cases.

### When to Use It

Use post-ingest enrichment when:

- The metadata can be computed **deterministically** from existing entity content (no LLM needed)
- The computation is **batch-oriented** (applies to a whole corpus or subset)
- The data is better analyzed in a DataFrame (cross-entity joins, regex extraction, vectorized ops)
- You want a **CSV snapshot** for offline analysis before committing back to the DB

### Reference Implementation: `extract_deployment_dates.py`

`scripts/extract_deployment_dates.py` is the canonical example. It extracts `date_created` and `date_updated` from deployment corpus entities:

1. **Load** — Pull entities + `document_chunk.heading_hierarchy` arrays into a pandas DataFrame via SQL
2. **Compute** — Extract dates deterministically:
   - ADR `**Date**:` field (highest priority)
   - Session note `## YYYY-MM-DD` heading hierarchy entries
   - Filename prefix fallback (`YYYY-MM-DD-title.md`)
3. **Write back** — `UPDATE entity SET metadata = metadata || '{"date_created": "..."}'` for each entity
4. **Snapshot** — Export results to `data/deployment_dates.csv`

**Coverage:** 233/252 deployment entities (92%). The 19 missing predate the date-heading convention.

### Running the Enrichment Script

```bash
# Dry run — shows what would be updated
python scripts/extract_deployment_dates.py --dry-run

# Apply updates
python scripts/extract_deployment_dates.py

# Export snapshot only (no DB writes)
python scripts/extract_deployment_dates.py --export-only
```

### The `data/` Directory

`data/` stores CSV snapshots from enrichment runs. These snapshots enable:

- Offline analysis without a DB connection
- Auditing before applying updates
- Comparing enrichment runs over time

### Pattern Summary

```text
DB (entities + chunks)
    → pandas DataFrame (load)
    → compute fields (regex / vectorized logic)
    → UPDATE entity.metadata (write back)
    → data/*.csv (snapshot)
```

This is **not** a replacement for LLM classification. Use it when the answer is in the text and the extraction rule is deterministic.

---

## Troubleshooting

### Database Connection

Scripts connect automatically via `SEMOPS_DB_*` env vars configured in `.env`. Direct PostgreSQL access is available on port 5434.

**Error:** `connection refused`
**Cause:** Docker services not running, or `.env` missing `SEMOPS_DB_*` variables.
**Fix:** Start services with `python start_services.py --skip-clone` and verify `.env` has the `SEMOPS_DB_*` connection settings.

### Embedding Generation

**Error:** `OPENAI_API_KEY not set in environment`
**Fix:** The `.env` file has the key, but the script's `.env` loader may not pick it up. Export it explicitly:

```bash
export $(grep OPENAI_API_KEY .env)
```

### Ingestion

**Error:** `column "X" of relation "entity" does not exist`
**Cause:** Database schema doesn't match what the script expects.
**Fix:** Verify entity table structure: `docker exec semops-hub-pg psql -U postgres -d postgres -c "\d entity"`

**All entities show status FAILED:**
**Cause:** First entity fails, then all subsequent fail with "current transaction is aborted". This is a single-transaction issue.
**Fix:** Fix the root cause of the first failure, then re-run.
