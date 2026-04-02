-- Unified Lineage Schema
-- Issue: #168
-- Purpose: Generalize ingestion_run/ingestion_episode to run/episode
--          to support all process types (ingestion, slash commands, scheduled jobs)
--
-- Design principles:
-- - One event model for everything — the run carries architectural context,
--   the episode carries per-target metadata
-- - Manifest-driven: every process reads a config, episodes inherit context from run
-- - Dimensional model: run = dimension table, episode = fact table
-- - Episodes are entities in the corpus — queryable by agents
--
-- Migration strategy: CREATE new tables, migrate data, drop old tables.
-- Old views (recent_episodes, entity_lineage, pattern_emergence) are recreated
-- against the new tables.

-- ============================================================================
-- RUN (replaces ingestion_run)
-- ============================================================================
-- A bounded execution driven by a manifest. Carries constant dimensions:
-- source, corpus, agent, model. All episodes in a run inherit these.

CREATE TABLE IF NOT EXISTS run (
  id TEXT PRIMARY KEY,                              -- ULID for time-ordering
  run_type TEXT NOT NULL
    CHECK (run_type IN ('manual', 'scheduled', 'interactive', 'agent')),

  -- Manifest context (what drove this run)
  manifest_ref TEXT,                                -- config path or command name
  source_id TEXT,                                   -- source identifier from manifest
  corpus TEXT,                                      -- target corpus (from manifest routing)

  -- Agent context (who executed)
  agent_name TEXT,                                  -- script name, 'claude-code', etc.
  model_name TEXT,                                  -- LLM model used (if any)

  -- Scope and timing
  scope JSONB NOT NULL DEFAULT '{}',                -- what was covered (file count, repo list, etc.)
  started_at TIMESTAMPTZ NOT NULL DEFAULT now(),
  completed_at TIMESTAMPTZ,
  status TEXT NOT NULL DEFAULT 'running'
    CHECK (status IN ('running', 'completed', 'failed', 'cancelled')),

  -- Reproducibility
  manifest_snapshot JSONB NOT NULL DEFAULT '{}',    -- full config snapshot at execution time

  -- Aggregate metrics (populated on completion)
  metrics JSONB NOT NULL DEFAULT '{}'               -- entities_processed, errors, etc.
);

COMMENT ON TABLE run IS 'Bounded execution driven by a manifest — ingestion batch, slash command, scheduled job';
COMMENT ON COLUMN run.run_type IS 'manual = human CLI, scheduled = cron, interactive = slash command, agent = autonomous';
COMMENT ON COLUMN run.manifest_ref IS 'Config that drove this run: YAML path, command name, cron schedule';
COMMENT ON COLUMN run.source_id IS 'Source identifier from manifest (e.g., github-publisher-pr)';
COMMENT ON COLUMN run.corpus IS 'Target corpus from manifest routing (e.g., core_kb, deployment)';
COMMENT ON COLUMN run.manifest_snapshot IS 'Full config snapshot for reproducibility — what was active when this ran';
COMMENT ON COLUMN run.scope IS 'What was covered: {file_count, repo_list, entity_count, etc.}';

-- ============================================================================
-- EPISODE (replaces ingestion_episode)
-- ============================================================================
-- One meaningful operation within a run. The unit of lineage.
-- Lightweight by design — most context inherited from run via run_id join.
-- Episodes are entities in the corpus, queryable by agents.

CREATE TABLE IF NOT EXISTS episode (
  id TEXT PRIMARY KEY,                              -- ULID for time-ordering
  run_id TEXT REFERENCES run(id) ON DELETE SET NULL, -- NULL if standalone

  -- What happened (episode-specific)
  operation TEXT NOT NULL
    CHECK (operation IN (
      -- Content operations
      'ingest',           -- entity created from source document
      'classify',         -- entity assigned primary pattern
      'embed',            -- embedding generated for entity/chunk
      'create_edge',      -- relationship established
      'declare_pattern',  -- new pattern created from synthesis
      'publish',          -- delivery created
      'synthesize',       -- new content generated from multiple sources
      -- Governance operations
      'audit',            -- checked state against expected state
      'evaluate',         -- assessed input against domain model
      'measure',          -- computed coherence/drift score
      'sync',             -- updated state to match authority
      'bridge'            -- mapped concepts to patterns (HITL)
    )),

  -- What was acted on
  target_type TEXT NOT NULL
    CHECK (target_type IN ('entity', 'pattern', 'edge', 'delivery', 'corpus', 'repo')),
  target_id TEXT NOT NULL,

  -- Quality signals
  coherence_score DECIMAL(4,3)                      -- 0.000-1.000, NULL if not computed
    CHECK (coherence_score IS NULL OR (coherence_score >= 0 AND coherence_score <= 1)),

  -- Relationships discovered
  detected_edges JSONB DEFAULT '[]',                -- [{predicate, target_id, strength, rationale}]

  -- Provenance
  input_hash TEXT,                                  -- SHA256 of input for deduplication
  error_message TEXT,                               -- if operation failed

  -- Operation-specific data (file path, chunk refs, classification result, etc.)
  metadata JSONB NOT NULL DEFAULT '{}',

  -- Context used (what informed the operation)
  context_pattern_ids TEXT[] DEFAULT '{}',           -- patterns retrieved/considered
  context_entity_ids TEXT[] DEFAULT '{}',            -- entities used as context

  -- Run-inherited overrides (only set when they differ from the run)
  agent_name TEXT,                                  -- override when episode uses different agent
  model_name TEXT,                                  -- override when episode uses different model
  prompt_hash TEXT,                                 -- SHA256 of prompt for this operation
  token_usage JSONB DEFAULT '{}',                   -- {prompt_tokens, completion_tokens, total_tokens}

  created_at TIMESTAMPTZ NOT NULL DEFAULT now()
);

COMMENT ON TABLE episode IS 'One meaningful operation within a run — the unit of lineage';
COMMENT ON COLUMN episode.operation IS 'Operation type: content (ingest/classify/embed/...) or governance (audit/evaluate/measure/...)';
COMMENT ON COLUMN episode.target_type IS 'What was acted on: entity, pattern, edge, delivery, corpus, repo';
COMMENT ON COLUMN episode.coherence_score IS 'Semantic coherence 0-1 (embedding similarity or LLM evaluation)';
COMMENT ON COLUMN episode.detected_edges IS 'Relationships discovered: [{predicate, target_id, strength, rationale}]';
COMMENT ON COLUMN episode.metadata IS 'Operation-specific data: file_path, chunk_refs, classification_result, audit_findings, etc.';
COMMENT ON COLUMN episode.agent_name IS 'Override — only set when this episode uses a different agent than the run';
COMMENT ON COLUMN episode.model_name IS 'Override — only set when this episode uses a different model than the run';

-- ============================================================================
-- DATA MIGRATION
-- ============================================================================
-- Migrate existing ingestion_run → run, ingestion_episode → episode

INSERT INTO run (id, run_type, manifest_ref, source_id, agent_name, started_at, completed_at, status, manifest_snapshot, metrics)
SELECT
  id,
  run_type,
  source_name,              -- manifest_ref ← source_name (best available)
  source_name,              -- source_id ← source_name
  agent_name,
  started_at,
  completed_at,
  status,
  source_config,            -- manifest_snapshot ← source_config
  metrics
FROM ingestion_run
ON CONFLICT (id) DO NOTHING;

INSERT INTO episode (
  id, run_id, operation, target_type, target_id,
  coherence_score, detected_edges, input_hash, error_message, metadata,
  context_pattern_ids, context_entity_ids,
  agent_name, model_name, prompt_hash, token_usage,
  created_at
)
SELECT
  id, run_id, operation, target_type, target_id,
  coherence_score, detected_edges, input_hash, error_message, metadata,
  context_pattern_ids, context_entity_ids,
  agent_name, model_name, prompt_hash, token_usage,
  created_at
FROM ingestion_episode
ON CONFLICT (id) DO NOTHING;

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Run indexes
CREATE INDEX IF NOT EXISTS idx_run_status ON run(status);
CREATE INDEX IF NOT EXISTS idx_run_type ON run(run_type);
CREATE INDEX IF NOT EXISTS idx_run_started ON run(started_at);
CREATE INDEX IF NOT EXISTS idx_run_source ON run(source_id);
CREATE INDEX IF NOT EXISTS idx_run_corpus ON run(corpus);
CREATE INDEX IF NOT EXISTS idx_run_manifest ON run(manifest_ref);

-- Episode indexes
CREATE INDEX IF NOT EXISTS idx_ep_run ON episode(run_id);
CREATE INDEX IF NOT EXISTS idx_ep_operation ON episode(operation);
CREATE INDEX IF NOT EXISTS idx_ep_target ON episode(target_type, target_id);
CREATE INDEX IF NOT EXISTS idx_ep_target_id ON episode(target_id);
CREATE INDEX IF NOT EXISTS idx_ep_created ON episode(created_at);
CREATE INDEX IF NOT EXISTS idx_ep_coherence ON episode(coherence_score) WHERE coherence_score IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ep_errors ON episode(run_id) WHERE error_message IS NOT NULL;
CREATE INDEX IF NOT EXISTS idx_ep_context_patterns ON episode USING gin(context_pattern_ids);
CREATE INDEX IF NOT EXISTS idx_ep_context_entities ON episode USING gin(context_entity_ids);

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Recent episodes with full run context
CREATE OR REPLACE VIEW recent_episodes AS
SELECT
  e.id AS episode_id,
  e.operation,
  e.target_type,
  e.target_id,
  e.coherence_score,
  e.error_message,
  e.created_at,
  r.id AS run_id,
  r.run_type,
  r.manifest_ref,
  r.source_id,
  r.corpus,
  COALESCE(e.agent_name, r.agent_name) AS agent_name,
  COALESCE(e.model_name, r.model_name) AS model_name,
  r.status AS run_status
FROM episode e
LEFT JOIN run r ON e.run_id = r.id
ORDER BY e.created_at DESC
LIMIT 100;

COMMENT ON VIEW recent_episodes IS 'Last 100 episodes with run context — COALESCE resolves agent/model inheritance';

-- Entity lineage: all episodes that touched an entity
CREATE OR REPLACE VIEW entity_lineage AS
SELECT
  e.target_id AS entity_id,
  e.operation,
  e.context_pattern_ids,
  e.coherence_score,
  COALESCE(e.agent_name, r.agent_name) AS agent_name,
  COALESCE(e.model_name, r.model_name) AS model_name,
  e.detected_edges,
  e.created_at AS episode_at,
  r.source_id,
  r.corpus,
  r.run_type,
  r.manifest_ref
FROM episode e
LEFT JOIN run r ON e.run_id = r.id
WHERE e.target_type = 'entity'
ORDER BY e.target_id, e.created_at;

COMMENT ON VIEW entity_lineage IS 'Full lineage history for entities — join provides corpus and source dimensions';

-- Pattern emergence: episodes that declared patterns
CREATE OR REPLACE VIEW pattern_emergence AS
SELECT
  e.target_id AS pattern_id,
  e.context_pattern_ids AS source_patterns,
  e.context_entity_ids AS source_entities,
  e.coherence_score,
  COALESCE(e.agent_name, r.agent_name) AS agent_name,
  e.detected_edges AS proposed_edges,
  e.created_at AS declared_at,
  r.source_id AS research_source,
  r.corpus
FROM episode e
LEFT JOIN run r ON e.run_id = r.id
WHERE e.operation = 'declare_pattern'
ORDER BY e.created_at DESC;

COMMENT ON VIEW pattern_emergence IS 'Track how patterns emerge from research synthesis';

-- Staleness: entities by last episode timestamp
CREATE OR REPLACE VIEW entity_staleness AS
SELECT
  e.target_id AS entity_id,
  MAX(e.created_at) AS last_episode_at,
  COUNT(*) AS total_episodes,
  COUNT(*) FILTER (WHERE e.error_message IS NOT NULL) AS error_episodes,
  r.corpus
FROM episode e
LEFT JOIN run r ON e.run_id = r.id
WHERE e.target_type = 'entity'
GROUP BY e.target_id, r.corpus
ORDER BY last_episode_at ASC;

COMMENT ON VIEW entity_staleness IS 'Staleness report — entities ordered by oldest last-episode';

-- Corpus coherence summary: avg coherence by corpus over time
CREATE OR REPLACE VIEW corpus_coherence_summary AS
SELECT
  r.corpus,
  DATE_TRUNC('week', e.created_at) AS week,
  AVG(e.coherence_score) AS avg_coherence,
  MIN(e.coherence_score) AS min_coherence,
  MAX(e.coherence_score) AS max_coherence,
  COUNT(*) AS episode_count,
  COUNT(*) FILTER (WHERE e.error_message IS NOT NULL) AS error_count
FROM episode e
JOIN run r ON e.run_id = r.id
WHERE e.coherence_score IS NOT NULL
GROUP BY r.corpus, DATE_TRUNC('week', e.created_at)
ORDER BY r.corpus, week DESC;

COMMENT ON VIEW corpus_coherence_summary IS 'Weekly corpus coherence trends — core SC metric';

-- Source reliability: error rate by source and operation
CREATE OR REPLACE VIEW source_reliability AS
SELECT
  r.source_id,
  e.operation,
  COUNT(*) AS total_episodes,
  COUNT(*) FILTER (WHERE e.error_message IS NOT NULL) AS failed_episodes,
  ROUND(
    COUNT(*) FILTER (WHERE e.error_message IS NOT NULL)::DECIMAL / NULLIF(COUNT(*), 0),
    3
  ) AS error_rate,
  AVG(e.coherence_score) AS avg_coherence
FROM episode e
JOIN run r ON e.run_id = r.id
GROUP BY r.source_id, e.operation
ORDER BY error_rate DESC NULLS LAST;

COMMENT ON VIEW source_reliability IS 'Error rate and avg coherence by source and operation type';

-- ============================================================================
-- DROP OLD TABLES (after migration)
-- ============================================================================
-- Keep old tables for now — uncomment when confident migration is complete.
-- Old views were already replaced above (CREATE OR REPLACE).
--
-- DROP TABLE IF EXISTS ingestion_episode;
-- DROP TABLE IF EXISTS ingestion_run;

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

INSERT INTO schema_version (version, description) VALUES
  ('9.0.0', 'Unified Lineage: run + episode generalize ingestion_run/ingestion_episode for all process types (ingestion, slash commands, scheduled jobs, investigations)')
ON CONFLICT (version) DO NOTHING;
