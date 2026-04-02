-- Reasoning Trace Extension for Episode Model
-- Issue: #160
-- ADR: ADR-0017
-- Purpose: Capture agentic reasoning strategy metadata on episodes,
--          enabling SC measurement to correlate coherence with how agents reason.
--
-- This is a MINOR (additive) change — all columns are nullable.
-- Existing episodes and instrumentation continue unchanged.

-- ============================================================================
-- NEW COLUMNS ON EPISODE
-- ============================================================================

-- Reasoning strategy: how the agent processed context
ALTER TABLE episode ADD COLUMN IF NOT EXISTS reasoning_pattern TEXT
  CHECK (reasoning_pattern IS NULL OR reasoning_pattern IN (
    'workflow',           -- sequential steps, no branching (simple pipelines)
    'cot',               -- chain of thought: sequential reasoning, single path
    'react',             -- observation-action cycles interleaving reasoning with tool use
    'tree-of-thoughts',  -- branching exploration with evaluation/backtracking
    'reflexion',         -- self-critique and revision loops
    'llm-p',             -- LLM-based planning (plan generation + execution)
    'direct'             -- no explicit reasoning strategy (single-shot)
  ));

ALTER TABLE episode ADD COLUMN IF NOT EXISTS chain_depth INT
  CHECK (chain_depth IS NULL OR chain_depth >= 0);

ALTER TABLE episode ADD COLUMN IF NOT EXISTS branching_factor INT
  CHECK (branching_factor IS NULL OR branching_factor >= 0);

ALTER TABLE episode ADD COLUMN IF NOT EXISTS observation_action_cycles INT
  CHECK (observation_action_cycles IS NULL OR observation_action_cycles >= 0);

-- Context assembly: how context was constructed and utilized
ALTER TABLE episode ADD COLUMN IF NOT EXISTS context_assembly_method TEXT
  CHECK (context_assembly_method IS NULL OR context_assembly_method IN (
    'rag',        -- retrieval-augmented generation
    'full_doc',   -- full document(s) loaded
    'summary',    -- pre-summarized context
    'hybrid'      -- combination of methods
  ));

ALTER TABLE episode ADD COLUMN IF NOT EXISTS context_token_count INT
  CHECK (context_token_count IS NULL OR context_token_count >= 0);

ALTER TABLE episode ADD COLUMN IF NOT EXISTS context_utilization DECIMAL(4,3)
  CHECK (context_utilization IS NULL OR (context_utilization >= 0 AND context_utilization <= 1));

-- ============================================================================
-- COMMENTS
-- ============================================================================

COMMENT ON COLUMN episode.reasoning_pattern IS 'Reasoning strategy used: workflow, cot, react, tree-of-thoughts, reflexion, llm-p, direct (vocabulary from semops-orchestrator#202)';
COMMENT ON COLUMN episode.chain_depth IS 'Number of reasoning steps in the chain (CoT, ReAct)';
COMMENT ON COLUMN episode.branching_factor IS 'Branches explored (ToT) or revision cycles (reflexion)';
COMMENT ON COLUMN episode.observation_action_cycles IS 'Tool-use cycles in ReAct pattern';
COMMENT ON COLUMN episode.context_assembly_method IS 'How context was constructed: rag, full_doc, summary, hybrid';
COMMENT ON COLUMN episode.context_token_count IS 'Tokens in assembled context window';
COMMENT ON COLUMN episode.context_utilization IS 'Fraction of context tokens actually referenced in output (0.000-1.000)';

-- ============================================================================
-- INDEXES
-- ============================================================================

-- Enable grouping/filtering by reasoning pattern for SC analysis
CREATE INDEX IF NOT EXISTS idx_ep_reasoning_pattern ON episode(reasoning_pattern)
  WHERE reasoning_pattern IS NOT NULL;

-- Enable context utilization analysis
CREATE INDEX IF NOT EXISTS idx_ep_context_utilization ON episode(context_utilization)
  WHERE context_utilization IS NOT NULL;

-- ============================================================================
-- VIEWS
-- ============================================================================

-- Reasoning pattern coherence: SC grouped by reasoning strategy
CREATE OR REPLACE VIEW reasoning_pattern_coherence AS
SELECT
  e.reasoning_pattern,
  e.context_assembly_method,
  COUNT(*) AS episode_count,
  AVG(e.coherence_score) AS avg_coherence,
  MIN(e.coherence_score) AS min_coherence,
  MAX(e.coherence_score) AS max_coherence,
  AVG(e.chain_depth) AS avg_chain_depth,
  AVG(e.observation_action_cycles) AS avg_observation_cycles,
  AVG(e.context_token_count) AS avg_context_tokens,
  AVG(e.context_utilization) AS avg_context_utilization
FROM episode e
WHERE e.reasoning_pattern IS NOT NULL
  AND e.coherence_score IS NOT NULL
GROUP BY e.reasoning_pattern, e.context_assembly_method
ORDER BY avg_coherence DESC;

COMMENT ON VIEW reasoning_pattern_coherence IS 'SC scores grouped by reasoning strategy and context assembly method — answers "which approach produces highest coherence?"';

-- Context efficiency: utilization vs coherence
CREATE OR REPLACE VIEW context_efficiency AS
SELECT
  e.id AS episode_id,
  e.operation,
  e.reasoning_pattern,
  e.context_token_count,
  e.context_utilization,
  e.coherence_score,
  COALESCE(e.agent_name, r.agent_name) AS agent_name,
  COALESCE(e.model_name, r.model_name) AS model_name,
  r.corpus,
  e.created_at
FROM episode e
LEFT JOIN run r ON e.run_id = r.id
WHERE e.context_token_count IS NOT NULL
  AND e.context_utilization IS NOT NULL
ORDER BY e.context_utilization ASC;

COMMENT ON VIEW context_efficiency IS 'Context efficiency report — low utilization = wasted context, correlate with coherence to diagnose retrieval quality';

-- ============================================================================
-- SCHEMA VERSION
-- ============================================================================

INSERT INTO schema_version (version, description) VALUES
  ('9.1.0', 'Reasoning trace: reasoning_pattern, chain_depth, branching_factor, observation_action_cycles, context_assembly_method, context_token_count, context_utilization on episode (#160)')
ON CONFLICT (version) DO NOTHING;
