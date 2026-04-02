-- Consulting Deployment Edge Predicates
-- Issue: semops-orchestrator#196
-- ADR: semops-orchestrator/ADR-0008 (Consulting Deployment Pattern, D7)
-- Purpose: Add edge predicates for consulting deployment relationships:
--   deploys_to     — capability -> deployment repo
--   adopts_from    — deployment pattern -> core pattern
--   incubates      — host repo -> embedded engagement
--   originated_from — deployment repo -> origin repo
--
-- This is a MINOR (additive) change — extends the CHECK constraint on edge.predicate.
-- Existing edges are unaffected.

-- ============================================================================
-- Extend edge predicate CHECK constraint
-- ============================================================================
-- PostgreSQL requires dropping and recreating CHECK constraints.

ALTER TABLE edge DROP CONSTRAINT IF EXISTS edge_predicate_check;

ALTER TABLE edge ADD CONSTRAINT edge_predicate_check CHECK (predicate IN (
  -- PROV-O / Schema.org / Domain predicates
  'derived_from', 'cites', 'version_of', 'part_of',
  'documents', 'depends_on', 'related_to',
  -- Strategic DDD predicates (ADR-0009)
  'implements',
  'delivered_by',
  'integration',
  -- Aggregate structure predicates
  'described_by',
  -- Design doc predicates (#211)
  'references',
  'covers',
  -- Consulting deployment predicates (semops-orchestrator#196, ADR-0008 D7)
  'deploys_to',
  'adopts_from',
  'incubates',
  'originated_from'
));

-- Update comment
COMMENT ON COLUMN edge.predicate IS
  'PROV-O: derived_from, cites, version_of; Schema.org: part_of, documents; '
  'Domain: depends_on, related_to; Strategic DDD: implements, delivered_by, integration; '
  'Aggregate: described_by; Design: references, covers; '
  'Deployment: deploys_to, adopts_from, incubates, originated_from (ADR-0008)';

-- Record migration
INSERT INTO schema_version (version, description)
VALUES ('8.1.0', 'Consulting deployment edge predicates (semops-orchestrator#196, ADR-0008 D7)')
ON CONFLICT DO NOTHING;
