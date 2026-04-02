-- Project Ike Schema Fitness Functions (Phase 2 + Tactical DDD)
-- Validation rules to ensure schema integrity and business rules.
--
-- Updated for Phase 2 schema (pattern/entity/edge) and ADR-0009
-- entity_type discriminator (content/capability/repository).

-- =============================================================================
-- CORE BUSINESS RULE VALIDATIONS
-- =============================================================================

-- Rule: Pattern IDs must be valid kebab-case slugs
create or replace function check_pattern_id_format()
returns table(invalid_id text, issue text) as $$
begin
  return query
  select
    p.id,
    'Invalid pattern ID format - must be lowercase kebab-case, 3-50 chars' as issue
  from pattern p
  where p.id !~ '^[a-z0-9]+(-[a-z0-9]+)*$'
     or length(p.id) < 3
     or length(p.id) > 50;
end;
$$ language plpgsql;

-- Rule: Entity IDs must be valid kebab-case slugs
create or replace function check_entity_id_format()
returns table(invalid_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Invalid entity ID format - must be lowercase kebab-case, 3-80 chars' as issue
  from entity e
  where e.id !~ '^[a-z0-9]+(-[a-z0-9]+)*$'
     or length(e.id) < 3
     or length(e.id) > 80;
end;
$$ language plpgsql;

-- Rule: Content entities must have asset_type set (file or link)
-- Capability and repository entities should have NULL asset_type
create or replace function check_content_entity_asset_type()
returns table(invalid_id text, entity_type text, issue text) as $$
begin
  -- Content entities missing asset_type
  return query
  select
    e.id,
    e.entity_type,
    'Content entity missing asset_type (must be file or link)' as issue
  from entity e
  where e.entity_type = 'content'
    and e.asset_type is null;

  -- Non-content entities with asset_type set (warning, not error)
  return query
  select
    e.id,
    e.entity_type,
    'Non-content entity has asset_type set (should be NULL)' as issue
  from entity e
  where e.entity_type != 'content'
    and e.asset_type is not null;
end;
$$ language plpgsql;

-- Rule: Every capability must trace to at least one pattern (ADR-0009 coherence signal)
-- Via primary_pattern_id OR implements edge
create or replace function check_capability_pattern_coverage()
returns table(invalid_id text, capability_name text, issue text) as $$
begin
  return query
  select
    cap.id,
    cap.title,
    'Capability has no pattern link (needs primary_pattern_id or implements edge)' as issue
  from entity cap
  where cap.entity_type = 'capability'
    and cap.primary_pattern_id is null
    and not exists (
      select 1 from edge e
      where e.src_type = 'entity' and e.src_id = cap.id
        and e.dst_type = 'pattern' and e.predicate = 'implements'
    );
end;
$$ language plpgsql;

-- Rule: Edge strength must be between 0.0 and 1.0
create or replace function check_edge_strength_bounds()
returns table(edge_key text, current_strength decimal, issue text) as $$
begin
  return query
  select
    e.src_type || ':' || e.src_id || ' -> ' || e.dst_type || ':' || e.dst_id || ' (' || e.predicate || ')',
    e.strength,
    'Edge strength out of bounds' as issue
  from edge e
  where e.strength < 0.0 or e.strength > 1.0;
end;
$$ language plpgsql;

-- =============================================================================
-- REFERENCE INTEGRITY CHECKS
-- =============================================================================

-- Rule: Edge references must point to existing entities/patterns/surfaces
create or replace function check_edge_referential_integrity()
returns table(edge_key text, issue text) as $$
begin
  -- Check source references
  return query
  select
    e.src_type || ':' || e.src_id || ' -> ' || e.dst_type || ':' || e.dst_id || ' (' || e.predicate || ')',
    'Source ' || e.src_type || ' does not exist' as issue
  from edge e
  where (e.src_type = 'entity' and not exists (select 1 from entity where id = e.src_id))
     or (e.src_type = 'pattern' and not exists (select 1 from pattern where id = e.src_id))
     or (e.src_type = 'surface' and not exists (select 1 from surface where id = e.src_id));

  -- Check destination references
  return query
  select
    e.src_type || ':' || e.src_id || ' -> ' || e.dst_type || ':' || e.dst_id || ' (' || e.predicate || ')',
    'Destination ' || e.dst_type || ' does not exist' as issue
  from edge e
  where (e.dst_type = 'entity' and not exists (select 1 from entity where id = e.dst_id))
     or (e.dst_type = 'pattern' and not exists (select 1 from pattern where id = e.dst_id))
     or (e.dst_type = 'surface' and not exists (select 1 from surface where id = e.dst_id));
end;
$$ language plpgsql;

-- Rule: Integration edges must have required metadata
-- integration_pattern and direction are required for repo-to-repo integration edges
create or replace function check_integration_edge_metadata()
returns table(edge_key text, issue text) as $$
begin
  return query
  select
    e.src_id || ' -> ' || e.dst_id,
    'Integration edge missing required metadata (needs integration_pattern and direction)' as issue
  from edge e
  where e.predicate = 'integration'
    and (
      not e.metadata ? 'integration_pattern'
      or not e.metadata ? 'direction'
    );
end;
$$ language plpgsql;

-- =============================================================================
-- DATA QUALITY CHECKS
-- =============================================================================

-- Rule: Patterns should have complete definitions
create or replace function check_pattern_completeness()
returns table(pattern_id text, issue text) as $$
begin
  return query
  select
    p.id,
    'Pattern definition is too short (< 10 chars)' as issue
  from pattern p
  where p.definition is null
     or length(trim(p.definition)) < 10;
end;
$$ language plpgsql;

-- Rule: Entities should have meaningful titles
create or replace function check_entity_title_quality()
returns table(invalid_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Entity title is too short or generic' as issue
  from entity e
  where e.title is not null
    and (
      length(trim(e.title)) < 5
      or e.title ilike any(array['untitled%', 'draft%', 'temp%', 'test%'])
    );
end;
$$ language plpgsql;

-- Rule: Agent entities must exercise at least one capability (ADR-0013 coherence signal)
create or replace function check_agent_capability_coverage()
returns table(invalid_id text, agent_name text, issue text) as $$
begin
  return query
  select
    a.id,
    a.title,
    'Agent has no capability link (needs implements edge to a capability)' as issue
  from entity a
  where a.entity_type = 'agent'
    and not exists (
      select 1 from edge e
      where e.src_type = 'entity' and e.src_id = a.id
        and e.dst_type = 'entity' and e.predicate = 'implements'
    );
end;
$$ language plpgsql;

-- Rule: Capabilities should have at least one agent exercising them (ADR-0013 coverage gap)
create or replace function check_capability_agent_coverage()
returns table(invalid_id text, capability_name text, issue text) as $$
begin
  return query
  select
    cap.id,
    cap.title,
    'Capability has no agent exercising it (no implements edge from any agent)' as issue
  from entity cap
  where cap.entity_type = 'capability'
    and not exists (
      select 1 from edge e
      where e.dst_type = 'entity' and e.dst_id = cap.id
        and e.src_type = 'entity' and e.predicate = 'implements'
        and exists (select 1 from entity a where a.id = e.src_id and a.entity_type = 'agent')
    );
end;
$$ language plpgsql;

-- Rule: Explicit Architecture coherence — patterns with capabilities should have
-- bidirectional coverage (pattern->capability via implements, capability->repo via delivered_by)
-- This is the EA invariant: architecture is queryable, gaps are visible as query results.
create or replace function check_explicit_architecture_coverage()
returns table(pattern_id text, pattern_name text, issue text) as $$
begin
  -- Patterns with no implementing capabilities (orphan patterns)
  return query
  select
    p.id,
    p.preferred_label,
    'Pattern has no implementing capability (EA gap: invisible to capability layer)' as issue
  from pattern p
  where p.metadata->>'pattern_type' = 'domain'
    and p.metadata->>'status' != 'retired'
    and not exists (
      select 1 from edge e
      where e.dst_type = 'pattern' and e.dst_id = p.id
        and e.src_type = 'entity' and e.predicate = 'implements'
        and exists (select 1 from entity cap where cap.id = e.src_id and cap.entity_type = 'capability')
    );
end;
$$ language plpgsql;

-- Rule: Published deliveries should have published_at timestamp
create or replace function check_published_deliveries_have_timestamp()
returns table(delivery_id text, issue text) as $$
begin
  return query
  select
    d.id,
    'Published delivery missing published_at timestamp' as issue
  from delivery d
  where d.status = 'published'
    and d.published_at is null;
end;
$$ language plpgsql;

-- Rule: Content entities in core_kb should have LLM-enriched metadata
-- Detects entities missing primary_concept, summary, subject_area, or detected_edges
create or replace function check_content_metadata_completeness()
returns table(entity_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Missing ' || string_agg(field, ', ' order by field) as issue
  from entity e,
  lateral (
    values
      ('primary_concept', e.metadata->>'primary_concept'),
      ('summary', e.metadata->>'summary'),
      ('subject_area', e.metadata->>'subject_area'),
      ('detected_edges', e.metadata->>'detected_edges')
  ) as checks(field, val)
  where e.entity_type = 'content'
    and e.metadata->>'corpus' = 'core_kb'
    and (val is null or val = '[]' or val = 'null' or val = '')
  group by e.id;
end;
$$ language plpgsql;

-- =============================================================================
-- CATALOG COMPLETENESS (metadata contracts — #174)
-- =============================================================================

-- Rule: Capability entities must have complete metadata (registry.yaml contract)
create or replace function check_capability_completeness()
returns table(entity_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Missing ' || string_agg(field, ', ' order by field) as issue
  from entity e,
  lateral (
    values
      ('description', e.metadata->>'description'),
      ('domain_classification', e.metadata->>'domain_classification'),
      ('implements_patterns', e.metadata->>'implements_patterns'),
      ('delivered_by_repos', e.metadata->>'delivered_by_repos'),
      ('status', e.metadata->>'status')
  ) as checks(field, val)
  where e.entity_type = 'capability'
    and (val is null or val = '[]' or val = 'null' or val = '')
  group by e.id;
end;
$$ language plpgsql;

-- Rule: Repository entities must have complete metadata (repos.yaml contract)
create or replace function check_repository_completeness()
returns table(entity_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Missing ' || string_agg(field, ', ' order by field) as issue
  from entity e,
  lateral (
    values
      ('role', e.metadata->>'role'),
      ('github_url', e.metadata->>'github_url'),
      ('delivers_capabilities', e.metadata->>'delivers_capabilities'),
      ('status', e.metadata->>'status')
  ) as checks(field, val)
  where e.entity_type = 'repository'
    and (val is null or val = '[]' or val = 'null' or val = '')
  group by e.id;
end;
$$ language plpgsql;

-- Rule: Agent entities must have complete metadata (agents.yaml contract)
create or replace function check_agent_completeness()
returns table(entity_id text, issue text) as $$
begin
  return query
  select
    e.id,
    'Missing ' || string_agg(field, ', ' order by field) as issue
  from entity e,
  lateral (
    values
      ('agent_type', e.metadata->>'agent_type'),
      ('deployed_as', e.metadata->>'deployed_as')
  ) as checks(field, val)
  where e.entity_type = 'agent'
    and (val is null or val = 'null' or val = '')
  group by e.id;
end;
$$ language plpgsql;

-- Rule: Patterns must have complete registration (pattern_v1.yaml contract)
create or replace function check_pattern_registration_completeness()
returns table(pattern_id text, issue text) as $$
begin
  return query
  select
    p.id,
    'Missing ' || string_agg(field, ', ' order by field) as issue
  from pattern p,
  lateral (
    values
      ('definition', p.definition),
      ('provenance', p.metadata->>'provenance'),
      ('pattern_type', p.metadata->>'pattern_type')
  ) as checks(field, val)
  where (val is null or val = 'null' or val = '' or length(trim(coalesce(val, ''))) < 3)
  group by p.id;
end;
$$ language plpgsql;

comment on function check_capability_completeness() is
  '#174 catalog contract: capabilities must have description, domain_classification, implements_patterns, delivered_by_repos, status';

comment on function check_repository_completeness() is
  '#174 catalog contract: repositories must have role, github_url, delivers_capabilities, status';

comment on function check_agent_completeness() is
  '#174 catalog contract: agents must have agent_type, deployed_as';

comment on function check_pattern_registration_completeness() is
  '#174 catalog contract: patterns must have definition, provenance, pattern_type';

-- =============================================================================
-- COMPREHENSIVE FITNESS CHECK
-- =============================================================================

-- Master function to run all fitness functions
create or replace function run_all_fitness_functions()
returns table(
  check_name text,
  entity_id text,
  issue_description text,
  severity text
) as $$
begin
  -- Critical: structural integrity
  return query
  select 'edge_references' as check_name, split_part(edge_key, ' ', 1), issue, 'CRITICAL' as severity
  from check_edge_referential_integrity();

  return query
  select 'capability_coverage' as check_name, invalid_id, issue, 'CRITICAL' as severity
  from check_capability_pattern_coverage();

  return query
  select 'agent_capability_coverage' as check_name, invalid_id, issue, 'CRITICAL' as severity
  from check_agent_capability_coverage();

  return query
  select 'capability_agent_coverage' as check_name, invalid_id, issue, 'MEDIUM' as severity
  from check_capability_agent_coverage();

  -- High: schema format
  return query
  select 'pattern_id_format' as check_name, invalid_id, issue, 'HIGH' as severity
  from check_pattern_id_format();

  return query
  select 'entity_id_format' as check_name, invalid_id, issue, 'HIGH' as severity
  from check_entity_id_format();

  return query
  select 'content_asset_type' as check_name, invalid_id, issue, 'HIGH' as severity
  from check_content_entity_asset_type();

  return query
  select 'integration_metadata' as check_name, edge_key, issue, 'HIGH' as severity
  from check_integration_edge_metadata();

  -- Medium: EA coverage
  return query
  select 'ea_pattern_coverage' as check_name, pattern_id, issue, 'MEDIUM' as severity
  from check_explicit_architecture_coverage();

  -- Medium: data consistency
  return query
  select 'edge_strength' as check_name, split_part(edge_key, ' ', 1), issue, 'MEDIUM' as severity
  from check_edge_strength_bounds();

  return query
  select 'published_timestamp' as check_name, delivery_id, issue, 'MEDIUM' as severity
  from check_published_deliveries_have_timestamp();

  -- Low: data quality
  return query
  select 'pattern_completeness' as check_name, pattern_id, issue, 'LOW' as severity
  from check_pattern_completeness();

  return query
  select 'title_quality' as check_name, invalid_id, issue, 'LOW' as severity
  from check_entity_title_quality();

  return query
  select 'content_metadata' as check_name, entity_id, issue, 'MEDIUM' as severity
  from check_content_metadata_completeness();

  -- Catalog completeness (#174)
  return query
  select 'capability_completeness' as check_name, entity_id, issue, 'MEDIUM' as severity
  from check_capability_completeness();

  return query
  select 'repository_completeness' as check_name, entity_id, issue, 'MEDIUM' as severity
  from check_repository_completeness();

  return query
  select 'agent_completeness' as check_name, entity_id, issue, 'MEDIUM' as severity
  from check_agent_completeness();

  return query
  select 'pattern_registration' as check_name, pattern_id, issue, 'MEDIUM' as severity
  from check_pattern_registration_completeness();
end;
$$ language plpgsql;

-- =============================================================================
-- USAGE EXAMPLES
-- =============================================================================

/*
-- Run all fitness functions
select * from run_all_fitness_functions()
order by
  case severity
    when 'CRITICAL' then 1
    when 'HIGH' then 2
    when 'MEDIUM' then 3
    when 'LOW' then 4
  end,
  check_name;

-- Run specific check
select * from check_capability_pattern_coverage();

-- Check fitness before deployment
do $$
declare
  violation_count int;
begin
  select count(*) into violation_count
  from run_all_fitness_functions()
  where severity in ('CRITICAL', 'HIGH');

  if violation_count > 0 then
    raise exception 'Schema fitness check failed: % critical/high violations found', violation_count;
  end if;

  raise notice 'Schema fitness check passed';
end $$;
*/

-- Add comments for documentation
comment on function check_pattern_id_format() is
  'Validates pattern IDs follow kebab-case naming convention (3-50 chars)';

comment on function check_entity_id_format() is
  'Validates entity IDs follow kebab-case naming convention (3-80 chars)';

comment on function check_content_entity_asset_type() is
  'Ensures content entities have asset_type set and non-content entities have NULL';

comment on function check_capability_pattern_coverage() is
  'ADR-0009 coherence signal: every capability must trace to at least one pattern';

comment on function check_edge_referential_integrity() is
  'Validates that all edge endpoints reference existing entities, patterns, or surfaces';

comment on function check_integration_edge_metadata() is
  'Validates that integration edges have required metadata (integration_pattern, direction)';

comment on function check_agent_capability_coverage() is
  'ADR-0013 coherence signal: every agent must exercise at least one capability';

comment on function check_capability_agent_coverage() is
  'ADR-0013 coverage gap: capabilities without agents are unimplemented at the application layer';

comment on function check_explicit_architecture_coverage() is
  'Explicit Architecture coherence: domain patterns without implementing capabilities are invisible to the capability layer';

comment on function check_content_metadata_completeness() is
  'Detects core_kb content entities missing LLM-enriched metadata (primary_concept, summary, subject_area, detected_edges)';

comment on function run_all_fitness_functions() is
  'Master function to execute all schema validation rules with severity levels';
