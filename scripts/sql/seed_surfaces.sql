-- Project Ike Surface Records
-- Owned surfaces for Tim Mitchell / TJMConsulting
--
-- Run with: psql -f scripts/sql/seed_surfaces.sql
-- Or: python scripts/run_sql.py scripts/sql/seed_surfaces.sql

-- GitHub profile (primary for source ingestion)
INSERT INTO surface (id, platform, surface_type, direction, constraints, metadata) VALUES
('github-semops-ai', 'github', 'profile', 'bidirectional',
 '{"formats": ["markdown", "code", "json", "yaml"]}',
 '{"username": "semops-ai", "organization": "TJMConsulting"}')
ON CONFLICT (id) DO UPDATE SET
  metadata = EXCLUDED.metadata,
  constraints = EXCLUDED.constraints,
  updated_at = now();

-- WordPress site
INSERT INTO surface (id, platform, surface_type, direction, constraints, metadata) VALUES
('wordpress-semops-ai', 'wordpress', 'site', 'bidirectional',
 '{"formats": ["html", "markdown"], "max_post_size_bytes": 1000000}',
 '{"site_url": "https://www.semops-ai.com", "site_name": "Tim J Mitchell"}')
ON CONFLICT (id) DO UPDATE SET
  metadata = EXCLUDED.metadata,
  constraints = EXCLUDED.constraints,
  updated_at = now();

-- LinkedIn profile
INSERT INTO surface (id, platform, surface_type, direction, constraints, metadata) VALUES
('linkedin-semops-ai', 'linkedin', 'profile', 'publish',
 '{"max_post_length": 3000, "formats": ["text", "markdown"]}',
 '{"profile_url": "https://linkedin.com/in/semops-ai"}')
ON CONFLICT (id) DO UPDATE SET
  metadata = EXCLUDED.metadata,
  constraints = EXCLUDED.constraints,
  updated_at = now();

-- Surface addresses

-- GitHub public profile
INSERT INTO surface_address (surface_id, kind, uri, active) VALUES
('github-semops-ai', 'public', 'https://github.com/semops-ai', true)
ON CONFLICT (surface_id, kind, uri) DO NOTHING;

-- GitHub API endpoint
INSERT INTO surface_address (surface_id, kind, uri, active) VALUES
('github-semops-ai', 'api', 'https://api.github.com/users/semops-ai', true)
ON CONFLICT (surface_id, kind, uri) DO NOTHING;

-- WordPress public site
INSERT INTO surface_address (surface_id, kind, uri, active) VALUES
('wordpress-semops-ai', 'public', 'https://www.semops-ai.com', true)
ON CONFLICT (surface_id, kind, uri) DO NOTHING;

-- WordPress API
INSERT INTO surface_address (surface_id, kind, uri, active) VALUES
('wordpress-semops-ai', 'api', 'https://www.semops-ai.com/wp-json/wp/v2', true)
ON CONFLICT (surface_id, kind, uri) DO NOTHING;

-- LinkedIn profile
INSERT INTO surface_address (surface_id, kind, uri, active) VALUES
('linkedin-semops-ai', 'public', 'https://linkedin.com/in/semops-ai', true)
ON CONFLICT (surface_id, kind, uri) DO NOTHING;
