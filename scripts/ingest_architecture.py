#!/usr/bin/env python3
"""
Architecture layer ingestion for Project SemOps.

Parses structured architecture sources (REPOS.yaml, config/registry.yaml) and
creates repository entities, capability entities, agent entities, and typed
edges (implements, delivered_by, integration) in the entity catalog.

This is a deterministic parser — no LLM classification needed. The architecture
data is already structured in YAML.

Usage:
    python scripts/ingest_architecture.py                    # Full ingestion
    python scripts/ingest_architecture.py --dry-run          # Parse only, no DB
    python scripts/ingest_architecture.py --verify           # Ingest + test queries
    python scripts/ingest_architecture.py --skip-neo4j       # Skip Neo4j sync
    python scripts/ingest_architecture.py --dry-run -v       # Verbose dry run
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import psycopg
import yaml
from rich.console import Console
from rich.table import Table

console = Console()

# ---------------------------------------------------------------------------
# Source file locations
# ---------------------------------------------------------------------------
REPOS_YAML = Path.home() / "GitHub" / "semops-orchestrator" / "config" / "repos.yaml"
REGISTRY_YAML = Path(__file__).parent.parent / "config" / "registry.yaml"
DESIGN_DOCS_YAML = Path.home() / "GitHub" / "semops-orchestrator" / "config" / "design-docs.yaml"

# Repos outside the SemOps bounded context
SKIP_REPOS = {"motorsport-consulting"}

# All SemOps repos (for expanding "all repos" in integration map)
SEMOPS_REPOS = {
    "semops-orchestrator",
    "semops-data",
    "publisher-pr",
    "docs-pr",
    "data-pr",
    "sites-pr",
    "semops-backoffice",
}

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection
from yaml_comments import extract_comments, merge_comments_into_metadata

NEO4J_URL = os.environ.get("NEO4J_URL", "http://localhost:7474")


# ---------------------------------------------------------------------------
# Neo4j utilities (reused from materialize_graph.py)
# ---------------------------------------------------------------------------
def neo4j_escape(s: str) -> str:
    """Escape string for Cypher."""
    return s.replace("\\", "\\\\").replace("'", "\\'")


def run_cypher(cypher: str) -> dict | None:
    """Execute Cypher statement via Neo4j HTTP API."""
    try:
        result = subprocess.run(
            [
                "curl", "-s",
                "-H", "Content-Type: application/json",
                "-d", json.dumps({"statements": [{"statement": cypher}]}),
                f"{NEO4J_URL}/db/neo4j/tx/commit",
            ],
            capture_output=True,
            text=True,
            timeout=10,
        )
        return json.loads(result.stdout) if result.returncode == 0 else None
    except Exception:
        return None


# ---------------------------------------------------------------------------
# Parsers
# ---------------------------------------------------------------------------
def parse_repos(yaml_path: Path) -> list[dict]:
    """Parse REPOS.yaml into repository entity dicts."""
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    repos = []
    for entry in data.get("repos", []):
        name = entry.get("name", "")
        if name in SKIP_REPOS:
            continue
        if entry.get("status") != "active":
            continue

        now = datetime.now(timezone.utc).isoformat()
        repos.append({
            "id": name,
            "entity_type": "repository",
            "asset_type": None,
            "title": name,
            "version": "1.0",
            "filespec": json.dumps({
                "$schema": "filespec_v1",
                "uri": f"github://semops-ai/{name}",
                "platform": "github",
            }),
            "attribution": json.dumps({
                "$schema": "attribution_v2",
                "creator": ["Tim Mitchell"],
                "organization": "TJMConsulting",
                "platform": "github",
            }),
            "metadata": {
                "$schema": "repository_metadata_v1",
                "role": entry.get("role", ""),
                "context": entry.get("context", ""),
                "github_url": f"https://github.com/semops-ai/{name}",
                "delivers_capabilities": [],  # populated after capability parsing
                "status": "active",
                "layer": entry.get("layer", ""),
                "context_type": entry.get("context_type", ""),
                "bounded_context": entry.get("bounded_context", ""),
                "depends_on": entry.get("depends_on", []),
                "provides": entry.get("provides", []),
                "subdomains": entry.get("subdomains", []),
            },
            "created_at": now,
            "updated_at": now,
            "_raw": entry,  # keep for cross-referencing
        })

    return repos


def load_registry(yaml_path: Path) -> dict:
    """Load the centralized architecture registry YAML."""
    with open(yaml_path) as f:
        return yaml.safe_load(f)


def load_capabilities(registry: dict) -> list[dict]:
    """Load capabilities from registry YAML into entity dicts."""
    capabilities = []
    caps_section = registry.get("capabilities", {})

    # Core capabilities: grouped by layer
    core = caps_section.get("core", {})
    for _layer_name, layer_caps in core.items():
        if not isinstance(layer_caps, list):
            continue
        for entry in layer_caps:
            capabilities.append(_capability_to_entity(entry, "core"))

    # Generic capabilities: flat list
    for entry in caps_section.get("generic", []):
        capabilities.append(_capability_to_entity(entry, "generic"))

    return capabilities


def _capability_to_entity(entry: dict, domain_classification: str) -> dict:
    """Convert a registry YAML capability entry to an entity dict."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": entry["id"],
        "entity_type": "capability",
        "asset_type": None,
        "title": entry["name"],
        "version": "1.0",
        "filespec": json.dumps({}),
        "attribution": json.dumps({
            "$schema": "attribution_v2",
            "creator": ["Tim Mitchell"],
            "organization": "TJMConsulting",
        }),
        "metadata": {
            "$schema": "capability_metadata_v1",
            "domain_classification": domain_classification,
            "implements_patterns": entry.get("implements_patterns", []),
            "delivered_by_repos": entry.get("delivered_by", []),
            "status": entry.get("status", "draft"),
            "projects": entry.get("projects", []),
            "governance": entry.get("governance", {}),
        },
        "created_at": now,
        "updated_at": now,
    }


def build_repo_capability_map(capabilities: list[dict]) -> dict[str, list[str]]:
    """Derive repo→capability map from capability delivered_by fields."""
    repo_map: dict[str, list[str]] = {}
    for cap in capabilities:
        for repo_id in cap["metadata"].get("delivered_by_repos", []):
            repo_map.setdefault(repo_id, []).append(cap["id"])
    return repo_map


def load_integration_edges(registry: dict) -> list[dict]:
    """Load integration map from registry YAML → edge dicts."""
    edges = []
    for entry in registry.get("integrations", []):
        src = entry["source"]
        dst = entry["target"]
        pattern_kebab = entry["pattern"]
        shared = entry.get("shared_artifact", "")
        direction = entry.get("direction", "downstream").lower()

        # Expand "all repos" to individual edges
        if dst.lower() == "all repos":
            for repo in sorted(SEMOPS_REPOS - {src}):
                edges.append({
                    "src_type": "entity",
                    "src_id": src,
                    "dst_type": "entity",
                    "dst_id": repo,
                    "predicate": "integration",
                    "strength": 1.0,
                    "metadata": {
                        "integration_pattern": pattern_kebab,
                        "shared_artifact": shared,
                        "direction": direction,
                    },
                })
        else:
            edges.append({
                "src_type": "entity",
                "src_id": src,
                "dst_type": "entity",
                "dst_id": dst,
                "predicate": "integration",
                "strength": 1.0,
                "metadata": {
                    "integration_pattern": pattern_kebab,
                    "shared_artifact": shared,
                    "direction": direction,
                },
            })

    return edges


def load_agents(registry: dict) -> list[dict]:
    """Load agents from registry YAML into entity dicts.

    Agents may have been moved to a separate file (semops-orchestrator/config/agents.yaml).
    If the registry value is a string pointer instead of a list, return empty.
    """
    agents = []
    agents_list = registry.get("agents", [])

    if not isinstance(agents_list, list):
        return agents

    for entry in agents_list:
        agents.append(_agent_to_entity(entry))

    return agents


def _agent_to_entity(entry: dict) -> dict:
    """Convert a registry YAML agent entry to an entity dict."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": entry["id"],
        "entity_type": "agent",
        "asset_type": None,
        "title": entry["name"],
        "version": "1.0",
        "filespec": json.dumps({}),
        "attribution": json.dumps({
            "$schema": "attribution_v2",
            "creator": ["Tim Mitchell"],
            "organization": "TJMConsulting",
        }),
        "metadata": {
            "$schema": "agent_metadata_v1",
            "agent_type": entry.get("agent_type", ""),
            "deployed_as": entry.get("deployed_as", ""),
            "tools": entry.get("tools", []),
            "memory": entry.get("memory", ""),
            "exercises_capabilities": entry.get("exercises_capabilities", []),
            "delivered_by_repo": entry.get("delivered_by", ""),
            "lifecycle_stage": entry.get("lifecycle_stage", "active"),
            "layer": entry.get("layer", ""),
        },
        "created_at": now,
        "updated_at": now,
    }


def load_design_docs(yaml_path: Path) -> list[dict]:
    """Load design docs from design-docs.yaml into entity dicts."""
    if not yaml_path.exists():
        console.print(f"[yellow]Design docs source not found: {yaml_path} — skipping[/yellow]")
        return []

    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    docs = []
    for entry in data.get("design_docs", []):
        docs.append(_design_doc_to_entity(entry))
    return docs


def _design_doc_to_entity(entry: dict) -> dict:
    """Convert a design-docs.yaml entry to an entity dict."""
    now = datetime.now(timezone.utc).isoformat()
    return {
        "id": entry["id"],
        "entity_type": "design_doc",
        "asset_type": None,
        "title": entry.get("title", entry["id"]),
        "version": str(entry.get("version", "0.1.0")),
        "filespec": json.dumps({
            "$schema": "filespec_v1",
            "uri": f"file://{entry.get('path', '')}",
            "platform": "local",
        }),
        "attribution": json.dumps({
            "$schema": "attribution_v2",
            "creator": ["Tim Mitchell"],
            "organization": "TJMConsulting",
        }),
        "metadata": {
            "$schema": "design_doc_metadata_v1",
            "slug": entry.get("slug", ""),
            "status": entry.get("status", "Draft"),
            "date": str(entry.get("date", "")),
            "path": entry.get("path", ""),
            "adrs": entry.get("adrs", []),
            "issues": entry.get("issues", []),
            "domain_concepts": entry.get("domain_concepts", []),
        },
        "created_at": now,
        "updated_at": now,
    }


def build_design_doc_edges(design_docs: list[dict]) -> list[dict]:
    """Build edges from design docs → ADRs (references) and → domain concepts (covers)."""
    edges = []
    for doc in design_docs:
        meta = doc["metadata"]

        # references edges to ADRs (ADR IDs like "semops-data/ADR-0015")
        for adr_ref in meta.get("adrs", []):
            edges.append({
                "src_type": "entity",
                "src_id": doc["id"],
                "dst_type": "entity",
                "dst_id": adr_ref,
                "predicate": "references",
                "strength": 1.0,
                "metadata": {},
            })

        # covers edges to domain concepts
        for concept in meta.get("domain_concepts", []):
            edges.append({
                "src_type": "entity",
                "src_id": doc["id"],
                "dst_type": "entity",
                "dst_id": concept,
                "predicate": "covers",
                "strength": 1.0,
                "metadata": {},
            })

    return edges


def build_agent_implements_edges(agents: list[dict]) -> list[dict]:
    """Build implements edges from agent → capability."""
    edges = []
    for agent in agents:
        meta = agent["metadata"]
        for cap_id in meta.get("exercises_capabilities", []):
            edges.append({
                "src_type": "entity",
                "src_id": agent["id"],
                "dst_type": "entity",
                "dst_id": cap_id,
                "predicate": "implements",
                "strength": 1.0,
                "metadata": {},
            })
    return edges


def build_agent_delivered_by_edges(agents: list[dict]) -> list[dict]:
    """Build delivered_by edges from agent → repository."""
    edges = []
    for agent in agents:
        meta = agent["metadata"]
        repo_id = meta.get("delivered_by_repo")
        if repo_id:
            edges.append({
                "src_type": "entity",
                "src_id": agent["id"],
                "dst_type": "entity",
                "dst_id": repo_id,
                "predicate": "delivered_by",
                "strength": 1.0,
                "metadata": {},
            })
    return edges


def derive_lifecycle_stages(
    capabilities: list[dict],
    repos: list[dict],
    implements_edges: list[dict],
    delivered_by_edges: list[dict],
) -> None:
    """Set lifecycle_stage for capabilities and repos.

    Lifecycle evolution model (Issue #146):
      v1 (current): Human-declared status from STRATEGIC_DDD.md Status column.
         Capabilities use the 5-state model: planned → draft → in_progress → active → retired.
         Repos use lifecycle_stage from REPOS.yaml (defaulting to 'active').
      v2 (future): System validates declared vs edge-derived, reports delta.
      v3 (future): Computed becomes primary, declared becomes override.

    Sets lifecycle_stage in entity.metadata (JSONB), consistent with how
    source_config.py / entity_builder.py set it for content entities.
    """
    # v1: Use declared status from parsed table (stored in metadata.status)
    valid_states = {"planned", "draft", "in_progress", "active", "retired"}

    for cap in capabilities:
        declared = cap["metadata"].get("status", "draft")
        cap["metadata"]["lifecycle_stage"] = declared if declared in valid_states else "draft"

    # Repos: use status from REPOS.yaml parse (defaults to 'active')
    for repo in repos:
        declared = repo["metadata"].get("status", "active")
        repo["metadata"]["lifecycle_stage"] = declared if declared in valid_states else "active"


def build_delivered_by_edges(capabilities: list[dict]) -> list[dict]:
    """Build delivered_by edges from capability metadata."""
    edges = []
    for cap in capabilities:
        meta = cap["metadata"]
        for repo_id in meta.get("delivered_by_repos", []):
            edges.append({
                "src_type": "entity",
                "src_id": cap["id"],
                "dst_type": "entity",
                "dst_id": repo_id,
                "predicate": "delivered_by",
                "strength": 1.0,
                "metadata": {},
            })
    return edges


def build_depends_on_edges(repos: list[dict]) -> list[dict]:
    """Build depends_on edges from repository metadata (#211)."""
    edges = []
    for repo in repos:
        meta = repo["metadata"]
        for dep_id in meta.get("depends_on", []):
            edges.append({
                "src_type": "entity",
                "src_id": repo["id"],
                "dst_type": "entity",
                "dst_id": dep_id,
                "predicate": "depends_on",
                "strength": 1.0,
                "metadata": {},
            })
    return edges


def build_implements_edges(
    capabilities: list[dict], registered_patterns: set[str]
) -> tuple[list[dict], list[str]]:
    """Build implements edges for patterns that exist in DB.

    Returns (edges, warnings) where warnings lists unregistered pattern IDs.
    """
    edges = []
    unregistered: set[str] = set()

    for cap in capabilities:
        meta = cap["metadata"]
        for pattern_id in meta.get("implements_patterns", []):
            if pattern_id in registered_patterns:
                edges.append({
                    "src_type": "entity",
                    "src_id": cap["id"],
                    "dst_type": "pattern",
                    "dst_id": pattern_id,
                    "predicate": "implements",
                    "strength": 1.0,
                    "metadata": {},
                })
            else:
                unregistered.add(pattern_id)

    return edges, sorted(unregistered)


# ---------------------------------------------------------------------------
# Database operations
# ---------------------------------------------------------------------------
def upsert_entity(entity: dict, cursor: Any) -> bool:
    """Insert or update an entity."""
    try:
        cursor.execute(
            """
            INSERT INTO entity (
                id, entity_type, asset_type, title, version,
                filespec, attribution, metadata,
                created_at, updated_at
            ) VALUES (
                %(id)s, %(entity_type)s, %(asset_type)s, %(title)s, %(version)s,
                %(filespec)s, %(attribution)s, %(metadata)s,
                %(created_at)s, %(updated_at)s
            )
            ON CONFLICT (id) DO UPDATE SET
                entity_type = EXCLUDED.entity_type,
                title = EXCLUDED.title,
                filespec = EXCLUDED.filespec,
                attribution = EXCLUDED.attribution,
                metadata = EXCLUDED.metadata,
                updated_at = EXCLUDED.updated_at
            """,
            {
                "id": entity["id"],
                "entity_type": entity["entity_type"],
                "asset_type": entity["asset_type"],
                "title": entity["title"],
                "version": entity["version"],
                "filespec": entity["filespec"] if isinstance(entity["filespec"], str)
                    else json.dumps(entity["filespec"]),
                "attribution": entity["attribution"] if isinstance(entity["attribution"], str)
                    else json.dumps(entity["attribution"]),
                "metadata": json.dumps(entity["metadata"]),
                "created_at": entity["created_at"],
                "updated_at": entity["updated_at"],
            },
        )
        return True
    except Exception as e:
        console.print(f"[red]Entity upsert error ({entity['id']}): {e}[/red]")
        return False


def upsert_edge(edge: dict, cursor: Any) -> bool:
    """Insert or update an edge."""
    try:
        cursor.execute(
            """
            INSERT INTO edge (
                src_type, src_id, dst_type, dst_id, predicate,
                strength, metadata
            ) VALUES (
                %(src_type)s, %(src_id)s, %(dst_type)s, %(dst_id)s, %(predicate)s,
                %(strength)s, %(metadata)s
            )
            ON CONFLICT (src_type, src_id, dst_type, dst_id, predicate) DO UPDATE SET
                strength = EXCLUDED.strength,
                metadata = EXCLUDED.metadata
            """,
            {
                "src_type": edge["src_type"],
                "src_id": edge["src_id"],
                "dst_type": edge["dst_type"],
                "dst_id": edge["dst_id"],
                "predicate": edge["predicate"],
                "strength": edge["strength"],
                "metadata": json.dumps(edge["metadata"]),
            },
        )
        return True
    except Exception as e:
        console.print(
            f"[red]Edge upsert error ({edge['src_id']} "
            f"-[{edge['predicate']}]-> {edge['dst_id']}): {e}[/red]"
        )
        return False


def get_registered_patterns(cursor: Any) -> set[str]:
    """Query existing pattern IDs from the pattern table."""
    cursor.execute("SELECT id FROM pattern")
    return {row[0] for row in cursor.fetchall()}


# ---------------------------------------------------------------------------
# Neo4j materialization
# ---------------------------------------------------------------------------
def materialize_neo4j(
    repos: list[dict],
    capabilities: list[dict],
    delivered_by_edges: list[dict],
    implements_edges: list[dict],
    integration_edges: list[dict],
    agents: list[dict] | None = None,
    agent_implements_edges: list[dict] | None = None,
    agent_delivered_by_edges: list[dict] | None = None,
    design_docs: list[dict] | None = None,
    design_doc_edges: list[dict] | None = None,
    verbose: bool = False,
) -> dict[str, int]:
    """Materialize architecture layer to Neo4j graph."""
    counts = {"nodes": 0, "relationships": 0}

    # Check connectivity
    health = run_cypher("RETURN 1")
    if health is None:
        console.print(f"[yellow]Cannot connect to Neo4j at {NEO4J_URL} — skipping[/yellow]")
        return counts

    # Constraints
    run_cypher(
        "CREATE CONSTRAINT repository_id IF NOT EXISTS "
        "FOR (r:Repository) REQUIRE r.id IS UNIQUE"
    )
    run_cypher(
        "CREATE CONSTRAINT capability_id IF NOT EXISTS "
        "FOR (c:Capability) REQUIRE c.id IS UNIQUE"
    )

    # Repository nodes
    for repo in repos:
        meta = repo["metadata"]
        cypher = (
            f"MERGE (r:Repository {{id: '{neo4j_escape(repo['id'])}'}}) "
            f"SET r.title = '{neo4j_escape(repo['title'])}', "
            f"r.role = '{neo4j_escape(meta.get('role', ''))}'"
        )
        run_cypher(cypher)
        counts["nodes"] += 1

    # Capability nodes
    for cap in capabilities:
        meta = cap["metadata"]
        cypher = (
            f"MERGE (c:Capability {{id: '{neo4j_escape(cap['id'])}'}}) "
            f"SET c.title = '{neo4j_escape(cap['title'])}', "
            f"c.domain = '{neo4j_escape(meta.get('domain_classification', ''))}'"
        )
        run_cypher(cypher)
        counts["nodes"] += 1

    # DELIVERED_BY relationships
    for edge in delivered_by_edges:
        cypher = (
            f"MATCH (c:Capability {{id: '{neo4j_escape(edge['src_id'])}'}}) "
            f"MATCH (r:Repository {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
            f"MERGE (c)-[:DELIVERED_BY]->(r)"
        )
        run_cypher(cypher)
        counts["relationships"] += 1

    # IMPLEMENTS relationships
    for edge in implements_edges:
        cypher = (
            f"MATCH (c:Capability {{id: '{neo4j_escape(edge['src_id'])}'}}) "
            f"MERGE (p:Pattern {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
            f"MERGE (c)-[:IMPLEMENTS]->(p)"
        )
        run_cypher(cypher)
        counts["relationships"] += 1

    # INTEGRATION relationships
    for edge in integration_edges:
        meta = edge["metadata"]
        cypher = (
            f"MATCH (s:Repository {{id: '{neo4j_escape(edge['src_id'])}'}}) "
            f"MATCH (t:Repository {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
            f"MERGE (s)-[r:INTEGRATION]->(t) "
            f"SET r.pattern = '{neo4j_escape(meta.get('integration_pattern', ''))}', "
            f"r.direction = '{neo4j_escape(meta.get('direction', ''))}', "
            f"r.shared_artifact = '{neo4j_escape(meta.get('shared_artifact', ''))}'"
        )
        run_cypher(cypher)
        counts["relationships"] += 1

    # Agent nodes
    if agents:
        run_cypher(
            "CREATE CONSTRAINT agent_id IF NOT EXISTS "
            "FOR (a:Agent) REQUIRE a.id IS UNIQUE"
        )
        for agent in agents:
            meta = agent["metadata"]
            cypher = (
                f"MERGE (a:Agent {{id: '{neo4j_escape(agent['id'])}'}}) "
                f"SET a.title = '{neo4j_escape(agent['title'])}', "
                f"a.agent_type = '{neo4j_escape(meta.get('agent_type', ''))}', "
                f"a.deployed_as = '{neo4j_escape(meta.get('deployed_as', ''))}', "
                f"a.memory = '{neo4j_escape(meta.get('memory', ''))}', "
                f"a.lifecycle_stage = '{neo4j_escape(meta.get('lifecycle_stage', ''))}', "
                f"a.layer = '{neo4j_escape(meta.get('layer', ''))}'"
            )
            run_cypher(cypher)
            counts["nodes"] += 1

    # Agent IMPLEMENTS relationships (agent → capability)
    if agent_implements_edges:
        for edge in agent_implements_edges:
            cypher = (
                f"MATCH (a:Agent {{id: '{neo4j_escape(edge['src_id'])}'}}) "
                f"MATCH (c:Capability {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
                f"MERGE (a)-[:IMPLEMENTS]->(c)"
            )
            run_cypher(cypher)
            counts["relationships"] += 1

    # Agent DELIVERED_BY relationships (agent → repository)
    if agent_delivered_by_edges:
        for edge in agent_delivered_by_edges:
            cypher = (
                f"MATCH (a:Agent {{id: '{neo4j_escape(edge['src_id'])}'}}) "
                f"MATCH (r:Repository {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
                f"MERGE (a)-[:DELIVERED_BY]->(r)"
            )
            run_cypher(cypher)
            counts["relationships"] += 1

    # DesignDoc nodes (#215)
    design_docs = design_docs or []
    design_doc_edges = design_doc_edges or []
    if design_docs:
        run_cypher(
            "CREATE CONSTRAINT design_doc_id IF NOT EXISTS "
            "FOR (d:DesignDoc) REQUIRE d.id IS UNIQUE"
        )
        for doc in design_docs:
            meta = doc["metadata"]
            cypher = (
                f"MERGE (d:DesignDoc {{id: '{neo4j_escape(doc['id'])}'}}) "
                f"SET d.title = '{neo4j_escape(doc['title'])}', "
                f"d.status = '{neo4j_escape(meta.get('status', ''))}', "
                f"d.version = '{neo4j_escape(doc.get('version', ''))}'"
            )
            run_cypher(cypher)
            counts["nodes"] += 1

    # DesignDoc REFERENCES relationships (design_doc → entity, typically ADR)
    # DesignDoc COVERS relationships (design_doc → entity, typically concept)
    for edge in design_doc_edges:
        predicate = edge["predicate"]
        if predicate == "references":
            rel_type = "REFERENCES"
        elif predicate == "covers":
            rel_type = "COVERS"
        else:
            continue
        cypher = (
            f"MERGE (d:DesignDoc {{id: '{neo4j_escape(edge['src_id'])}'}}) "
            f"MERGE (t {{id: '{neo4j_escape(edge['dst_id'])}'}}) "
            f"MERGE (d)-[:{rel_type}]->(t)"
        )
        run_cypher(cypher)
        counts["relationships"] += 1

    return counts


# ---------------------------------------------------------------------------
# Verification
# ---------------------------------------------------------------------------
def run_verification(conn: psycopg.Connection) -> None:
    """Run test queries from Issue #117."""
    cursor = conn.cursor()
    console.print()
    console.print("[bold]Verification Queries[/bold]")
    console.print("=" * 60)

    # Test Query 2: "Which repo owns the Pattern model?"
    console.print()
    console.print("[bold]Query 2:[/bold] Which repo owns the Pattern model?")
    cursor.execute(
        """
        SELECT repo_id, repo_name, repo_role, capability_id, capability_name
        FROM repo_capabilities
        WHERE capability_id = 'domain-data-model'
        """
    )
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            console.print(f"  [green]{row[0]}[/green] ({row[2]}) delivers {row[4]}")
    else:
        console.print("  [yellow]No results — domain-data-model not found[/yellow]")

    # Test Query 3: "How does content flow from docs-pr to sites-pr?"
    console.print()
    console.print("[bold]Query 3:[/bold] How does content flow from docs-pr to sites-pr?")
    cursor.execute(
        """
        SELECT source_repo_id, target_repo_id, integration_pattern,
               shared_artifact, direction
        FROM integration_map
        WHERE source_repo_id IN ('docs-pr', 'sites-pr')
           OR target_repo_id IN ('docs-pr', 'sites-pr')
        """
    )
    rows = cursor.fetchall()
    if rows:
        for row in rows:
            console.print(
                f"  [green]{row[0]}[/green] -> [green]{row[1]}[/green] "
                f"({row[2]}, {row[4]}): {row[3]}"
            )
    else:
        console.print("  [yellow]No integration edges involving docs-pr or sites-pr[/yellow]")

    # Capability coverage
    console.print()
    console.print("[bold]Capability Coverage (coherence signal):[/bold]")
    cursor.execute(
        """
        SELECT capability_id, capability_name, domain_classification,
               pattern_count, repo_count
        FROM capability_coverage
        ORDER BY domain_classification, capability_name
        """
    )
    rows = cursor.fetchall()
    tbl = Table(title="Capability Coverage")
    tbl.add_column("ID")
    tbl.add_column("Name")
    tbl.add_column("Domain")
    tbl.add_column("Patterns", justify="right")
    tbl.add_column("Repos", justify="right")
    for row in rows:
        tbl.add_row(row[0], row[1], row[2] or "", str(row[3]), str(row[4]))
    console.print(tbl)


# ---------------------------------------------------------------------------
# Display helpers
# ---------------------------------------------------------------------------
def print_dry_run(
    repos: list[dict],
    capabilities: list[dict],
    agents: list[dict],
    delivered_by: list[dict],
    implements: list[dict],
    integration: list[dict],
    agent_impl_edges: list[dict],
    agent_db_edges: list[dict],
    unregistered: list[str],
    verbose: bool = False,
    design_docs: list[dict] | None = None,
    design_doc_edges: list[dict] | None = None,
    depends_on_edges: list[dict] | None = None,
) -> None:
    """Print what would be created."""
    design_docs = design_docs or []
    design_doc_edges = design_doc_edges or []
    depends_on_edges = depends_on_edges or []

    console.print()
    console.print("[bold]Dry Run Summary[/bold]")
    console.print("=" * 60)

    # Entities
    tbl = Table(title="Entities")
    tbl.add_column("Type")
    tbl.add_column("Count", justify="right")
    tbl.add_row("Repository", str(len(repos)))
    tbl.add_row("Capability", str(len(capabilities)))
    tbl.add_row("Agent", str(len(agents)))
    tbl.add_row("Design Doc", str(len(design_docs)))
    total_entities = len(repos) + len(capabilities) + len(agents) + len(design_docs)
    tbl.add_row("[bold]Total[/bold]", f"[bold]{total_entities}[/bold]")
    console.print(tbl)

    # Edges
    tbl = Table(title="Edges")
    tbl.add_column("Predicate")
    tbl.add_column("Count", justify="right")
    tbl.add_row("delivered_by", str(len(delivered_by)))
    tbl.add_row("depends_on", str(len(depends_on_edges)))
    tbl.add_row("implements", str(len(implements)))
    tbl.add_row("integration", str(len(integration)))
    tbl.add_row("agent→capability", str(len(agent_impl_edges)))
    tbl.add_row("agent→repository", str(len(agent_db_edges)))
    # Split design doc edges by predicate for clarity
    dd_refs = [e for e in design_doc_edges if e["predicate"] == "references"]
    dd_covers = [e for e in design_doc_edges if e["predicate"] == "covers"]
    tbl.add_row("dd→adr (references)", str(len(dd_refs)))
    tbl.add_row("dd→concept (covers)", str(len(dd_covers)))
    total = (len(delivered_by) + len(depends_on_edges) + len(implements)
             + len(integration) + len(agent_impl_edges) + len(agent_db_edges)
             + len(design_doc_edges))
    tbl.add_row("[bold]Total[/bold]", f"[bold]{total}[/bold]")
    console.print(tbl)

    if unregistered:
        console.print()
        console.print(
            f"[yellow]Unregistered patterns ({len(unregistered)}) — "
            f"implements edges skipped:[/yellow]"
        )
        for p in unregistered:
            console.print(f"  - {p}")

    if verbose:
        console.print()
        console.print("[bold]Repository Entities:[/bold]")
        for r in repos:
            meta = r["metadata"]
            console.print(
                f"  {r['id']} — {meta['role']} "
                f"({len(meta['delivers_capabilities'])} capabilities)"
            )

        console.print()
        console.print("[bold]Capability Entities:[/bold]")
        for c in capabilities:
            meta = c["metadata"]
            console.print(
                f"  {c['id']} [{meta['domain_classification']}] — "
                f"{c['title']} "
                f"(patterns: {', '.join(meta['implements_patterns'])})"
            )

        console.print()
        console.print("[bold]Agent Entities:[/bold]")
        for a in agents:
            meta = a["metadata"]
            tools = meta.get('tools', [])
            tools_str = f" tools={tools}" if tools else ""
            console.print(
                f"  {a['id']} \\[{meta['agent_type']}|{meta.get('deployed_as', '')}] — "
                f"{a['title']} → {meta['exercises_capabilities']}"
                f" memory={meta.get('memory', '')}{tools_str}"
            )

        console.print()
        console.print("[bold]Integration Edges:[/bold]")
        for e in integration:
            m = e["metadata"]
            console.print(
                f"  {e['src_id']} -> {e['dst_id']} "
                f"({m['integration_pattern']}, {m['direction']})"
            )

        if design_docs:
            console.print()
            console.print("[bold]Design Doc Entities:[/bold]")
            for d in design_docs:
                meta = d["metadata"]
                concepts = ", ".join(meta.get("domain_concepts", [])[:3])
                if len(meta.get("domain_concepts", [])) > 3:
                    concepts += "..."
                console.print(
                    f"  {d['id']} [{meta['status']}] — {d['title']} "
                    f"(concepts: {concepts})"
                )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main() -> int:
    parser = argparse.ArgumentParser(
        description="Ingest architecture layer (repos, capabilities, integration edges)"
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Parse and show what would be created, without writing to DB",
    )
    parser.add_argument(
        "--skip-neo4j", action="store_true",
        help="Skip Neo4j graph materialization",
    )
    parser.add_argument(
        "--verify", action="store_true",
        help="Run test queries after ingestion",
    )
    parser.add_argument(
        "-v", "--verbose", action="store_true",
        help="Show detailed output",
    )
    args = parser.parse_args()

    console.print()
    console.print("[bold]Architecture Layer Ingestion[/bold]")
    console.print("=" * 60)

    # ------------------------------------------------------------------
    # 1. Validate source files
    # ------------------------------------------------------------------
    if not REPOS_YAML.exists():
        console.print(f"[red]Source not found: {REPOS_YAML}[/red]")
        return 1
    if not REGISTRY_YAML.exists():
        console.print(f"[red]Source not found: {REGISTRY_YAML}[/red]")
        return 1

    console.print(f"Sources: {REPOS_YAML.name}, {REGISTRY_YAML.name}")

    # ------------------------------------------------------------------
    # 2. Parse sources
    # ------------------------------------------------------------------
    console.print("Parsing sources...")

    # Load centralized registry
    registry = load_registry(REGISTRY_YAML)

    # Parse capabilities first (need repo→capability map for repo metadata)
    capabilities = load_capabilities(registry)
    repo_cap_map = build_repo_capability_map(capabilities)

    # Extract YAML comments and merge into entity metadata (#215)
    registry_comments = extract_comments(REGISTRY_YAML)
    enriched = merge_comments_into_metadata(capabilities, registry_comments)
    if enriched:
        console.print(f"  YAML comments: enriched {enriched} capabilities")

    # Parse repos and inject delivers_capabilities from registry
    repos = parse_repos(REPOS_YAML)
    for repo in repos:
        repo["metadata"]["delivers_capabilities"] = repo_cap_map.get(repo["id"], [])
        # Remove internal tracking field
        repo.pop("_raw", None)

    # Load agents (ADR-0013)
    agents = load_agents(registry)

    # Load design docs (#211)
    design_docs = load_design_docs(DESIGN_DOCS_YAML)
    design_doc_edges = build_design_doc_edges(design_docs)

    # Build edges
    delivered_by_edges = build_delivered_by_edges(capabilities)
    depends_on_edges = build_depends_on_edges(repos)
    integration_edges = load_integration_edges(registry)
    agent_impl_edges = build_agent_implements_edges(agents)
    agent_db_edges = build_agent_delivered_by_edges(agents)

    # For implements edges, check which patterns exist (need DB for live run)
    if args.dry_run:
        # In dry-run, assume seed patterns only
        seed_patterns = {"ddd", "skos", "prov-o", "dublin-core", "dam"}
        implements_edges, unregistered = build_implements_edges(
            capabilities, seed_patterns
        )
    else:
        conn = get_db_connection()
        conn.autocommit = False
        cursor = conn.cursor()
        registered = get_registered_patterns(cursor)
        conn.commit()  # clear transaction state from SELECT
        implements_edges, unregistered = build_implements_edges(
            capabilities, registered
        )

    all_edge_count = (
        len(delivered_by_edges) + len(depends_on_edges)
        + len(implements_edges) + len(integration_edges)
        + len(agent_impl_edges) + len(agent_db_edges) + len(design_doc_edges)
    )
    console.print(
        f"Parsed: {len(repos)} repos, {len(capabilities)} capabilities, "
        f"{len(agents)} agents, {len(design_docs)} design docs, {all_edge_count} edges"
    )

    # ------------------------------------------------------------------
    # 2b. Derive lifecycle_stage from edge coverage (ADR-0011)
    # ------------------------------------------------------------------
    derive_lifecycle_stages(
        capabilities, repos, implements_edges, delivered_by_edges
    )

    # Count capabilities by lifecycle stage
    from collections import Counter
    cap_stages = Counter(c["metadata"].get("lifecycle_stage", "draft") for c in capabilities)
    stage_parts = [f"{count} {stage}" for stage, count in sorted(cap_stages.items())]
    console.print(f"Capability lifecycle: {', '.join(stage_parts)}")

    repo_stages = Counter(r["metadata"].get("lifecycle_stage", "active") for r in repos)
    stage_parts = [f"{count} {stage}" for stage, count in sorted(repo_stages.items())]
    console.print(f"Repo lifecycle: {', '.join(stage_parts)}")

    # ------------------------------------------------------------------
    # 3. Dry run — just print
    # ------------------------------------------------------------------
    if args.dry_run:
        print_dry_run(
            repos, capabilities, agents,
            delivered_by_edges, implements_edges, integration_edges,
            agent_impl_edges, agent_db_edges,
            unregistered, args.verbose,
            design_docs=design_docs,
            design_doc_edges=design_doc_edges,
            depends_on_edges=depends_on_edges,
        )
        return 0

    # ------------------------------------------------------------------
    # 4. Ingest to PostgreSQL
    # ------------------------------------------------------------------
    console.print()
    console.print("Ingesting to PostgreSQL...")

    try:
        # Upsert repository entities
        repo_ok = 0
        for repo in repos:
            if upsert_entity(repo, cursor):
                repo_ok += 1
        console.print(f"  Repositories: {repo_ok}/{len(repos)}")

        # Upsert capability entities
        cap_ok = 0
        for cap in capabilities:
            if upsert_entity(cap, cursor):
                cap_ok += 1
        console.print(f"  Capabilities: {cap_ok}/{len(capabilities)}")

        # Upsert depends_on edges (#211)
        dep_ok = 0
        for edge in depends_on_edges:
            if upsert_edge(edge, cursor):
                dep_ok += 1
        console.print(f"  depends_on edges: {dep_ok}/{len(depends_on_edges)}")

        # Upsert delivered_by edges
        db_ok = 0
        for edge in delivered_by_edges:
            if upsert_edge(edge, cursor):
                db_ok += 1
        console.print(f"  delivered_by edges: {db_ok}/{len(delivered_by_edges)}")

        # Upsert implements edges
        impl_ok = 0
        for edge in implements_edges:
            if upsert_edge(edge, cursor):
                impl_ok += 1
        console.print(f"  implements edges: {impl_ok}/{len(implements_edges)}")

        if unregistered:
            console.print(
                f"  [yellow]Skipped {len(unregistered)} unregistered patterns: "
                f"{', '.join(unregistered)}[/yellow]"
            )

        # Upsert integration edges
        intg_ok = 0
        for edge in integration_edges:
            if upsert_edge(edge, cursor):
                intg_ok += 1
        console.print(f"  integration edges: {intg_ok}/{len(integration_edges)}")

        # Upsert agent entities (ADR-0013)
        agent_ok = 0
        for agent in agents:
            if upsert_entity(agent, cursor):
                agent_ok += 1
        console.print(f"  Agents: {agent_ok}/{len(agents)}")

        # Upsert agent → capability edges
        agent_impl_ok = 0
        for edge in agent_impl_edges:
            if upsert_edge(edge, cursor):
                agent_impl_ok += 1
        console.print(f"  agent implements edges: {agent_impl_ok}/{len(agent_impl_edges)}")

        # Upsert agent → repository edges
        agent_db_ok = 0
        for edge in agent_db_edges:
            if upsert_edge(edge, cursor):
                agent_db_ok += 1
        console.print(f"  agent delivered_by edges: {agent_db_ok}/{len(agent_db_edges)}")

        # Upsert design doc entities (#211)
        dd_ok = 0
        for doc in design_docs:
            if upsert_entity(doc, cursor):
                dd_ok += 1
        console.print(f"  Design docs: {dd_ok}/{len(design_docs)}")

        # Upsert design doc edges (references + covers)
        dd_edge_ok = 0
        for edge in design_doc_edges:
            if upsert_edge(edge, cursor):
                dd_edge_ok += 1
        console.print(f"  design doc edges: {dd_edge_ok}/{len(design_doc_edges)}")

        conn.commit()
        console.print("[green]PostgreSQL commit successful[/green]")

    except Exception as e:
        conn.rollback()
        console.print(f"[red]Transaction rolled back: {e}[/red]")
        return 1

    # ------------------------------------------------------------------
    # 5. Neo4j materialization
    # ------------------------------------------------------------------
    if not args.skip_neo4j:
        console.print()
        console.print("Materializing to Neo4j...")
        counts = materialize_neo4j(
            repos, capabilities,
            delivered_by_edges, implements_edges, integration_edges,
            agents=agents,
            agent_implements_edges=agent_impl_edges,
            agent_delivered_by_edges=agent_db_edges,
            design_docs=design_docs,
            design_doc_edges=design_doc_edges,
            verbose=args.verbose,
        )
        console.print(
            f"  Neo4j: {counts['nodes']} nodes, "
            f"{counts['relationships']} relationships"
        )

    # ------------------------------------------------------------------
    # 6. Verification
    # ------------------------------------------------------------------
    if args.verify:
        run_verification(conn)

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    console.print()
    total_entities = len(repos) + len(capabilities) + len(agents) + len(design_docs)
    total_edges = (
        len(delivered_by_edges) + len(depends_on_edges)
        + len(implements_edges) + len(integration_edges)
        + len(agent_impl_edges) + len(agent_db_edges) + len(design_doc_edges)
    )
    console.print(f"[bold green]Done:[/bold green] {total_entities} entities, {total_edges} edges")
    console.print()

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
