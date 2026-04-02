"""
Shared Neo4j graph query functions for the SemOps knowledge base.

All query functions return plain dicts. Consumers (MCP, API) handle
error handling and response formatting.

Uses Neo4j HTTP API with parameterized Cypher to avoid injection.

Used by:
    - api/mcp_server.py (MCP tools for Claude Code agents)
    - api/query.py (FastAPI endpoints)
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request


def _neo4j_url() -> str:
    return os.environ.get("NEO4J_URL", "http://localhost:7474")


def _execute_cypher(statement: str, parameters: dict | None = None) -> list[dict]:
    """Execute a parameterized Cypher statement via Neo4j HTTP API.

    Returns the raw row data from the first result set.
    """
    body = {
        "statements": [
            {
                "statement": statement,
                "parameters": parameters or {},
            }
        ]
    }

    url = f"{_neo4j_url()}/db/neo4j/tx/commit"
    req = urllib.request.Request(
        url,
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"},
        method="POST",
    )

    with urllib.request.urlopen(req, timeout=10) as resp:
        data = json.loads(resp.read())

    errors = data.get("errors", [])
    if errors:
        raise RuntimeError(f"Neo4j query error: {errors[0].get('message', errors)}")

    results = data.get("results", [{}])
    if not results:
        return []

    return [row_data["row"] for row_data in results[0].get("data", [])]


def get_neighbors(entity_id: str) -> list[dict]:
    """Get graph neighbors (incoming and outgoing) for an entity.

    Returns list of dicts with: id, label, relationship, direction, strength.
    """
    # Outgoing edges
    outgoing_rows = _execute_cypher(
        "MATCH (s {id: $id})-[r]->(t) "
        "RETURN labels(t)[0] as label, t.id as id, type(r) as rel, r.strength as strength",
        {"id": entity_id},
    )

    # Incoming edges
    incoming_rows = _execute_cypher(
        "MATCH (s)-[r]->(t {id: $id}) "
        "RETURN labels(s)[0] as label, s.id as id, type(r) as rel, r.strength as strength",
        {"id": entity_id},
    )

    neighbors = []
    for r in outgoing_rows:
        neighbors.append(
            {
                "id": r[1],
                "label": r[0] or "Unknown",
                "relationship": r[2],
                "direction": "outgoing",
                "strength": r[3],
            }
        )

    for r in incoming_rows:
        neighbors.append(
            {
                "id": r[1],
                "label": r[0] or "Unknown",
                "relationship": r[2],
                "direction": "incoming",
                "strength": r[3],
            }
        )

    return neighbors
