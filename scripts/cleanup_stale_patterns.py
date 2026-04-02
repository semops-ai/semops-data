#!/usr/bin/env python3
"""
Clean stale entries from the pattern table that aren't in pattern_v1.yaml.

Compares DB pattern IDs against the YAML registry and removes entries that
were left over from prior registration attempts.

Usage:
    python scripts/cleanup_stale_patterns.py --dry-run   # Show what would be removed
    python scripts/cleanup_stale_patterns.py              # Remove stale patterns
"""

from __future__ import annotations

import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection

PATTERN_V1_PATH = Path(__file__).parent.parent.parent / "semops-orchestrator" / "config" / "patterns" / "pattern_v1.yaml"


def load_yaml_pattern_ids(path: Path) -> set[str]:
    """Load all pattern IDs from pattern_v1.yaml."""
    with open(path) as f:
        data = yaml.safe_load(f)
    return {p["id"] for p in data.get("patterns", [])}


def main() -> int:
    dry_run = "--dry-run" in sys.argv

    if not PATTERN_V1_PATH.exists():
        print(f"ERROR: {PATTERN_V1_PATH} not found")
        return 1

    yaml_ids = load_yaml_pattern_ids(PATTERN_V1_PATH)
    print(f"Patterns in pattern_v1.yaml: {len(yaml_ids)}")

    conn = get_db_connection()
    cursor = conn.cursor()

    cursor.execute("SELECT id, preferred_label, provenance FROM pattern ORDER BY id")
    db_patterns = cursor.fetchall()
    print(f"Patterns in DB: {len(db_patterns)}")

    stale = [(pid, label, prov) for pid, label, prov in db_patterns if pid not in yaml_ids]

    if not stale:
        print("\nNo stale patterns found.")
        conn.close()
        return 0

    print(f"\nStale patterns (in DB but not in YAML): {len(stale)}")
    for pid, label, prov in stale:
        print(f"  - {pid} ({label}, {prov})")

    if dry_run:
        print("\nDRY RUN — no changes made.")
        conn.close()
        return 0

    # Check for edges referencing stale patterns before deleting
    stale_ids = [s[0] for s in stale]
    cursor.execute(
        """
        SELECT dst_id, COUNT(*) as edge_count
        FROM edge
        WHERE dst_type = 'pattern' AND dst_id = ANY(%s)
        GROUP BY dst_id
        """,
        (stale_ids,),
    )
    edge_refs = {row[0]: row[1] for row in cursor.fetchall()}

    # Check for entities with primary_pattern_id referencing stale patterns
    cursor.execute(
        """
        SELECT primary_pattern_id, COUNT(*) as entity_count
        FROM entity
        WHERE primary_pattern_id = ANY(%s)
        GROUP BY primary_pattern_id
        """,
        (stale_ids,),
    )
    entity_refs = {row[0]: row[1] for row in cursor.fetchall()}

    blocked = []
    removable = []
    for pid, label, prov in stale:
        edges = edge_refs.get(pid, 0)
        entities = entity_refs.get(pid, 0)
        if edges > 0 or entities > 0:
            blocked.append((pid, label, edges, entities))
        else:
            removable.append(pid)

    if blocked:
        print(f"\nBlocked (have references — clean edges/entities first):")
        for pid, label, edges, entities in blocked:
            print(f"  - {pid}: {edges} edges, {entities} entities")

    if removable:
        # Delete pattern_edge rows referencing stale patterns
        cursor.execute(
            "DELETE FROM pattern_edge WHERE src_id = ANY(%s) OR dst_id = ANY(%s)",
            (removable, removable),
        )
        pe_deleted = cursor.rowcount
        if pe_deleted:
            print(f"\nDeleted {pe_deleted} pattern_edge rows")

        cursor.execute("DELETE FROM pattern WHERE id = ANY(%s)", (removable,))
        deleted = cursor.rowcount
        conn.commit()
        print(f"\nRemoved {deleted} stale patterns.")
    else:
        print("\nNo patterns safe to remove (all have references).")

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
