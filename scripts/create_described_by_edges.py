#!/usr/bin/env python3
"""Create described_by edges from patterns to their concept entities.

Part of the concept-to-pattern promotion workflow (#176).
Links pattern aggregate roots to content entities that describe them,
making the relationship explicit and navigable via get_pattern().

Sources for described_by edges:
1. entity.primary_pattern_id FK (content entities directly owned by pattern)
2. entity→pattern 'documents' edges (content that documents the pattern)

Does NOT create edges for:
- entity→pattern 'related_to' edges (too loose; these are not value objects)
- Entities already linked via described_by (idempotent)
"""

import argparse
import json
from db_utils import get_db_connection


def create_described_by_edges(conn, *, pattern_id: str | None = None, dry_run: bool = False) -> int:
    """Create described_by edges for patterns from their concept entities.

    Args:
        conn: Database connection
        pattern_id: Optional specific pattern to process (all if None)
        dry_run: If True, show what would be done without writing

    Returns:
        Number of edges created
    """
    cur = conn.cursor()

    pattern_filter = ""
    params: list = []
    if pattern_id:
        pattern_filter = "AND p.id = %s"
        params.append(pattern_id)

    # Find content entities linked to patterns via primary_pattern_id FK
    # that don't already have a described_by edge
    cur.execute(f"""
        SELECT p.id AS pattern_id, e.id AS entity_id, 'primary_pattern_id' AS source
        FROM pattern p
        JOIN entity e ON e.primary_pattern_id = p.id AND e.entity_type = 'content'
        WHERE NOT EXISTS (
            SELECT 1 FROM edge ed
            WHERE ed.src_type = 'pattern' AND ed.src_id = p.id
              AND ed.dst_type = 'entity' AND ed.dst_id = e.id
              AND ed.predicate = 'described_by'
        )
        {pattern_filter}

        UNION

        -- Content entities linked via documents edges (entity→pattern)
        SELECT ed.dst_id AS pattern_id, ed.src_id AS entity_id, 'documents_edge' AS source
        FROM edge ed
        JOIN entity e ON ed.src_id = e.id AND e.entity_type = 'content'
        WHERE ed.dst_type = 'pattern' AND ed.src_type = 'entity'
          AND ed.predicate = 'documents'
          AND NOT EXISTS (
              SELECT 1 FROM edge ed2
              WHERE ed2.src_type = 'pattern' AND ed2.src_id = ed.dst_id
                AND ed2.dst_type = 'entity' AND ed2.dst_id = ed.src_id
                AND ed2.predicate = 'described_by'
          )
          {pattern_filter.replace('p.id', 'ed.dst_id') if pattern_filter else ''}

        ORDER BY pattern_id, entity_id
    """, params * 2 if params else [])

    candidates = cur.fetchall()

    if not candidates:
        print("No new described_by edges needed.")
        return 0

    print(f"Found {len(candidates)} candidate described_by edges:")
    created = 0

    for pattern_id_val, entity_id, source in candidates:
        print(f"  {pattern_id_val} --described_by--> {entity_id}  (from: {source})")

        if not dry_run:
            cur.execute("""
                INSERT INTO edge (src_type, src_id, dst_type, dst_id, predicate, strength, metadata)
                VALUES ('pattern', %s, 'entity', %s, 'described_by', 1.0, %s)
                ON CONFLICT DO NOTHING
            """, (
                pattern_id_val,
                entity_id,
                json.dumps({"created_by": "create_described_by_edges", "source": source}),
            ))
            if cur.rowcount > 0:
                created += 1

    if not dry_run:
        conn.commit()
        print(f"\nCreated {created} described_by edges")
    else:
        print(f"\n[DRY RUN] Would create {len(candidates)} described_by edges")

    return created


def main():
    parser = argparse.ArgumentParser(
        description="Create described_by edges from patterns to concept entities"
    )
    parser.add_argument(
        "--pattern", "-p",
        help="Process specific pattern by ID (default: all patterns)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be done without writing to database",
    )
    args = parser.parse_args()

    conn = get_db_connection()
    try:
        create_described_by_edges(conn, pattern_id=args.pattern, dry_run=args.dry_run)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
