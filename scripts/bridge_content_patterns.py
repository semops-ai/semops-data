#!/usr/bin/env python3
"""
Bridge content entities to the pattern layer in PostgreSQL.

Automated workflow (ADR-0012: concept-entity model):
  1. --extract: Extract concepts from detected_edges, match against patterns,
               apply orphan rules, generate concept-pattern-map.yaml
  2. --apply:   Create PostgreSQL edges from edge-classified concepts
  3. --verify:  Report on bridging results and governance impact

Concepts are entities with typed edges to patterns. Classification is
rule-based (no human review step):
  - edge:   valid relationship (default for all concepts)
  - orphan: corpus noise (single entity + strength < 0.5 + cites-only)

Promotion candidates are flagged but not auto-registered.

Usage:
    python scripts/bridge_content_patterns.py --extract
    python scripts/bridge_content_patterns.py --extract --dry-run
    python scripts/bridge_content_patterns.py --apply
    python scripts/bridge_content_patterns.py --apply --dry-run
    python scripts/bridge_content_patterns.py --verify
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from datetime import datetime, UTC
from pathlib import Path

import psycopg
import yaml
from rich.console import Console
from rich.table import Table

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection

console = Console()

DEFAULT_MAPPING_PATH = Path(__file__).parent.parent / "config" / "mappings" / "concept-pattern-map.yaml"


# ---------------------------------------------------------------------------
# Data structures
# ---------------------------------------------------------------------------

@dataclass
class ConceptInfo:
    """Aggregated info about a unique concept across all detected_edges."""
    concept_id: str
    occurrences: int = 0
    entities: list[str] = field(default_factory=list)
    predicates: list[str] = field(default_factory=list)
    max_strength: float = 0.0
    sample_rationale: str = ""


@dataclass
class MatchResult:
    """Result of matching a concept against the pattern table."""
    match_type: str  # exact | alt_label | none
    pattern_id: str | None = None
    pattern_label: str | None = None


# ---------------------------------------------------------------------------
# Extract: query entities and collect concepts
# ---------------------------------------------------------------------------

def extract_concept_entities(
    conn: psycopg.Connection,
    corpus: str | None = None,
    content_type: str | None = None,
) -> list[dict]:
    """Query content entities directly (no detected_edges required).

    Each entity's id is treated as the concept. Used for --direct mode
    where the entity *is* the concept (e.g., content_type=concept).

    Args:
        conn: Database connection
        corpus: Optional corpus filter
        content_type: Optional content_type filter

    Returns:
        List of {id, title, corpus} dicts
    """
    cursor = conn.cursor()
    query = """
        SELECT id, title, metadata
        FROM entity
        WHERE entity_type = 'content'
    """
    params: list = []
    if corpus:
        query += " AND metadata->>'corpus' = %s"
        params.append(corpus)
    if content_type:
        query += " AND metadata->>'content_type' = %s"
        params.append(content_type)
    query += " ORDER BY id"

    cursor.execute(query, params)
    results = []
    for row in cursor.fetchall():
        entity_id, title, metadata = row
        metadata = metadata or {}
        results.append({
            "id": entity_id,
            "title": title,
            "corpus": metadata.get("corpus", ""),
        })
    return results


def extract_detected_edges(
    conn: psycopg.Connection,
    corpus: str | None = None,
    content_type: str | None = None,
) -> list[dict]:
    """Query all content entities with detected_edges in metadata.

    Args:
        conn: Database connection
        corpus: Optional corpus filter (e.g., 'core_kb')
        content_type: Optional content_type filter (e.g., 'concept')

    Returns:
        List of {id, title, corpus, detected_edges} dicts
    """
    cursor = conn.cursor()
    query = """
        SELECT id, title, metadata
        FROM entity
        WHERE entity_type = 'content'
          AND metadata->'detected_edges' IS NOT NULL
          AND jsonb_array_length(metadata->'detected_edges') > 0
    """
    params: list = []
    if corpus:
        query += " AND metadata->>'corpus' = %s"
        params.append(corpus)
    if content_type:
        query += " AND metadata->>'content_type' = %s"
        params.append(content_type)
    query += " ORDER BY id"

    cursor.execute(query, params)
    results = []
    for row in cursor.fetchall():
        entity_id, title, metadata = row
        metadata = metadata or {}
        results.append({
            "id": entity_id,
            "title": title,
            "corpus": metadata.get("corpus", ""),
            "primary_concept": metadata.get("primary_concept", ""),
            "detected_edges": metadata.get("detected_edges", []),
        })
    return results


def collect_unique_concepts(entities: list[dict]) -> dict[str, ConceptInfo]:
    """Aggregate detected_edges across entities into unique concepts.

    Returns:
        {concept_id: ConceptInfo} sorted by occurrence count descending
    """
    concepts: dict[str, ConceptInfo] = {}

    for entity in entities:
        for edge in entity.get("detected_edges", []):
            concept_id = edge.get("target_concept", "").strip()
            if not concept_id:
                continue

            if concept_id not in concepts:
                concepts[concept_id] = ConceptInfo(concept_id=concept_id)

            info = concepts[concept_id]
            info.occurrences += 1
            if entity["id"] not in info.entities:
                info.entities.append(entity["id"])

            predicate = edge.get("predicate", "related_to")
            if predicate not in info.predicates:
                info.predicates.append(predicate)

            strength = edge.get("strength", 0.0)
            if strength > info.max_strength:
                info.max_strength = strength
                info.sample_rationale = edge.get("rationale", "")

    # Sort by occurrence count descending
    return dict(sorted(concepts.items(), key=lambda x: -x[1].occurrences))


def get_registered_patterns(conn: psycopg.Connection) -> dict[str, dict]:
    """Load all registered patterns with their alt_labels.

    Returns:
        {pattern_id: {preferred_label, alt_labels, provenance}}
    """
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, preferred_label, alt_labels, provenance
        FROM pattern
        ORDER BY id
    """)
    patterns = {}
    for row in cursor.fetchall():
        pattern_id, label, alt_labels, provenance = row
        patterns[pattern_id] = {
            "preferred_label": label,
            "alt_labels": alt_labels or [],
            "provenance": provenance,
        }
    return patterns


def match_concept_to_pattern(
    concept_id: str,
    patterns: dict[str, dict],
) -> MatchResult:
    """Try to match a concept to a registered pattern.

    Resolution order:
    1. Exact match: concept_id == pattern.id
    2. Alt-label match: concept_id in any pattern's alt_labels
    3. No match

    Args:
        concept_id: LLM-generated concept identifier
        patterns: Registered patterns from get_registered_patterns()

    Returns:
        MatchResult with match_type and pattern_id
    """
    # 1. Exact match
    if concept_id in patterns:
        return MatchResult(
            match_type="exact",
            pattern_id=concept_id,
            pattern_label=patterns[concept_id]["preferred_label"],
        )

    # 2. Alt-label match
    for pid, pdata in patterns.items():
        if concept_id in (pdata.get("alt_labels") or []):
            return MatchResult(
                match_type="alt_label",
                pattern_id=pid,
                pattern_label=pdata["preferred_label"],
            )

    # 3. No match
    return MatchResult(match_type="none")


def classify_concept(info: ConceptInfo, match: MatchResult) -> tuple[str, str | None]:
    """Apply orphan/edge rules and promotion detection.

    Orphan rules (all must be true):
      - Single entity reference
      - Low strength (max_strength < 0.5)
      - Only 'cites' predicate

    Promotion candidate (any true):
      - 5+ entities with max_strength > 0.8
      - extends/derived_from edges to 2+ patterns (requires match)

    Returns:
        (action, promotion) tuple
    """
    # Orphan check
    is_orphan = (
        len(info.entities) == 1
        and info.max_strength < 0.5
        and all(p == "cites" for p in info.predicates)
    )
    if is_orphan:
        return "orphan", None

    # Promotion candidate check
    promotion = None
    if len(info.entities) >= 5 and info.max_strength > 0.8:
        promotion = "candidate"
    elif (
        match.match_type != "none"
        and any(p in ("extends", "derived_from") for p in info.predicates)
    ):
        # Has strong derivation edges — flag if high strength
        if info.max_strength > 0.8:
            promotion = "candidate"

    return "edge", promotion


def generate_mapping_file(
    concepts: dict[str, ConceptInfo],
    matches: dict[str, MatchResult],
    entities: list[dict],
    output_path: Path,
    *,
    corpus: str | None = None,
    content_type: str | None = None,
) -> None:
    """Write concept-pattern-map.yaml with automated classification.

    Classification is rule-based (ADR-0012):
      - edge: valid concept-to-pattern or concept-to-concept relationship
      - orphan: corpus noise (single entity + low strength + cites-only)

    Args:
        concepts: Unique concepts with occurrence info
        matches: Match results per concept
        entities: Source entities (for context)
        output_path: Where to write the YAML
        corpus: Corpus filter used (for header)
        content_type: Content type filter used (for header)
    """
    now = datetime.now(UTC).isoformat()

    filters = {}
    if corpus:
        filters["corpus"] = corpus
    if content_type:
        filters["content_type"] = content_type

    edge_count = 0
    orphan_count = 0
    promotion_count = 0

    # Build the mapping structure
    concept_entries = {}
    for concept_id, info in concepts.items():
        match = matches[concept_id]
        action, promotion = classify_concept(info, match)

        if action == "edge":
            edge_count += 1
        else:
            orphan_count += 1
        if promotion:
            promotion_count += 1

        entry: dict = {
            "occurrences": info.occurrences,
            "max_strength": round(info.max_strength, 2),
            "predicates": info.predicates,
            "entities": info.entities,
            "sample_rationale": info.sample_rationale,
            "action": action,
            "pattern_ids": [match.pattern_id] if match.pattern_id else [],
            "match_type": match.match_type if match.match_type != "none" else "inferred",
        }
        if promotion:
            entry["promotion"] = promotion

        concept_entries[concept_id] = entry

    mapping = {
        "version": "2.0",
        "status": "generated",
        "generated_at": now,
        "entity_count": len(entities),
        "concept_count": len(concepts),
        "edge_count": edge_count,
        "orphan_count": orphan_count,
        "promotion_candidates": promotion_count,
    }
    if filters:
        mapping["filters"] = filters
    mapping["concepts"] = concept_entries

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w") as f:
        filter_parts = [f"--{k} {v}" for k, v in filters.items()]
        filter_desc = (" " + " ".join(filter_parts)) if filter_parts else ""
        f.write(f"# Concept-to-Pattern Map (ADR-0012: concept-entity model)\n")
        f.write(f"# Generated: {now} by bridge_content_patterns.py --extract{filter_desc}\n")
        f.write("#\n")
        f.write("# Classification is automated (no human review step):\n")
        f.write("#   edge:   valid concept relationship (default)\n")
        f.write("#   orphan: corpus noise (single entity + strength < 0.5 + cites-only)\n")
        f.write("#\n")
        f.write("# promotion: candidate — flagged for pattern registration review\n")
        f.write("#\n")
        f.write("# To apply: python scripts/bridge_content_patterns.py --apply\n")
        f.write("#\n\n")
        yaml.dump(mapping, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    console.print(f"\n[green]Mapping file written:[/green] {output_path}")
    console.print(f"  Concepts: {len(concepts)} ({edge_count} edge, {orphan_count} orphan, {promotion_count} promotion candidates)")
    console.print(f"  Entities: {len(entities)}")
    console.print(f"\n[bold]Next:[/bold] Run --apply to materialize edges")


def run_extract(args: argparse.Namespace) -> int:
    """Extract mode: query entities, match concepts, generate mapping."""
    console.print()
    console.print("[bold]Bridge Content→Pattern: Extract[/bold]")
    console.print("=" * 50)

    conn = get_db_connection()

    filters = []
    if args.corpus:
        filters.append(f"corpus={args.corpus}")
    if args.content_type:
        filters.append(f"content_type={args.content_type}")
    filter_desc = f" ({', '.join(filters)})" if filters else ""

    if args.direct:
        # Direct mode: entity id = concept, no detected_edges required
        console.print(f"\n[bold]Querying entities directly{filter_desc}...[/bold]")
        entities = extract_concept_entities(conn, corpus=args.corpus, content_type=args.content_type)
        console.print(f"  Found {len(entities)} entities")

        if not entities:
            console.print("[yellow]No entities found matching filters.[/yellow]")
            conn.close()
            return 0

        # Each entity is its own concept (1:1)
        concepts: dict[str, ConceptInfo] = {}
        for entity in entities:
            concepts[entity["id"]] = ConceptInfo(
                concept_id=entity["id"],
                occurrences=1,
                entities=[entity["id"]],
                predicates=["is"],
                max_strength=1.0,
                sample_rationale=f"Direct concept entity: {entity['title'][:80]}",
            )
        console.print(f"  Concepts (direct): {len(concepts)}")
    else:
        # Standard mode: extract from detected_edges
        console.print(f"\n[bold]Querying entities with detected_edges{filter_desc}...[/bold]")
        entities = extract_detected_edges(conn, corpus=args.corpus, content_type=args.content_type)
        console.print(f"  Found {len(entities)} entities with detected edges")

        if not entities:
            console.print("[yellow]No entities with detected_edges found.[/yellow]")
            conn.close()
            return 0

        # Collect unique concepts from detected_edges
        concepts = collect_unique_concepts(entities)
        console.print(f"  Unique concepts: {len(concepts)}")

        total_edges = sum(c.occurrences for c in concepts.values())
        console.print(f"  Total edge references: {total_edges}")

        # Apply min-count filter
        if args.min_count > 0:
            before = len(concepts)
            concepts = {k: v for k, v in concepts.items() if v.occurrences >= args.min_count}
            console.print(f"  Filtered to {len(concepts)} concepts with >= {args.min_count} occurrences (removed {before - len(concepts)})")

    # 3. Load registered patterns
    console.print("\n[bold]Loading registered patterns...[/bold]")
    patterns = get_registered_patterns(conn)
    console.print(f"  Registered patterns: {len(patterns)}")

    # 4. Match concepts to patterns
    console.print("\n[bold]Matching concepts to patterns...[/bold]")
    matches: dict[str, MatchResult] = {}
    for concept_id in concepts:
        matches[concept_id] = match_concept_to_pattern(concept_id, patterns)

    # 5. Print summary table
    table = Table(title="Concept Match Results")
    table.add_column("Concept", style="cyan", max_width=35)
    table.add_column("Count", justify="right")
    table.add_column("Strength", justify="right")
    table.add_column("Match", style="green")
    table.add_column("Pattern", style="magenta")

    for concept_id, info in concepts.items():
        match = matches[concept_id]
        match_display = match.match_type if match.match_type != "none" else "[red]none[/red]"
        pattern_display = match.pattern_id or "-"
        table.add_row(
            concept_id,
            str(info.occurrences),
            f"{info.max_strength:.2f}",
            match_display,
            pattern_display,
        )

    console.print()
    console.print(table)

    # 6. Generate mapping file
    if not args.dry_run:
        generate_mapping_file(
            concepts, matches, entities, args.mapping_file,
            corpus=args.corpus, content_type=args.content_type,
        )
    else:
        matched = sum(1 for m in matches.values() if m.match_type != "none")
        console.print(f"\n[yellow]DRY RUN:[/yellow] Would write mapping file to {args.mapping_file}")
        console.print(f"  {matched} auto-matched, {len(matches) - matched} need review")

    conn.close()
    return 0


# ---------------------------------------------------------------------------
# Apply: read mapping, create edges and register patterns
# ---------------------------------------------------------------------------

def load_mapping(mapping_path: Path) -> dict:
    """Load and validate the concept-pattern mapping file.

    Supports both v2.0 (edge/orphan) and legacy v1.0 (map/dismiss) formats.

    Args:
        mapping_path: Path to concept-pattern-map.yaml

    Returns:
        Parsed mapping dict

    Raises:
        SystemExit: If file is missing or invalid
    """
    if not mapping_path.exists():
        console.print(f"[red]Mapping file not found:[/red] {mapping_path}")
        console.print("Run --extract first to generate it.")
        sys.exit(1)

    with open(mapping_path) as f:
        mapping = yaml.safe_load(f)

    if not mapping or not isinstance(mapping, dict):
        console.print("[red]Invalid mapping file format[/red]")
        sys.exit(1)

    concepts = mapping.get("concepts", {})
    if not concepts:
        console.print("[yellow]No concepts in mapping file[/yellow]")
        sys.exit(1)

    version = mapping.get("version", "1.0")

    if version >= "2.0":
        # v2 schema: edge/orphan actions, pattern_ids list
        valid_actions = {"edge", "orphan"}
        for concept_id, entry in concepts.items():
            action = entry.get("action", "")
            if action not in valid_actions:
                console.print(f"[red]Invalid action '{action}' for concept '{concept_id}'[/red]")
                console.print(f"  Valid actions: {', '.join(sorted(valid_actions))}")
                sys.exit(1)
    else:
        # Legacy v1 schema: map/dismiss/register — convert on the fly
        if mapping.get("status") != "reviewed":
            console.print(f"[red]Legacy v1 mapping requires status 'reviewed', got '{mapping.get('status', 'unknown')}'[/red]")
            sys.exit(1)
        for concept_id, entry in concepts.items():
            action = entry.get("action", "")
            # Translate legacy actions
            if action == "map":
                entry["action"] = "edge"
                pid = entry.pop("pattern_id", None)
                entry["pattern_ids"] = [pid] if pid else []
            elif action == "dismiss":
                entry["action"] = "orphan"
                entry.pop("pattern_id", None)
                entry["pattern_ids"] = []
            elif action == "register":
                entry["action"] = "edge"
                reg = entry.get("register", {})
                entry["pattern_ids"] = [reg["id"]] if reg else []
        console.print("[yellow]Converted legacy v1 mapping to v2 format[/yellow]")

    return mapping


# Predicate mapping: LLM predicate → edge table predicate
PREDICATE_MAP = {
    "derived_from": "documents",
    "cites": "documents",
    "extends": "documents",
    "related_to": "related_to",
    "contradicts": "related_to",
}


def register_pattern(
    pattern_data: dict,
    cursor: psycopg.cursor.Cursor,
    dry_run: bool = False,
) -> bool:
    """Register a new pattern (roadmap or otherwise).

    Args:
        pattern_data: Dict with id, preferred_label, definition, provenance
        cursor: DB cursor
        dry_run: If True, don't write

    Returns:
        True if successful
    """
    pattern_id = pattern_data["id"]
    label = pattern_data["preferred_label"]
    definition = pattern_data["definition"]
    provenance = pattern_data.get("provenance", "1p")

    metadata = {
        "$schema": "pattern_registry_v1",
        "pattern_type": pattern_data.get("pattern_type", "concept"),
        "lifecycle_stage": pattern_data.get("lifecycle_stage", "proposed"),
        "registered_by": "bridge_content_patterns.py",
    }

    status = "[DRY]" if dry_run else "  +"
    console.print(f"  {status} register pattern: {pattern_id} ({provenance})")

    if dry_run:
        return True

    try:
        cursor.execute(
            """
            INSERT INTO pattern (id, preferred_label, definition, provenance, metadata)
            VALUES (%s, %s, %s, %s, %s)
            ON CONFLICT (id) DO UPDATE SET
                preferred_label = EXCLUDED.preferred_label,
                definition = EXCLUDED.definition,
                metadata = EXCLUDED.metadata,
                updated_at = now()
            """,
            (pattern_id, label, definition, provenance, json.dumps(metadata)),
        )
        return True
    except Exception as e:
        console.print(f"[red]Pattern register error ({pattern_id}): {e}[/red]")
        return False


def create_documents_edge(
    entity_id: str,
    pattern_id: str,
    strength: float,
    edge_metadata: dict,
    cursor: psycopg.cursor.Cursor,
    dry_run: bool = False,
) -> bool:
    """Create entity→pattern documents edge using upsert.

    Args:
        entity_id: Source content entity
        pattern_id: Target pattern
        strength: Relationship strength 0.0-1.0
        edge_metadata: Traceability metadata
        cursor: DB cursor
        dry_run: If True, don't write

    Returns:
        True if successful
    """
    if dry_run:
        return True

    predicate = edge_metadata.pop("_predicate", "documents")

    try:
        cursor.execute(
            """
            INSERT INTO edge (src_type, src_id, dst_type, dst_id, predicate, strength, metadata)
            VALUES ('entity', %s, 'pattern', %s, %s, %s, %s)
            ON CONFLICT (src_type, src_id, dst_type, dst_id, predicate) DO UPDATE SET
                strength = EXCLUDED.strength,
                metadata = EXCLUDED.metadata
            """,
            (entity_id, pattern_id, predicate, strength, json.dumps(edge_metadata)),
        )
        return True
    except Exception as e:
        console.print(f"[red]Edge error ({entity_id} → {pattern_id}): {e}[/red]")
        return False


def set_primary_pattern(
    entity_id: str,
    pattern_id: str,
    cursor: psycopg.cursor.Cursor,
    dry_run: bool = False,
) -> bool:
    """Set primary_pattern_id on entity (only if currently NULL).

    Args:
        entity_id: Entity to update
        pattern_id: Pattern to set as primary
        cursor: DB cursor
        dry_run: If True, don't write

    Returns:
        True if updated (was NULL), False if already set
    """
    if dry_run:
        return True

    try:
        cursor.execute(
            """
            UPDATE entity
            SET primary_pattern_id = %s, updated_at = now()
            WHERE id = %s AND primary_pattern_id IS NULL
            """,
            (pattern_id, entity_id),
        )
        return cursor.rowcount > 0
    except Exception as e:
        console.print(f"[red]Primary pattern error ({entity_id}): {e}[/red]")
        return False


def add_alt_label(
    pattern_id: str,
    alt_label: str,
    cursor: psycopg.cursor.Cursor,
    dry_run: bool = False,
) -> bool:
    """Add a concept as alt_label on a pattern (if not already present).

    Args:
        pattern_id: Pattern to update
        alt_label: Alt label to add
        cursor: DB cursor
        dry_run: If True, don't write

    Returns:
        True if successful
    """
    if dry_run:
        return True

    try:
        cursor.execute(
            """
            UPDATE pattern
            SET alt_labels = array_append(alt_labels, %s)
            WHERE id = %s AND NOT (%s = ANY(alt_labels))
            """,
            (alt_label, pattern_id, alt_label),
        )
        return True
    except Exception as e:
        console.print(f"[red]Alt label error ({pattern_id}): {e}[/red]")
        return False


def run_apply(args: argparse.Namespace) -> int:
    """Apply mode: create edges for edge-classified concepts."""
    console.print()
    console.print("[bold]Bridge Content→Pattern: Apply[/bold]")
    console.print("=" * 50)

    if args.dry_run:
        console.print("[yellow]DRY RUN — no database changes[/yellow]")

    # Load and validate mapping
    mapping = load_mapping(args.mapping_file)
    concepts = mapping.get("concepts", {})

    # Tally actions
    action_counts = {"edge": 0, "orphan": 0}
    promotion_count = 0
    for entry in concepts.values():
        action_counts[entry.get("action", "edge")] = action_counts.get(entry.get("action", "edge"), 0) + 1
        if entry.get("promotion"):
            promotion_count += 1

    console.print(f"\n  Concepts: {len(concepts)}")
    console.print(f"  Edge: {action_counts.get('edge', 0)}, Orphan: {action_counts.get('orphan', 0)}, Promotion candidates: {promotion_count}")

    # Load entities with detected_edges (to get per-entity edge details)
    conn = get_db_connection()
    conn.autocommit = False
    cursor = conn.cursor()

    # Build entity→edges lookup from DB
    entities = extract_detected_edges(conn)
    entity_edges: dict[str, list[dict]] = {}
    for entity in entities:
        entity_edges[entity["id"]] = entity.get("detected_edges", [])

    # Verify pattern references exist
    registered = get_registered_patterns(conn)

    now = datetime.now(UTC).isoformat()
    stats = {
        "edges_created": 0,
        "orphans_skipped": 0,
        "patterns_missing": 0,
        "errors": 0,
    }

    try:
        console.print("\n[bold]Creating edges for edge-classified concepts[/bold]")
        for concept_id, entry in concepts.items():
            if entry.get("action") == "orphan":
                stats["orphans_skipped"] += 1
                continue

            # Get pattern_ids (v2) or fall back to pattern_id (legacy)
            pattern_ids = entry.get("pattern_ids", [])
            if not pattern_ids:
                # No pattern match — concept-to-concept edge only (no DB edge to create)
                continue

            entity_ids = entry.get("entities", [])

            for pattern_id in pattern_ids:
                # Verify pattern exists
                if pattern_id not in registered:
                    console.print(f"  [red]Pattern not found: {pattern_id} (concept: {concept_id})[/red]")
                    stats["patterns_missing"] += 1
                    continue

                # Create edge for each entity that references this concept
                for entity_id in entity_ids:
                    # Find the original detected_edge for this entity+concept
                    edges_for_entity = entity_edges.get(entity_id, [])
                    original_edge = next(
                        (e for e in edges_for_entity if e.get("target_concept") == concept_id),
                        None,
                    )

                    original_predicate = (original_edge or {}).get("predicate", "related_to")
                    strength = (original_edge or {}).get("strength", 0.5)
                    mapped_predicate = PREDICATE_MAP.get(original_predicate, "documents")

                    edge_metadata = {
                        "source": "bridge_content_patterns",
                        "original_predicate": original_predicate,
                        "original_concept": concept_id,
                        "mapped_at": now,
                        "_predicate": mapped_predicate,
                    }

                    status_str = "[DRY]" if args.dry_run else "  +"
                    console.print(
                        f"  {status_str} edge: {entity_id} --{mapped_predicate}--> {pattern_id}"
                        f" (str={strength:.2f})"
                    )

                    ok = create_documents_edge(
                        entity_id, pattern_id, strength, edge_metadata,
                        cursor, dry_run=args.dry_run,
                    )
                    if ok:
                        stats["edges_created"] += 1
                    else:
                        stats["errors"] += 1

        # Commit
        if not args.dry_run:
            conn.commit()
            console.print("\n[green]Transaction committed.[/green]")
        else:
            conn.rollback()

    except Exception as e:
        conn.rollback()
        console.print(f"\n[red]Error: {e}[/red]")
        import traceback
        traceback.print_exc()
        return 1
    finally:
        conn.close()

    # Summary
    console.print()
    table = Table(title="Apply Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Count", justify="right")
    table.add_row("Edges created", str(stats["edges_created"]))
    table.add_row("Orphans skipped", str(stats["orphans_skipped"]))
    table.add_row("Patterns missing", str(stats["patterns_missing"]))
    table.add_row("Errors", str(stats["errors"]))
    console.print(table)

    if args.dry_run:
        console.print("\n[yellow]DRY RUN — no changes made[/yellow]")

    return 0 if stats["errors"] == 0 else 1


# ---------------------------------------------------------------------------
# Verify: report on bridging results
# ---------------------------------------------------------------------------

def run_verify(args: argparse.Namespace) -> int:
    """Verify mode: report on bridging results and governance impact."""
    console.print()
    console.print("[bold]Bridge Content→Pattern: Verify[/bold]")
    console.print("=" * 50)

    conn = get_db_connection()
    cursor = conn.cursor()

    # 1. Documents edges created by this script
    console.print("\n[bold]Documents edges (bridge_content_patterns)[/bold]")
    cursor.execute("""
        SELECT
            e.src_id AS entity_id,
            e.dst_id AS pattern_id,
            e.predicate,
            e.strength,
            p.preferred_label
        FROM edge e
        JOIN pattern p ON e.dst_id = p.id
        WHERE e.src_type = 'entity'
          AND e.dst_type = 'pattern'
          AND e.predicate IN ('documents', 'related_to')
          AND e.metadata->>'source' = 'bridge_content_patterns'
        ORDER BY e.dst_id, e.src_id
    """)
    bridge_edges = cursor.fetchall()

    if bridge_edges:
        table = Table(title=f"Bridge Edges ({len(bridge_edges)} total)")
        table.add_column("Entity", style="cyan", max_width=35)
        table.add_column("Predicate")
        table.add_column("Pattern", style="magenta")
        table.add_column("Label")
        table.add_column("Strength", justify="right")
        for row in bridge_edges:
            table.add_row(row[0], row[2], row[1], row[4], f"{row[3]:.2f}")
        console.print(table)
    else:
        console.print("  [yellow]No bridge edges found. Run --apply first.[/yellow]")

    # 2. Pattern coverage with edge-based content
    console.print("\n[bold]Pattern coverage (content via FK + documents edges)[/bold]")
    cursor.execute("""
        SELECT
            p.id,
            p.preferred_label,
            p.provenance,
            COUNT(DISTINCT fk.id) AS fk_content,
            COUNT(DISTINCT ed.src_id) AS edge_content,
            COUNT(DISTINCT COALESCE(fk.id, ed.src_id)) AS total_content
        FROM pattern p
        LEFT JOIN entity fk ON fk.primary_pattern_id = p.id AND fk.entity_type = 'content'
        LEFT JOIN edge ed ON ed.dst_type = 'pattern' AND ed.dst_id = p.id
            AND ed.src_type = 'entity' AND ed.predicate IN ('documents', 'related_to')
        GROUP BY p.id, p.preferred_label, p.provenance
        HAVING COUNT(DISTINCT fk.id) > 0 OR COUNT(DISTINCT ed.src_id) > 0
        ORDER BY COUNT(DISTINCT COALESCE(fk.id, ed.src_id)) DESC
    """)
    rows = cursor.fetchall()

    if rows:
        table = Table(title="Pattern Coverage (content entities)")
        table.add_column("Pattern", style="cyan")
        table.add_column("Label")
        table.add_column("FK", justify="right", style="dim")
        table.add_column("Edge", justify="right", style="green")
        table.add_column("Total", justify="right", style="bold")
        for row in rows:
            table.add_row(row[0], row[1], str(row[3]), str(row[4]), str(row[5]))
        console.print(table)
    else:
        console.print("  [yellow]No patterns with content coverage[/yellow]")

    # 3. Remaining orphans (content with no pattern link)
    console.print("\n[bold]Remaining orphan content entities[/bold]")
    cursor.execute("""
        SELECT e.id, e.title, e.metadata->>'corpus' AS corpus
        FROM entity e
        WHERE e.entity_type = 'content'
          AND e.primary_pattern_id IS NULL
          AND NOT EXISTS (
              SELECT 1 FROM edge ed
              WHERE ed.src_type = 'entity' AND ed.src_id = e.id
                AND ed.dst_type = 'pattern'
                AND ed.predicate IN ('documents', 'related_to')
          )
        ORDER BY e.id
    """)
    orphans = cursor.fetchall()
    console.print(f"  Orphan entities: {len(orphans)}")
    if orphans and args.verbose:
        for row in orphans:
            console.print(f"    {row[0]} | {row[1]} | {row[2]}")

    # 4. Summary counts
    console.print("\n[bold]Summary[/bold]")
    cursor.execute("SELECT count(*) FROM edge WHERE metadata->>'source' = 'bridge_content_patterns'")
    console.print(f"  Bridge edges: {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT count(*) FROM pattern
        WHERE metadata->>'registered_by' = 'bridge_content_patterns.py'
    """)
    console.print(f"  Registered patterns: {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT count(*) FROM entity
        WHERE entity_type = 'content' AND primary_pattern_id IS NOT NULL
    """)
    console.print(f"  Content with primary_pattern_id: {cursor.fetchone()[0]}")

    cursor.execute("""
        SELECT count(*) FROM entity
        WHERE entity_type = 'content' AND primary_pattern_id IS NULL
    """)
    console.print(f"  Content without primary_pattern_id: {cursor.fetchone()[0]}")

    conn.close()
    console.print()
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    parser = argparse.ArgumentParser(
        description="Bridge content entities to pattern layer (automated, ADR-0012)"
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--extract", action="store_true",
                       help="Extract concepts and generate mapping YAML")
    group.add_argument("--apply", action="store_true",
                       help="Apply reviewed mapping (create edges, register patterns)")
    group.add_argument("--verify", action="store_true",
                       help="Report on bridging results")

    parser.add_argument("--dry-run", action="store_true",
                        help="Show what would be done without writing")
    parser.add_argument("--mapping-file", type=Path, default=DEFAULT_MAPPING_PATH,
                        help=f"Mapping file path (default: {DEFAULT_MAPPING_PATH})")
    parser.add_argument("--direct", action="store_true",
                        help="Direct mode: treat entity id as concept (no detected_edges required)")
    parser.add_argument("--corpus", type=str, default=None,
                        help="Filter entities by corpus (e.g., 'core_kb')")
    parser.add_argument("--content-type", type=str, default=None,
                        help="Filter entities by content_type (e.g., 'concept')")
    parser.add_argument("--min-count", type=int, default=0,
                        help="Minimum occurrence count to include in mapping (default: 0 = all)")
    parser.add_argument("-v", "--verbose", action="store_true",
                        help="Show detailed output")

    args = parser.parse_args()

    if args.extract:
        return run_extract(args)
    elif args.apply:
        return run_apply(args)
    elif args.verify:
        return run_verify(args)

    return 0


if __name__ == "__main__":
    sys.exit(main())
