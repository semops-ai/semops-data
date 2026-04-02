#!/usr/bin/env python3
"""
Cross-corpus concept synthesis document generator.

Searches all corpora in the SemOps knowledge base for a concept query,
groups results by corpus role, pulls pattern/capability graph edges,
and generates a structured markdown synthesis document.

Usage:
    python scripts/synthesize_concept.py "explicit enterprise"
    python scripts/synthesize_concept.py "scale projection" --output docs/synthesis-scale-projection.md
    python scripts/synthesize_concept.py "semantic coherence" --similarity-threshold 0.5
"""

from __future__ import annotations

import argparse
import sys
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from textwrap import dedent

from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection
from schema_queries import get_pattern, list_capabilities, search_patterns
from search import (
    EMBEDDING_DIMENSIONS,
    EMBEDDING_MODEL,
    list_corpora,
    search_chunks,
    search_entities,
)

# Corpus role labels for document sections
CORPUS_ROLES = {
    "core_kb": ("Authoritative Definitions", "Primary concept documents, framework pillars, patterns"),
    "deployment": ("Design History & Operations", "Session notes, issues, ADRs, operational context"),
    "published": ("Published State", "Public-facing articles and documentation"),
    "research_ai": ("Research", "Research papers and AI analysis"),
}


def get_openai_client() -> OpenAI:
    import os
    from db_utils import load_env
    load_env()
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise ValueError("OPENAI_API_KEY not set in environment or .env")
    return OpenAI(api_key=api_key)


def generate_embedding(client: OpenAI, text: str) -> list[float]:
    response = client.embeddings.create(
        model=EMBEDDING_MODEL,
        input=text,
        dimensions=EMBEDDING_DIMENSIONS,
    )
    return response.data[0].embedding


def get_capability_edges(conn, entity_ids: set[str]) -> list[dict]:
    """Find capabilities that reference any of the discovered entity IDs."""
    if not entity_ids:
        return []
    cursor = conn.cursor()
    placeholders = ", ".join(["%s"] * len(entity_ids))
    cursor.execute(
        f"""
        SELECT DISTINCT
            e.src_id, cap.title AS capability_name,
            e.dst_id, e.dst_type, e.predicate, e.strength
        FROM edge e
        JOIN entity cap ON e.src_id = cap.id AND cap.entity_type = 'capability'
        WHERE e.dst_id IN ({placeholders})
           OR e.src_id IN ({placeholders})
        ORDER BY cap.title, e.predicate
        """,
        list(entity_ids) + list(entity_ids),
    )
    return [
        {
            "src_id": r[0],
            "capability_name": r[1],
            "dst_id": r[2],
            "dst_type": r[3],
            "predicate": r[4],
            "strength": float(r[5]) if r[5] is not None else 1.0,
        }
        for r in cursor.fetchall()
    ]


def get_entity_edges(conn, entity_id: str) -> list[dict]:
    """Get all edges for a specific entity (both directions)."""
    cursor = conn.cursor()
    cursor.execute(
        """
        SELECT
            e.src_id, e.src_type, e.dst_id, e.dst_type, e.predicate,
            e.strength, e.metadata
        FROM edge e
        WHERE e.src_id = %s OR e.dst_id = %s
        ORDER BY e.predicate
        """,
        [entity_id, entity_id],
    )
    return [
        {
            "src_id": r[0],
            "src_type": r[1],
            "dst_id": r[2],
            "dst_type": r[3],
            "predicate": r[4],
            "strength": float(r[5]) if r[5] is not None else 1.0,
            "metadata": r[6],
        }
        for r in cursor.fetchall()
    ]


def dedup_chunks(chunks: list[dict], threshold: float = 0.95) -> list[dict]:
    """Remove near-duplicate chunks (same content appearing in multiple corpora)."""
    seen_content: list[str] = []
    deduped = []
    for chunk in chunks:
        content = chunk.get("content", "")[:200]
        is_dup = False
        for seen in seen_content:
            # Simple overlap check
            if content and seen and content[:100] == seen[:100]:
                is_dup = True
                break
        if not is_dup:
            deduped.append(chunk)
            seen_content.append(content)
    return deduped


def group_by_corpus(items: list[dict]) -> dict[str, list[dict]]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for item in items:
        corpus = item.get("corpus", "unknown")
        groups[corpus].append(item)
    return dict(groups)


def format_entity_table(entities: list[dict]) -> str:
    lines = ["| Entity | Type | Similarity | Source |", "|--------|------|------------|--------|"]
    for e in entities:
        uri = e.get("uri", "") or ""
        source = uri.split("/")[-1] if uri else "-"
        lines.append(
            f"| `{e['id']}` | {e.get('content_type', '-')} | {e['similarity']:.3f} | `{source}` |"
        )
    return "\n".join(lines)


def format_chunk_section(chunks: list[dict], max_chunks: int = 5) -> str:
    lines = []
    for chunk in chunks[:max_chunks]:
        heading = " > ".join(chunk.get("heading_hierarchy", [])[-2:]) or "(root)"
        content = (chunk.get("content", "") or "")[:300]
        if content:
            content = content.replace("\n", " ").strip()
        lines.append(f"**{heading}** (sim: {chunk['similarity']:.3f})")
        if content:
            lines.append(f"> {content}")
        lines.append("")
    if len(chunks) > max_chunks:
        lines.append(f"*... and {len(chunks) - max_chunks} more chunks*")
    return "\n".join(lines)


def format_pattern_section(pattern: dict) -> str:
    lines = []
    lines.append(f"**{pattern['preferred_label']}** (`{pattern['id']}`)")
    lines.append(f"- Provenance: {pattern.get('provenance', '-')}")
    if pattern.get("definition"):
        lines.append(f"- Definition: {pattern['definition'][:200]}")
    cov = pattern.get("coverage", {})
    if cov:
        lines.append(
            f"- Coverage: {cov.get('content_count', 0)} content, "
            f"{cov.get('capability_count', 0)} capabilities, "
            f"{cov.get('repo_count', 0)} repos"
        )
    edges = pattern.get("edges", [])
    if edges:
        lines.append("")
        lines.append("SKOS/Adoption edges:")
        lines.append("")
        for edge in edges:
            direction = "→" if edge["src_id"] == pattern["id"] else "←"
            lines.append(
                f"- `{pattern['id']}` {direction} `{edge['predicate']}` "
                f"{direction} `{edge['related_label']}`"
            )
    return "\n".join(lines)


def format_timeline(entities: list[dict]) -> str:
    """Extract dated items and format as a timeline."""
    dated = []
    for e in entities:
        date = e.get("date_created")
        if date:
            dated.append((date, e))
    if not dated:
        return ""
    dated.sort(key=lambda x: x[0])
    lines = ["| Date | Entity | Title |", "|------|--------|-------|"]
    for date, e in dated:
        title = (e.get("title") or "-")[:60]
        lines.append(f"| {date} | `{e['id']}` | {title} |")
    return "\n".join(lines)


def generate_synthesis(
    query: str,
    entities: list[dict],
    chunks: list[dict],
    pattern: dict | None,
    pattern_matches: list[dict],
    entity_edges: list[dict],
    corpora: list[dict],
    similarity_threshold: float,
) -> str:
    """Generate the synthesis markdown document."""
    # Filter by threshold
    entities = [e for e in entities if e["similarity"] >= similarity_threshold]
    chunks = [c for c in chunks if c["similarity"] >= similarity_threshold]

    # Group by corpus
    entity_groups = group_by_corpus(entities)
    chunk_groups = group_by_corpus(chunks)

    # Deduplicate chunks
    chunks = dedup_chunks(chunks)

    # Get primary entity (highest similarity)
    primary = entities[0] if entities else None

    sections = []

    # Header
    title = primary["title"] if primary else query.title()
    sections.append(f"# {title} — Cross-Corpus Synthesis")
    sections.append("")
    sections.append(f"> Generated: {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    sections.append(f"> Query: \"{query}\"")
    sections.append(
        f"> Threshold: {similarity_threshold:.2f} | "
        f"Entities: {len(entities)} | Chunks: {len(chunks)} | "
        f"Corpora: {len(entity_groups)}/{len(corpora)}"
    )
    sections.append("")
    sections.append("---")
    sections.append("")

    # Definition (from primary entity summary or top chunk)
    if primary:
        sections.append("## Definition")
        sections.append("")
        if primary.get("summary"):
            sections.append(f"> {primary['summary']}")
        else:
            # Use the top chunk from the primary entity
            primary_chunks = [c for c in chunks if c.get("entity_id") == primary["id"]]
            if primary_chunks:
                content = (primary_chunks[0].get("content", "") or "")[:500]
                sections.append(f"> {content}")
            else:
                sections.append(f"*Primary entity: `{primary['id']}` ({primary.get('corpus', '-')})*")
        sections.append("")

    # Pattern layer (if matched)
    if pattern:
        sections.append("## Pattern Layer")
        sections.append("")
        sections.append(format_pattern_section(pattern))
        sections.append("")
    elif pattern_matches:
        sections.append("## Related Patterns")
        sections.append("")
        for pm in pattern_matches[:5]:
            sections.append(
                f"- **{pm['preferred_label']}** (`{pm['id']}`, "
                f"{pm.get('provenance', '-')}) — sim: {pm['similarity']:.3f}"
            )
        sections.append("")

    # Per-corpus sections
    for corpus_id, (section_title, section_desc) in CORPUS_ROLES.items():
        corpus_entities = entity_groups.get(corpus_id, [])
        corpus_chunks = chunk_groups.get(corpus_id, [])
        if not corpus_entities and not corpus_chunks:
            continue

        sections.append(f"## {section_title}")
        sections.append(f"*{section_desc}*")
        sections.append("")

        if corpus_entities:
            sections.append("### Entities")
            sections.append("")
            sections.append(format_entity_table(corpus_entities))
            sections.append("")

        if corpus_chunks:
            deduped_corpus_chunks = dedup_chunks(corpus_chunks)
            sections.append("### Key Passages")
            sections.append("")
            sections.append(format_chunk_section(deduped_corpus_chunks))
            sections.append("")

    # Timeline
    timeline = format_timeline(entities)
    if timeline:
        sections.append("## Timeline")
        sections.append("")
        sections.append(timeline)
        sections.append("")

    # Graph edges
    if entity_edges:
        sections.append("## Graph Edges")
        sections.append("")
        edge_by_predicate: dict[str, list] = defaultdict(list)
        for edge in entity_edges:
            edge_by_predicate[edge["predicate"]].append(edge)
        for predicate, edges in sorted(edge_by_predicate.items()):
            sections.append(f"### {predicate}")
            sections.append("")
            for edge in edges:
                if edge["src_id"] == (primary["id"] if primary else ""):
                    sections.append(f"- → `{edge['dst_id']}` ({edge['dst_type']})")
                else:
                    sections.append(f"- ← `{edge['src_id']}` ({edge['src_type']})")
            sections.append("")

    # Corpus coverage summary
    sections.append("## Corpus Coverage")
    sections.append("")
    sections.append("| Corpus | Entities | Chunks | Role |")
    sections.append("|--------|----------|--------|------|")
    for c in corpora:
        corpus_id = c["corpus"]
        role = CORPUS_ROLES.get(corpus_id, ("Unknown", ""))[0]
        e_count = len(entity_groups.get(corpus_id, []))
        c_count = len(chunk_groups.get(corpus_id, []))
        marker = " *" if e_count == 0 and c_count == 0 else ""
        sections.append(f"| {corpus_id} | {e_count} | {c_count} | {role}{marker} |")
    sections.append("")

    return "\n".join(sections)


def main():
    parser = argparse.ArgumentParser(
        description="Generate a cross-corpus synthesis document for a concept",
    )
    parser.add_argument("query", type=str, help="Concept to synthesize")
    parser.add_argument(
        "--output", "-o",
        type=str,
        help="Output file path (default: stdout)",
    )
    parser.add_argument(
        "--similarity-threshold",
        type=float,
        default=0.35,
        help="Minimum similarity score to include (default: 0.35)",
    )
    parser.add_argument(
        "--entity-limit",
        type=int,
        default=20,
        help="Max entities per search (default: 20)",
    )
    parser.add_argument(
        "--chunk-limit",
        type=int,
        default=25,
        help="Max chunks per search (default: 25)",
    )
    args = parser.parse_args()

    print(f"Synthesizing: \"{args.query}\"", file=sys.stderr)

    openai_client = get_openai_client()
    conn = get_db_connection()

    # Generate embedding once
    print("  Generating embedding...", file=sys.stderr)
    query_embedding = generate_embedding(openai_client, args.query)

    # Search all corpora
    print("  Searching entities...", file=sys.stderr)
    entities = search_entities(
        conn, query_embedding, limit=args.entity_limit,
    )

    print("  Searching chunks...", file=sys.stderr)
    chunks = search_chunks(
        conn, query_embedding, limit=args.chunk_limit, content_max_chars=500,
    )

    # Search patterns
    print("  Searching patterns...", file=sys.stderr)
    pattern_matches = search_patterns(conn, query_embedding, limit=5)

    # Try exact pattern match from top entity ID
    pattern = None
    if entities:
        pattern = get_pattern(conn, entities[0]["id"])
    if not pattern and pattern_matches:
        pattern = get_pattern(conn, pattern_matches[0]["id"])

    # Get graph edges for primary entity
    entity_edges = []
    if entities:
        entity_edges = get_entity_edges(conn, entities[0]["id"])

    # List corpora for coverage summary
    corpora = list_corpora(conn)

    print("  Generating document...", file=sys.stderr)
    doc = generate_synthesis(
        query=args.query,
        entities=entities,
        chunks=chunks,
        pattern=pattern,
        pattern_matches=pattern_matches,
        entity_edges=entity_edges,
        corpora=corpora,
        similarity_threshold=args.similarity_threshold,
    )

    if args.output:
        Path(args.output).write_text(doc)
        print(f"  Written to {args.output}", file=sys.stderr)
    else:
        print(doc)

    conn.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
