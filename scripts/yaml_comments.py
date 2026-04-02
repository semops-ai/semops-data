#!/usr/bin/env python3
"""
YAML comment extraction utility.

Standard YAML parsers strip comments. This module pre-processes YAML files
to extract inline and block comments as a parallel data structure that can
be merged into entity metadata during ingestion.

Usage:
    from yaml_comments import extract_comments

    comments = extract_comments("config/registry.yaml")
    # Returns: {"data-system-mix": "data-due-diligence decomposed per #208 ..."}
"""

from __future__ import annotations

import re
from pathlib import Path


def extract_comments(yaml_path: str | Path) -> dict[str, str]:
    """Extract comments from a YAML file, keyed by the next entity ID.

    Strategy: accumulate consecutive comment lines, then associate them with
    the next ``id:`` value found. This is simple and robust for registry-style
    YAML where comments describe the following list entry.

    Returns a dict mapping entity IDs to their associated comment text.
    Comment text is stripped of leading ``# `` prefixes and joined with
    newlines for multi-line blocks.
    """
    path = Path(yaml_path)
    if not path.exists():
        return {}

    lines = path.read_text().splitlines()
    comments: dict[str, str] = {}

    pending_comment_lines: list[str] = []

    for line in lines:
        stripped = line.strip()

        # Accumulate comment lines
        if stripped.startswith("#"):
            text = re.sub(r"^#\s?", "", stripped)
            pending_comment_lines.append(text)
            continue

        # Skip blank lines (preserve pending comments across blanks)
        if not stripped:
            continue

        # Check for id: value — this anchors comments to an entity
        id_match = re.match(r"^\s*(?:-\s+)?id:\s*(.+)", line)
        if id_match and pending_comment_lines:
            entity_id = id_match.group(1).strip()
            comments[entity_id] = "\n".join(pending_comment_lines)
            pending_comment_lines.clear()
            continue

        # Check for a top-level key that should anchor section comments
        # (e.g., "agents:" with a comment above it)
        key_match = re.match(r"^(\w[\w-]*):", line)
        if key_match and pending_comment_lines:
            key_name = key_match.group(1)
            comments[f"_section:{key_name}"] = "\n".join(pending_comment_lines)
            pending_comment_lines.clear()
            continue

        # Any other content line — if it has an inline comment, capture it
        inline_match = re.search(r"\s+#\s+(.+)$", line)
        if inline_match:
            # Try to find an id on this line
            inline_id = re.match(r"^\s*(?:-\s+)?id:\s*(\S+)", line)
            if inline_id:
                entity_id = inline_id.group(1)
                comments.setdefault(entity_id, "")
                if comments[entity_id]:
                    comments[entity_id] += "\n"
                comments[entity_id] += inline_match.group(1)

        # Non-id content line — discard pending comments (they were
        # associated with removed/commented-out content, not the next entry)
        pending_comment_lines.clear()

    return comments


def merge_comments_into_metadata(entities: list[dict], comments: dict[str, str]) -> int:
    """Merge extracted YAML comments into entity metadata JSONB.

    For each entity, if a matching comment exists in the comments dict
    (keyed by entity ID), adds it as ``metadata["yaml_comment"]``.
    Returns the count of entities enriched.
    """
    enriched = 0
    for entity in entities:
        entity_id = entity.get("id", "")
        comment = comments.get(entity_id)
        if comment:
            meta = entity.get("metadata", {})
            meta["yaml_comment"] = comment
            entity["metadata"] = meta
            enriched += 1
    return enriched
