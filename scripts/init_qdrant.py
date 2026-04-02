#!/usr/bin/env python3
"""
Initialize Qdrant collections for the multi-corpus strategy (ADR-0005).

Creates one collection per corpus with consistent vector configuration
(1536-dim, cosine distance) matching the pgvector embedding setup.

Collections:
    core_kb          — DDD core: theory, patterns, ADRs, architecture
    deployment       — Infrastructure: ADRs, session notes, architecture docs
    published        — Blog posts, public docs
    research_ai      — AI/ML research and experiments
    research_general — General research and explorations

Usage:
    python scripts/init_qdrant.py              # Create missing collections
    python scripts/init_qdrant.py --recreate   # Drop and recreate all collections
    python scripts/init_qdrant.py --status     # Show collection status only
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams

# Reuse embedding config from shared search module
sys.path.insert(0, str(Path(__file__).parent))
from search import EMBEDDING_DIMENSIONS

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

QDRANT_HOST = os.environ.get("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.environ.get("QDRANT_PORT", "6333"))

COLLECTIONS = {
    "core_kb": "DDD core: theory, patterns, ADRs, architecture",
    "deployment": "Infrastructure: ADRs, session notes, architecture docs",
    "published": "Blog posts, public docs",
    "research_ai": "AI/ML research and experiments",
    "research_general": "General research and explorations",
}

VECTOR_PARAMS = VectorParams(
    size=EMBEDDING_DIMENSIONS,
    distance=Distance.COSINE,
)


# ---------------------------------------------------------------------------
# Qdrant connection
# ---------------------------------------------------------------------------


def get_qdrant_client() -> QdrantClient:
    """Get a Qdrant client using env vars or defaults."""
    api_key = os.environ.get("QDRANT_API_KEY")
    return QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, api_key=api_key)


# ---------------------------------------------------------------------------
# Operations
# ---------------------------------------------------------------------------


def show_status(client: QdrantClient) -> None:
    """Print status of all expected collections."""
    existing = {c.name for c in client.get_collections().collections}
    print(f"Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
    print(f"Collections: {len(existing)} found\n")

    for name, description in COLLECTIONS.items():
        if name in existing:
            info = client.get_collection(name)
            count = info.points_count
            print(f"  {name:20s}  {count:>6} points  {description}")
        else:
            print(f"  {name:20s}  MISSING           {description}")

    extra = existing - set(COLLECTIONS)
    if extra:
        print(f"\nUnexpected collections: {', '.join(sorted(extra))}")


def create_collections(client: QdrantClient, *, recreate: bool = False) -> None:
    """Create Qdrant collections for each corpus."""
    existing = {c.name for c in client.get_collections().collections}

    for name, description in COLLECTIONS.items():
        if name in existing and not recreate:
            print(f"  {name:20s}  exists (skipped)")
            continue

        if name in existing and recreate:
            client.delete_collection(name)
            print(f"  {name:20s}  deleted")

        client.create_collection(
            collection_name=name,
            vectors_config=VECTOR_PARAMS,
        )
        print(f"  {name:20s}  created  ({EMBEDDING_DIMENSIONS}d, cosine)")

    print("\nDone.")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Initialize Qdrant collections for multi-corpus strategy"
    )
    parser.add_argument(
        "--recreate",
        action="store_true",
        help="Drop and recreate all collections (destroys data)",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="Show collection status only, don't create anything",
    )
    args = parser.parse_args()

    try:
        client = get_qdrant_client()
        # Quick connectivity check
        client.get_collections()
    except Exception as e:
        print(f"Failed to connect to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}: {e}", file=sys.stderr)
        sys.exit(1)

    if args.status:
        show_status(client)
        return

    if args.recreate:
        print("Recreating all Qdrant collections...\n")
    else:
        print("Initializing Qdrant collections...\n")

    create_collections(client, recreate=args.recreate)
    print()
    show_status(client)


if __name__ == "__main__":
    main()
