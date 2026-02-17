"""
CLI for Research RAG Pipeline

Usage:
    python -m data_systems_toolkit.research.cli ingest
    python -m data_systems_toolkit.research.cli query "What causes AI transformation to fail?"
    python -m data_systems_toolkit.research.cli search "measurement ROI"
    python -m data_systems_toolkit.research.cli sources
    python -m data_systems_toolkit.research.cli status
    python -m data_systems_toolkit.research.cli cleanup
"""
import argparse
import json
import sys

from .config import config
from .ingest import load_manifest, ingest_sources
from .embed import create_collection, delete_collection, get_collection_info, embed_chunks
from .query import query_rag, search_similar, list_sources


def cmd_status(args):
    """Show status of the research pipeline."""
    print(f"Research RAG Pipeline Status")
    print(f"=" * 40)
    print(f"Collection: {config.collection_name}")
    print(f"Qdrant URL: {config.qdrant_url}")
    print(f"Docling URL: {config.docling_url}")
    print()

    # Check collection
    info = get_collection_info()
    if info:
        print(f"Collection exists: Yes")
        print(f"Points count: {info.points_count}")
    else:
        print(f"Collection exists: No")

    # List sources in manifest
    sources = load_manifest()
    print(f"\nSources in manifest: {len(sources)}")
    for s in sources:
        print(f"  - {s.title} ({s.source_type})")


def cmd_ingest(args):
    """Ingest sources from manifest."""
    # Load sources
    sources = load_manifest()
    if not sources:
        print("No sources found in manifest.json")
        return

    print(f"Ingesting {len(sources)} sources...")

    # Filter by type if specified
    if args.type:
        sources = [s for s in sources if s.source_type == args.type]
        print(f"Filtered to {len(sources)} {args.type} sources")

    # Create/recreate collection if requested
    if args.recreate:
        create_collection(recreate=True)
    else:
        create_collection(recreate=False)

    # Ingest
    chunks = ingest_sources(sources)

    if chunks:
        print(f"\nEmbedding {len(chunks)} chunks...")
        embed_chunks(chunks)
        print("Done!")
    else:
        print("No chunks to embed.")


def cmd_query(args):
    """Query the RAG pipeline."""
    result = query_rag(
        question=args.question,
        limit=args.limit,
    )

    print("\n" + "=" * 60)
    print("ANSWER:")
    print("=" * 60)
    print(result["answer"])

    print("\n" + "-" * 60)
    print("SOURCES USED:")
    print("-" * 60)
    for i, source in enumerate(result["sources"], 1):
        print(f"[{i}] {source.source_title} (score: {source.score:.3f})")
        print(f"    {source.source_url}")


def cmd_search(args):
    """Search for similar chunks."""
    results = search_similar(
        query=args.query,
        limit=args.limit,
    )

    print(f"\nFound {len(results)} results for: '{args.query}'")
    print("=" * 60)

    for i, result in enumerate(results, 1):
        print(f"\n[{i}] {result.source_title} (score: {result.score:.3f})")
        print(f"URL: {result.source_url}")
        print("-" * 40)
        # Truncate long text
        text = result.text
        if len(text) > 500:
            text = text[:500] + "..."
        print(text)


def cmd_sources(args):
    """List sources in the collection."""
    sources = list_sources()

    if not sources:
        print("No sources found in collection.")
        return

    print(f"\nSources in collection: {len(sources)}")
    print("=" * 60)

    for source in sources:
        print(f"\n{source['title']}")
        print(f"  ID: {source['source_id']}")
        print(f"  URL: {source['url']}")
        if source.get('authors'):
            print(f"  Authors: {', '.join(source['authors'])}")
        if source.get('year'):
            print(f"  Year: {source['year']}")


def cmd_cleanup(args):
    """Delete the collection (cleanup)."""
    if not args.force:
        confirm = input(f"Delete collection '{config.collection_name}'? [y/N] ")
        if confirm.lower() != 'y':
            print("Aborted.")
            return

    delete_collection()


def main():
    parser = argparse.ArgumentParser(description="Research RAG Pipeline CLI")
    subparsers = parser.add_subparsers(dest="command", help="Commands")

    # status
    parser_status = subparsers.add_parser("status", help="Show pipeline status")
    parser_status.set_defaults(func=cmd_status)

    # ingest
    parser_ingest = subparsers.add_parser("ingest", help="Ingest sources from manifest")
    parser_ingest.add_argument("--type", choices=["pdf", "web"], help="Only ingest specific type")
    parser_ingest.add_argument("--recreate", action="store_true", help="Recreate collection")
    parser_ingest.set_defaults(func=cmd_ingest)

    # query
    parser_query = subparsers.add_parser("query", help="Query the RAG pipeline")
    parser_query.add_argument("question", help="Question to ask")
    parser_query.add_argument("--limit", type=int, default=5, help="Number of sources to use")
    parser_query.set_defaults(func=cmd_query)

    # search
    parser_search = subparsers.add_parser("search", help="Search for similar chunks")
    parser_search.add_argument("query", help="Search query")
    parser_search.add_argument("--limit", type=int, default=10, help="Number of results")
    parser_search.set_defaults(func=cmd_search)

    # sources
    parser_sources = subparsers.add_parser("sources", help="List sources in collection")
    parser_sources.set_defaults(func=cmd_sources)

    # cleanup
    parser_cleanup = subparsers.add_parser("cleanup", help="Delete the collection")
    parser_cleanup.add_argument("--force", action="store_true", help="Skip confirmation")
    parser_cleanup.set_defaults(func=cmd_cleanup)

    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        return

    args.func(args)


if __name__ == "__main__":
    main()
