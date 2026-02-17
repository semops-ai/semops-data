"""
Research RAG Pipeline

Ephemeral knowledge stores for research meta-analysis.
Lifecycle: create → use → evaluate → archive|promote|discard
"""

from .ingest import ingest_pdf, ingest_url, ingest_sources
from .embed import embed_chunks, create_collection, delete_collection
from .query import query_rag, search_similar

__all__ = [
    "ingest_pdf",
    "ingest_url",
    "ingest_sources",
    "embed_chunks",
    "create_collection",
    "delete_collection",
    "query_rag",
    "search_similar",
]
