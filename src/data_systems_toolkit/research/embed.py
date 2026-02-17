"""
Embed module: Chunks → embeddings → Qdrant

Uses OpenAI for embeddings and Qdrant for vector storage.
"""
import uuid
from typing import Optional
from qdrant_client import QdrantClient
from qdrant_client.models import (
    Distance,
    VectorParams,
    PointStruct,
    CollectionInfo,
)

from .config import config
from .ingest import Chunk

# Lazy import openai to avoid dependency issues
_openai_client = None


def get_openai_client():
    """Get or create OpenAI client."""
    global _openai_client
    if _openai_client is None:
        from openai import OpenAI
        _openai_client = OpenAI(api_key=config.openai_api_key)
    return _openai_client


def get_qdrant_client() -> QdrantClient:
    """Get Qdrant client."""
    # Parse URL to get host and port
    url = config.qdrant_url
    if url.startswith("http://"):
        url = url[7:]
    elif url.startswith("https://"):
        url = url[8:]

    host, port = url.split(":") if ":" in url else (url, 6333)
    return QdrantClient(host=host, port=int(port))


def create_collection(
    collection_name: Optional[str] = None,
    recreate: bool = False
) -> bool:
    """Create a Qdrant collection for storing embeddings."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    # Check if collection exists
    collections = client.get_collections().collections
    exists = any(c.name == collection_name for c in collections)

    if exists:
        if recreate:
            print(f"Deleting existing collection: {collection_name}")
            client.delete_collection(collection_name)
        else:
            print(f"Collection already exists: {collection_name}")
            return False

    # Create collection
    client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=config.embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    print(f"Created collection: {collection_name}")
    return True


def delete_collection(collection_name: Optional[str] = None) -> bool:
    """Delete a Qdrant collection."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    try:
        client.delete_collection(collection_name)
        print(f"Deleted collection: {collection_name}")
        return True
    except Exception as e:
        print(f"Error deleting collection: {e}")
        return False


def get_collection_info(collection_name: Optional[str] = None) -> Optional[CollectionInfo]:
    """Get information about a collection."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    try:
        return client.get_collection(collection_name)
    except Exception:
        return None


def embed_text(text: str) -> list[float]:
    """Generate embedding for a single text."""
    client = get_openai_client()

    response = client.embeddings.create(
        model=config.embedding_model,
        input=text,
    )

    return response.data[0].embedding


def embed_texts(texts: list[str], batch_size: int = 100) -> list[list[float]]:
    """Generate embeddings for multiple texts."""
    client = get_openai_client()
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch = texts[i:i + batch_size]
        response = client.embeddings.create(
            model=config.embedding_model,
            input=batch,
        )
        batch_embeddings = [item.embedding for item in response.data]
        all_embeddings.extend(batch_embeddings)

        print(f"Embedded {min(i + batch_size, len(texts))}/{len(texts)} texts")

    return all_embeddings


def embed_chunks(
    chunks: list[Chunk],
    collection_name: Optional[str] = None,
    batch_size: int = 100
) -> int:
    """Embed chunks and store in Qdrant."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    # Ensure collection exists
    create_collection(collection_name, recreate=False)

    # Generate embeddings
    texts = [chunk.text for chunk in chunks]
    embeddings = embed_texts(texts, batch_size)

    # Create points
    points = []
    for chunk, embedding in zip(chunks, embeddings):
        point = PointStruct(
            id=str(uuid.uuid4()),
            vector=embedding,
            payload={
                "text": chunk.text,
                "source_id": chunk.source_id,
                "source_title": chunk.source_title,
                "source_url": chunk.source_url,
                "chunk_index": chunk.chunk_index,
                **chunk.metadata,
            }
        )
        points.append(point)

    # Upsert in batches
    for i in range(0, len(points), batch_size):
        batch = points[i:i + batch_size]
        client.upsert(
            collection_name=collection_name,
            points=batch,
        )
        print(f"Stored {min(i + batch_size, len(points))}/{len(points)} chunks")

    print(f"Total chunks stored: {len(points)}")
    return len(points)
