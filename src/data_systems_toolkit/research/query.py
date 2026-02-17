"""
Query module: RAG query interface

Search and retrieve from the vector store, optionally with LLM synthesis.
"""
from typing import Optional
from dataclasses import dataclass, field
from qdrant_client.models import Filter, FieldCondition, MatchValue

from .config import config
from .embed import get_qdrant_client, embed_text, get_openai_client


@dataclass
class SearchResult:
    """A search result from the vector store."""
    text: str
    score: float
    source_title: str
    source_url: str
    source_id: str
    chunk_index: int
    metadata: dict = field(default_factory=dict)


def search_similar(
    query: str,
    limit: int = 10,
    collection_name: Optional[str] = None,
    source_filter: Optional[str] = None,
) -> list[SearchResult]:
    """Search for similar chunks in the vector store."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    # Generate query embedding
    query_embedding = embed_text(query)

    # Build filter if source specified
    search_filter = None
    if source_filter:
        search_filter = Filter(
            must=[
                FieldCondition(
                    key="source_id",
                    match=MatchValue(value=source_filter)
                )
            ]
        )

    # Search (using query_points for newer qdrant-client versions)
    results = client.query_points(
        collection_name=collection_name,
        query=query_embedding,
        limit=limit,
        query_filter=search_filter,
    ).points

    # Convert to SearchResult objects
    search_results = []
    for result in results:
        payload = result.payload
        search_result = SearchResult(
            text=payload.get("text", ""),
            score=result.score,
            source_title=payload.get("source_title", ""),
            source_url=payload.get("source_url", ""),
            source_id=payload.get("source_id", ""),
            chunk_index=payload.get("chunk_index", 0),
            metadata={
                k: v for k, v in payload.items()
                if k not in ["text", "source_title", "source_url", "source_id", "chunk_index"]
            }
        )
        search_results.append(search_result)

    return search_results


def format_context(results: list[SearchResult]) -> str:
    """Format search results as context for LLM."""
    context_parts = []

    for i, result in enumerate(results, 1):
        part = f"[{i}] Source: {result.source_title}\n{result.text}"
        context_parts.append(part)

    return "\n\n---\n\n".join(context_parts)


def query_rag(
    question: str,
    limit: int = 5,
    collection_name: Optional[str] = None,
    model: str = "gpt-4o-mini",
    system_prompt: Optional[str] = None,
) -> dict:
    """
    RAG query: search for relevant chunks and synthesize answer with LLM.

    Returns dict with:
    - answer: LLM-generated answer
    - sources: list of SearchResult objects used
    - context: formatted context string
    """
    # Search for relevant chunks
    results = search_similar(
        query=question,
        limit=limit,
        collection_name=collection_name,
    )

    if not results:
        return {
            "answer": "No relevant information found in the knowledge base.",
            "sources": [],
            "context": "",
        }

    # Format context
    context = format_context(results)

    # Build prompt
    if system_prompt is None:
        system_prompt = """You are a research assistant analyzing academic and industry reports on AI transformation in enterprises.

Your task is to answer questions based on the provided context from research sources.

Guidelines:
- Base your answers strictly on the provided context
- Cite sources using [N] notation when making claims
- If the context doesn't contain enough information, say so
- Be concise but thorough
- Note any disagreements or nuances between sources"""

    user_prompt = f"""Context from research sources:

{context}

---

Question: {question}

Please provide a well-reasoned answer based on the context above."""

    # Call LLM
    client = get_openai_client()
    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
    )

    answer = response.choices[0].message.content

    return {
        "answer": answer,
        "sources": results,
        "context": context,
    }


def list_sources(collection_name: Optional[str] = None) -> list[dict]:
    """List all unique sources in the collection."""
    collection_name = collection_name or config.collection_name
    client = get_qdrant_client()

    # Scroll through all points to get unique sources
    # This is not efficient for large collections but works for our use case
    sources = {}

    offset = None
    while True:
        results, offset = client.scroll(
            collection_name=collection_name,
            limit=100,
            offset=offset,
            with_payload=True,
            with_vectors=False,
        )

        for point in results:
            source_id = point.payload.get("source_id")
            if source_id and source_id not in sources:
                sources[source_id] = {
                    "source_id": source_id,
                    "title": point.payload.get("source_title"),
                    "url": point.payload.get("source_url"),
                    "authors": point.payload.get("authors", []),
                    "year": point.payload.get("year"),
                }

        if offset is None:
            break

    return list(sources.values())
