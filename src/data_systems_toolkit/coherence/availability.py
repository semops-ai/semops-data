"""Availability component: embedding recall via cosine similarity."""

import math
from typing import Optional

import requests

from .config import config
from .models import Pattern

# Lazy-loaded session
_session: Optional[requests.Session] = None


def _get_session -> requests.Session:
 """Get or create HTTP session for Ollama."""
 global _session
 if _session is None:
 _session = requests.Session
 return _session


def embed_text(text: str) -> list[float]:
 """Generate embedding using nomic-embed-text via Ollama API.

 Args:
 text: Text to embed.

 Returns:
 Embedding vector of dimension 768.
 """
 session = _get_session
 response = session.post(
 f"{config.ollama_url}/api/embeddings",
 json={"model": config.embedding_model, "prompt": text},
 timeout=30,
 )
 response.raise_for_status
 return response.json["embedding"]


def embed_texts(texts: list[str]) -> list[list[float]]:
 """Embed multiple texts sequentially.

 Args:
 texts: List of texts to embed.

 Returns:
 List of embedding vectors.
 """
 return [embed_text(t) for t in texts]


def cosine_similarity(a: list[float], b: list[float]) -> float:
 """Compute cosine similarity between two vectors."""
 dot = sum(x * y for x, y in zip(a, b))
 norm_a = math.sqrt(sum(x * x for x in a))
 norm_b = math.sqrt(sum(x * x for x in b))
 if norm_a == 0 or norm_b == 0:
 return 0.0
 return dot / (norm_a * norm_b)


def compute_centroid(embeddings: list[list[float]]) -> list[float]:
 """Compute mean centroid of embedding vectors."""
 if not embeddings:
 raise ValueError("Cannot compute centroid of empty list")
 dim = len(embeddings[0])
 n = len(embeddings)
 centroid = [0.0] * dim
 for emb in embeddings:
 for i in range(dim):
 centroid[i] += emb[i]
 return [c / n for c in centroid]


def compute_availability(
 pattern: Pattern,
 corpus_embeddings: list[list[float]],
 pattern_embedding: Optional[list[float]] = None,
) -> float:
 """Compute availability as cosine similarity to corpus centroid.

 Args:
 pattern: Pattern to score.
 corpus_embeddings: Pre-computed embeddings of corpus patterns.
 pattern_embedding: Pre-computed pattern embedding (optional, avoids re-embedding).

 Returns:
 Availability score normalized to 0-1.
 """
 if not corpus_embeddings:
 return 0.0

 if pattern_embedding is None:
 pattern_embedding = embed_text(pattern.text)

 centroid = compute_centroid(corpus_embeddings)
 sim = cosine_similarity(pattern_embedding, centroid)

 # Cosine similarity is [-1, 1]; normalize to [0, 1]
 return max(0.0, min(1.0, (sim + 1.0) / 2.0))
