"""
Research RAG Pipeline Configuration
"""
import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class ResearchConfig:
    """Configuration for the research RAG pipeline."""

    # Docling service (PDF/doc processing)
    docling_url: str = "http://localhost:5001"

    # Qdrant service (vector storage)
    qdrant_url: str = "http://localhost:6333"

    # Collection settings
    collection_name: str = "research-ai-transformation"
    embedding_model: str = "text-embedding-3-small"  # OpenAI
    embedding_dim: int = 1536

    # Chunking settings
    chunk_size: int = 1000
    chunk_overlap: int = 200

    # Local paths
    sources_dir: Path = field(default_factory=lambda: Path(__file__).parent / "sources")
    cache_dir: Path = field(default_factory=lambda: Path(__file__).parent / "cache")

    # OpenAI API (for embeddings)
    openai_api_key: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_API_KEY"))

    def __post_init__(self):
        self.sources_dir = Path(self.sources_dir)
        self.cache_dir = Path(self.cache_dir)
        self.sources_dir.mkdir(parents=True, exist_ok=True)
        self.cache_dir.mkdir(parents=True, exist_ok=True)


# Default config instance
config = ResearchConfig()
