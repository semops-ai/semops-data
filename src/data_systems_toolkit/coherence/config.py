"""Configuration for the coherence scoring pipeline."""

import os
from dataclasses import dataclass, field
from typing import Optional


@dataclass
class CoherenceConfig:
    """Configuration for coherence scoring."""

    # Ollama service (local embeddings)
    ollama_url: str = "http://localhost:11434"
    embedding_model: str = "nomic-embed-text"
    embedding_dim: int = 768

    # NLI model (local GPU)
    nli_model: str = "cross-encoder/nli-deberta-v3-base"

    # Claude judge (API)
    anthropic_api_key: Optional[str] = field(
        default_factory=lambda: os.getenv("ANTHROPIC_API_KEY")
    )

    # MLflow
    mlflow_tracking_uri: str = "file:./mlruns"

    # Stability (future)
    temporal_window_days: int = 30


# Singleton instance
config = CoherenceConfig()
