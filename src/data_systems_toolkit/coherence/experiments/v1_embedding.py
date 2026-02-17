"""MLflow experiment: v1-embedding baseline (cosine similarity to centroid)."""

import json
import tempfile
from pathlib import Path

import mlflow

from ..config import config
from ..models import Pattern
from ..score import score_batch


def run(
    test_patterns: list[Pattern],
    corpus_patterns: list[Pattern],
    experiment_name: str = "coherence-v1-embedding",
    run_name: str = "embedding-baseline",
) -> list:
    """Run v1-embedding experiment with MLflow tracking.

    Args:
        test_patterns: Patterns to score.
        corpus_patterns: Reference corpus.
        experiment_name: MLflow experiment name.
        run_name: MLflow run name.

    Returns:
        List of CoherenceScore results.
    """
    mlflow.set_tracking_uri(config.mlflow_tracking_uri)
    mlflow.set_experiment(experiment_name)

    with mlflow.start_run(run_name=run_name):
        mlflow.log_param("method", "v1-embedding")
        mlflow.log_param("embedding_model", config.embedding_model)
        mlflow.log_param("num_test_patterns", len(test_patterns))
        mlflow.log_param("corpus_size", len(corpus_patterns))

        scores = score_batch(test_patterns, corpus_patterns, method="v1-embedding")

        avg_availability = sum(s.availability for s in scores) / len(scores)
        avg_composite = sum(s.composite_score for s in scores) / len(scores)
        min_availability = min(s.availability for s in scores)
        max_availability = max(s.availability for s in scores)

        mlflow.log_metric("avg_availability", avg_availability)
        mlflow.log_metric("avg_composite_score", avg_composite)
        mlflow.log_metric("min_availability", min_availability)
        mlflow.log_metric("max_availability", max_availability)

        results = [
            {
                "pattern_id": s.pattern_id,
                "availability": round(s.availability, 4),
                "composite_score": round(s.composite_score, 4),
            }
            for s in scores
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(results, f, indent=2)
            mlflow.log_artifact(f.name, "results")

        print(f"v1-embedding: avg_availability={avg_availability:.3f}, "
              f"range=[{min_availability:.3f}, {max_availability:.3f}]")

        return scores
