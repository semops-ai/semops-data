"""MLflow experiment: v2-nli (embedding + NLI consistency)."""

import json
import tempfile

import mlflow

from ..config import config
from ..models import Pattern
from ..score import score_batch


def run(
    test_patterns: list[Pattern],
    corpus_patterns: list[Pattern],
    experiment_name: str = "coherence-v2-nli",
    run_name: str = "nli-consistency",
) -> list:
    """Run v2-nli experiment with MLflow tracking.

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
        mlflow.log_param("method", "v2-nli")
        mlflow.log_param("embedding_model", config.embedding_model)
        mlflow.log_param("nli_model", config.nli_model)
        mlflow.log_param("num_test_patterns", len(test_patterns))
        mlflow.log_param("corpus_size", len(corpus_patterns))

        scores = score_batch(test_patterns, corpus_patterns, method="v2-nli")

        avg_availability = sum(s.availability for s in scores) / len(scores)
        avg_consistency = sum(s.consistency for s in scores) / len(scores)
        avg_composite = sum(s.composite_score for s in scores) / len(scores)
        min_composite = min(s.composite_score for s in scores)
        max_composite = max(s.composite_score for s in scores)

        mlflow.log_metric("avg_availability", avg_availability)
        mlflow.log_metric("avg_consistency", avg_consistency)
        mlflow.log_metric("avg_composite_score", avg_composite)
        mlflow.log_metric("min_composite_score", min_composite)
        mlflow.log_metric("max_composite_score", max_composite)

        results = [
            {
                "pattern_id": s.pattern_id,
                "availability": round(s.availability, 4),
                "consistency": round(s.consistency, 4),
                "stability": round(s.stability, 4),
                "composite_score": round(s.composite_score, 4),
            }
            for s in scores
        ]

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(results, f, indent=2)
            mlflow.log_artifact(f.name, "results")

        print(f"v2-nli: availability={avg_availability:.3f}, "
              f"consistency={avg_consistency:.3f}, "
              f"composite={avg_composite:.3f}")

        return scores
