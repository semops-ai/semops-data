"""SC formula implementation and scoring orchestration."""

from .availability import compute_availability, embed_text, embed_texts
from .consistency import compute_consistency
from .models import CoherenceScore, Pattern


def compute_stability(pattern: Pattern) -> float:
    """Compute stability via bi-temporal delta.

    STUB: Returns 1.0 until Graphiti temporal data is available.
    """
    # TODO: Query Graphiti for pattern history
    # TODO: Compute delta over config.temporal_window_days
    return 1.0


def score_pattern(
    pattern: Pattern,
    corpus_patterns: list[Pattern],
    corpus_embeddings: list[list[float]],
    method: str = "v1-embedding",
) -> CoherenceScore:
    """Score a single pattern using the SC formula.

    SC = (Availability × Consistency × Stability)^(1/3)

    Args:
        pattern: Pattern to score.
        corpus_patterns: Reference corpus patterns.
        corpus_embeddings: Pre-computed corpus embeddings.
        method: Scoring method ('v1-embedding' or 'v2-nli').

    Returns:
        CoherenceScore with component and composite scores.
    """
    if method == "v1-embedding":
        availability = compute_availability(pattern, corpus_embeddings)
        consistency = 1.0
        stability = 1.0
    elif method == "v2-nli":
        availability = compute_availability(pattern, corpus_embeddings)
        consistency = compute_consistency(pattern, corpus_patterns)
        stability = compute_stability(pattern)
    else:
        raise ValueError(f"Unknown method: {method}")

    composite = (availability * consistency * stability) ** (1 / 3)

    return CoherenceScore(
        pattern_id=pattern.pattern_id,
        availability=availability,
        consistency=consistency,
        stability=stability,
        composite_score=composite,
        method=method,
    )


def score_batch(
    patterns: list[Pattern],
    corpus_patterns: list[Pattern],
    method: str = "v1-embedding",
) -> list[CoherenceScore]:
    """Score multiple patterns.

    Pre-computes corpus embeddings once for efficiency.

    Args:
        patterns: Patterns to score.
        corpus_patterns: Reference corpus.
        method: Scoring method.

    Returns:
        List of CoherenceScore results.
    """
    corpus_embeddings = embed_texts([p.text for p in corpus_patterns])
    return [
        score_pattern(p, corpus_patterns, corpus_embeddings, method)
        for p in patterns
    ]
