"""Coherence Scoring Pipeline

Semantic Coherence (SC) measurement for knowledge pattern drift detection.
SC = (Availability × Consistency × Stability)^(1/3)
"""

from .models import CoherenceScore, Pattern
from .score import score_batch, score_pattern

__all__ = [
 "score_pattern",
 "score_batch",
 "Pattern",
 "CoherenceScore",
]
