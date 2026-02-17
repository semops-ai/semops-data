"""Tests for coherence scoring pipeline."""

import math

import pytest

from data_systems_toolkit.coherence.availability import (
    compute_centroid,
    cosine_similarity,
)
from data_systems_toolkit.coherence.fixtures import (
    generate_corpus_patterns,
    generate_test_patterns,
)
from data_systems_toolkit.coherence.models import CoherenceScore, Pattern
from data_systems_toolkit.coherence.score import compute_stability


class TestModels:
    """Test data models."""

    def test_pattern_creation(self):
        p = Pattern(text="test", pattern_id="p1")
        assert p.text == "test"
        assert p.pattern_id == "p1"
        assert p.corpus_id is None
        assert p.metadata == {}

    def test_coherence_score_creation(self):
        s = CoherenceScore(
            pattern_id="p1",
            availability=0.8,
            consistency=0.9,
            stability=1.0,
            composite_score=(0.8 * 0.9 * 1.0) ** (1 / 3),
            method="v2-nli",
        )
        assert s.composite_score == pytest.approx(0.8963, abs=0.001)


class TestAvailabilityMath:
    """Test availability math (no external services)."""

    def test_cosine_similarity_identical(self):
        v = [1.0, 2.0, 3.0]
        assert cosine_similarity(v, v) == pytest.approx(1.0)

    def test_cosine_similarity_orthogonal(self):
        a = [1.0, 0.0]
        b = [0.0, 1.0]
        assert cosine_similarity(a, b) == pytest.approx(0.0)

    def test_cosine_similarity_opposite(self):
        a = [1.0, 0.0]
        b = [-1.0, 0.0]
        assert cosine_similarity(a, b) == pytest.approx(-1.0)

    def test_cosine_similarity_zero_vector(self):
        a = [0.0, 0.0]
        b = [1.0, 1.0]
        assert cosine_similarity(a, b) == 0.0

    def test_compute_centroid(self):
        embeddings = [[1.0, 2.0], [3.0, 4.0]]
        centroid = compute_centroid(embeddings)
        assert centroid == [2.0, 3.0]

    def test_compute_centroid_single(self):
        embeddings = [[1.0, 2.0, 3.0]]
        centroid = compute_centroid(embeddings)
        assert centroid == [1.0, 2.0, 3.0]

    def test_compute_centroid_empty_raises(self):
        with pytest.raises(ValueError):
            compute_centroid([])


class TestSCFormula:
    """Test SC formula math."""

    def test_perfect_scores(self):
        composite = (1.0 * 1.0 * 1.0) ** (1 / 3)
        assert composite == pytest.approx(1.0)

    def test_geometric_mean(self):
        composite = (0.8 * 0.6 * 1.0) ** (1 / 3)
        assert composite == pytest.approx(0.7830, abs=0.001)

    def test_zero_component_zeros_composite(self):
        composite = (0.0 * 0.9 * 1.0) ** (1 / 3)
        assert composite == pytest.approx(0.0)

    def test_stability_stub(self):
        p = Pattern(text="test", pattern_id="p1")
        assert compute_stability(p) == 1.0


class TestFixtures:
    """Test synthetic fixture generators."""

    def test_corpus_patterns_default(self):
        patterns = generate_corpus_patterns()
        assert len(patterns) == 8
        assert all(isinstance(p, Pattern) for p in patterns)
        assert all(p.corpus_id == "synthetic-v1" for p in patterns)

    def test_corpus_patterns_limited(self):
        patterns = generate_corpus_patterns(n=3)
        assert len(patterns) == 3

    def test_test_patterns_default(self):
        patterns = generate_test_patterns()
        assert len(patterns) == 7
        assert all(isinstance(p, Pattern) for p in patterns)

    def test_test_patterns_limited(self):
        patterns = generate_test_patterns(n=2)
        assert len(patterns) == 2


def _ollama_available() -> bool:
    """Check if Ollama is running."""
    try:
        import requests
        requests.get("http://localhost:11434/api/tags", timeout=1)
        return True
    except Exception:
        return False


def _gpu_available() -> bool:
    """Check if GPU is available for NLI."""
    try:
        import torch
        return torch.cuda.is_available()
    except ImportError:
        return False


@pytest.mark.skipif(not _ollama_available(), reason="Ollama not running")
class TestAvailabilityIntegration:
    """Integration tests requiring Ollama."""

    def test_embed_text(self):
        from data_systems_toolkit.coherence.availability import embed_text
        embedding = embed_text("test text")
        assert len(embedding) == 768
        assert all(isinstance(x, float) for x in embedding)

    def test_compute_availability(self):
        from data_systems_toolkit.coherence.availability import (
            compute_availability,
            embed_texts,
        )

        corpus = generate_corpus_patterns(n=3)
        corpus_embeddings = embed_texts([p.text for p in corpus])

        # Coherent pattern should score higher than unrelated
        coherent = Pattern(text="Data validation prevents quality issues", pattern_id="c")
        unrelated = Pattern(text="The weather is sunny today", pattern_id="u")

        score_coherent = compute_availability(coherent, corpus_embeddings)
        score_unrelated = compute_availability(unrelated, corpus_embeddings)

        assert 0.0 <= score_coherent <= 1.0
        assert 0.0 <= score_unrelated <= 1.0
        assert score_coherent > score_unrelated
