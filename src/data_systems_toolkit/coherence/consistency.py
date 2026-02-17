"""Consistency component: NLI-based contradiction detection."""

from typing import Optional

import torch

from .config import config
from .models import Pattern

# Lazy-loaded NLI model and tokenizer
_nli_model = None
_nli_tokenizer = None
_device = None

# DeBERTa NLI label mapping: 0=contradiction, 1=neutral, 2=entailment
CONTRADICTION = 0
NEUTRAL = 1
ENTAILMENT = 2


def _get_device() -> torch.device:
    """Get compute device (GPU if available)."""
    global _device
    if _device is None:
        _device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return _device


def get_nli_model():
    """Load cross-encoder NLI model on GPU (lazy)."""
    global _nli_model, _nli_tokenizer
    if _nli_model is None:
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        _nli_tokenizer = AutoTokenizer.from_pretrained(config.nli_model)
        _nli_model = AutoModelForSequenceClassification.from_pretrained(
            config.nli_model
        )
        _nli_model = _nli_model.to(_get_device())
        _nli_model.eval()
    return _nli_model, _nli_tokenizer


def nli_scores(premise: str, hypothesis: str) -> dict[str, float]:
    """Run NLI inference on a premise-hypothesis pair.

    Args:
        premise: The premise text.
        hypothesis: The hypothesis text.

    Returns:
        Dict with keys 'contradiction', 'neutral', 'entailment' (probabilities).
    """
    model, tokenizer = get_nli_model()
    device = _get_device()

    inputs = tokenizer(
        premise,
        hypothesis,
        return_tensors="pt",
        truncation=True,
        max_length=512,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        logits = model(**inputs).logits
        probs = torch.softmax(logits, dim=-1)[0]

    return {
        "contradiction": probs[CONTRADICTION].item(),
        "neutral": probs[NEUTRAL].item(),
        "entailment": probs[ENTAILMENT].item(),
    }


def compute_consistency(
    pattern: Pattern,
    corpus_patterns: list[Pattern],
    aggregation: str = "mean",
) -> float:
    """Compute consistency via NLI entailment/contradiction scoring.

    For each corpus pattern, run bidirectional NLI and aggregate
    contradiction scores. High contradiction = low consistency.

    Args:
        pattern: Pattern to score.
        corpus_patterns: Other patterns in corpus to check against.
        aggregation: How to aggregate scores ('mean' or 'max').

    Returns:
        Consistency score (0-1), where 1 = no contradictions.
    """
    if not corpus_patterns:
        return 1.0

    contradiction_scores: list[float] = []

    for corpus_pattern in corpus_patterns:
        if corpus_pattern.pattern_id == pattern.pattern_id:
            continue

        # Bidirectional: check both directions
        scores_fwd = nli_scores(pattern.text, corpus_pattern.text)
        scores_rev = nli_scores(corpus_pattern.text, pattern.text)

        # Take max contradiction from either direction
        contradiction = max(
            scores_fwd["contradiction"], scores_rev["contradiction"]
        )
        contradiction_scores.append(contradiction)

    if not contradiction_scores:
        return 1.0

    if aggregation == "max":
        avg_contradiction = max(contradiction_scores)
    else:
        avg_contradiction = sum(contradiction_scores) / len(contradiction_scores)

    return 1.0 - avg_contradiction
