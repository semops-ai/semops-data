# ADR-0003: Coherence Scoring Pipeline

> **Status:** In Progress
> **Date:** 2026-01-30
> **Related Issue:** [](https://github.com/semops-ai/semops-data/issues/38)
> **Related ADR:** [ADR-0005 (semops-dx-orchestrator)](https://github.com/semops-ai/semops-dx-orchestrator) — Phase D drift measurement strategy

## Executive Summary

Implement the Semantic Coherence (SC) scoring pipeline as defined in GLOBAL_ARCHITECTURE.proposed.md PD-4. This provides the measurement infrastructure that Phase D  requires for theory-layer drift detection.

## Context

ADR-0005 defines the drift measurement strategy but has no implementation. The SC formula and MLflow experiment design are documented in the proposed architecture but need to be built as a concrete pipeline in semops-data.

The scoring pipeline must integrate with:
- `ingestion_episode.coherence_score` column 
- `@emit_lineage` decorator's `set_coherence_score` helper
- Graphiti bi-temporal model for Stability measurement
- Corpus-aware routing  for scoring context

## Decision

### SC Formula

```
SC = (Availability × Consistency × Stability)^(1/3)
```

- **Availability:** Can the pattern be retrieved? (embedding recall)
- **Consistency:** Does it contradict other knowledge? (NLI / entailment)
- **Stability:** Has it changed over time? (bi-temporal delta)

### Experiment Strategy

Use MLflow to evaluate five scoring approaches incrementally:

| Experiment | Approach | Rationale |
|------------|----------|-----------|
| `coherence-v1-embedding` | Cosine similarity to pattern centroid | Baseline — fast, interpretable |
| `coherence-v2-nli` | DeBERTa NLI for logical consistency | Catches contradictions embeddings miss |
| `coherence-v3-judge` | LLM-as-judge evaluation | Semantic understanding, higher cost |
| `coherence-v4-hybrid` | Embedding + NLI threshold | Best of v1+v2, tunable |
| `coherence-v5-multi-judge` | Multiple LLM passes, consensus | Gold standard, calibration reference |

Experiments run sequentially — each builds on learnings from the previous. v1 and v2 are MVP; v3-v5 are stretch.

### Technology Choices

| Component | Choice | Rationale |
|-----------|--------|-----------|
| **Embeddings** | `nomic-embed-text` via Ollama | Same model as semops-core ingestion — cosine similarity must use matching embedding space |
| **NLI Model** | `cross-encoder/nli-deberta-v3-base` (86M params, ~1.5GB VRAM) | Standard NLI cross-encoder, full precision, local GPU inference |
| **LLM Judge** | Claude via Anthropic API (`anthropic` SDK) | Already in research deps; quality matters more than cost for judge scoring |
| **Experiment Tracking** | MLflow file-based (`mlruns/`) | Current setup sufficient, no need for SQLite backend yet |
| **Module Location** | `src/data_systems_toolkit/coherence/` | Parallel to existing `research/` module |
| **Dependencies** | New `[coherence]` optional group in pyproject.toml | `transformers`, `sentence-transformers` |

### Threshold Calibration

Thresholds for "drift detected" will be determined empirically via the MLflow experiments. No hardcoded thresholds until calibration data exists.

## Consequences

### Positive

- Unblocks Phase D 
- Provides measurable drift detection for the theory layer
- MLflow tracking enables reproducible experiment comparison
- Incremental approach reduces risk — v1 delivers value immediately

### Negative

- Depends on (corpus-aware routing) for full context
- Stability component requires temporal data that may not exist yet
- Multiple experiment variants add maintenance surface

### Risks

- Threshold calibration may require manual review of edge cases
- LLM-as-judge approaches (v3, v5) have cost implications at scale

## Implementation Plan

1. Set up MLflow tracking infrastructure in semops-data
2. Implement SC formula components (Availability, Consistency, Stability)
3. Build v1-embedding experiment
4. Build v2-nli experiment
5. Evaluate v1/v2 results, calibrate thresholds
6. Stretch: v3-v5 experiments

## Session Log

- 2026-01-30: ADR created, issue #38 assigned
- 2026-01-30: Technology choices confirmed (nomic-embed-text, DeBERTa-v3-base, Claude judge)
- 2026-01-30: MVP implemented — v1-embedding + v2-nli experiments, CLI, tests, user guide

## References

- GLOBAL_ARCHITECTURE.proposed.md PD-4
- ADR-0005 §Drift Strategy (semops-dx-orchestrator)
- (Phase D)
- (episode provenance)
- (corpus-aware routing)
