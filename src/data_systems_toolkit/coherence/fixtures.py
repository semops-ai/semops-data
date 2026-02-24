"""Synthetic test pattern generators for coherence experiments."""

from .models import Pattern

# Coherent corpus: patterns about data quality that should be consistent
_CORPUS_PATTERNS = [
 "Data validation should occur at ingestion time to prevent bad data from entering the pipeline.",
 "Schema evolution must be backward compatible to avoid breaking downstream consumers.",
 "Data lineage tracking enables root cause analysis when quality issues are detected.",
 "Idempotent transformations ensure reprocessing produces identical results.",
 "Partitioning by date enables efficient incremental processing of time-series data.",
 "Data contracts between producers and consumers prevent schema drift.",
 "Monitoring data freshness ensures SLAs are met for downstream dashboards.",
 "Deduplication at the source prevents inflated metrics in aggregate tables.",
]

# Test patterns: mix of coherent, contradictory, and unrelated
_TEST_PATTERNS = [
 # Should score high (coherent with corpus)
 ("t1", "Validating data at the point of ingestion prevents downstream quality issues."),
 ("t2", "Tracking data lineage helps identify the root cause of data quality problems."),
 ("t3", "Schema changes should maintain backward compatibility with existing consumers."),
 # Should score lower (contradicts corpus)
 ("t4", "Data validation is unnecessary overhead and should be removed from pipelines."),
 ("t5", "Schema evolution should break backward compatibility to force consumer updates."),
 # Should score lower (unrelated/off-topic)
 ("t6", "The weather forecast predicts rain tomorrow afternoon."),
 ("t7", "Renaissance art influenced modern architectural design principles."),
]


def generate_corpus_patterns(n: int | None = None) -> list[Pattern]:
 """Generate synthetic corpus patterns.

 Args:
 n: Number of patterns to return (default: all).

 Returns:
 List of Pattern objects.
 """
 patterns = [
 Pattern(text=text, pattern_id=f"corpus-{i}", corpus_id="synthetic-v1")
 for i, text in enumerate(_CORPUS_PATTERNS)
 ]
 if n is not None:
 patterns = patterns[:n]
 return patterns


def generate_test_patterns(n: int | None = None) -> list[Pattern]:
 """Generate synthetic test patterns.

 Args:
 n: Number of patterns to return (default: all).

 Returns:
 List of Pattern objects with mix of coherent, contradictory, and unrelated.
 """
 patterns = [
 Pattern(text=text, pattern_id=pid, corpus_id="synthetic-v1")
 for pid, text in _TEST_PATTERNS
 ]
 if n is not None:
 patterns = patterns[:n]
 return patterns
