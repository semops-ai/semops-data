"""Data models for coherence scoring."""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Pattern:
 """A semantic pattern to score."""

 text: str
 pattern_id: str
 corpus_id: Optional[str] = None
 metadata: dict = field(default_factory=dict)


@dataclass
class CoherenceScore:
 """Result of coherence scoring."""

 pattern_id: str
 availability: float # 0-1
 consistency: float # 0-1
 stability: float # 0-1
 composite_score: float # geometric mean
 method: str # "v1-embedding", "v2-nli", etc.
 metadata: dict = field(default_factory=dict)
