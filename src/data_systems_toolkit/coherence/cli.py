"""CLI for coherence scoring pipeline."""

import argparse
import json
import sys

from .models import Pattern


def _load_patterns(path: str) -> list[Pattern]:
 """Load patterns from a JSON file."""
 with open(path) as f:
 data = json.load(f)
 return [
 Pattern(
 text=p["text"],
 pattern_id=p["pattern_id"],
 corpus_id=p.get("corpus_id"),
 metadata=p.get("metadata", {}),
 )
 for p in data
 ]


def cmd_run_v1(args):
 """Run v1-embedding experiment."""
 from .experiments import v1_embedding
 from .fixtures import generate_corpus_patterns, generate_test_patterns

 if args.input and args.corpus:
 test_patterns = _load_patterns(args.input)
 corpus_patterns = _load_patterns(args.corpus)
 else:
 print("Using synthetic fixtures...")
 test_patterns = generate_test_patterns
 corpus_patterns = generate_corpus_patterns

 scores = v1_embedding.run(test_patterns, corpus_patterns)

 for s in scores:
 print(f" {s.pattern_id}: availability={s.availability:.3f} "
 f"composite={s.composite_score:.3f}")


def cmd_run_v2(args):
 """Run v2-nli experiment."""
 from .experiments import v2_nli
 from .fixtures import generate_corpus_patterns, generate_test_patterns

 if args.input and args.corpus:
 test_patterns = _load_patterns(args.input)
 corpus_patterns = _load_patterns(args.corpus)
 else:
 print("Using synthetic fixtures...")
 test_patterns = generate_test_patterns
 corpus_patterns = generate_corpus_patterns

 scores = v2_nli.run(test_patterns, corpus_patterns)

 for s in scores:
 print(f" {s.pattern_id}: availability={s.availability:.3f} "
 f"consistency={s.consistency:.3f} composite={s.composite_score:.3f}")


def cmd_score(args):
 """Score a single pattern."""
 from .score import score_pattern
 from .availability import embed_texts

 corpus_patterns = _load_patterns(args.corpus)
 corpus_embeddings = embed_texts([p.text for p in corpus_patterns])

 pattern = Pattern(text=args.text, pattern_id="ad-hoc")
 result = score_pattern(pattern, corpus_patterns, corpus_embeddings, method=args.method)

 print(json.dumps({
 "pattern_id": result.pattern_id,
 "availability": round(result.availability, 4),
 "consistency": round(result.consistency, 4),
 "stability": round(result.stability, 4),
 "composite_score": round(result.composite_score, 4),
 "method": result.method,
 }, indent=2))


def main:
 parser = argparse.ArgumentParser(description="Coherence Scoring CLI")
 subparsers = parser.add_subparsers(dest="command", help="Commands")

 # run-v1
 p_v1 = subparsers.add_parser("run-v1", help="Run v1-embedding experiment")
 p_v1.add_argument("--input", help="Test patterns JSON (default: synthetic)")
 p_v1.add_argument("--corpus", help="Corpus patterns JSON (default: synthetic)")
 p_v1.set_defaults(func=cmd_run_v1)

 # run-v2
 p_v2 = subparsers.add_parser("run-v2", help="Run v2-nli experiment")
 p_v2.add_argument("--input", help="Test patterns JSON (default: synthetic)")
 p_v2.add_argument("--corpus", help="Corpus patterns JSON (default: synthetic)")
 p_v2.set_defaults(func=cmd_run_v2)

 # score
 p_score = subparsers.add_parser("score", help="Score a single pattern")
 p_score.add_argument("--text", required=True, help="Pattern text to score")
 p_score.add_argument("--corpus", required=True, help="Corpus patterns JSON")
 p_score.add_argument(
 "--method", default="v2-nli", choices=["v1-embedding", "v2-nli"]
 )
 p_score.set_defaults(func=cmd_score)

 args = parser.parse_args

 if not args.command:
 parser.print_help
 sys.exit(1)

 args.func(args)


if __name__ == "__main__":
 main
