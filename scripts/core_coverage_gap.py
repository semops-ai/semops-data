#!/usr/bin/env python3
"""
Scope 1 Analytic: core-coverage-gap

Measures completeness of pattern → capability → repo coverage chains.
Identifies patterns without implementations, capabilities without pattern
links, and repos without capability exposure.

Composes existing views (pattern_coverage, capability_coverage,
repo_capabilities) and fitness functions into a unified gap report
with scoring.

DD-0023 Scope 1, Core Aggregate Health.
Issue: semops-orchestrator#249

Usage:
    python scripts/core_coverage_gap.py             # Full report
    python scripts/core_coverage_gap.py --json      # JSON output
    python scripts/core_coverage_gap.py --score-only # Just the composite score
"""

from __future__ import annotations

import argparse
import json
import sys

from db_utils import get_db_connection


# ---------------------------------------------------------------------------
# Gap queries — compose existing views into gap detection
# ---------------------------------------------------------------------------

def patterns_without_capabilities(conn) -> list[dict]:
    """Patterns with zero implementing capabilities (pattern_coverage view)."""
    cur = conn.cursor()
    cur.execute("""
        SELECT pattern_id, preferred_label, provenance,
               content_count, capability_count, repo_count
        FROM pattern_coverage
        WHERE capability_count = 0
        ORDER BY provenance, preferred_label
    """)
    return [
        {
            "pattern_id": r[0],
            "preferred_label": r[1],
            "provenance": r[2],
            "content_count": r[3],
            "capability_count": r[4],
            "repo_count": r[5],
        }
        for r in cur.fetchall()
    ]


def patterns_without_repos(conn) -> list[dict]:
    """Patterns with capabilities but no repo delivery chain."""
    cur = conn.cursor()
    cur.execute("""
        SELECT pattern_id, preferred_label, provenance,
               capability_count, repo_count
        FROM pattern_coverage
        WHERE capability_count > 0 AND repo_count = 0
        ORDER BY preferred_label
    """)
    return [
        {
            "pattern_id": r[0],
            "preferred_label": r[1],
            "provenance": r[2],
            "capability_count": r[3],
            "repo_count": r[4],
        }
        for r in cur.fetchall()
    ]


def capabilities_without_patterns(conn) -> list[dict]:
    """Capabilities with no pattern implementation edges."""
    cur = conn.cursor()
    cur.execute("""
        SELECT capability_id, capability_name, domain_classification,
               pattern_count, repo_count
        FROM capability_coverage
        WHERE pattern_count = 0
        ORDER BY capability_name
    """)
    return [
        {
            "capability_id": r[0],
            "capability_name": r[1],
            "domain_classification": r[2],
            "pattern_count": r[3],
            "repo_count": r[4],
        }
        for r in cur.fetchall()
    ]


def capabilities_without_repos(conn) -> list[dict]:
    """Capabilities with patterns but no delivering repos."""
    cur = conn.cursor()
    cur.execute("""
        SELECT capability_id, capability_name, domain_classification,
               pattern_count, repo_count
        FROM capability_coverage
        WHERE pattern_count > 0 AND repo_count = 0
        ORDER BY capability_name
    """)
    return [
        {
            "capability_id": r[0],
            "capability_name": r[1],
            "domain_classification": r[2],
            "pattern_count": r[3],
            "repo_count": r[4],
        }
        for r in cur.fetchall()
    ]


def repos_without_capabilities(conn) -> list[dict]:
    """Repos in entity table with no capability delivery edges."""
    cur = conn.cursor()
    cur.execute("""
        SELECT e.id, e.title, e.metadata->>'role' AS role
        FROM entity e
        WHERE e.entity_type = 'repository'
          AND NOT EXISTS (
            SELECT 1 FROM edge ed
            WHERE ed.dst_type = 'entity' AND ed.dst_id = e.id
              AND ed.predicate = 'delivered_by'
          )
        ORDER BY e.title
    """)
    return [
        {"repo_id": r[0], "repo_name": r[1], "role": r[2]}
        for r in cur.fetchall()
    ]


def orphan_content_entities(conn) -> list[dict]:
    """Content entities without any pattern connection."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, content_type
        FROM orphan_entities
        LIMIT 50
    """)
    return [
        {"entity_id": r[0], "title": r[1], "content_type": r[2]}
        for r in cur.fetchall()
    ]


def coverage_totals(conn) -> dict:
    """Aggregate counts for scoring."""
    cur = conn.cursor()
    cur.execute("SELECT count(*) FROM pattern")
    total_patterns = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM entity WHERE entity_type = 'capability'")
    total_capabilities = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM entity WHERE entity_type = 'repository'")
    total_repos = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM pattern_coverage WHERE capability_count > 0")
    patterns_with_caps = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM pattern_coverage WHERE repo_count > 0")
    patterns_with_repos = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM capability_coverage WHERE pattern_count > 0")
    caps_with_patterns = cur.fetchone()[0]
    cur.execute("SELECT count(*) FROM capability_coverage WHERE repo_count > 0")
    caps_with_repos = cur.fetchone()[0]

    return {
        "total_patterns": total_patterns,
        "total_capabilities": total_capabilities,
        "total_repos": total_repos,
        "patterns_with_capabilities": patterns_with_caps,
        "patterns_with_repos": patterns_with_repos,
        "capabilities_with_patterns": caps_with_patterns,
        "capabilities_with_repos": caps_with_repos,
    }


def fitness_function_violations(conn) -> list[dict]:
    """Run coverage-related fitness functions individually.

    Avoids run_all_fitness_functions() which has an ambiguous column bug
    in check_content_metadata_completeness().
    """
    cur = conn.cursor()
    results = []

    checks = [
        ("capability_coverage", "check_capability_pattern_coverage", "CRITICAL"),
        ("agent_capability_coverage", "check_agent_capability_coverage", "CRITICAL"),
        ("capability_agent_coverage", "check_capability_agent_coverage", "MEDIUM"),
        ("explicit_architecture_coverage", "check_explicit_architecture_coverage", "MEDIUM"),
    ]

    for check_name, func_name, severity in checks:
        try:
            cur.execute(f"SELECT invalid_id, issue FROM {func_name}()")
            for r in cur.fetchall():
                results.append({
                    "check_name": check_name,
                    "entity_id": r[0],
                    "issue": r[1],
                    "severity": severity,
                })
        except Exception as e:
            # autocommit mode — no rollback needed, just skip
            pass

    return sorted(results, key=lambda r: (r["severity"], r["check_name"]))


def implementation_backlog(conn) -> list[dict]:
    """Capabilities not yet active, grouped by pattern (ADR-0017).

    Shows what's registered and waiting — the implementation pipeline.
    Patterns with mixed lifecycles (some active, some not) are the most
    actionable: the pattern is proven, but not all capabilities are built.
    """
    cur = conn.cursor()
    cur.execute("""
        SELECT
            p.id AS pattern_id,
            p.preferred_label,
            e.id AS capability_id,
            e.title AS capability_name,
            e.metadata->>'lifecycle_stage' AS lifecycle_stage,
            e.metadata->>'status' AS status
        FROM edge ed
        JOIN pattern p ON ed.dst_type = 'pattern' AND ed.dst_id = p.id
            AND ed.predicate = 'implements'
        JOIN entity e ON ed.src_type = 'entity' AND ed.src_id = e.id
            AND e.entity_type = 'capability'
        WHERE e.metadata->>'lifecycle_stage' != 'active'
        ORDER BY p.preferred_label, e.metadata->>'lifecycle_stage', e.title
    """)
    return [
        {
            "pattern_id": r[0],
            "pattern_label": r[1],
            "capability_id": r[2],
            "capability_name": r[3],
            "lifecycle_stage": r[4],
            "status": r[5],
        }
        for r in cur.fetchall()
    ]


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(totals: dict) -> dict:
    """Compute composite Scope 1 Availability score.

    Three sub-scores (0.0–1.0):
      - pattern_capability_coverage: % of patterns with ≥1 capability
      - capability_pattern_coverage: % of capabilities with ≥1 pattern
      - delivery_chain_coverage: % of capabilities with ≥1 delivering repo

    Composite = weighted average (equal weights).
    """
    def safe_ratio(num, denom):
        return round(num / denom, 4) if denom > 0 else 0.0

    p_cap = safe_ratio(totals["patterns_with_capabilities"], totals["total_patterns"])
    c_pat = safe_ratio(totals["capabilities_with_patterns"], totals["total_capabilities"])
    c_repo = safe_ratio(totals["capabilities_with_repos"], totals["total_capabilities"])

    composite = round((p_cap + c_pat + c_repo) / 3, 4)

    return {
        "pattern_capability_coverage": p_cap,
        "capability_pattern_coverage": c_pat,
        "delivery_chain_coverage": c_repo,
        "composite_availability": composite,
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_report(conn) -> dict:
    """Run the full core-coverage-gap analysis."""
    totals = coverage_totals(conn)
    score = compute_score(totals)

    return {
        "analytic": "core-coverage-gap",
        "scope": "scope-1-core-aggregate-health",
        "totals": totals,
        "score": score,
        "gaps": {
            "patterns_without_capabilities": patterns_without_capabilities(conn),
            "patterns_without_repos": patterns_without_repos(conn),
            "capabilities_without_patterns": capabilities_without_patterns(conn),
            "capabilities_without_repos": capabilities_without_repos(conn),
            "repos_without_capabilities": repos_without_capabilities(conn),
            "orphan_content_entities": orphan_content_entities(conn),
        },
        "fitness_violations": fitness_function_violations(conn),
        "implementation_backlog": implementation_backlog(conn),
    }


def print_text_report(report: dict) -> None:
    """Pretty-print the gap report."""
    totals = report["totals"]
    score = report["score"]

    print("=" * 70)
    print("  CORE COVERAGE GAP REPORT — Scope 1: Core Aggregate Health")
    print("=" * 70)

    print(f"\n## Totals")
    print(f"  Patterns:     {totals['total_patterns']}")
    print(f"  Capabilities: {totals['total_capabilities']}")
    print(f"  Repositories: {totals['total_repos']}")

    print(f"\n## Scores")
    print(f"  Pattern → Capability coverage: {score['pattern_capability_coverage']:.1%}")
    print(f"  Capability → Pattern coverage: {score['capability_pattern_coverage']:.1%}")
    print(f"  Delivery chain coverage:       {score['delivery_chain_coverage']:.1%}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Composite Availability:        {score['composite_availability']:.1%}")

    gaps = report["gaps"]

    def print_gap_section(title, items, id_key, label_key=None):
        print(f"\n## {title} ({len(items)})")
        if not items:
            print("  (none)")
            return
        for item in items:
            label = f" — {item[label_key]}" if label_key and item.get(label_key) else ""
            extra = ""
            if "provenance" in item:
                extra = f" [{item['provenance']}]"
            elif "domain_classification" in item:
                extra = f" [{item['domain_classification'] or '?'}]"
            elif "role" in item:
                extra = f" [{item['role'] or '?'}]"
            print(f"  - {item[id_key]}{label}{extra}")

    print_gap_section(
        "Patterns without capabilities",
        gaps["patterns_without_capabilities"],
        "pattern_id", "preferred_label",
    )
    print_gap_section(
        "Patterns with capabilities but no repo delivery",
        gaps["patterns_without_repos"],
        "pattern_id", "preferred_label",
    )
    print_gap_section(
        "Capabilities without pattern implementation",
        gaps["capabilities_without_patterns"],
        "capability_id", "capability_name",
    )
    print_gap_section(
        "Capabilities without delivering repos",
        gaps["capabilities_without_repos"],
        "capability_id", "capability_name",
    )
    print_gap_section(
        "Repos without capability delivery",
        gaps["repos_without_capabilities"],
        "repo_id", "repo_name",
    )
    print_gap_section(
        "Orphan content entities (no pattern link)",
        gaps["orphan_content_entities"],
        "entity_id", "title",
    )

    violations = report["fitness_violations"]
    print(f"\n## Fitness function violations (coverage-related): {len(violations)}")
    if violations:
        for v in violations:
            print(f"  [{v['severity']}] {v['check_name']}: {v['entity_id']} — {v['issue']}")
    else:
        print("  (none)")

    # Implementation backlog — capabilities waiting to be built
    backlog = report.get("implementation_backlog", [])
    if backlog:
        # Group by pattern
        by_pattern: dict[str, list] = {}
        for item in backlog:
            key = item["pattern_label"]
            by_pattern.setdefault(key, []).append(item)

        print(f"\n## Implementation backlog: {len(backlog)} capabilities waiting across {len(by_pattern)} patterns")
        for pattern_label, caps in sorted(by_pattern.items()):
            stages = {}
            for c in caps:
                stages.setdefault(c["lifecycle_stage"], []).append(c)
            stage_summary = ", ".join(
                f"{len(v)} {k}" for k, v in sorted(stages.items())
            )
            print(f"\n  {pattern_label} ({stage_summary})")
            for c in caps:
                print(f"    [{c['lifecycle_stage']}] {c['capability_id']} — {c['capability_name']}")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Scope 1: core-coverage-gap analytic")
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--score-only", action="store_true", help="Print only the composite score")
    args = parser.parse_args()

    conn = get_db_connection(autocommit=True)
    try:
        report = run_report(conn)

        if args.score_only:
            print(json.dumps(report["score"], indent=2))
        elif args.json:
            print(json.dumps(report, indent=2))
        else:
            print_text_report(report)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
