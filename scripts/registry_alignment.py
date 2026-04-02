#!/usr/bin/env python3
"""
Scope 1 Analytic: registry-alignment

Measures consistency between bronze YAML registries and silver SQL domain
model. Detects mismatches, stale entries, and ingestion gaps.

Compares:
  - pattern_v1.yaml  → pattern table
  - registry.yaml    → entity table (capabilities)
  - repos.yaml       → entity table (repositories)

DD-0023 Scope 1, Core Aggregate Health.
DD-0001 v0.6.0 ingestion gap audit — known field losses.
Issue: semops-orchestrator#249

Usage:
    python scripts/registry_alignment.py             # Full report
    python scripts/registry_alignment.py --json      # JSON output
    python scripts/registry_alignment.py --score-only # Just the composite score
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from db_utils import get_db_connection

# ---------------------------------------------------------------------------
# YAML file paths
# ---------------------------------------------------------------------------
DX_HUB = Path.home() / "GitHub" / "semops-orchestrator"
SEMOPS_HUB = Path.home() / "GitHub" / "semops-data"

PATTERN_YAML = DX_HUB / "config" / "patterns" / "pattern_v1.yaml"
REGISTRY_YAML = SEMOPS_HUB / "config" / "registry.yaml"
REPOS_YAML = DX_HUB / "config" / "repos.yaml"


# ---------------------------------------------------------------------------
# YAML loaders
# ---------------------------------------------------------------------------

def load_yaml_patterns() -> dict[str, dict]:
    """Load pattern_v1.yaml and return {id: pattern_dict}."""
    with open(PATTERN_YAML) as f:
        data = yaml.safe_load(f)
    return {p["id"]: p for p in data.get("patterns", [])}


def load_yaml_capabilities() -> dict[str, dict]:
    """Load registry.yaml and flatten capabilities into {id: cap_dict}."""
    with open(REGISTRY_YAML) as f:
        data = yaml.safe_load(f)
    caps = {}
    for tier_name, tier in data.get("capabilities", {}).items():
        if isinstance(tier, dict):
            for layer_name, cap_list in tier.items():
                if isinstance(cap_list, list):
                    for cap in cap_list:
                        cap["_tier"] = tier_name
                        cap["_layer"] = layer_name
                        caps[cap["id"]] = cap
        elif isinstance(tier, list):
            for cap in tier:
                cap["_tier"] = tier_name
                cap["_layer"] = tier_name
                caps[cap["id"]] = cap
    return caps


def load_yaml_repos() -> dict[str, dict]:
    """Load repos.yaml and return {name: repo_dict}."""
    with open(REPOS_YAML) as f:
        data = yaml.safe_load(f)
    return {r["name"]: r for r in data.get("repos", [])}


# ---------------------------------------------------------------------------
# SQL loaders
# ---------------------------------------------------------------------------

def load_sql_patterns(conn) -> dict[str, dict]:
    """Load all patterns from SQL."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, preferred_label, definition, provenance, metadata
        FROM pattern
    """)
    return {
        r[0]: {
            "id": r[0],
            "preferred_label": r[1],
            "definition": r[2],
            "provenance": r[3],
            "metadata": r[4] or {},
        }
        for r in cur.fetchall()
    }


def load_sql_capabilities(conn) -> dict[str, dict]:
    """Load all capability entities from SQL."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, metadata
        FROM entity
        WHERE entity_type = 'capability'
    """)
    return {
        r[0]: {
            "id": r[0],
            "title": r[1],
            "metadata": r[2] or {},
        }
        for r in cur.fetchall()
    }


def load_sql_repos(conn) -> dict[str, dict]:
    """Load all repository entities from SQL."""
    cur = conn.cursor()
    cur.execute("""
        SELECT id, title, metadata
        FROM entity
        WHERE entity_type = 'repository'
    """)
    return {
        r[0]: {
            "id": r[0],
            "title": r[1],
            "metadata": r[2] or {},
        }
        for r in cur.fetchall()
    }


# ---------------------------------------------------------------------------
# Comparison checks
# ---------------------------------------------------------------------------

def compare_patterns(yaml_patterns: dict, sql_patterns: dict) -> dict:
    """Compare pattern_v1.yaml against pattern table."""
    yaml_ids = set(yaml_patterns.keys())
    sql_ids = set(sql_patterns.keys())

    in_yaml_not_sql = sorted(yaml_ids - sql_ids)
    in_sql_not_yaml = sorted(sql_ids - yaml_ids)
    in_both = sorted(yaml_ids & sql_ids)

    mismatches = []
    field_losses = []

    for pid in in_both:
        yp = yaml_patterns[pid]
        sp = sql_patterns[pid]
        meta = sp.get("metadata", {})

        # Name mismatch
        if yp.get("name") and sp["preferred_label"] != yp["name"]:
            mismatches.append({
                "pattern_id": pid,
                "field": "name/preferred_label",
                "yaml_value": yp["name"],
                "sql_value": sp["preferred_label"],
            })

        # Status → lifecycle_stage
        # ADR-0017: pattern lifecycle is edge-derived from capability
        # lifecycles, not a direct copy of YAML status. Skip this comparison
        # for patterns — divergence is expected and correct.

        # pattern_type
        yaml_type = yp.get("pattern_type")
        sql_type = meta.get("pattern_type")
        if yaml_type and sql_type and yaml_type != sql_type:
            mismatches.append({
                "pattern_id": pid,
                "field": "pattern_type",
                "yaml_value": yaml_type,
                "sql_value": sql_type,
            })

        # DD-0001 documented field loss: shape
        if yp.get("shape"):
            field_losses.append({
                "pattern_id": pid,
                "field": "shape",
                "yaml_value": yp["shape"],
                "note": "DD-0001: shape field not ingested (TBD placeholder)",
            })

    return {
        "entity_type": "pattern",
        "yaml_count": len(yaml_ids),
        "sql_count": len(sql_ids),
        "in_yaml_not_sql": in_yaml_not_sql,
        "in_sql_not_yaml": in_sql_not_yaml,
        "mismatches": mismatches,
        "field_losses": field_losses,
    }


def compare_capabilities(yaml_caps: dict, sql_caps: dict) -> dict:
    """Compare registry.yaml capabilities against entity table."""
    yaml_ids = set(yaml_caps.keys())
    sql_ids = set(sql_caps.keys())

    in_yaml_not_sql = sorted(yaml_ids - sql_ids)
    in_sql_not_yaml = sorted(sql_ids - yaml_ids)
    in_both = sorted(yaml_ids & sql_ids)

    mismatches = []
    field_losses = []

    for cid in in_both:
        yc = yaml_caps[cid]
        sc = sql_caps[cid]
        meta = sc.get("metadata", {})

        # Name mismatch
        if yc.get("name") and sc["title"] != yc["name"]:
            mismatches.append({
                "capability_id": cid,
                "field": "name/title",
                "yaml_value": yc["name"],
                "sql_value": sc["title"],
            })

        # Status
        yaml_status = yc.get("status")
        sql_status = meta.get("status")
        if yaml_status and sql_status and yaml_status != sql_status:
            mismatches.append({
                "capability_id": cid,
                "field": "status",
                "yaml_value": yaml_status,
                "sql_value": sql_status,
            })

        # DD-0001 documented field losses
        if yc.get("projects"):
            field_losses.append({
                "capability_id": cid,
                "field": "projects",
                "yaml_value": yc["projects"],
                "note": "DD-0001: projects not ingested — breaks project→capability traceability",
            })
        gov = yc.get("governance", {})
        if gov.get("issue"):
            field_losses.append({
                "capability_id": cid,
                "field": "governance.issue",
                "yaml_value": gov["issue"],
                "note": "DD-0001: governance issue not ingested — lifecycle invisible in SQL",
            })
        if gov.get("criteria"):
            field_losses.append({
                "capability_id": cid,
                "field": "governance.criteria",
                "yaml_value": str(gov["criteria"])[:80],
                "note": "DD-0001: governance criteria not ingested",
            })
        if yc.get("strategic_classification"):
            field_losses.append({
                "capability_id": cid,
                "field": "strategic_classification",
                "yaml_value": yc["strategic_classification"],
                "note": "DD-0001: strategic_classification not ingested",
            })

    return {
        "entity_type": "capability",
        "yaml_count": len(yaml_ids),
        "sql_count": len(sql_ids),
        "in_yaml_not_sql": in_yaml_not_sql,
        "in_sql_not_yaml": in_sql_not_yaml,
        "mismatches": mismatches,
        "field_losses": field_losses,
    }


def compare_repos(yaml_repos: dict, sql_repos: dict) -> dict:
    """Compare repos.yaml against entity table."""
    yaml_ids = set(yaml_repos.keys())
    sql_ids = set(sql_repos.keys())

    in_yaml_not_sql = sorted(yaml_ids - sql_ids)
    in_sql_not_yaml = sorted(sql_ids - yaml_ids)
    in_both = sorted(yaml_ids & sql_ids)

    mismatches = []
    field_losses = []

    for rid in in_both:
        yr = yaml_repos[rid]
        sr = sql_repos[rid]
        meta = sr.get("metadata", {})

        # Role mismatch
        yaml_role = yr.get("role")
        sql_role = meta.get("role")
        if yaml_role and sql_role and yaml_role != sql_role:
            mismatches.append({
                "repo_id": rid,
                "field": "role",
                "yaml_value": yaml_role,
                "sql_value": sql_role,
            })

        # DD-0001 documented field losses
        if yr.get("layer"):
            field_losses.append({
                "repo_id": rid,
                "field": "layer",
                "yaml_value": yr["layer"],
                "note": "DD-0001: layer not ingested (semops-core/bolt-on/deployment)",
            })
        if yr.get("context_type"):
            field_losses.append({
                "repo_id": rid,
                "field": "context_type",
                "yaml_value": yr["context_type"],
                "note": "DD-0001: context_type not ingested (core/generic DDD subdomain)",
            })
        if yr.get("bounded_context"):
            field_losses.append({
                "repo_id": rid,
                "field": "bounded_context",
                "yaml_value": yr["bounded_context"],
                "note": "DD-0001: bounded_context not ingested (DDD subdomain role)",
            })
        if yr.get("depends_on"):
            field_losses.append({
                "repo_id": rid,
                "field": "depends_on",
                "yaml_value": yr["depends_on"],
                "note": "DD-0001: depends_on not ingested (dependency graph)",
            })
        if yr.get("provides"):
            field_losses.append({
                "repo_id": rid,
                "field": "provides",
                "yaml_value": yr["provides"],
                "note": "DD-0001: provides not ingested (capability exposure graph)",
            })

    return {
        "entity_type": "repository",
        "yaml_count": len(yaml_ids),
        "sql_count": len(sql_ids),
        "in_yaml_not_sql": in_yaml_not_sql,
        "in_sql_not_yaml": in_sql_not_yaml,
        "mismatches": mismatches,
        "field_losses": field_losses,
    }


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

def compute_score(comparisons: list[dict]) -> dict:
    """Compute composite Scope 1 Availability score for registry alignment.

    Sub-scores (0.0–1.0):
      - entity_presence: % of YAML entities present in SQL (across all types)
      - field_fidelity: 1 - (mismatches / entities_in_both)
      - ingestion_completeness: 1 - (field_losses / total_fields_checked)

    Composite = weighted average (entity_presence 40%, fidelity 30%, completeness 30%).
    """
    total_yaml = sum(c["yaml_count"] for c in comparisons)
    total_missing = sum(len(c["in_yaml_not_sql"]) for c in comparisons)
    total_in_both = sum(
        c["yaml_count"] - len(c["in_yaml_not_sql"]) for c in comparisons
    )
    total_mismatches = sum(len(c["mismatches"]) for c in comparisons)
    total_field_losses = sum(len(c["field_losses"]) for c in comparisons)

    def safe_ratio(num, denom):
        return round(num / denom, 4) if denom > 0 else 1.0

    entity_presence = safe_ratio(total_yaml - total_missing, total_yaml)
    field_fidelity = 1.0 - safe_ratio(total_mismatches, max(total_in_both, 1))
    # Field losses are structural (DD-0001 known gaps), not per-entity errors.
    # Score based on unique field types lost, not total instances.
    unique_loss_fields = len({
        (fl.get("field"),)
        for c in comparisons
        for fl in c["field_losses"]
    })
    # Total checkable field types: ~12 (estimated from DD-0001 audit)
    total_field_types = 12
    ingestion_completeness = 1.0 - safe_ratio(unique_loss_fields, total_field_types)

    composite = round(
        entity_presence * 0.4 + field_fidelity * 0.3 + ingestion_completeness * 0.3,
        4,
    )

    return {
        "entity_presence": round(entity_presence, 4),
        "field_fidelity": round(field_fidelity, 4),
        "ingestion_completeness": round(ingestion_completeness, 4),
        "composite_alignment": composite,
        "detail": {
            "total_yaml_entities": total_yaml,
            "total_missing_from_sql": total_missing,
            "total_mismatches": total_mismatches,
            "total_field_losses": total_field_losses,
            "unique_lost_field_types": unique_loss_fields,
        },
    }


# ---------------------------------------------------------------------------
# Report
# ---------------------------------------------------------------------------

def run_report(conn) -> dict:
    """Run the full registry-alignment analysis."""
    yaml_patterns = load_yaml_patterns()
    yaml_caps = load_yaml_capabilities()
    yaml_repos = load_yaml_repos()

    sql_patterns = load_sql_patterns(conn)
    sql_caps = load_sql_capabilities(conn)
    sql_repos = load_sql_repos(conn)

    comparisons = [
        compare_patterns(yaml_patterns, sql_patterns),
        compare_capabilities(yaml_caps, sql_caps),
        compare_repos(yaml_repos, sql_repos),
    ]

    score = compute_score(comparisons)

    return {
        "analytic": "registry-alignment",
        "scope": "scope-1-core-aggregate-health",
        "score": score,
        "comparisons": {c["entity_type"]: c for c in comparisons},
    }


def print_text_report(report: dict) -> None:
    """Pretty-print the alignment report."""
    score = report["score"]

    print("=" * 70)
    print("  REGISTRY ALIGNMENT REPORT — Scope 1: Core Aggregate Health")
    print("=" * 70)

    print(f"\n## Scores")
    print(f"  Entity presence (YAML→SQL):  {score['entity_presence']:.1%}")
    print(f"  Field fidelity (no drift):   {score['field_fidelity']:.1%}")
    print(f"  Ingestion completeness:      {score['ingestion_completeness']:.1%}")
    print(f"  ─────────────────────────────────────────")
    print(f"  Composite Alignment:         {score['composite_alignment']:.1%}")

    detail = score["detail"]
    print(f"\n  ({detail['total_yaml_entities']} YAML entities, "
          f"{detail['total_missing_from_sql']} missing from SQL, "
          f"{detail['total_mismatches']} value mismatches, "
          f"{detail['unique_lost_field_types']} field types lost)")

    for entity_type in ("pattern", "capability", "repository"):
        comp = report["comparisons"][entity_type]
        print(f"\n{'─' * 70}")
        print(f"## {entity_type.title()}s — YAML: {comp['yaml_count']}, SQL: {comp['sql_count']}")

        if comp["in_yaml_not_sql"]:
            print(f"\n  ### In YAML but not SQL ({len(comp['in_yaml_not_sql'])})")
            for eid in comp["in_yaml_not_sql"]:
                print(f"    - {eid}")

        if comp["in_sql_not_yaml"]:
            print(f"\n  ### In SQL but not YAML ({len(comp['in_sql_not_yaml'])})")
            for eid in comp["in_sql_not_yaml"]:
                print(f"    - {eid}")

        if comp["mismatches"]:
            print(f"\n  ### Value mismatches ({len(comp['mismatches'])})")
            for m in comp["mismatches"]:
                id_key = next(k for k in m if k.endswith("_id"))
                print(f"    - {m[id_key]}.{m['field']}: "
                      f"YAML={m['yaml_value']!r} → SQL={m['sql_value']!r}")

        if comp["field_losses"]:
            # Deduplicate by field name for summary
            fields_seen = {}
            for fl in comp["field_losses"]:
                fname = fl["field"]
                if fname not in fields_seen:
                    fields_seen[fname] = {"count": 0, "note": fl["note"]}
                fields_seen[fname]["count"] += 1

            print(f"\n  ### Field losses (DD-0001 documented gaps)")
            for fname, info in sorted(fields_seen.items()):
                print(f"    - {fname} ({info['count']} entities): {info['note']}")

        if not comp["in_yaml_not_sql"] and not comp["in_sql_not_yaml"] \
                and not comp["mismatches"] and not comp["field_losses"]:
            print("  (fully aligned)")

    print()


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="Scope 1: registry-alignment analytic"
    )
    parser.add_argument("--json", action="store_true", help="JSON output")
    parser.add_argument("--score-only", action="store_true",
                        help="Print only the composite score")
    args = parser.parse_args()

    conn = get_db_connection(autocommit=True)
    try:
        report = run_report(conn)

        if args.score_only:
            print(json.dumps(report["score"], indent=2))
        elif args.json:
            print(json.dumps(report, indent=2, default=str))
        else:
            print_text_report(report)
    finally:
        conn.close()


if __name__ == "__main__":
    main()
