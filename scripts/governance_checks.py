#!/usr/bin/env python3
"""
Governance fitness checks for lifecycle state validation.

Validates registry.yaml declarations against observable signals (GitHub API).
Implements the data-contracts pattern: manifest declares expected state,
conditional checks validate against reality.

Output format matches run_all_fitness_functions() for MCP compatibility:
    {check_name, entity_id, issue, severity}

Usage:
    python scripts/governance_checks.py                    # All checks
    python scripts/governance_checks.py --severity HIGH    # Filter by severity
    python scripts/governance_checks.py --check-github     # Include GitHub API checks
    python scripts/governance_checks.py --json             # JSON output
    python scripts/governance_checks.py --dry-run          # Show what would be checked
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

import yaml

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
REGISTRY_PATH = REPO_ROOT / "config" / "registry.yaml"

# GitHub org for repo lookups
GH_OWNER = os.environ.get("GH_OWNER", "semops-ai")


# ---------------------------------------------------------------------------
# Registry loading
# ---------------------------------------------------------------------------
def load_registry() -> dict:
    """Load and flatten registry.yaml into a list of capabilities."""
    with open(REGISTRY_PATH) as f:
        return yaml.safe_load(f)


def flatten_capabilities(registry: dict) -> list[dict]:
    """Flatten nested capability structure into a flat list."""
    capabilities = []
    caps = registry.get("capabilities", {})
    for tier_name, tier in caps.items():
        if isinstance(tier, dict):
            for layer_name, cap_list in tier.items():
                if isinstance(cap_list, list):
                    for cap in cap_list:
                        cap["_tier"] = tier_name
                        cap["_layer"] = layer_name
                        capabilities.append(cap)
        elif isinstance(tier, list):
            for cap in tier:
                cap["_tier"] = tier_name
                cap["_layer"] = tier_name
                capabilities.append(cap)
    return capabilities


# ---------------------------------------------------------------------------
# GitHub API helpers
# ---------------------------------------------------------------------------
def gh_issue_state(repo: str, issue_num: int) -> str | None:
    """Query GitHub issue state via gh CLI. Returns 'OPEN'|'CLOSED' or None."""
    try:
        result = subprocess.run(
            ["gh", "issue", "view", str(issue_num),
             "--repo", f"{GH_OWNER}/{repo}",
             "--json", "state", "-q", ".state"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            return result.stdout.strip()
    except (subprocess.TimeoutExpired, FileNotFoundError):
        pass
    return None


def gh_repo_has_open_issues(repo: str, capability_id: str) -> bool:
    """Check if a repo has any open issues mentioning this capability."""
    try:
        result = subprocess.run(
            ["gh", "issue", "list",
             "--repo", f"{GH_OWNER}/{repo}",
             "--state", "open",
             "--search", capability_id,
             "--json", "number", "-q", "length"],
            capture_output=True, text=True, timeout=10,
        )
        if result.returncode == 0:
            count = result.stdout.strip()
            return int(count) > 0 if count else False
    except (subprocess.TimeoutExpired, FileNotFoundError, ValueError):
        pass
    return False


def parse_issue_ref(ref: str) -> tuple[str, int] | None:
    """Parse 'repo#123' into (repo, 123). Returns None if invalid."""
    if not ref or ref == "null":
        return None
    parts = ref.split("#")
    if len(parts) == 2:
        try:
            return parts[0], int(parts[1])
        except ValueError:
            pass
    return None


# ---------------------------------------------------------------------------
# Manifest-only checks (no external queries)
# ---------------------------------------------------------------------------
def check_governance_issue_required(capabilities: list[dict]) -> list[dict]:
    """in_progress capabilities MUST have a governance issue."""
    results = []
    for cap in capabilities:
        if cap["status"] == "in_progress" and not cap.get("governance", {}).get("issue"):
            results.append({
                "check_name": "governance_issue_required",
                "entity_id": cap["id"],
                "issue": f"Capability is in_progress but has no governance issue "
                         f"(status={cap['status']}, layer={cap.get('_layer', '?')})",
                "severity": "HIGH",
            })
    return results


def check_project_assignment(capabilities: list[dict]) -> list[dict]:
    """Non-retired capabilities should have at least one project assignment."""
    results = []
    for cap in capabilities:
        if cap["status"] not in ("retired",) and not cap.get("projects"):
            results.append({
                "check_name": "project_assignment",
                "entity_id": cap["id"],
                "issue": f"Capability has no project assignment "
                         f"(status={cap['status']})",
                "severity": "MEDIUM",
            })
    return results


def check_delivered_by_required(capabilities: list[dict]) -> list[dict]:
    """Active capabilities must have at least one delivering repo."""
    results = []
    for cap in capabilities:
        if cap["status"] == "active" and not cap.get("delivered_by"):
            results.append({
                "check_name": "delivered_by_required",
                "entity_id": cap["id"],
                "issue": "Active capability has no delivering repository",
                "severity": "CRITICAL",
            })
    return results


def check_pattern_implementation(capabilities: list[dict]) -> list[dict]:
    """All capabilities should implement at least one pattern."""
    results = []
    for cap in capabilities:
        if not cap.get("implements_patterns"):
            results.append({
                "check_name": "pattern_implementation",
                "entity_id": cap["id"],
                "issue": f"Capability implements no patterns "
                         f"(status={cap['status']})",
                "severity": "HIGH",
            })
    return results


def check_criteria_defined(capabilities: list[dict]) -> list[dict]:
    """Active/in_progress capabilities should have acceptance criteria."""
    results = []
    for cap in capabilities:
        if cap["status"] in ("active", "in_progress"):
            criteria = cap.get("governance", {}).get("criteria")
            if not criteria or len(criteria.strip()) < 10:
                results.append({
                    "check_name": "criteria_defined",
                    "entity_id": cap["id"],
                    "issue": f"Capability is {cap['status']} but has no meaningful acceptance criteria",
                    "severity": "MEDIUM",
                })
    return results


def derive_pattern_lifecycle(capabilities: list[dict]) -> list[dict]:
    """Derive pattern lifecycle from implementing capabilities and flag orphans."""
    pattern_states: dict[str, list[str]] = {}
    for cap in capabilities:
        for pattern_id in cap.get("implements_patterns", []):
            pattern_states.setdefault(pattern_id, []).append(cap["status"])

    results = []
    for pattern_id, statuses in pattern_states.items():
        if all(s in ("planned", "draft") for s in statuses):
            results.append({
                "check_name": "pattern_lifecycle_derived",
                "entity_id": pattern_id,
                "issue": f"Pattern only referenced by planned/draft capabilities "
                         f"({len(statuses)} capability refs, all pre-active)",
                "severity": "LOW",
            })
    return results


# ---------------------------------------------------------------------------
# GitHub-backed checks (require API access)
# ---------------------------------------------------------------------------
def check_issue_state_alignment(capabilities: list[dict]) -> list[dict]:
    """Validate governance issue state matches capability status."""
    results = []
    for cap in capabilities:
        ref = cap.get("governance", {}).get("issue")
        parsed = parse_issue_ref(ref)
        if not parsed:
            continue

        repo, issue_num = parsed
        state = gh_issue_state(repo, issue_num)
        if state is None:
            results.append({
                "check_name": "issue_state_alignment",
                "entity_id": cap["id"],
                "issue": f"Could not resolve governance issue {ref}",
                "severity": "MEDIUM",
            })
            continue

        # in_progress should have OPEN issue
        if cap["status"] == "in_progress" and state == "CLOSED":
            results.append({
                "check_name": "issue_state_alignment",
                "entity_id": cap["id"],
                "issue": f"Capability is in_progress but governance issue {ref} is CLOSED",
                "severity": "HIGH",
            })

        # active should have CLOSED issue (work is done)
        if cap["status"] == "active" and state == "OPEN":
            results.append({
                "check_name": "issue_state_alignment",
                "entity_id": cap["id"],
                "issue": f"Capability is active but governance issue {ref} is still OPEN",
                "severity": "MEDIUM",
            })

        # retired should have CLOSED issue
        if cap["status"] == "retired" and state == "OPEN":
            results.append({
                "check_name": "issue_state_alignment",
                "entity_id": cap["id"],
                "issue": f"Capability is retired but governance issue {ref} is still OPEN",
                "severity": "HIGH",
            })

    return results


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
MANIFEST_CHECKS = [
    check_governance_issue_required,
    check_project_assignment,
    check_delivered_by_required,
    check_pattern_implementation,
    check_criteria_defined,
    derive_pattern_lifecycle,
]

GITHUB_CHECKS = [
    check_issue_state_alignment,
]


def run_governance_checks(
    *,
    include_github: bool = False,
    severity: list[str] | None = None,
) -> list[dict]:
    """Run all governance checks and return violations."""
    registry = load_registry()
    capabilities = flatten_capabilities(registry)

    results = []
    for check_fn in MANIFEST_CHECKS:
        results.extend(check_fn(capabilities))

    if include_github:
        for check_fn in GITHUB_CHECKS:
            results.extend(check_fn(capabilities))

    if severity:
        severity_upper = [s.upper() for s in severity]
        results = [r for r in results if r["severity"] in severity_upper]

    # Sort by severity
    severity_order = {"CRITICAL": 0, "HIGH": 1, "MEDIUM": 2, "LOW": 3}
    results.sort(key=lambda r: (severity_order.get(r["severity"], 99), r["check_name"], r["entity_id"]))

    return results


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Governance fitness checks for lifecycle state validation")
    parser.add_argument("--check-github", action="store_true", help="Include GitHub API checks (slower)")
    parser.add_argument("--severity", nargs="+", help="Filter by severity (CRITICAL, HIGH, MEDIUM, LOW)")
    parser.add_argument("--json", action="store_true", dest="json_output", help="JSON output")
    parser.add_argument("--dry-run", action="store_true", help="Show capabilities that would be checked")
    args = parser.parse_args()

    if args.dry_run:
        registry = load_registry()
        capabilities = flatten_capabilities(registry)
        print(f"Registry: {REGISTRY_PATH}")
        print(f"Capabilities: {len(capabilities)}")
        print()
        for cap in capabilities:
            issue_ref = cap.get("governance", {}).get("issue") or "—"
            projects = ", ".join(cap.get("projects", [])) or "—"
            print(f"  [{cap['status']:12s}] {cap['id']:<35s} issue={issue_ref:<20s} projects={projects}")
        return

    results = run_governance_checks(
        include_github=args.check_github,
        severity=args.severity,
    )

    if args.json_output:
        print(json.dumps(results, indent=2))
    else:
        if not results:
            print("All governance checks passed.")
            return

        # Group by severity
        for sev in ("CRITICAL", "HIGH", "MEDIUM", "LOW"):
            sev_results = [r for r in results if r["severity"] == sev]
            if sev_results:
                print(f"\n{sev} ({len(sev_results)})")
                print("-" * 60)
                for r in sev_results:
                    print(f"  {r['check_name']}: {r['entity_id']}")
                    print(f"    {r['issue']}")

        print(f"\nTotal violations: {len(results)}")


if __name__ == "__main__":
    main()
