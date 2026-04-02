#!/usr/bin/env python3
"""
Ingest GitHub Issues from all deployment repos into the deployment corpus.

Each issue becomes a content entity with:
- corpus: deployment
- content_type: issue
- date_created / date_updated from GitHub API (deterministic)
- LLM classification: summary, subject_area, primary_concept, detected_edges

Usage:
    python scripts/ingest_github_issues.py
    python scripts/ingest_github_issues.py --no-llm
    python scripts/ingest_github_issues.py --dry-run
    python scripts/ingest_github_issues.py --repo semops-data
    python scripts/ingest_github_issues.py --state open
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

load_dotenv(override=False)

import psycopg
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

sys.path.insert(0, str(Path(__file__).parent))

from db_utils import get_db_connection
from ingest_from_source import ingest_chunks, ingest_entity, materialize_edges_neo4j

console = Console()

OWNER = "semops-ai"

DEPLOYMENT_REPOS = [
    "semops-backoffice",
    "data-pr",
    "docs-pr",
    "semops-orchestrator",
    "publisher-pr",
    "semops-research",
    "semops-data",
    "sites-pr",
]

LLM_MODEL = "claude-haiku-4-5-20251001"

ATTRIBUTION = {
    "$schema": "attribution_v2",
    "creator": ["Tim Mitchell"],
    "rights": "CC-BY-4.0",
    "organization": "TJMConsulting",
    "platform": "github",
    "channel": "semops-ai",
    "epistemic_status": "operational",
}


def list_issues(owner: str, repo: str, state: str = "all") -> list[dict]:
    """Fetch all issues from a repo via gh CLI (paginated, excludes PRs)."""
    issues = []
    page = 1
    while True:
        result = subprocess.run(
            [
                "gh", "api",
                f"repos/{owner}/{repo}/issues",
                "--method", "GET",
                "-f", f"state={state}",
                "-f", f"per_page=100",
                "-f", f"page={page}",
            ],
            capture_output=True,
            text=True,
            check=False,
        )
        if result.returncode != 0:
            console.print(f"[red]API error for {repo}: {result.stderr.strip()}[/red]")
            break

        batch = json.loads(result.stdout)
        if not batch:
            break

        # Exclude pull requests (they appear in the issues endpoint too)
        for issue in batch:
            if "pull_request" not in issue:
                issues.append(issue)

        if len(batch) < 100:
            break
        page += 1

    return issues


def format_issue_content(issue: dict, repo: str) -> str:
    """Format a GitHub issue as markdown content for chunking/search."""
    lines = [
        f"# Issue #{issue['number']}: {issue['title']}",
        "",
        f"**Repository:** {OWNER}/{repo}",
        f"**State:** {issue['state']}",
    ]

    labels = [lb["name"] for lb in issue.get("labels", [])]
    if labels:
        lines.append(f"**Labels:** {', '.join(labels)}")

    lines.append(f"**Created:** {issue['created_at']}")
    if issue.get("closed_at"):
        lines.append(f"**Closed:** {issue['closed_at']}")

    lines.extend(["", "## Description", ""])

    body = (issue.get("body") or "").strip()
    if body:
        lines.append(body)
    else:
        lines.append("*(no description)*")

    return "\n".join(lines)


def derive_entity_id(repo: str, number: int) -> str:
    """Derive a deterministic entity ID for a GitHub issue."""
    return f"{repo}-issue-{number}"


def build_entity(issue: dict, repo: str, classification=None) -> dict:
    """Build an entity dict from a GitHub issue."""
    entity_id = derive_entity_id(repo, issue["number"])
    content = format_issue_content(issue, repo)
    content_hash = hashlib.sha256(content.encode()).hexdigest()

    labels = [lb["name"] for lb in issue.get("labels", [])]

    created_at = issue["created_at"]
    closed_at = issue.get("closed_at")
    updated_at = issue.get("updated_at") or created_at

    # Normalise to naive UTC datetimes for psycopg
    def parse_dt(s: Optional[str]) -> Optional[datetime]:
        if not s:
            return None
        return datetime.fromisoformat(s.replace("Z", "+00:00")).replace(tzinfo=None)

    metadata: dict = {
        "$schema": "content_metadata_v1",
        "corpus": "deployment",
        "content_type": "issue",
        "media_type": "text",
        "language": "en",
        "word_count": len(content.split()),
        "reading_time_minutes": max(1, len(content.split()) // 200),
        "state": issue["state"],
        "labels": labels,
        "date_created": created_at[:10],  # YYYY-MM-DD
        "date_updated": (closed_at or updated_at)[:10],
    }

    if classification:
        from entity_builder import LLMClassification
        llm_dict = classification.to_dict()
        llm_dict.pop("concept_ownership", None)
        llm_dict.pop("content_type", None)  # preserve explicit content_type: issue
        metadata.update(llm_dict)

    filespec = {
        "$schema": "filespec_v1",
        "uri": f"github://{OWNER}/{repo}/issues/{issue['number']}",
        "hash": f"sha256:{content_hash}",
        "format": "markdown",
        "platform": "github",
        "accessible": True,
        "size_bytes": len(content.encode()),
        "last_checked": datetime.now(timezone.utc).isoformat(),
    }

    return {
        "id": entity_id,
        "entity_type": "content",
        "asset_type": "file",
        "title": f"Issue #{issue['number']}: {issue['title']}",
        "version": "1.0",
        "filespec": filespec,
        "attribution": ATTRIBUTION,
        "metadata": metadata,
        "created_at": parse_dt(created_at),
        "updated_at": parse_dt(updated_at),
    }


def classify_issue(content: str) -> Optional[object]:
    """LLM-classify issue content."""
    try:
        from llm_classifier import LLMClassifier
        classifier = LLMClassifier(model=LLM_MODEL)
        return classifier.classify(content)
    except Exception as e:
        console.print(f"[yellow]LLM classification failed: {e}[/yellow]")
        return None


def ingest_repo_issues(
    repo: str,
    conn: psycopg.Connection,
    openai_client,
    state: str = "all",
    no_llm: bool = False,
    dry_run: bool = False,
) -> dict:
    """Ingest all issues from a single repo."""
    console.print(f"\n[bold cyan]{repo}[/bold cyan]")

    issues = list_issues(OWNER, repo, state=state)
    if not issues:
        console.print("  No issues found.")
        return {"ingested": 0, "classified": 0, "chunks": 0, "edges": 0}

    console.print(f"  Found {len(issues)} issues")

    counts = {"ingested": 0, "classified": 0, "chunks": 0, "edges": 0}

    with Progress(SpinnerColumn(), TextColumn("{task.description}"), console=console, transient=True) as progress:
        task = progress.add_task("Processing...", total=len(issues))

        for issue in issues:
            progress.update(task, description=f"  #{issue['number']}: {issue['title'][:50]}...")

            content = format_issue_content(issue, repo)

            classification = None
            if not no_llm:
                classification = classify_issue(content)
                if classification:
                    counts["classified"] += 1

            entity = build_entity(issue, repo, classification)

            success = ingest_entity(entity, conn, dry_run=dry_run)
            if success:
                counts["ingested"] += 1

            if success and openai_client:
                try:
                    chunk_count = ingest_chunks(
                        entity_id=entity["id"],
                        content=content,
                        source_file=entity["filespec"]["uri"],
                        corpus="deployment",
                        content_type="issue",
                        conn=conn,
                        openai_client=openai_client,
                        dry_run=dry_run,
                    )
                    counts["chunks"] += chunk_count
                except Exception as e:
                    console.print(f"[yellow]  Chunking failed for #{issue['number']}: {e}[/yellow]")

            if success and not dry_run:
                try:
                    counts["edges"] += materialize_edges_neo4j(entity)
                except Exception as e:
                    console.print(f"[yellow]  Graph failed for #{issue['number']}: {e}[/yellow]")

    if not dry_run:
        conn.commit()

    console.print(
        f"  Ingested: {counts['ingested']} | "
        f"LLM: {counts['classified']} | "
        f"Chunks: {counts['chunks']} | "
        f"Edges: {counts['edges']}"
    )
    return counts


def main() -> int:
    parser = argparse.ArgumentParser(description="Ingest GitHub Issues into the deployment corpus")
    parser.add_argument("--repo", help="Single repo name (e.g. semops-data). Default: all 8 repos.")
    parser.add_argument("--state", choices=["open", "closed", "all"], default="all", help="Issue state filter")
    parser.add_argument("--no-llm", action="store_true", help="Skip LLM classification")
    parser.add_argument("--dry-run", action="store_true", help="Don't write to database")
    args = parser.parse_args()

    repos = [args.repo] if args.repo else DEPLOYMENT_REPOS

    console.print(f"[bold]GitHub Issues Ingestion[/bold]")
    console.print(f"Repos: {', '.join(repos)}")
    console.print(f"State: {args.state} | LLM: {'disabled' if args.no_llm else 'enabled'} | Dry-run: {args.dry_run}")

    conn = get_db_connection()

    openai_client = None
    if not args.dry_run:
        try:
            api_key = os.environ.get("OPENAI_API_KEY")
            if api_key:
                from openai import OpenAI
                openai_client = OpenAI(api_key=api_key)
        except Exception as e:
            console.print(f"[yellow]Warning: OpenAI init failed (no embeddings): {e}[/yellow]")

    totals = {"ingested": 0, "classified": 0, "chunks": 0, "edges": 0}

    for repo in repos:
        counts = ingest_repo_issues(
            repo=repo,
            conn=conn,
            openai_client=openai_client,
            state=args.state,
            no_llm=args.no_llm,
            dry_run=args.dry_run,
        )
        for k in totals:
            totals[k] += counts[k]

    if not args.dry_run:
        conn.commit()
    conn.close()

    console.print(f"\n[bold green]Total:[/bold green] "
                  f"Ingested={totals['ingested']} | "
                  f"LLM={totals['classified']} | "
                  f"Chunks={totals['chunks']} | "
                  f"Edges={totals['edges']}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
