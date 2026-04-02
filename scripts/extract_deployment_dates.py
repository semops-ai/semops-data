#!/usr/bin/env python3
"""
Extract dates from deployment corpus entities (ADRs, session notes) and write
them back to entity.metadata as structured fields.

Date extraction is fully deterministic — no LLM needed:

  ADR (ADR-NNNN-*.md):    **Date:** or **Date**: YYYY-MM-DD  in chunk content
  ADR (YYYY-MM-DD-*.md):  date prefix in filename
  Session note:           YYYY-MM-DD headings in document_chunk.heading_hierarchy
                          first heading → date_created, last → date_updated
  All types:              YYYY-MM-DD prefix in filename URI (fallback)

Usage:
    python scripts/extract_deployment_dates.py           # Preview (dry run)
    python scripts/extract_deployment_dates.py --write   # Write to DB
    python scripts/extract_deployment_dates.py --csv     # Export summary CSV
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection

# --- Regex patterns ---
# Filename date prefix: 2024-11-24-some-title.md
RE_FILENAME_DATE = re.compile(r"^(\d{4}-\d{2}-\d{2})-")

# ADR content — two bold-date formats:
#   **Date**: YYYY-MM-DD   (colon outside **)
#   **Date:** YYYY-MM-DD   (colon inside **)
RE_ADR_DATE = re.compile(r"\*\*Date:?\*\*:?\s*(\d{4}-\d{2}-\d{2})")
RE_ADR_UPDATED = re.compile(r"\*\*Updated:?\*\*:?\s*(\d{4}-\d{2}-\d{2})")

# Date string that appears as a heading level in heading_hierarchy arrays
RE_DATE_STR = re.compile(r"^\d{4}-\d{2}-\d{2}$")


def load_entities(conn) -> pd.DataFrame:
    """Load deployment entities with chunk content and heading hierarchies."""
    # Entities with aggregated content
    entity_query = """
        SELECT
            e.id,
            e.title,
            e.metadata->>'content_type'   AS content_type,
            e.metadata->>'source_id'      AS source_id,
            e.filespec->>'uri'            AS uri,
            e.metadata->>'date_created'   AS existing_date_created,
            e.metadata->>'date_updated'   AS existing_date_updated,
            e.metadata->>'summary'        AS summary,
            string_agg(dc.content, E'\\n' ORDER BY dc.chunk_index) AS full_content
        FROM entity e
        LEFT JOIN document_chunk dc ON dc.entity_id = e.id
        WHERE e.entity_type = 'content'
          AND e.metadata->>'corpus' = 'deployment'
        GROUP BY e.id, e.title, e.metadata, e.filespec
        ORDER BY content_type, e.id
    """
    # Heading hierarchies per entity (for session note date extraction)
    # Use json_agg to avoid dimensionality conflicts between text[] arrays
    heading_query = """
        SELECT entity_id, json_agg(heading_hierarchy ORDER BY chunk_index) AS hierarchies
        FROM document_chunk
        WHERE corpus = 'deployment'
        GROUP BY entity_id
    """
    with conn.cursor() as cur:
        cur.execute(entity_query)
        rows = cur.fetchall()
        cols = [d.name for d in cur.description]
        df = pd.DataFrame(rows, columns=cols)

        cur.execute(heading_query)
        hrows = cur.fetchall()

    hdf = pd.DataFrame(hrows, columns=["id", "hierarchies"])
    return df.merge(hdf, on="id", how="left")


def extract_filename_date(uri: str) -> str | None:
    if not uri:
        return None
    filename = uri.split("/")[-1].replace(".md", "")
    m = RE_FILENAME_DATE.match(filename)
    return m.group(1) if m else None


def extract_adr_dates(content: str) -> tuple[str | None, str | None]:
    """Return (date_created, date_updated) from ADR chunk content."""
    created = None
    updated = None
    m = RE_ADR_DATE.search(content or "")
    if m:
        created = m.group(1)
    m = RE_ADR_UPDATED.search(content or "")
    if m:
        updated = m.group(1)
    return created, updated


def extract_heading_dates(hierarchies: list) -> tuple[str | None, str | None]:
    """Extract YYYY-MM-DD dates from heading_hierarchy arrays across all chunks.

    Session notes use ## YYYY-MM-DD as section headings, captured in
    heading_hierarchy by the chunker. Returns (first_date, last_date).
    """
    if not hierarchies:
        return None, None
    dates: list[str] = []
    for hierarchy in hierarchies:
        if not hierarchy:
            continue
        for level in hierarchy:
            if isinstance(level, str) and RE_DATE_STR.match(level):
                if not dates or level != dates[-1]:  # deduplicate consecutive
                    dates.append(level)
    if not dates:
        return None, None
    return dates[0], dates[-1]


def extract_dates(row: pd.Series) -> pd.Series:
    """Apply extraction rules for a single entity row."""
    ct = row["content_type"]
    content = row["full_content"] or ""
    uri = row["uri"] or ""
    hierarchies = row.get("hierarchies") or []

    filename_date = extract_filename_date(uri)
    date_created = None
    date_updated = None

    if ct == "adr":
        adr_created, adr_updated = extract_adr_dates(content)
        date_created = adr_created or filename_date
        date_updated = adr_updated
    elif ct == "session_note":
        first, last = extract_heading_dates(hierarchies)
        date_created = first or filename_date
        date_updated = last if last and last != first else None
    else:
        date_created = filename_date

    return pd.Series({"date_created": date_created, "date_updated": date_updated})


def write_dates(conn, df: pd.DataFrame) -> int:
    """Write date_created / date_updated back to entity.metadata JSONB."""
    updated_count = 0
    with conn.cursor() as cur:
        for _, row in df.iterrows():
            if not row["date_created"] and not row["date_updated"]:
                continue
            patch = {}
            if row["date_created"] and not pd.isna(row["date_created"]):
                patch["date_created"] = row["date_created"]
            if row["date_updated"] and not pd.isna(row["date_updated"]):
                patch["date_updated"] = row["date_updated"]
            cur.execute(
                "UPDATE entity SET metadata = metadata || %s::jsonb WHERE id = %s",
                (json.dumps(patch), row["id"]),
            )
            updated_count += 1
    conn.commit()
    return updated_count


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract dates from deployment corpus entities")
    parser.add_argument("--write", action="store_true", help="Write extracted dates to DB")
    parser.add_argument("--csv", metavar="FILE", help="Export summary to CSV")
    args = parser.parse_args()

    conn = get_db_connection()
    df = load_entities(conn)
    print(f"Loaded {len(df)} deployment entities")

    # Extract dates
    dates = df.apply(extract_dates, axis=1)
    df["date_created"] = dates["date_created"]
    df["date_updated"] = dates["date_updated"]

    # Coverage stats
    with_created = df["date_created"].notna().sum()
    with_updated = df["date_updated"].notna().sum()
    no_date = df["date_created"].isna().sum()

    print(f"\nExtraction results:")
    print(f"  date_created found:  {with_created}/{len(df)}")
    print(f"  date_updated found:  {with_updated}/{len(df)}")
    print(f"  No date found:       {no_date}")

    # Show sample
    print("\nSample (first 10 with dates):")
    sample = df[df["date_created"].notna()][
        ["id", "content_type", "date_created", "date_updated"]
    ].head(10)
    print(sample.to_string(index=False))

    # Show missing
    missing = df[df["date_created"].isna()][["id", "content_type", "uri"]]
    if not missing.empty:
        print(f"\nEntities with no date extracted ({len(missing)}):")
        print(missing.to_string(index=False))

    if args.csv:
        out = df[["id", "title", "content_type", "source_id", "uri", "date_created", "date_updated", "summary"]]
        out.to_csv(args.csv, index=False)
        print(f"\nExported to {args.csv}")

    if args.write:
        count = write_dates(conn, df)
        print(f"\nWrote date metadata for {count} entities")
    else:
        print("\n[Dry run] Use --write to persist to DB")

    conn.close()


if __name__ == "__main__":
    main()
