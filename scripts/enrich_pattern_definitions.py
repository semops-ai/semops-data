"""
Enrich pattern definitions in the database with substantive descriptions.

Replaces label-echo definitions (where definition = preferred_label or
definition = "label (attribution)") with real definitions that explain:
- 3P patterns: what the standard provides and why you'd adopt it
- 1P patterns: what it innovates and what 3P foundations it synthesizes

Run with: python scripts/enrich_pattern_definitions.py [--dry-run]

Issue: semops-data#145 (Pattern Refinement)
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from db_utils import get_db_connection

# ---------- 3P Pattern Definitions ----------

DEFINITIONS_3P = {
    "agentic-rag": (
        "Retrieval-Augmented Generation enhanced with autonomous agent reasoning, "
        "where agents dynamically plan retrieval strategies rather than following "
        "fixed pipelines."
    ),
    "arc42": (
        "Template-based software architecture documentation providing a standardized "
        "structure for describing system building blocks, runtime behavior, and "
        "cross-cutting concepts."
    ),
    "backstage-software-catalog": (
        "Developer portal software catalog that models entities, their lifecycle "
        "stages, and ownership relationships. Provides the lifecycle model "
        "(planned, draft, in_progress, active, retired) adopted by SemOps."
    ),
    "business-domain": (
        "Industry-standard classification systems (NIST, NAICS, KIBO) for "
        "categorizing business domains. Provides the vocabulary for domain "
        "classification in reference architectures."
    ),
    "cms": (
        "Content management systems for creating, editing, and publishing digital "
        "content with editorial workflows and version control."
    ),
    "data-lineage": (
        "Tracking the flow and transformation of data through processing systems, "
        "providing visibility into where data comes from, how it changes, and where "
        "it goes."
    ),
    "data-modeling": (
        "Structured representation of data elements, their relationships, and "
        "constraints. Foundation for schema design, entity-relationship modeling, "
        "and data dictionary creation."
    ),
    "data-profiling": (
        "Statistical analysis of data to understand its structure, quality, "
        "completeness, and distribution. Foundation for data quality governance "
        "and anomaly detection."
    ),
    "dam": (
        "Digital asset lifecycle management covering ingestion, storage, approval "
        "workflows, multi-channel distribution, and version control for published "
        "content."
    ),
    "ddd": (
        "Software design approach centered on rich domain models, bounded contexts, "
        "ubiquitous language, and aggregate boundaries. The primary architecture "
        "pattern for SemOps."
    ),
    "dublin-core": (
        "Metadata standard for describing digital resources using fifteen core "
        "elements including creator, title, date, rights, and publisher. Adopted "
        "for content entity attribution."
    ),
    "edge-predicates": (
        "Typed semantic relationships between entities using standardized predicates "
        "(cites, derived_from, version_of, implements). Provides the vocabulary for "
        "the edge table."
    ),
    "episode-provenance": (
        "Extension of W3C PROV-O that groups agent actions into episodes — bounded "
        "sequences of activities with context, intent, and outcomes. Foundation for "
        "agentic lineage tracking."
    ),
    "etl": (
        "Extract-Transform-Load pipeline pattern for moving data between systems "
        "with transformation logic applied during transit."
    ),
    "medallion-architecture": (
        "Staged data quality pattern (Bronze, Silver, Gold) where each tier applies "
        "increasing levels of curation, validation, and enrichment."
    ),
    "open-lineage": (
        "Open standard for collecting and analyzing data lineage metadata across "
        "heterogeneous systems, providing a common event model for pipeline "
        "observability."
    ),
    "pattern-language": (
        "Interconnected system of design patterns where each pattern addresses a "
        "specific problem in context and links to related patterns. Foundation for "
        "semantic object pattern management."
    ),
    "platform-engineering": (
        "Internal developer platforms that provide self-service infrastructure, "
        "standardized tooling, and golden paths for development teams."
    ),
    "pim": (
        "Product information management for centralizing and governing product data "
        "(descriptions, attributes, media) across channels and touchpoints."
    ),
    "prov-o": (
        "W3C standard ontology for representing provenance as relationships between "
        "entities, activities, and agents. Adopted for content lineage tracking "
        "(derived_from, cites, version_of)."
    ),
    "raptor": (
        "Recursive abstractive processing that builds tree-organized summaries at "
        "multiple abstraction levels, enabling retrieval at the right level of "
        "detail for a given query."
    ),
    "rlhf": (
        "Training AI models using human feedback as a reward signal. In SemOps, "
        "applied to scale projection: human-in-the-loop processes generate "
        "structured training data for style learning."
    ),
    "seci": (
        "Knowledge creation model describing how knowledge transforms between tacit "
        "and explicit forms through socialization, externalization, combination, and "
        "internalization. Applied to scale projection and style learning."
    ),
    "shape": (
        "Data structure validation using constraint languages (W3C SHACL, JSON "
        "Schema) to define and enforce expected data shapes. Applied to JSONB "
        "schema versioning in the domain model."
    ),
    "skos": (
        "W3C standard for representing concept taxonomies using broader, narrower, "
        "and related relationships. Adopted for the pattern hierarchy — every "
        "pattern-to-pattern edge uses SKOS predicates."
    ),
    "zettelkasten": (
        "Note-linking knowledge management method where atomic ideas are connected "
        "through explicit links, enabling emergence of structure through association "
        "rather than imposed hierarchy."
    ),
}

# ---------- 1P Pattern Definitions ----------

DEFINITIONS_1P = {
    "semantic-coherence": (
        "Measurable signal that quantifies how well a system's reality matches its "
        "intended domain model, defined as SC = (Availability * Consistency * "
        "Stability)^(1/3). The objective function of the Semantic Optimization Loop "
        "— Pattern sets the target, Coherence measures the gap."
    ),
    "semantic-ingestion": (
        "Ingestion pipeline where every byproduct — classifications, detected edges, "
        "coherence scores, embeddings — becomes a queryable knowledge artifact rather "
        "than being discarded. Synthesizes ETL and Medallion Architecture with "
        "domain-aware semantic enrichment."
    ),
    "agentic-lineage": (
        "Lineage tracking extended with agent decision context and trust provenance, "
        "recording not just what happened but why an agent chose that action. "
        "Synthesizes OpenLineage and Episode Provenance with agentic reasoning "
        "metadata."
    ),
    "semantic-object-pattern": (
        "Patterns treated as first-class semantic objects — provenance-tracked, "
        "lineage-measured, AI-agent-usable domain concepts with SKOS taxonomy and "
        "adoption history. The aggregate root itself is the 1P innovation. "
        "Synthesizes Pattern Language with knowledge organization systems."
    ),
    "scale-projection": (
        "Validates domain coherence by projecting architecture to scale. Manual "
        "human-in-the-loop processes intentionally generate structured ML training "
        "data, creating a path from current-state to autonomous execution. "
        "Synthesizes RLHF, SECI, and data profiling with domain-model-aware scaling."
    ),
    "explicit-enterprise": (
        "Enterprise systems where architecture, data, and AI are treated as "
        "first-class, agent-addressable concerns. Humble tools (inbox, financial "
        "ledger) become signal streams that agents can reason about. Synthesizes "
        "Platform Engineering with agent-first system design."
    ),
    "explicit-architecture": (
        "Making architectural decisions, patterns, and their rationale explicit, "
        "queryable, and traceable rather than implicit in code or tribal knowledge. "
        "Extends DDD with formalized architecture documentation and pattern "
        "traceability."
    ),
    "mirror-architecture": (
        "Architecture that mirrors domain structure in system structure — "
        "organizational boundaries, team structures, and code boundaries reflect "
        "the domain model. Extends DDD strategic design with explicit structural "
        "mirroring."
    ),
    "provenance-first-design": (
        "Design principle where every entity carries provenance classification "
        "(1P original, 2P derivative, 3P third-party standard) as a first-class "
        "attribute, enabling trust assessment and licensing governance. Extends "
        "PROV-O with provenance tiers."
    ),
    "derivative-work-lineage": (
        "Tracking the lineage chain of derivative works — how original content "
        "transforms through AI composition, human editing, and republication. "
        "Extends PROV-O with derivative-specific provenance for content ownership "
        "and attribution."
    ),
    "unified-catalog": (
        "Single catalog that treats owned content and referenced external content "
        "as peers with different provenance, enabling seamless discovery across both. "
        "Extends DAM and SKOS to unify owned assets with curated external references."
    ),
}


def main():
    dry_run = "--dry-run" in sys.argv
    all_definitions = {**DEFINITIONS_3P, **DEFINITIONS_1P}

    if dry_run:
        print("=== DRY RUN — no changes will be made ===\n")
        for pattern_id, definition in sorted(all_definitions.items()):
            print(f"  {pattern_id}:")
            print(f"    {definition}\n")
        print(f"Total: {len(all_definitions)} patterns")
        return

    conn = get_db_connection()
    try:
        with conn.cursor() as cur:
            updated = 0
            skipped = 0
            missing = 0
            for pattern_id, definition in sorted(all_definitions.items()):
                # Check pattern exists
                cur.execute(
                    "SELECT definition FROM pattern WHERE id = %s",
                    (pattern_id,),
                )
                row = cur.fetchone()
                if row is None:
                    print(f"  MISSING: {pattern_id} — not in database")
                    missing += 1
                    continue

                old_def = row[0]
                if old_def == definition:
                    print(f"  SKIP: {pattern_id} — already enriched")
                    skipped += 1
                    continue

                cur.execute(
                    "UPDATE pattern SET definition = %s WHERE id = %s",
                    (definition, pattern_id),
                )
                print(f"  UPDATE: {pattern_id}")
                print(f"    OLD: {old_def}")
                print(f"    NEW: {definition[:80]}...")
                updated += 1

        conn.commit()
        print(f"\nDone: {updated} updated, {skipped} already enriched, {missing} missing")
    finally:
        conn.close()


if __name__ == "__main__":
    main()
