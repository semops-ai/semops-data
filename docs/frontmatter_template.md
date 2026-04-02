# Frontmatter Template for Markdown Docs

Use this template when adding explicit metadata to your canonical pillar docs.

## Minimal Example (Recommended - Start Here)

```yaml
---
# === Identity ===
entity_id: semantic-coherence
# Custom entity ID (optional - auto-generated if not provided)

# === Core Attributes ===
title: Semantic Coherence
# Override auto-extracted title (optional)

description: >
  The degree of shared meaning and mutual understanding within a semantic context.

# === Classification ===
category: semantic-operations

tags:
  - semantic-coherence
  - semantic-operations
  - knowledge-management

# === Provenance & Status ===
provenance: 1p
# 1p (your content) | 2p (collaborative) | 3p (external)

visibility: public
# public | private

approval_status: approved
# pending | approved | rejected

# === Content Metadata (SKOS - Core Fields Only) ===
metadata:
  semantic_type: concept
  # Available: concept | framework | methodology | definition | principle | pattern | guide

  primary_concept: semantic-coherence
  # Main concept this content addresses (entity_id reference)

  abstraction_level: intermediate
  # foundational | intermediate | advanced

  broader_concepts:
    - semantic-operations
  # W3C SKOS: skos:broader - Parent concepts (use entity_id slugs)

  narrower_concepts:
    - semantic-drift
    - coherence-audit
  # W3C SKOS: skos:narrower - Child concepts (use entity_id slugs)

# === Relationships (W3C PROV-O based) ===
relationships:
  - predicate: part_of
    target_file: SEMANTIC_OPERATIONS.md
    strength: 1.0

  - predicate: depends_on
    target_file: DIKW.md
    strength: 0.7
---
```

**That's it!** Start with these core fields. Add more SKOS fields (below) only when needed.

---

## Full Example (Advanced - Use When Needed)

```yaml
---
# === Identity ===
entity_id: semantic-operations-intro
# Custom entity ID (optional - auto-generated UUID if not provided)
# Use kebab-case, descriptive names

# === Core Attributes ===
title: Introduction to Semantic Operations
# Override auto-extracted title (optional - uses H1 or filename if not provided)

description: >
  A comprehensive introduction to semantic operations, covering first principles,
  the DIKW hierarchy, and practical applications in knowledge management.
# Custom description (optional - uses first paragraph if not provided)

# === Classification ===
category: semantic-operations
# Available: first-principles | semantic-operations | ai-transformation |
#            architecture | data-systems | examples | uncategorized

tags:
  - semantic-operations
  - dikw
  - knowledge-ops
  - ddd
# Custom tags (optional - auto-derived from content if not provided)

# === Content Metadata (W3C SKOS-based) ===
metadata:
  semantic_type: framework
  # Available: concept | framework | methodology | definition | principle | pattern | guide

  primary_concept: semantic-operations
  # Main concept this content addresses (entity_id reference)

  abstraction_level: intermediate
  # foundational | intermediate | advanced

  broader_concepts:
    - knowledge-management
  # W3C SKOS: skos:broader - More general/parent concepts (use entity_id slugs)

  narrower_concepts:
    - semantic-coherence
    - knowledge-promotion
  # W3C SKOS: skos:narrower - More specific/child concepts (use entity_id slugs)

  # === Optional SKOS Fields (add only when needed) ===

  preferred_label: "Semantic Operations"
  # W3C SKOS: skos:prefLabel - Use if different from title

  alt_labels:
    - "SemOps"
    - "Semantic Content Operations"
  # W3C SKOS: skos:altLabel - Alternative labels and synonyms

  definition: >
    A methodology for managing knowledge assets through semantic relationships
    and graph-based organization.
  # W3C SKOS: skos:definition - Formal definition (or put in content)

  related_concepts:
    - domain-driven-design
    - knowledge-graphs
  # W3C SKOS: skos:related - Use edges instead when possible

  scope_note: "Use this framework when..."
  # W3C SKOS: skos:scopeNote - Usage context

  example: "A team using semantic operations to..."
  # W3C SKOS: skos:example - Example usage

# === Provenance & Status ===
provenance: 1p
# 1p (your content) | 2p (collaborative) | 3p (external)

visibility: internal
# internal (private) | public

approval_status: approved
# pending | approved | rejected

# === Relationships (W3C PROV-O based) ===
# Define provenance relationships BETWEEN entities
# Based on W3C PROV-O (Provenance Ontology): https://www.w3.org/TR/prov-o/
relationships:
  # Example 1: Reference by target_id (if you know the entity_id)
  - predicate: documents
    target_id: dikw-hierarchy-entity
    strength: 1.0
    # Strength: 0.0-1.0, where 1.0 is strongest

  # Example 2: Reference by filename (resolved during ingestion)
  - predicate: cites  # prov:wasQuotedFrom
    target_file: DIKW.md
    strength: 0.8

  # Example 3: Derived work (provenance lineage)
  - predicate: derived_from  # prov:wasDerivedFrom
    target_file: semantic_first_principles.md
    strength: 0.9

  # Example 4: Related concept
  - predicate: related_to
    target_file: Domain_Driven_Design.md
    strength: 0.6

# Available predicates (W3C PROV-O based):
#   derived_from - W3C PROV-O: prov:wasDerivedFrom (transformation/extraction)
#   cites        - W3C PROV-O: prov:wasQuotedFrom (references/attribution)
#   version_of   - W3C PROV-O: prov:wasRevisionOf (version succession)
#   part_of      - Schema.org: schema:isPartOf (compositional hierarchy)
#   documents    - Schema.org: schema:about inverted (explanation/documentation)
#   depends_on   - Project Ike: logical dependency (prerequisite knowledge)
#   related_to   - W3C SKOS: skos:related (conceptual association)
---

# Your Content Starts Here

This is where your actual markdown content begins.
The frontmatter above will be parsed and merged with auto-derived attributes.

## How It Works

1. **Frontmatter takes precedence** - If you define it, it overrides auto-detection
2. **Auto-derived as fallback** - Missing fields are auto-generated from content
3. **Relationships resolved in Pass 2** - After all entities are loaded

## Minimal Example

If you only want to define relationships:

```yaml
---
relationships:
  - predicate: cites
    target_file: lineage-provenance-bp.md
    strength: 0.9
---
```

Everything else will be auto-derived!

## Available Predicates (W3C PROV-O Based)

| Predicate      | W3C Standard | Meaning                                      | Example                          |
|----------------|--------------|----------------------------------------------|----------------------------------|
| `derived_from` | PROV-O `prov:wasDerivedFrom` | Transformation of source | Transcript → YouTube video |
| `cites`        | PROV-O `prov:wasQuotedFrom` | References/attributes source | Blog post → Research paper |
| `version_of`   | PROV-O `prov:wasRevisionOf` | Different version | v2.0 → v1.0 |
| `part_of`      | Schema.org `schema:isPartOf` | Component relationship | Chapter → Book |
| `documents`    | Schema.org `schema:about` (inverted) | Provides documentation | Guide → Software component |
| `depends_on`   | Project Ike extension | Logical dependency | Advanced guide → Basic tutorial |
| `related_to`   | SKOS `skos:related` | Semantically related | DDD concept → DIKW framework |

See [EDGE_PREDICATES.md](./EDGE_PREDICATES.md) for complete documentation.

## Relationship Strength Guidelines

- **1.0** - Direct, essential relationship (transcript derives from video)
- **0.8-0.9** - Strong relationship (primary citation, core dependency)
- **0.5-0.7** - Moderate relationship (supporting citation, related concept)
- **0.3-0.4** - Weak relationship (tangential reference)

## Tips

1. **Start simple** - Add frontmatter only where relationships matter
2. **Use `target_file`** - Easier than tracking entity IDs manually
3. **Run `--dry-run`** first to preview what will be ingested
4. **Define relationships** for your pillar docs to build the knowledge graph
