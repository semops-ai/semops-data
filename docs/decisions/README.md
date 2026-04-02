# Architecture Decision Records (ADRs)

This directory contains Architecture Decision Records documenting significant technical and design decisions for ike-semantic-ops.

## Document Naming

| Pattern | Use Case | Example |
|---------|----------|---------|
| `ADR-NNNN.md` | Standalone architectural decisions | `ADR-0001.md` |
| `ISSUE-XX-DESCRIPTION.md` | GitHub issue-driven decisions | `ISSUE-47-CONCEPT-STRATEGY.md` |

## Document Statuses

- **Draft** - Under development, not yet approved
- **In Progress** - Approved and being implemented
- **Complete** - Fully implemented
- **Superseded** - Replaced by a newer decision (link to successor)

## Session Log Convention

For multi-session implementation work, ADRs include a **Session Log** section to track progress across sessions.

### When to Use

- Multi-phase implementations (e.g., Schema v2 rollout)
- Complex refactoring spanning multiple sessions
- Cross-repo coordination work
- Any ADR with an "Implementation Plan" section

### Benefits

- **Context Continuity** - Easy to resume after interruptions
- **Progress Visibility** - See what's been done at a glance
- **Handoff Support** - Others can pick up work
- **Decision History** - Why we approached things in certain order

### Format

```markdown
---

## Session Log

### YYYY-MM-DD: Brief Session Description
**Status:** In Progress | Completed | Blocked
**Tracking Issue:** [#XX Issue Title](issue-url)

**Completed:**
- Bullet list of what was accomplished
- Specific files/components worked on
- Key decisions made

**Next Session Should Start With:**
1. Clear next steps
2. Files to review
3. Dependencies to check
```

Each session entry should:
1. Record the date and a brief description
2. List completed work with specifics
3. Provide clear handoff instructions for the next session

## Creating a New ADR

1. Copy [TEMPLATE.md](./TEMPLATE.md)
2. Rename to match the naming convention
3. Fill in all relevant sections
4. Add Session Log entries as work progresses

## Current ADRs

| Document | Status | Description |
|----------|--------|-------------|
| [ADR-0001](./ADR-0001.md) | Proposed | Use Supabase as Global Knowledge DB |
| [ADR-0002](./ADR-0002.md) | In Progress | Simplified Content Data Entry |
| [ISSUE-47](./ISSUE-47-CONCEPT-STRATEGY.md) | Draft | Concept Entity Strategy |
| [ISSUE-48](./ISSUE-48-FOUNDATION-CATALOG.md) | Complete | Foundation Entity Catalog |

## Related

- [Semantic Operations](https://github.com/semops-ai/project-ike/blob/main/docs/SEMANTIC_OPERATIONS/SEMANTIC_OPERATIONS.md)
- [schemas/UBIQUITOUS_LANGUAGE.md](../../schemas/UBIQUITOUS_LANGUAGE.md)
