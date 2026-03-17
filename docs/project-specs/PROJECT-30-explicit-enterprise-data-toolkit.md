# PROJECT-30: Explicit Enterprise Data Toolkit

> **Project:** [@explicit-enterprise-data-toolkit](https://github.com/users/semops-ai/projects/30)
> **Status:** Active
> **Parent Issue:** 

## Outcome

Build an architecture-first data toolkit and deploy it as SemOps' own analytics data system — separate from runtime infrastructure. Ingest existing data sources (PostgreSQL, Qdrant, Python dicts, YAML configs) into a proper analytical layer. Provide historical data for scale projection and semantic coherence scoring. Develop new metrics. Ingest 3rd-party sources (Google Analytics, etc.). The toolkit is both the product (for generalists) and the proof (running on SemOps).

## Acceptance Criteria

### Toolkit (Product)
- [x] Decision matrix connects domain understanding → architecture → infrastructure in a single decision flow
- [ ] At least 3 strategy playbooks ("I want X without the expensive way") following the decision matrix
- [ ] At least 2 industry verticals with synthetic datasets generated from domain models
- [ ] At least 1 runnable reference implementation proving the toolkit end-to-end
- [ ] Narrative packaging: at least one publishable piece for the target audience (generalists)

### SemOps Deployment (Proof)
- [ ] Analytics layer exists, separate from runtime infrastructure (PostgreSQL, Qdrant, Neo4j)
- [ ] Existing SemOps data sources ingested (operational DBs, Python dicts, YAML configs)
- [ ] Historical data available for scale projection and semantic coherence scoring
- [ ] At least 1 new metric developed on the analytical layer
- [ ] At least 1 3rd-party source ingested (e.g., Google Analytics)

## Execution Sequence

| # | Step | Issue(s) | Requires | Delivers |
|---|------|----------|----------|----------|
| 1 | Decision matrix: unify frameworks into single architecture-before-infrastructure flow |  |  research (done) | Architecture checklist a generalist can follow |
| 2 | Strategy playbooks: "I want X without the expensive way" guides |  | Step 1 | Practical guides for BI, pipeline, governance |
| 2 | Company domain modeling for playbook examples |  | Step 1 | Industry domain models with DDD mapping |
| 3 | Synthetic data generation from domain models |  | Step 2 ( domain models) | Realistic test datasets per industry vertical |
| 4 | Reference implementations with synthetic data |  | Step 3 | Proof the toolkit works end-to-end |
| 5 | SemOps analytics layer: ingest operational data into analytical substrate | TBD | Steps 1-2 (architecture decided) | Runtime → analytics separation, historical data for coherence/scale-projection |
| 5 | 3rd-party ingestion (Google Analytics, etc.) | TBD | Step 5 (analytics layer exists) | External data feeding into analytical layer |
| 6 | New metrics development on analytical layer | TBD | Step 5 | Metrics beyond coherence scoring and scale projection |
| 7 | Narrative packaging for publication |  | Steps 2-6 | "Explicit Enterprise Data Toolkit" as coherent offering |

**Parallel work:** Steps at the same sequence number can run concurrently. Step 2 has two parallel tracks (playbooks + domain modeling). Step 5 has two parallel tracks (SemOps ingestion + 3rd-party sources).

## Child Issues

| Issue | Repo | Scope | Status |
|-------|------|-------|--------|
|  | semops-data | Parent issue — vision, phasing, core argument | Open |
|  | semops-data | DDD → data architecture research (foundation) | Done |
|  | semops-data | Decision matrix: unified architecture-before-infrastructure framework | Done |
|  | semops-data | Strategy playbooks | Open |
|  | semops-data | Company domain modeling analysis | Done |
|  | semops-data | Synthetic data generation from domain models | Open |
|  | semops-data | Reference implementations with synthetic data | Open |
| TBD | semops-data | SemOps analytics layer: runtime → analytics separation | Planned |
| TBD | semops-data | 3rd-party ingestion (Google Analytics, etc.) | Planned |
| TBD | semops-data | New metrics development | Planned |
|  | semops-data | Open Source Data Hub (lineage/catalog layer — related) | Open |
|  | semops-publisher | Narrative packaging for publication | Open |

## Dependencies

| Dependency | Source | Status | Blocks |
|------------|--------|--------|--------|
| DDD → data architecture research |  | Done | Step 1 |
| Strategic Data frameworks (four-types, surface analysis, vendor comparison) | docs-pr | Ready | Step 1 |
| DDD validation doc | docs-pr | Done | Step 1 |
| Decision matrix |  | Done | Steps 2, 5 |
| Company domain modeling |  | Done | Step 3 |
| SemOps runtime infrastructure (PostgreSQL, Qdrant, Neo4j) | semops-core | Active | Step 5 (data sources) |

## References

- **Research foundation:** [Data Architecture Through the DDD Lens](https://github.com/semops-ai/semops-data/blob/main/docs/research/ddd-data-architecture.md)
- **Framework validation:** [Data System Classification DDD Validation](https://github.com/semops-ai/docs-pr/blob/main/docs/SEMOPS_DOCS/SEMANTIC_OPERATIONS_FRAMEWORK/STRATEGIC_DATA/data-system-classification-ddd-validation.md)
- **Intake evaluation:**  — Governance mode, 2026-02-22
- **Related Projects:**  (predecessor — accumulated work that fed into this)
- **Key Docs:**
  - `docs-pr/.../STRATEGIC_DATA/data-system-classification.md`
  - `docs-pr/.../STRATEGIC_DATA/surface-analysis.md`
  - `docs-pr/.../EXPLICIT_ARCHITECTURE/what-is-architecture.md`
  - `semops-data/docs/research/ddd-data-architecture.md`
