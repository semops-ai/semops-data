# Domain-Driven Design Architecture
This document outlines the Domain-Driven Design (DDD) architecture for the Ike Knowledge Ops platform, explaining the core aggregates, their relationships, and how they align with best practices in content and product management systems. Our hypothesis is that DDD is a strong fit for organizations that are focused on treating data and knowledge as first-class citizens in order to truly benefit from AI and modern software systems.

# Ike vs. Broader Adoption
Ike is my testing ground leveraging data and AI in business with a system that is optimized to do just that.  Our AI First and Data First principles, combine DIKW, DDD, Knowledge Operations, and other core principles and guardrails. The repos, tools, and software built wil will coordinate components of content management, publishing, advertising, analytics, CRM, product information management (PIM), digital asset management (DAM) systems, while also incorporating lessons from LLM research and AI ops. We will also apply real-world learning from consulting clients and other pet projects to refine and validate the model.


  
# DDD Architecture v1 — Small Core with flexible edges
Goal: a tiny, stable core that everything snaps into; flexible edges (typed metadata, projections); simple public/private auth; graph-aware RAG; clear schema governance. Phase 2 adds lightweight CRM on top—without bloating the core.
## Scope & Principles

- **Small stable core**: five aggregates — `DomainEntity`, `Item`, `Surface`, `Delivery`, `Edge`.
- **Provenance-first**: model “came from my factory?” explicitly (`1p' | '2P' |'3p' `).
- **Visibility is simple**: `public | private` on `Item`.
- **Typed metadata**: JSON sidecars with **discriminator + version**; promote to columns only when needed.
- **Graph + Vector**: use edges for relationships; re-rank vector hits with graph context.
- **UL & Governance**: Ubiquitous Language doc; treat enum/field changes as **product changes** with SemVer.

##  Ubiquitous Language (snapshot)

### Phase 1
DomainEntity — durable “thing” in the catalog. Not a URL.
Item — concrete artifact/content you produce or store; carries visibility, provenance.
Provenance — origin: 1p | 2p | 3p | 
Visibility — public | private.
Keep [UBIQUITOUS_LANGUAGE.md] beside migrations; update with every schema release.

### Phase 2
Surface — addressable place; has addresses; direction ∈ publish/ingest/bidirectional.
Delivery — record of Item on a Surface (original|syndication).
Edge — typed relation among nodes (documents / derived_from / depends_on / contains / delivered_to).
Steward — current responsible party: us | external.

## 3) Schema Governance

SemVer the schema (store current in a tiny schema_version table + SCHEMA_CHANGELOG.md).

MAJOR: breaking enum/field meaning changes or removals.
MINOR: additive (new enum, new nullable field).
PATCH: docs/constraints fixes.
PR checklist (must answer):
What fields/enums changed? Backward compat? Defaults/migrations?
Impacted adapters (Surfaces), prompts/agents, analytics, docs.
Rollout plan (dual-read/write? feature flag?) + deprecation date.
Tests updated (contract tests for affected Surfaces).
Fitness functions (CI):
Private→Public publish is blocked.
One original Delivery per Item.
derived requires at least one derived_from edge.
Enum removals require MAJOR bump.

## 1 Core Model (Phase 1)

### 1.1 Entities

**DomainEntity** — “thing in the catalog”  
Examples: application, repo, service, dataset, feature, initiative, concept.

Sample Data for Ike:

```sql
create table domain_entity (
  id           text primary key,
  kind         text not null,                              -- application | service | repo | api | dataset | feature | initiative | concept | topic
  name         text not null,
  lifecycle    text default 'active',                      -- planned | active | sunset
  provenance   text not null default '1p'         -- first_party | third_party | derived | partner
               check (provenance in ('1p','2p','3p')),
  steward      text default 'us'                           -- us | external (optional but useful)
               check (steward in ('us','external')),
  attribution  jsonb not null default '{}',                -- typed: attribution_v1 (sources, license)
  metadata     jsonb not null default '{}'                 -- typed metadata
);
```
Item — a concrete thing you create/store/publish (one type covers “knowledge” and “product”).
Examples: article, video, pdf, container image, binary, dataset file, note.
```sql
create table item (
  id            text primary key,
  content_kind  text not null,                             -- article | video | pdf | container | binary | image | dataset | note | ...
  title         text,
  version       text,
  visibility    text not null check (visibility in ('public','private')),
  status        text not null default 'draft'              -- draft | published | archived
               check (status in ('draft','published','archived')),
  provenance    text not null default '1p'
               check (provenance in ('1p','2p','derived','3p')),
  filespec      jsonb not null default '{}',               -- where bits live (URIs, checksums)
  metadata      jsonb not null default '{}',               -- typed metadata
  created_at    timestamptz not null default now,
  published_at  timestamptz
);
```
### Stable core, flexible edge

Core (relational, versioned):
DomainEntity, Representation (knowledge/product), Surface, Delivery, Aboutness, Identity.
Edge projections (polyglot):
Vector index over (Representations + selected DomainEntity fields) for retrieval.
Analytics warehouse (star schema) for Delivery outcomes and costs.
Search index (denormalized) for UI.
Treat edge schemas as throwaway/derived; only the core is sacred.

## 1.2 Invariants & Policies

Visibility: item.visibility ∈ {public, private}. Publishing a private Item to a public Surface is blocked.

## Provenance rules:

1P Items may be role='original' on owned Surfaces.
3P Items cannot be marked original; require attribution if public.
All items may have sub-items. For example, we might have a 3p video, and then we have a pull-quote with most of the same properties and linked IDs except 
derived Items must have at least one edge(item -> X, 'derived_from').
Original uniqueness: at most one Delivery.role='original' per Item.
Published immutability: once status='published', freeze write-once fields (e.g., content_kind, version, provenance).

## 1.3 Typed Metadata (pattern)

JSON sidecar with type + version + fields; validate with JSON Schema/Zod on write.
```json
// item.metadata for a video
{ "type": "video_v1", "version": 1, "duration_sec": 482, "captions": true, "source": "premiere" }

// item.metadata for container image
{ "type": "container_v1", "version": 1, "image_ref": "ghcr.io/acme/app:1.4.0", "digest": "sha256:..." }

// domain_entity.attribution (for third/derived)
{ "type":"attribution_v1","version":1,"sources":[{"title":"OpenAPI 3.1","uri":"https://..."}],"license":{"name":"Apache-2.0","uri":"https://..."} }
```
Promotion rule: promote a metadata field to a column only if (a) used across ≥2 workflows or (b) needs indexing/joining.

# Futre Phases
## Phase 2: Content and Publishing, Edge

The following is the DDD design for the public facing products we develop with Ike.

- **DomainEntity** → industry analogs: *Catalog Item / Entity / Product / Service / SoftwareApplication / Repository*. In DDD it’s your aggregate root. In PIM it’s the Product/Variant tree. In schema.org it’s `Thing` (often `Product`, `Service`, `SoftwareApplication`, `Dataset`, etc.).
- **Representation** → *Asset / Entry / Artifact / Release*. In DAM this is the asset; in package registries it’s the artifact; in CMS it’s the entry/document; in CD systems it’s the build/deployment.
    - Versioned & immutable once published = best practice across all of these.
- **Surface** → most vendors say *Channel / Destination / Publication / Property*. CDPs (e.g., Segment) say “Destination,” PIMs (Akeneo, Shopify) say “Channel,” marketing suites say “Channel/Property,” CMSs talk “Environments/Channels.” Your “Surface” is a clean, neutral umbrella.
- **Delivery** (Representation ↔ Surface) → *Placement / Distribution / Publication / Deployment*. This join/process is standard everywhere.
- **Aboutness** edges → *References/Relations/Topics/SubjectOf*. Every modern content system has typed relations; making it explicit is best practice.

### Best-practice Patterns

- **Identity ≠ address:** keep platform-native IDs and treat URLs as changeable value objects. (DAM/PIM/CMS norm.)
- **Many-to-many by design:** one Representation → many Surfaces; one DomainEntity ← many Representations. (Essential for omnichannel.)
- **Versioned Representations:** new version = new record; don’t mutate history. (Release & editorial workflow standard.)
- **Surface capabilities & constraints:** model “what this channel supports” rather than baking rules into code. (Common in headless CMS & PIM.)
- **Event-first pipeline:** Delivery planned/queued/published/failed events → idempotent, retryable workers. (DevOps/content ops best practice.)
- **Typed relations (“Aboutness”):** keeps your knowledge graph queryable (analytics, docs, changelogs, etc.).

### If you want to align naming with the broader ecosystem

Keep your structure, swap labels only if it helps onboard teammates:

- **Surface → Channel** (most familiar to marketers/PIM folks)
- **Delivery → Placement** (common in adtech/CMS)
- **Representation → Asset** (DAM) or **Entry** (CMS)
- **DomainEntity → CatalogItem** (business-friendly) or keep **DomainEntity** (DDD teams love it)

## Event-first spine

Use an append-only event log as the source of truth for change; materialize tables from it.
Events: RepresentationCreated, RepresentationPublished, DeliveryQueued/Published/Failed, SurfaceAddressChanged, IngestionDiscovered, AgentRunCompleted.
Benefits: replayable state, easier backfills, and agents can subscribe to domain events instead of polling.

Wrap it in practice:

* Codegen adapters from JSON Schemas (Surface mappings, Tool specs).
* Contract tests per Surface (golden fixtures).
* Data QA checks (freshness, completeness) as CI on data.
* Backfill/evolve: event log + reproducible materializers.
* Docs-as-schema: keep a living “Ubiquitous Language” doc next to the schema; treat PRs that change enums/fields as product changes.

---


Surface — addressable place to publish to / ingest from (channel/repo/site/registry/list).
```sql
create table surface (
  id            text primary key,
  platform      text not null,                             -- youtube | github | wordpress | medium | ghcr | appstore | ...
  surface_type  text not null,                             -- channel | repo | site | publication | registry | list | space | ...
  direction     text not null check (direction in ('publish','ingest','bidirectional')),
  constraints   jsonb not null default '{}',               -- limits/capabilities
  metadata      jsonb not null default '{}'                -- typed metadata (e.g., github_repo_v2)
);

create table surface_address (
  surface_id  text references surface(id) on delete cascade,
  kind        text not null,                               -- public | feed | api | deeplink | write_endpoint
  uri         text not null,
  active      boolean not null default true,
  first_seen  timestamptz default now,
  last_seen   timestamptz default now,
  primary key (surface_id, kind, uri)
);
```
Delivery — an Item on a Surface (status, URL, field mappings, lineage).
```sql
create table delivery (
  id             text primary key,
  item_id        text references item(id),
  surface_id     text references surface(id),
  role           text not null check (role in ('original','syndication')),
  status         text not null check (status in ('planned','queued','published','failed','removed')),
  url            text,
  field_mapping  jsonb not null default '{}',              -- mapping Item fields -> platform schema
  published_at   timestamptz,
  source_hash    text,                                     -- lineage for reproducibility/dedup
  remote_id      text
);
```
Edge — typed relationships across nodes (graph-friendly).
Use it for “documents”, “derived_from”, “depends_on”, etc.
```sql
create table edge (
  src_type   text not null check (src_type in ('domain_entity','item','surface')),
  src_id     text not null,
  dst_type   text not null check (dst_type in ('domain_entity','item','surface')),
  dst_id     text not null,
  predicate  text not null,                                -- documents | announces | demonstrates | changelog_for | derived_from | depends_on | contains | delivered_to
  created_at timestamptz default now,
  primary key (src_type,src_id,dst_type,dst_id,predicate)
);
```


## Graph + Vector RAG

Chunks table (derived projection): item_chunk(id,item_id,text,embedding,lang,metadata).

Query recipe:

Vector search top-k over item_chunk for the user query.

Graph re-rank: boost chunks where edge(item -> domain_entity, 'documents'|'changelog_for'|...) matches the user’s context; down-weight stale (published_at).

Prefer provenance='first_party'; include citations for third_party/derived.

Graph engine: start with Postgres (edge + CTEs). Add Neo4j/Neptune later only if needed.

1.5 Events & Projections

Events (append-only): ItemCreated, ItemPublished, DeliveryQueued/Published/Failed, EdgeAdded, SurfaceAddressChanged.

Projections (rebuildable): Search index, Vector index, Analytics (deliveries by surface/status), optional external Graph store.


---

## AI ops slice (small but crucial):

### Agents need their own first-class schema

* Add a small “Ops for AI” slice—don’t bury this in code:
* Agent (id, purpose, tool_ids[], policy_id, capabilities[])
* Tool (id, openapi/jsonschema spec, rate_limits, auth_ref)
* Run (agent_id, inputs, outputs, cost, latency, errors, trace_ids)
* Prompt/Policy (id, template, variables, safety_rules, eval_score)
* EvalResult (dataset_id, metric, score, run_id)
* Link Run → Delivery when an agent publishes, so you can answer “who did what, with what prompt, at what cost.”



# Phase 3: CRM Extension
Add people/orgs and roles—without touching the core invariants.
5.1 Party & Identity
```sql

create table party (
  id         text primary key,
  party_type text not null check (party_type in ('person','organization')),
  name       text not null,
  metadata   jsonb not null default '{}'    -- typed: person_v1 | org_v1
);

create table party_identity (
  party_id   text references party(id) on delete cascade,
  kind       text not null,                 -- email | github | twitter | linkedin | website | youtube | mastodon | ...
  value      text not null,
  verified   boolean default false,
  primary key (party_id, kind, value)
);
```
5.2 Roles (Party ↔ DomainEntity / Item)
```sql
create table domain_entity_party_role (
  domain_entity_id text references domain_entity(id),
  party_id         text references party(id),
  role             text not null,          -- maintainer | owner | author | contact | sponsor | customer | prospect | influencer | partner
  source           text,                   -- ingestion | manual | github_api | ...
  confidence       numeric check (confidence between 0 and 1) default 1,
  start_at         timestamptz default now,
  end_at           timestamptz,
  primary key (domain_entity_id, party_id, role)
);

create table item_party_role (
  item_id    text references item(id),
  party_id   text references party(id),
  role       text not null,               -- author | reviewer | publisher | presenter
  primary key (item_id, party_id, role)
);

alter table domain_entity add column steward_party_id text references party(id);
```

If you prefer ultra-minimal, you can model these as edge rows with src_type/dst_type='party' and predicates like maintained_by, but dedicated tables give better indexing and constraints.

5.3 Interactions & Tasks (optional but useful)
```sql
create table interaction (
  id              text primary key,
  party_id        text references party(id),
  domain_entity_id text references domain_entity(id),
  item_id         text references item(id),
  medium          text not null,          -- email | dm | call | meeting | issue_comment | pr | webinar
  direction       text not null,          -- inbound | outbound
  occurred_at     timestamptz not null,
  summary         text,
  metadata        jsonb not null default '{}'   -- links, transcript ids, consent/source
);

create table crm_task (
  id              text primary key,
  party_id        text references party(id),
  domain_entity_id text references domain_entity(id),
  kind            text not null,          -- outreach | invite | feedback | co_marketing | followup
  due_at          timestamptz,
  status          text not null check (status in ('open','done','canceled')) default 'open',
  metadata        jsonb not null default '{}'
);
```
5.4 CRM Policies & Examples

Attach maintainers/owners to third_party/partner DomainEntities for outreach.

Use steward_party_id to route approvals/requests.

Leverage interaction history for freshness and prioritization.

Quick queries
```sql
-- External entities with no maintainer contact
select de.id, de.name
from domain_entity de
left join domain_entity_party_role r
  on r.domain_entity_id = de.id and r.role in ('maintainer','owner')
where de.provenance in ('third_party','partner') and r.party_id is null;

-- Top maintainers by your interactions in last 90 days
select p.name, count(*) as touches
from interaction i
join party p on p.id = i.party_id
where i.occurred_at >= now - interval '90 days'
group by 1 order by 2 desc;
```
## 6) Appendix
### 6.1 Example UL entries

Provenance: origin of creation (first_party | third_party | derived | partner).
Invariants: third_party Items cannot be original on owned Surfaces; derived requires derived_from edges.
Steward: current responsible party (us | external) or steward_party_id.

### 6.2 Event List (seed)
ItemCreated, ItemUpdated, ItemPublished, ItemArchived
DeliveryPlanned, DeliveryQueued, DeliveryPublished, DeliveryFailed, DeliveryRemoved
EdgeAdded, EdgeRemoved
SurfaceAddressChanged
(CRM) PartyUpserted, RoleAttached, InteractionLogged, CrmTaskCreated/Completed
### 6.3 Fitness Functions (CI checks)
- Publishing private → public disallowed.
- One original Delivery per Item.
- derived Items must have derived_from edge(s).
- Enum removals → require MAJOR schema bump.
- Typed metadata must pass JSON Schema validation.
### 6.4 Typed Metadata Schemas (sketch)
```json
// video_v1.schema.json
{ "type":"object","required":["type","version","duration_sec"],
  "properties":{
    "type":{"const":"video_v1"},
    "version":{"const":1},
    "duration_sec":{"type":"integer","minimum":0},
    "captions":{"type":"boolean"},
    "source":{"type":"string"}
  } }
```
```json
// attribution_v1.schema.json
{ "type":"object","required":["type","version","sources"],
  "properties":{
    "type":{"const":"attribution_v1"},
    "version":{"const":1},
    "sources":{"type":"array","items":{
      "type":"object","required":["title","uri"],
      "properties":{"title":{"type":"string"},"uri":{"type":"string","format":"uri"},"accessed":{"type":"string","format":"date"}}
    }},
    "license":{"type":"object","properties":{"name":{"type":"string"},"uri":{"type":"string","format":"uri"}}}
  } }
```