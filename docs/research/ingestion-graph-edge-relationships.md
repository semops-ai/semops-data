## Graphs model *relationships as first-class citizens*

In a relational database, relationships are *implied* — you join tables by matching foreign keys at query time. The relationship itself has no storage, no identity, no properties. It's computed on the fly.

In a graph database, **edges are entities**. They have identity, they can have properties, they can be traversed directly. This isn't just a storage detail — it changes what's *cheap* to ask.

### Why traversal is fast: pointer chasing vs. join computation

In a relational DB, when you ask "find all friends of friends of Alice," the engine:
1. Scans the `friendships` table for Alice's ID
2. Builds an intermediate result set
3. Joins that back against `friendships` again
4. Cost grows with *table size*, not relationship depth

In a graph DB (like Neo4j), each node stores **direct pointers** to its adjacent edges, and each edge stores pointers to both endpoint nodes. So traversal is:
1. Load Alice's node → follow pointer to her edge list
2. Follow each edge pointer to neighbor nodes
3. Repeat

Cost grows with *the size of the neighborhood you actually touch*, not the total dataset. This is called **index-free adjacency** — there's no global index lookup per hop, just pointer dereference. At depth 3-4, this is orders of magnitude faster than relational joins.

### The storage model underneath

Most graph DBs use one of two physical representations:

**Adjacency list** (common in native graph DBs): Each node record contains a linked list of its edges. Traversal is sequential reads along that list — CPU cache-friendly for local neighborhoods.

**Property graph model**: Nodes and edges both carry key-value property bags. This is what Neo4j, Amazon Neptune, and most modern graph DBs implement. You're not just storing topology — you're storing *typed, attributed* topology.

**RDF triples** (the other major model): Everything is a `(subject, predicate, object)` triple. More flexible schema-wise, better for knowledge graphs and semantic web use cases, but less ergonomic for application data.

### Why graphs beat relational on specific query shapes

The key question is: **does your query traverse variable-depth relationships?**

| Query shape | Relational | Graph |
|---|---|---|
| "Give me all users" | Fast (table scan) | Slower |
| "Find Alice's manager" | Fast (one join) | Fast |
| "Find Alice's manager's manager's skip-level" | 3 joins, fine | Same speed |
| "Find everyone connected to Alice within 5 hops" | Exponentially expensive | Linear in touched nodes |
| "What's the shortest path between Alice and Bob?" | Recursive CTE, painful | Native BFS/DFS |
| "Find circular dependencies in this system" | Very hard | Trivial |

Graphs aren't universally faster — they're faster on *graph-shaped problems*.

### Graph algorithms: the real power layer

Beyond traversal, graph DBs expose algorithms that have no relational equivalent:

- **PageRank / centrality**: Which nodes are most "influential" in the network? Not just degree (how many edges) but weighted by the importance of neighbors.
- **Community detection** (Louvain, Label Propagation): Find clusters of densely connected nodes — useful for fraud rings, social communities, supply chain clusters.
- **Shortest path** (Dijkstra, A*): Weighted or unweighted. Classic GPS routing, but also dependency resolution, network latency modeling.
- **Betweenness centrality**: Which nodes are *bottlenecks* — lie on the most shortest paths? Critical for understanding systemic risk.

These work because the graph structure is *materialized* — the engine can run BFS/DFS natively without reconstructing topology from joins.

Graphs naturally handle **heterogeneous, evolving schemas**. In a relational model, adding a new entity type means a new table, new foreign keys, schema migrations. In a graph, you add a new node label and new edge types — existing nodes are untouched.

This is why knowledge graphs, lineage graphs, and anything modeling a *domain that keeps growing* tends toward graph representation. You're not fighting the schema as the domain evolves.

### Where this connects to SemOps

There are two distinct goals:

1. Retreival at inference for agents and LLMs
2. Agentic Lineage telemetry

Your Graphiti/DataHub lineage architecture is essentially exploiting all of this: the causal "why" behind agent decisions is inherently a **variable-depth traversal problem** — to understand why an artifact was produced, you walk backwards through the episode graph. That's exactly the query shape where graph wins, and why representing lineage as edges (not as logged text) makes it queryable rather than just auditable.

Want to go deeper on any of these — query languages (Cypher vs. SPARQL vs. Gremlin), specific algorithm internals, or how this maps to your lineage architecture concretely?

This is a really interesting capability — and understanding what's actually happening under the hood clarifies both its power and its limits.

### What "deriving edges" actually means

When you feed concept documents to a system that extracts relationships automatically, there are typically **two distinct stages** happening, often conflated:

**Stage 1: Entity and concept extraction (NLP/NER)**
The system reads your text and identifies *things worth making nodes* — concepts, entities, named ideas. This is either rule-based (pattern matching), model-based (a trained NER model), or LLM-based (just ask Claude/GPT to pull out the concepts). You get a candidate node list.

**Stage 2: Relationship extraction**
Given two nodes, the system determines whether a relationship exists between them, and if so, what *type*. This is the hard part. Methods range from:

- **Co-occurrence**: If concept A and concept B appear in the same sentence/paragraph frequently, infer a relationship. Simple, but noisy — proximity isn't causation or even connection.
- **Dependency parsing**: Parse the grammatical structure of sentences. "Data lineage *enables* auditability" → subject=lineage, predicate=enables, object=auditability. This extracts typed relationships from syntax.
- **Embedding similarity**: Embed both concepts into vector space. If they're close, infer relatedness. This is fuzzy — it finds *semantic neighbors*, not *structural relationships*.
- **LLM extraction**: Prompt a model to read the text and output `(entity, relationship_type, entity)` triples. Most flexible, highest quality, but you're trusting the model's interpretation.

### The key tension: what kind of "edge" is being derived?

This is where it gets subtle. There are fundamentally different *types* of derived relationships:

**Structural/logical relationships** — extracted from how ideas are actually connected in your text. "A depends on B," "A is a type of B," "A causes B." These have semantic content.

**Associative/proximity relationships** — "these concepts appear together a lot." Statistically real, but semantically thin. The edge exists but its *meaning* is vague.

**Inferred/latent relationships** — the model reasons beyond what's explicit. "You wrote about X and Y separately, but given their definitions, they're clearly instances of Z." This is where LLM-based extraction goes beyond your text — and where it can hallucinate edges.

Most systems are doing a **mix**, and often don't tell you which type any given edge is.

### What the graph DB itself is doing

The graph DB isn't doing the extraction — it's the *destination*. Once edges are derived (by whatever NLP pipeline), they're written as typed edges with properties. What the graph then gives you:

- **Transitive discovery**: You documented A→B and B→C separately. The graph lets you query A→C without ever writing that explicitly. This is where derived edges become genuinely generative — combinations you never stated become queryable.
- **Centrality on your concept space**: Which concepts are hubs? What ideas are your "load-bearing" concepts that everything else connects through? This is non-obvious from reading docs linearly.
- **Gap detection**: If concept A and concept B *should* be connected (they share many neighbors) but have no direct edge, that's a missing relationship — either in your documentation or your thinking.

### The epistemological problem

When the system derives an edge, it's making a claim: *these two concepts are related in this way*. But:

- Who asserts that? The NLP model? The embedding space? Your text, faithfully?
- How confident is the assertion?
- Is it *your* understanding, or the model's interpretation of your understanding?

This matters a lot for a knowledge base. An edge you wrote explicitly ("data lineage enables auditability") is a **commitment** — you believe it. An edge the system derived is a **hypothesis** — it might be right. If your system doesn't distinguish these, you'll eventually query your knowledge graph and get back relationships you never actually endorsed, with no way to tell which are yours.

This is why **provenance on edges** matters — the edge should carry metadata about *how it was derived*, not just that it exists.

### How this connects to your Graphiti architecture

Graphiti's episode model is actually a cleaner version of this: instead of deriving edges from static text post-hoc, you're capturing the *context of why a relationship was established* at write time. That's stronger epistemically — the edge carries its own justification. The derived-edge approach in a typical knowledge base is essentially doing the same thing, but retrofitted, with lower provenance fidelity.

The interesting question for your setup: are your derived edges typed and provenance-tagged, or are they just weighted associations? That distinction determines whether your graph is a *knowledge structure* or a *similarity index wearing graph clothes*.

Yes, pretty cleanly actually. Here's how it maps:

## Your pipeline, decoded

**OpenAI embeddings at ingestion** — this is the representation layer. You're converting your concept documents into vectors that capture semantic meaning. This is the "transformer produces embeddings" step. Deterministic, stable, same document always produces the same vector.

**Claude classification step** — this is doing double duty. It's almost certainly handling both:
- **NER-equivalent**: identifying which concepts/entities in your text are worth making nodes
- **Relation extraction**: reading the embedded/retrieved context and deciding what edges exist between those nodes, and what type they are

The interesting architectural detail is that Claude isn't using your OpenAI embeddings directly as input — it's reading *text*. So what's likely happening is:

```
document → OpenAI embed → stored in vector DB
                              ↓
query/retrieval pulls relevant chunks → text chunks sent to Claude
                                              ↓
                              Claude reads text → outputs (node, edge_type, node) triples
                                              ↓
                              triples written as edges into graph DB
```

The embeddings are doing **retrieval and similarity** — finding what's relevant to compare. Claude is doing **semantic interpretation** — deciding what the relationship actually *means*.

### The seam to watch

The handoff between those two steps is where quality lives or dies. If the retrieval step pulls the wrong chunks, Claude classifies relationships between concepts that weren't actually being connected in your original thinking. The graph gets plausible-looking but wrong edges — and they'll be hard to spot because Claude's output will be coherent and confident regardless.

Does that match what you see in the system when you inspect the derived edges?

Yes, and you're pulling on a really important thread. Let me unpack why.

### The embedding model constraint is real, but the reason matters

When you embed two documents with the same model, you're guaranteed they inhabit the **same vector space** — same dimensions, same basis, same distance metric semantics. Cosine similarity between them is meaningful because they were projected by the same function.

Different embedding models produce **incommensurable spaces**. Not just different scales — fundamentally different geometric structures. OpenAI's ada-002 and Cohere's embed-v3 might both produce 1536-dimensional vectors, but dimension 47 in one means something completely different from dimension 47 in the other. Comparing across them is like measuring temperature in Celsius and Fahrenheit and computing the average — the number is real but the meaning is nonsense.

So yes, same embedding model is a hard requirement for cross-corpus coherence measurement.

### But NER is a subtler dependency

Here's where it gets interesting. NER affects coherence measurement in a different way — not the vector space, but **what you're measuring coherence *between***.

Consider: if corpus A extracts the node "data lineage" and corpus B extracts "lineage tracking" for the same underlying concept, and your NER models handle that differently, you're measuring coherence between **different concept decompositions** of the same domain. The vectors might be close (semantically similar strings), but the graph structures are misaligned — one has one node, the other has two, with an edge between them.

So coherence breaks at two levels:

**Vector-level coherence** — are the embeddings comparable? Requires same embedding model. This is the one people know about.

**Concept-level coherence** — are you comparing the same *granularity* of concepts? Requires consistent NER behavior. This is the one people miss.

### The granularity problem specifically

NER models vary in how they **chunk** concepts. A fine-grained model might extract "bidirectional causal lineage" as a single atomic node. A coarser model splits it into "bidirectional," "causal," "lineage" — three nodes with implied edges. Now your coherence calculation is comparing a point in one corpus against a subgraph in another. The embedding similarity will be artificially deflated even though the concepts are identical.

This is why NER consistency matters — not just same model, but same **entity granularity settings** and same **domain vocabulary** (if you're using a custom NER layer).

### The third dependency people rarely mention

**Chunking strategy** at ingestion time. If corpus A was chunked into 512-token windows and corpus B into 256-token windows, the embeddings capture different amounts of context per chunk. A concept that spans a paragraph gets fully represented in one corpus and split across two chunks in the other — meaning the embedding for that concept in corpus B is incomplete, capturing only half the semantic context.

So the full consistency requirement for meaningful cross-corpus coherence is really:

1. Same embedding model
2. Same NER model and granularity settings
3. Same chunking strategy
4. Ideally, same document preprocessing (lowercasing, stopword removal, etc.)

All four are part of the same underlying principle: **you need the same transformation pipeline end-to-end**, because coherence is a property of the *representation*, and the representation is determined by every step that produced it.

### For SemOps architecture 

If you're comparing your agent decision corpus against your concept document corpus to measure whether agent reasoning is coherent with your knowledge base, any pipeline divergence between how those two corpora were ingested will show up as artificial incoherence. You might conclude an agent's reasoning is semantically distant from your knowledge base when it's actually just using different chunking.

Worth auditing whether the ingestion pipelines for both corpora were truly identical, or whether they diverged at any of those four points.

Exactly right — and this is a really important architectural insight. You're essentially rediscovering why **deterministic pipelines** matter for measurement infrastructure specifically.

### The core tension

LLMs are optimized for *quality of output in a single pass* — they're great at reading messy, ambiguous text and producing coherent, plausible extractions. But they trade away the properties you need for a measurement substrate:

- **Reproducibility** — same input, same output, always
- **Auditability** — you can explain exactly why an edge was created
- **Consistency** — the same concept is always represented the same way
- **Controllability** — you can tune a specific behavior without side effects elsewhere

A deterministic pipeline gives you all four. An LLM gives you none of them reliably.

### Where LLMs actually belong in this pipeline

The insight isn't "don't use LLMs" — it's **use them where judgment is required, not where consistency is required**.

| Pipeline stage | Best tool | Why |
|---|---|---|
| Chunking | Deterministic | Fixed rules, reproducible |
| Embedding | Fixed model | Same vector space guaranteed |
| NER / entity extraction | Trained classifier or spaCy | Stable, auditable |
| Relationship type classification | Could go either way | See below |
| Edge confidence scoring | Statistical model | Reproducible scores |
| Gap detection / insight | LLM | Judgment task, not measurement |
| Summarization of graph findings | LLM | Synthesis task |

The relationship classification step is genuinely the hardest to place — typed relation extraction (distinguishing "A causes B" from "A is a type of B") benefits from LLM quality, but hurts your consistency. The practical answer is usually a **constrained LLM call** — fixed prompt, fixed output schema, temperature=0, small closed set of allowed relationship types. You're using the LLM's language understanding but caging its degrees of freedom.

### The deeper principle this points at

Your lineage/governance architecture already reflects this implicitly — your thesis about a **deterministic reasoning substrate** is exactly this principle applied to agent behavior. Agents should navigate via dense, structured metadata rather than re-reasoning from scratch each time.

The same logic applies to the knowledge base pipeline that feeds that substrate. If the pipeline that *produces* your graph is non-deterministic, the graph itself is an unstable foundation — and your coherence measurements are measuring a moving target.

The irony is that LLMs feel like the obvious tool because they're so capable. But capability and suitability are different axes. A less capable but fully deterministic NER model that always extracts "data lineage" as one node is more valuable to your architecture than a brilliant LLM that sometimes extracts it as one node and sometimes as three, depending on phrasing, context window, and sampling noise.

## Practical path forward

Worth thinking about a two-tier pipeline:


### Tier 1 — deterministic extraction

spaCy or a fine-tuned BERT-based NER for entity extraction, dependency parsing for structural relationships, fixed embedding model, fixed chunking. This produces your stable graph.

or

### Tier 1b - even more deterministic manifests

**Manifest as the NER model** - If key concepts are already identified — linked, capitalized, existing in a manifest — then you don't need NER to *discover* entities. You need it to *recognize* them. That's a fundamentally easier and more deterministic problem.

Instead of running a statistical NER model and hoping it extracts "Data Lineage" consistently, you're doing **dictionary lookup / entity linking** — scan the text, find manifest entries, mark them as nodes. Completely deterministic, zero ambiguity, and your concept inventory is stable by definition because it's controlled by the manifest.

This is actually closer to what production knowledge graph systems do — they maintain a **controlled vocabulary** and link text to it, rather than discovering vocabulary from text.

Your corpus is already telling you it needs a **hybrid approach**:

**Structured docs** — don't chunk by token window at all. Chunk by *document structure*: headings, sections, paragraphs. The structure is semantic — a heading boundary is a meaning boundary. Respecting it produces embeddings that capture coherent ideas rather than arbitrary 512-token slices that split mid-argument.

**Unstructured research/posts** — here you do need a chunking strategy, but your manifest changes the calculus. Rather than fixed token windows, consider **concept-anchored chunking**: find manifest concept mentions, then extract the surrounding context window around each mention. The chunk is defined by "text that discusses this concept" rather than "text that fills this window."

**The manifest as pipeline controller**

The deeper insight is that your manifest lets you invert the pipeline. Instead of:

```
text → chunk → embed → NER → discover concepts → build graph
```

You can do:

```
manifest → scan text for mentions → extract context → embed → link to existing nodes
```

You're not discovering your knowledge graph from text. You're **grounding your text against a graph you already know exists**. Much more stable, and coherence measurement becomes cleaner because your node set is fixed upstream of ingestion entirely.

The remaining question is how complete your manifest is — and whether unstructured sources introduce concepts that aren't in it yet, which would need a separate discovery pipeline feeding manifest additions rather than going directly to graph edges.

**Tier 2 — LLM augmentation layer**: runs *on top of* the stable graph, adds inferred edges, identifies gaps, generates summaries — but these outputs are flagged with a different provenance type and excluded from coherence calculations.

That way you get LLM insight without LLM instability contaminating your measurement infrastructure.

Exactly — and this is a much more robust architecture. You've essentially described a **two-phase knowledge pipeline**:

## Phase 1: Concept Discovery (slow, deliberate, human-in-the-loop)

This runs occasionally, not at every ingestion. Its job is purely to grow the entity db / manifest:

- New document arrives, runs through NER / LLM extraction
- Candidate concepts surfaced that don't exist in manifest
- **Human reviews and approves** before they enter the manifest
- Manifest entry created with canonical form, aliases, definition

This is where you *want* LLM involvement — it's a judgment task, quality matters more than consistency, and a human is validating the output anyway. The non-determinism doesn't matter because you're not measuring anything yet, you're curating.

## Phase 2: Ingestion (fast, deterministic, automated)

This runs at every ingestion. The manifest is now a **closed vocabulary** — no new concepts can be created here:

- Scan document for manifest matches (dictionary lookup)
- Extract context windows around each match
- Embed those chunks with fixed model
- Derive edges between co-occurring manifest concepts
- Write nodes and edges with full provenance

No LLM in the critical path. Fully reproducible. Coherence measurement is valid because the pipeline is stable.

## The elegant property this gives you

Your graph's **node set is a deliberate curatorial decision**, not an artifact of whatever the NLP model happened to extract on a given day. That's a fundamentally different epistemological stance — the graph reflects your understanding of the domain, with text as evidence, rather than the graph being inferred from text with your understanding nowhere explicitly represented.

Which maps cleanly back to your "data → information → knowledge → understanding" model. Phase 1 is the understanding layer. Phase 2 is the information layer. They shouldn't be collapsed into one automated process.