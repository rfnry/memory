# rfnry-memory

rfnry-memory stores and retrieves knowledge for a tenant.

A **corpus** holds two kinds of thing. **Sources** are written down: documents you ingest as text,
chunked and indexed. **Stories** are told: small slices of experience a model or your code
remembers — a fact, an event, a decision — each true from a moment until it is superseded. Both are
indexed through the same three **pillars**: text, vector and graph. You choose one, two or all;
every search runs the pillars you chose and fuses their results into one ranked list whose entries
say which pillar found them.

Models enter at two seams. On write, an embedding model indexes the vector pillar, a generation
model extracts entities and relations for the graph, and a decision model resolves whether two
entities are the same thing. On read, a decision model can annotate each result with four
judgements — relevant, evidence, contradicts, injection — and leave the decision to you. A tool
surface exposes the same verbs to a model, pinned to one corpus.

## Philosophy

**Hand the controls to the model, keep the structure.** The library decides nothing about what is
worth remembering or which result is good enough. A model or your code calls `remember`; a model or
your code reads the ranked list. What the library owns is what a model cannot provide for itself:
durable storage, exact temporal semantics, provenance, tenancy, and the pillars.

**Annotate, never filter.** Nothing in the read path drops or reorders a result on the library's
authority. When a decision model judges results, its probabilities ride along on each hit; the
caller — a strong reader that can judge for itself, or a decision agent with a small state budget
that cannot — decides what to do with them.

**Text in.** Turning a PDF, a drawing, a program or an image into text is the parser's job, and the
parser knows things the library would only lose: page numbers, sheet names, the components a
diagram connects. Those travel in as sections with locators and as graph facts. The library takes
text and structure; it never takes a file.

**One storage, all three pillars.** Rows, vectors, full-text index and graph live in one database,
so a write is one transaction and a temporal filter runs inside the vector search. There is no
second system to keep in step.

**Tenancy by construction.** Every verb names its corpus. Every row carries it. There is no default
corpus and nothing reads across corpora. Deleting a corpus is provable, not promised.

## Models

Three protocols. Any object with the right method is a model; adapters to vendors are yours.

```python
class EmbeddingModel(Protocol):
    model: str
    dimension: int
    async def embed(self, texts: list[str]) -> list[list[float]]: ...

class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> Reply: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: dict[str, Question]) -> dict[str, Answer]: ...
```

The generation and decision contracts are the same ones rfnry-harness uses, so one adapter serves
both libraries. The library asks a generation model for structured output through a forced tool
call and asks a decision model `Choice`, `Score` and `Check` questions; it never streams and never
holds a conversation. A fake embedding is enough to run everything in tests:

```python
class HashEmbedding:
    model, dimension = "hash-8", 8
    async def embed(self, texts):
        return [[b / 255 for b in sha256(t.encode()).digest()[:8]] for t in texts]
```

## Getting started

```python
memory = Memory(PostgresStorage(pool), embedding=emb)
await memory.initialize()

await memory.ingest("acme", text=manual_text, name="Pump manual v3", metadata={"line": "A"})
await memory.remember("acme", "Ana moved the go-live to March 3.", subject="project:apollo")

hits = await memory.search("acme", "when is the go-live?")
for hit in hits.hits:
    print(hit.kind, hit.score, hit.provenance, hit.text)
hits.pillars["vector"].returned, hits.pillars["vector"].duration_ms, hits.usage
```

With only `embedding`, the text and vector pillars are on. Add `generation=` and the graph pillar
is on. Pass nothing and the text pillar alone still works. `hits.pillars` is the observability
payload for a search; every write returns its own.

## Sources

```python
ingested = await memory.ingest("acme", text=text, name="Pump manual v3", metadata={"line": "A", "rev": "3"})
ingested.source.chunks, ingested.entities, ingested.usage
```

Text is split into chunks — markdown-aware, headings and code fences kept whole — and each chunk is
indexed by every pillar. When your parser already knows the structure, pass sections instead of
text and each becomes its own run of chunks, carrying its locator:

```python
await memory.ingest("acme", name="P&ID sheet 7", sections=[
    Section(title="Feed line", text=feed_text, locator={"sheet": 7, "region": "A2"}),
    Section(title="Return line", text=return_text, locator={"sheet": 7, "region": "C4"}),
])
```

`metadata` is yours. Every key becomes a facet you can filter on, with one value or any of several:

```python
hits = await memory.search("acme", "seal replacement", filters={"metadata.line": "A", "metadata.rev": ["2", "3"]})
sources, cursor = await memory.list_sources("acme", filters={"metadata.line": "A"})
source = await memory.get_source("acme", source_id)
text = await memory.read("acme", source_id)                             # the whole source
part = await memory.read("acme", source_id, locator={"sheet": 7})      # the chunks that match
deleted = await memory.delete_source("acme", source_id)
```

## Stories

A story is a sentence or two, about a subject, true from a moment until it is superseded.

```python
story = await memory.remember("acme", "The go-live is February 20.", subject="project:apollo",
                              occurred_at=datetime(2026, 1, 10, tzinfo=UTC))
closed, current = await memory.supersede("acme", story.story_id, "Ana moved the go-live to March 3.",
                                         effective_at=datetime(2026, 2, 1, tzinfo=UTC))
```

`supersede` closes the old story at `effective_at` and opens the new one there, pointing back:

```python
closed.valid_to == current.occurred_at          # True
current.supersedes == closed.story_id           # True
```

Nothing is edited in place, so the past stays answerable:

```python
await memory.recall("acme", "go-live date", subject="project:apollo")                                # → March 3
await memory.recall("acme", "go-live date", subject="project:apollo", as_of=datetime(2026, 1, 20, tzinfo=UTC))   # → February 20
stories, cursor = await memory.list_stories("acme", subject="project:apollo", since=datetime(2026, 1, 1, tzinfo=UTC))
```

`recall` searches stories with the pillars you chose plus time: `as_of` asks what was true then,
`since` and `until` bound when it happened, `subject` narrows to one thing. `forget` deletes a story
outright. `source_ids` on a story link it to the sources it came from — records, not constraints.
Remembering the same text twice while the first is still open returns the first.

## Pillars

| pillar | finds | needs |
| --- | --- | --- |
| `text` | what the query's words match | nothing |
| `vector` | what the query means | `embedding=` |
| `graph` | what the query names, and what that is connected to | entities, from `generation=` or from you |

Choose at construction, narrow per query:

```python
memory = Memory(storage, embedding=emb, generation=llm, pillars={"text", "vector", "graph"},
                relation_types=["CONNECTS_TO", "PART_OF", "OWNS"], weights={"graph": 2.0})
hits = await memory.search("acme", query, pillars={"vector"})
```

Each pillar over-fetches, the lists are fused by reciprocal rank, and each hit's `provenance` says
what each pillar thought of it — a pillar that did not return the hit is absent, not zero.
`hits.pillars` reports what each pillar returned and how long it took; a pillar that fails is
dropped and named there, and the rest still answer.

## Graph

With a generation model, every passage written is read once for entities and typed relations. The
relation vocabulary is yours and the model cannot leave it. With a decision model, an entity whose
name resembles an existing one is judged — different, related, or the same thing under another name
— and merged when it is the same. Without one, identity is exact name and type.

Your parser can write graph facts directly, with no model in the path:

```python
await memory.ingest("acme", name="P&ID sheet 7", sections=sections,
    entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
    relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")])
```

And you can walk it:

```python
hood = await memory.neighbors("acme", "P-3", hops=2)
hood.entities, hood.relations, hood.passages
```

## Judging results

With a decision model, ask for judgements and every hit carries four probabilities:

```python
hits = await memory.search("acme", "is the March 3 date confirmed?", judge=True)
hit.judge.relevant, hit.judge.evidence, hit.judge.contradicts, hit.judge.injection
```

*relevant*: the passage is about what the query asks. *evidence*: it contains what answers the
query. *contradicts*: it contradicts a premise the query states. *injection*: it contains
instructions addressed to an AI rather than information. The list is unchanged; what to keep, what
to flag and what to refuse is your code:

```python
evidence = [h.text for h in hits.hits if h.judge.relevant > 0.7 and h.judge.injection < 0.2]
conflicts = [h for h in hits.hits if h.judge.contradicts > 0.7]
```

For a strong reader that judges for itself the annotations cost nothing; for a decision agent
consuming memory — a small, literal state budget — they are the difference between a clean state
and a noisy one.

## The tool surface

```python
tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
tools.definitions()                                   # search recall read_source list_sources remember
await tools.dispatch("recall", {"query": "go-live"})  # supersede forget list_stories neighbors
```

Nine tools, each the method of the same name with the same arguments. The corpus is fixed when the
tools are built and a model cannot name another; bound filters can be narrowed by the model, never
widened. Errors come back as results, so a model always gets something to read:

```python
await tools.dispatch("remember", {"text": "Ana prefers the Tuesday slot.", "subject": "user:ana"})
await tools.dispatch("supersede", {"story_id": "missing"})       # → {"error": {"type": "NotFound", "message": "…"}}
```

The definitions are plain dicts and `dispatch` is one call, which is the shape rfnry-harness mounts:

```python
run = await engine.run("support", scope="acme", input=text,
                       toolsets=[MemoryTools(memory, corpus_id="acme")])
```

## Storage

Two storages ship: `InMemoryStorage()` for tests and small things, `PostgresStorage(pool)` for
everything else, taking an asyncpg pool you built.

```python
memory = Memory(InMemoryStorage(), embedding=HashEmbedding())          # a complete memory, no services
memory = Memory(PostgresStorage(await asyncpg.create_pool(dsn)), embedding=emb, generation=llm, decision=dm)
```

`initialize()` creates the tables, the full-text and vector indexes, and is safe to run again; the
tables are prefixed `memory_` so one database can also hold rfnry-harness's. Vectors record which
embedding model produced them; after changing the model, `await memory.reembed("acme")` brings a
corpus up to date — sources could be re-ingested, stories cannot.

## Tenancy

`corpus_id` is required on every verb and is the first column of every table. It is a reserved
filter key, so a query can never smuggle another corpus in. A tenant can hold several corpora; a
search reads one. To offboard:

```python
deleted = await memory.delete_corpus("acme")
deleted.sources, deleted.chunks, deleted.stories, deleted.entities
residue = await memory.verify_absent("acme")
assert residue.clean                                  # every table counted independently, all zero
```

## Errors

`ConfigurationError` for a pillar without its model or a storage without its extension,
`InputError` for a bad argument or a reserved filter key, `ModelError` when an adapter breaks the
contract, `StorageError` for the database, `NotFound`. All under `MemoryError`.
