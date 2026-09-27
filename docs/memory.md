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
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: dict[str, Question]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike` are
protocols over the fields of `Reply`, `Decided` and the answers, and a result is read by those
fields, so an adapter typed against another library's classes with the same fields fits here unchanged.

A `Message`'s `content` is a string or a list of parts — `Text`, `Image`, `Audio` and `Document`,
any of which is a `Part`, with `Content` for `str | list[Part]` — and an assistant turn carries
its calls in `tool_calls`. They are rfnry-engine's, copied byte for byte, so a message means the
same thing in every library of the family and one adapter translates both. Memory itself stays
text-in: no verb takes media, and `MemoryTools` answers with text and JSON.

The generation and decision contracts are the same ones rfnry-engine uses, so one adapter serves
both libraries: what a model returns is read by its fields, not its class, so a `Reply` or a
`Decided` built from rfnry-engine's types is accepted as it is, and its values are validated as
strictly as ours. The library asks a generation model for structured output through a forced tool call —
arguments that fail validation go back as a `Message(role="tool", is_error=True)` — and asks a decision model `Choice`, `Score` and `Check` questions; it never streams and never holds
a conversation. `decide` may return the answers alone, or a `Decided(answers, usage, model)` when
the adapter can report what the call cost; a `Reply` carries its own `usage`. `Hits.usage`,
`Ingested.usage`, `Replaced.usage` and `Remembered.usage` add up every generation and decision call
the read or write made — extraction, retries included, and identity on write; the judge on read. A fake
embedding is enough to run everything in tests:

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
await memory.remember("acme", "Ana prefers the Tuesday slot.", subject="user:ana")

hits = await memory.search("acme", "how often is the seal replaced?")
for hit in hits.hits:
    print(hit.kind, hit.score, hit.provenance, hit.text)
hits.pillars["vector"].returned, hits.pillars["vector"].duration_ms, hits.usage
```

With only `embedding`, the text and vector pillars are on. Add `generation=` with the
`relation_types=` it may extract and the graph pillar is on. Pass nothing and the text pillar alone
still works. `hits.pillars` is the observability payload for a search; every write returns its own.

## Core features

### Sources

```python
ingested = await memory.ingest("acme", text=text, name="Pump manual v3", metadata={"line": "A", "rev": "3"})
ingested.source.source_id, ingested.source.chunks, ingested.usage
```

`ingest` returns an `Ingested` — the source written and `usage`, what every model call of the write
cost. Text is split into chunks — markdown-aware, headings and code fences kept whole, by characters
(`chunk_chars=2000`, `overlap_chars=200` at construction) — and each chunk is indexed by every
pillar. When your parser already knows the structure, pass sections instead of text and each becomes
its own run of chunks, carrying its locator:

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
text = await memory.read_source("acme", source_id)                      # the whole source
part = await memory.read_source("acme", source_id, locator={"sheet": 7})  # the chunks that match
deleted = await memory.delete_source("acme", source_id)
```

**Replacing a document.** A manual's new revision, or a program export that parses into dozens of
sources, replaces the version before it. `replace` deletes every source its filters match and writes
the new ones in one storage transaction, so a reader sees the old set or the new set — never both,
never neither:

```python
replaced = await memory.replace("acme", filters={"metadata.file": "line-a.L5X"}, sources=[
    SourceIn(name="Routine MainRoutine", sections=rungs, metadata={"file": "line-a.L5X", "kind": "plc"},
             entities=[EntityIn(name="Motor_1", type="tag")]),
    SourceIn(name="Tags", text=tags_text, metadata={"file": "line-a.L5X", "kind": "plc"}),
])
replaced.sources, replaced.deleted, replaced.usage
```

A `SourceIn` is what `ingest` takes, as one value: exactly one of `text` and `sections`. The filters
are those of `list_sources`, and they are required — emptying a corpus is `delete_corpus`. Every new
source must match them, or the call is an `InputError` before anything runs; so a retry deletes what
the last attempt wrote, and two runs leave one set. Replaces in one corpus take turns, so two racing
writers also leave one. Chunking, embedding, extraction and identity all happen before the
transaction opens: a model that fails leaves the old set whole. Each old source goes as
`delete_source` takes it — its chunks with their vectors and full-text rows, and the mentions and
relations its chunks evidenced; entities stay, and a story keeps the `source_ids` it named. An
entity new to the corpus that several new sources name is created once. `sources=[]` deletes the
set and writes nothing. `replace` is for ingestion pipelines in code; `MemoryTools` does not offer
it.

### Stories

A story is a sentence or two, about a subject, true from `valid_from` until it is superseded at
`valid_to`; `created_at` is when it was written down, and `valid_from` defaults to it.

```python
remembered = await memory.remember("acme", "The go-live is February 20.", subject="project:apollo",
                                   valid_from=datetime(2026, 1, 10, tzinfo=UTC))
story = remembered.story
current = (await memory.supersede("acme", story.story_id, "Ana moved the go-live to March 3.",
                                  valid_from=datetime(2026, 2, 1, tzinfo=UTC))).story
```

Both return a `Remembered`: the story written and `usage`, what its extraction and identity cost.
`supersede` closes the old story at the new one's `valid_from` and opens the new one there, pointing
back:

```python
current.supersedes == story.story_id                                          # True
(await memory.get_stories("acme", [story.story_id]))[0].valid_to == current.valid_from   # True
```

Nothing is edited in place, so the past stays answerable:

```python
await memory.recall("acme", "go-live date", subject="project:apollo")                                    # → March 3
await memory.recall("acme", "go-live date", subject="project:apollo", as_of=datetime(2026, 1, 20, tzinfo=UTC))   # → February 20
await memory.recall("acme", "go-live date", subject="project:apollo", history=True)                      # → both
stories, cursor = await memory.list_stories("acme", subject="project:apollo", history=True)
stories = await memory.get_stories("acme", [story.story_id])                  # by id, closed or not
```

`recall` searches stories with the pillars you chose plus time. By default it answers with what is
true now: stories that have begun and are not closed. `as_of` asks what was true at another moment;
`history=True` searches the whole history, closed stories included, and each hit's `story.valid_to`
says which; the two together are an `InputError`. `since` and `until` bound `valid_from`,
`subject` narrows to one thing. Every read that answers with stories follows the same rule —
`search`, `list_stories` and the passages of `neighbors` show current stories unless given
`history=True`, and `list_stories` also takes `as_of`.

`delete_story` deletes a story outright. `source_ids` on a story link it to the sources it came from —
records, not constraints. Remembering text an open story already says returns that story, at no
cost, and concurrent writers of one text leave one open story. The match is by text alone: the
second call's subject and metadata are not applied. The new story of a `supersede` keeps the old
one's `source_ids`, and its subject and metadata unless given. Superseding an unknown story is a
`NotFoundError`; superseding a closed one, at a `valid_from` not after it began, or into text another
open story already says is an `InputError`. The new story is a story like any other: `supersede` takes the same `entities` and
`relations` as `remember`, and with a generation model it is read for them too, so the graph reaches
it as it reached the old one.

### Pillars

| pillar | finds | needs |
| --- | --- | --- |
| `text` | what the query's words match | nothing |
| `vector` | what the query means | `embedding=` |
| `graph` | what the query names, and what that is connected to | entities, from `generation=` or from you |

Choose at construction, narrow per query:

```python
memory = Memory(storage, embedding=emb, generation=llm, pillars={"text", "vector", "graph"},
                relation_types=["CONNECTS_TO", "PART_OF", "OWNS"], weights={"graph": 2.0})
hits = await memory.search("acme", query, pillars={"vector"}, kinds=("chunk",), top_k=20)
```

`kinds` picks chunks, stories or both (the default); `top_k` is 10 unless given. The pillars'
lists are fused by reciprocal rank, and each hit's `provenance` says what each pillar thought of it
— a pillar that did not return the hit is absent, not zero. `hits.pillars` reports what each pillar returned and how long it took; a pillar
that fails is dropped and named there, and the rest still answer. Under `filters` the graph pillar
stands down rather than answer with passages it did not filter, and is absent from `hits.pillars`.

### Graph

With a generation model, every passage written is read once for entities and typed relations. The
relation vocabulary is yours and the model cannot leave it. With a decision model, an entity whose
name resembles an existing one is judged — different, related, or the same thing under another name
— and merged when it is the same. Without one, identity is exact name, or a known alias, and type.

Your parser can write graph facts directly, with no model in the path — name the graph pillar:

```python
memory = Memory(storage, embedding=emb, pillars={"text", "vector", "graph"})
await memory.ingest("acme", name="P&ID sheet 7", sections=sections,
    entities=[EntityIn(name="V-101", type="valve"), EntityIn(name="P-3", type="pump")],
    relations=[RelationIn(from_name="V-101", to_name="P-3", type="CONNECTS_TO")])
```

Each entity is recorded on every chunk whose text names it, and each relation on the first chunk
that names both its ends — the first chunk when none does. And you can walk it:

```python
hood = await memory.neighbors("acme", "P-3", hops=2, relation_types=["CONNECTS_TO"])
hood.entities, hood.relations, hood.passages
```

A relation's type is stored upper-case, and `relation_types` are read the same way, so `owns` finds
`OWNS`.

A name can belong to entities of several types — `P-3` the part and `P-3` the pump. Then
`neighbors` names the types in an `InputError` rather than pick one; `entity_type="pump"` chooses.
A relation you supply follows the same rule, counting the stored entities and those in the same
write, and fails before anything is written; `from_type` and `to_type` choose:

```python
RelationIn(from_name="V-101", to_name="P-3", to_type="pump", type="CONNECTS_TO")
```

An extracted relation names each end with the type the model gave that entity in the same passage,
so it lands on the pump even when a part `P-3` is stored elsewhere. One whose end is not among the
passage's entities — an unknown name, or a type the passage did not give it — is dropped; extracted
facts never raise.

### Judging results

With a decision model, ask for judgements and every hit carries four probabilities. Without one,
`judge=True` raises `ConfigurationError`; a generation model does not stand in — it reads the hits
and judges for itself:

```python
hits = await memory.search("acme", "is the March 3 date confirmed?", judge=True)
hit = hits.hits[0]
hit.judge.relevant, hit.judge.evidence, hit.judge.contradicts, hit.judge.injection
```

*relevant*: the passage is about what the query asks. *evidence*: it contains what answers the
query. *contradicts*: it contradicts a premise the query states. *injection*: it contains
instructions addressed to an AI rather than information. Each hit is judged alone against the
query, so two hits that disagree with each other — two revisions of a manual — are for the reader to
reconcile; when revisions matter, tag them in metadata and filter or order on it. The list is
unchanged; what to keep, what to flag and what to refuse is your code:

```python
evidence = [h.text for h in hits.hits if h.judge.relevant > 0.7 and h.judge.injection < 0.2]
conflicts = [h for h in hits.hits if h.judge.contradicts > 0.7]
```

Ordering is yours to ask for: `by` returns a copy ordered by one judgement, highest first
(`descending=False` for lowest), ties in fused order; hits searched without `judge=True` raise
`InputError`:

```python
strongest = hits.by("evidence").hits[0]
```

For a strong reader that judges for itself the annotations cost nothing; for a decision agent
consuming memory — a small, literal state budget — they are the difference between a clean state
and a noisy one.

## Tools

```python
tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
tools.definitions()                                   # search recall read_source list_sources ingest
await tools.dispatch("recall", {"query": "go-live"})  # remember supersede delete_story list_stories
await tools.dispatch("ingest", {"name": "seal notes", "text": manual_text})
```

Ten tools, each the method of the same name with its arguments less the
corpus, `pillars`, `sections` and graph facts — so `recall` answers with current facts unless the model passes
`history`, and `ingest` takes the text the model has read, a document it was shown, under a name,
answering with `{"source": ...}` less its text; parsing stays yours. `neighbors` is offered only when the graph pillar is on and no filters are bound — the
graph is not faceted, so a view cannot walk it — and `judge` and `order_by` only with a decision model; `include=` keeps a subset. Every tool's result is
bounded by `max_result_chars` (200,000): one over it loses rows from the end of its lists, then
text, and is marked `truncated`.
`search` and `recall` also take `order_by` — `relevant`, `evidence`, `contradicts` or `injection`
— which judges the hits and answers in that order, each hit showing its `judge`. The corpus is fixed when the tools are built and a model cannot name another.
Bound filters take one value each; the model can narrow them, never widen them, and they are
stamped on what `ingest`, `remember` and `supersede` write, so the model finds its own sources and stories — a story or
source outside the view is not found, exactly as a missing one, and text already open outside it is
refused without a word about that story. `remember` and `supersede` answer with `{"story": ...}`, the story written; like `search`
and `recall`, a tool result carries no `usage` — the bill is the host's, not the model's to read.
A call that fails raises, as any `Tools` does — the library's own exception, with its message — and
whatever runs the loop decides what the model reads; an engine run marks the step failed and shows
the model the message:

```python
await tools.dispatch("remember", {"text": "Ana prefers the Tuesday slot.", "subject": "user:ana"})
try:
    await tools.dispatch("delete_story", {"story_id": "missing"})
except NotFoundError as error:
    print(error)                                      # memory: story 'missing' was not found
```

`MemoryTools` is a `Tools` — the same two-method shape rfnry-engine uses — so it mounts on an
engine run as it is, and combines with tools of your own or with an agent's:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))

mine = Tools([Tool("now", "The current UTC time", body=now)])
tools = MemoryTools(memory, corpus_id="acme").only("search", "recall") + mine
```

## Storage

Two storages ship, both the `Storage` protocol: `InMemoryStorage()` for tests and small things,
`PostgresStorage(pool)` for everything else, taking an asyncpg pool you built.

```python
memory = Memory(InMemoryStorage(), embedding=HashEmbedding())          # a complete memory, no services
memory = Memory(PostgresStorage(await asyncpg.create_pool(dsn)), embedding=emb, generation=llm, decision=dm,
                relation_types=["CONNECTS_TO", "PART_OF"])
```

`initialize()` creates the tables and their full-text indexes, and is safe to run again. Vector
search is exact within a corpus — no approximate index, so a filter never costs a hit. The
tables are prefixed `memory_` so one database can also hold rfnry-engine's and rfnry-thread's. Each `PostgresStorage`
reads its vector dimension from the tables when initialized, so a write with vectors before
`initialize()` is a `StorageError`, never a row without its vector. The vector dimension
is fixed when the tables are first created — a different embedding dimension later is a
re-provision, not a migration, and `initialize()` says so with a `ConfigurationError` rather than
letting the first write fail: `postgres: tables were created for dimension 8; this memory embeds
1024 — re-provision or use another schema`. Tables created without an embedding model have no
vector column and are refused the same way. Before 1.0 the table layout itself may change between
versions, and that is a re-provision too: there are no migrations. Vectors record which
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

## At a glance

Everything the package exports, by role.

- **The system:** `Memory`; its model-facing door `MemoryTools`, a `Tools` of `Tool`s that combines with `+` and narrows with `only`/`without`.
- **What is stored:** `Source`, `Section`, `Chunk`, `Story`, `Entity`, `Relation`; inputs `SourceIn`, `EntityIn`, `RelationIn`.
- **What reads return:** `Hit`, `Hits`, `Judgement`, `PillarReport`, `Neighborhood`; writes return `Ingested`, `Replaced`, `Remembered`; deletes and `verify_absent` return `Counts`, rows by table.
- **Speak to models:** `EmbeddingModel`, `GenerationModel`, `DecisionModel`; `ReplyLike`, `DecidedLike`, `AnswerLike`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; `confidence`.
- **Store:** `Storage`, `InMemoryStorage`, `PostgresStorage`.
- **Fail loudly:** `MemoryException`, `ConfigurationError`, `InputError`, `ModelError`, `StorageError`, `NotFoundError`.

## Errors

`ConfigurationError` for a setup that cannot work: an unknown or unconfigured pillar, the vector
pillar without `embedding=`, an embedding dimension the storage was not created for, extraction without `relation_types`, `judge=True` without a decision
model, graph facts or `neighbors` with the graph pillar off, `reembed` without the vector pillar, a
storage without its extension, `weights` naming an unknown pillar or not positive, `chunk_chars`
below 1 or `overlap_chars` negative or not below it, a `MemoryTools` without a `corpus_id`, with a
filter bound to a list, or with `max_result_chars` below 1000. `InputError` for a bad argument — an unknown
kind or judgement, `Hits.by` on hits that were not judged, `top_k` or `limit` below 1, a `cursor` no
listing returned — a reserved or malformed filter key,
text containing a NUL character or a number that is NaN or infinite (on every storage, since Postgres
cannot store either), `history=True`
with `as_of`, a `neighbors` name or a relation endpoint shared by several types without
`entity_type` or `from_type`/`to_type`, a refused
`supersede`, a `SourceIn` with both or neither of `text` and `sections`, or a `replace` without
filters or with a source they do not match. `ModelError` when an adapter breaks the contract.
`StorageError` for the database. `NotFoundError` when `read_source`, `delete_source`, `delete_story`, `supersede`
or `neighbors` names something the corpus does not hold. All under `MemoryException` — the root
takes the `Exception` suffix so it never shadows Python's own `MemoryError`; every child ends in
`Error`.

Every message starts with the name of the module that raised it, then a lowercase sentence;
identifiers inside it keep their case: `memory: story 'abc' was not found`,
`filters: filter key 'corpus_id' is reserved — pass it as a call argument`. The library logs nothing.