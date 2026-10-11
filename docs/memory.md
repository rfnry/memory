# rfnry-memory

rfnry-memory is a hybrid-retrieval memory system for AI agents.

It holds both halves of what an agent knows: the knowledge it is given — documents, the job RAG does
— and the memory it keeps — facts and events that change over time. One search runs keyword,
semantic and graph search together and returns one ranked list, and every memory knows when it was
true and when it was learned.

Every public name, with its signature, is in [the API reference](api.md).

A **corpus** holds two kinds of thing. **Sources** are written down: documents you ingest as text,
chunked and indexed. **Stories** are told: small slices of experience a model or your code
remembers — a fact, an event, a decision — each true from a moment until it is superseded. Both are
indexed through the same three **pillars**: text (keyword search), vector (semantic search) and
graph (graph search). You choose one, two or all; every search runs the pillars you chose and fuses
their results into one ranked list — hybrid search — whose entries say which pillar found them.

Models enter at two seams. On write, an embedding model indexes the vector pillar, a generation
model extracts entities and relations for the graph, and a decision model resolves whether two
entities are the same thing. On read, a decision model can annotate each result with four
judgements — relevant, evidence, contradicts, injection — and leave the decision to you. A tool
surface exposes the same verbs to a model, pinned to the corpora the host binds.

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

**Tenancy by construction.** Every verb names its corpora — `search` and `recall` one or several,
every other verb one. Every row carries its corpus. There is no default corpus and nothing reads a
corpus the call does not name. Deleting a corpus is provable, not promised.

## Models

Four protocols. Any object with the right method is a model; adapters to vendors are yours.

```python
class EmbeddingModel(Protocol):
    model: str
    dimension: int
    async def embed(self, texts: list[str], *, purpose: Literal["query", "document"]) -> list[list[float]]: ...

class DocumentEmbeddingModel(EmbeddingModel, Protocol):
    async def embed_in_document(self, texts: list[str]) -> list[list[float]]: ...

class GenerationModel(Protocol):
    async def generate(self, system: str, messages: list[Message], tools: list[ToolSpec]) -> ReplyLike: ...

class DecisionModel(Protocol):
    async def decide(self, state: State, questions: Mapping[str, QuestionLike]) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

`embed` is told what it embeds: `purpose="query"` for a search's query, `"document"` for the chunks
and stories it indexes. A model that embeds the two sides differently uses it; one that does not
ignores it.

A model that embeds a chunk in light of its document — late chunking, context-aware embedding — is a `DocumentEmbeddingModel` as well: `embed_in_document` takes one document's chunks, in order. Memory then embeds each source's chunks in one call and reuses no vector across documents, since the same paragraph in two documents is two vectors. Give it a `model` of its own — staleness is by model name — and switching to one is a model change: the fingerprint shows the old vectors stale until `reembed`, which re-embeds a source's chunks together.

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike` are
protocols over the fields of `Reply`, `Decided` and the answers, and a result is read by those
fields, so an adapter typed against another library's classes with the same fields fits here unchanged.
`QuestionLike` is the same over the questions a decision model is handed; an adapter branches on
`question.type` and reads `options`, `levels` or `yes` and `no`. The instructions and each of those
are a string, or a JSON object or list when a question has parts of its own — memory's own are
strings — and an option may mean `None`; the adapter passes the structure to its model as it is, or
renders it where its model reads a prompt. Annotated with `QuestionLike` from either library,
rfnry-memory's or rfnry-engine's, one adapter serves both, statically and at runtime.

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
`Ingested.usage`, `Synced.usage` and `Remembered.usage` add up every generation and decision call
the read or write made — extraction, retries included, and identity on write; the judge on read. A fake
embedding is enough to run everything in tests:

```python
class HashEmbedding:
    model, dimension = "hash-8", 8
    async def embed(self, texts, *, purpose):
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
    if hit.chunk:
        print(hit.chunk.source_name, hit.chunk.metadata, hit.chunk.locator)   # cite it, no extra read
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
    Section(title="Feed line", text=feed_text, locator={"sheet": 7, "region": "A2"},
            context="P&ID sheet 7, unit 300"),
    Section(title="Return line", text=return_text, locator={"sheet": 7, "region": "C4"}),
])
```

A section's `context` — free text your parser, or your own summarizer, gives it — travels with its
chunks. Keyword search matches a chunk on its source's name, its headings, its context and its text,
so a passage is found by what it belongs to.

`metadata` is yours. Every key becomes a facet you can filter on, with one value or any of several:

```python
hits = await memory.search("acme", "seal replacement", filters={"metadata.line": "A", "metadata.rev": ["2", "3"]})
sources, cursor = await memory.list_sources("acme", filters={"metadata.line": "A"})
source = await memory.get_source("acme", source_id)
text = await memory.read_source("acme", source_id)                      # the whole source
part = await memory.read_source("acme", source_id, locator={"sheet": 7})  # the chunks that match
deleted = await memory.delete_source("acme", source_id)
```

**Syncing a document.** A manual's new revision, or a program export that parses into dozens of
sources, takes the place of the version before it. `sync` makes the sources matching `filters`
exactly `sources`, in one transaction: new ones are added, changed ones replaced, missing ones
deleted. The first sync of a set adds it, an empty list removes it, and a retry changes nothing.
A reader sees the old set or the new set — never both, never neither:

```python
synced = await memory.sync("acme", filters={"metadata.file": "line-a.L5X"}, sources=[
    SourceIn(name="Routine MainRoutine", sections=rungs, metadata={"file": "line-a.L5X", "kind": "plc"},
             entities=[EntityIn(name="Motor_1", type="tag")]),
    SourceIn(name="Tags", text=tags_text, metadata={"file": "line-a.L5X", "kind": "plc"}),
])
synced.sources, synced.kept, synced.deleted, synced.usage
```

A `SourceIn` is what `ingest` takes, as one value: exactly one of `text` and `sections`. The filters
are those of `list_sources`, and they are required — emptying a corpus is `delete_corpus`. Every new
source must match them, or the call is an `InputError` before anything runs; so a retry finds what
the last attempt wrote among the sources it syncs, and two runs leave one set. Syncs in one corpus take turns, so two racing
writers also leave one. Chunking, embedding, extraction and identity all happen before the
transaction opens: a model that fails leaves the old set whole.

A source the list repeats exactly — the same name, metadata, text or sections and graph facts, under
the same chunk sizes, pillars and extraction — is kept as it is: its `source_id`, chunks, vectors and
graph evidence stay, no model runs for it, and `synced.kept` names it. Every other source is written
anew with a new `source_id`, and a chunk whose exact text the corpus already holds, with a vector
from the same embedding model, takes that vector rather than a new embedding — in `ingest` as in
`sync`. So a redaction rewrites the one source it touched and embeds only the chunks it changed; a
document parsed into sections, one per page, keeps its other pages' chunks, while one long `text`
can shift every chunk after the change. A kept source keeps the vectors it had: after changing the
embedding model, `reembed` brings them up to date. It also keeps what the generation and decision
models made of it, whichever now run: to write a set again under a new extractor, sync it empty,
then full. If another sync moves a kept source before this one's transaction — or, in a corpus that
keeps history, a newer write to the set lands first — this one writes its whole list again, its
models running again. Each old source goes as one source is deleted — its chunks with their vectors
and full-text rows, the mentions and relations its chunks evidenced, and the entities and names only
they carried (see Graph); a story keeps the `source_ids` it named. An entity new to the corpus that
several new sources name is created once. `sources=[]` deletes the set and writes nothing. `sync` is
for ingestion pipelines in code; `MemoryTools` does not offer it.

**Keeping history.** A corpus can keep the versions it replaces. With `set_history`, `sync` retires
the sources it would delete: they leave current reads but stay whole — text, chunks, vectors, graph
evidence — for reads that ask for the past, and each records when it was retired and the new source
of the same name that replaced it. `ingest` never retires. Turning it on records from that moment:

```python
await memory.set_history("acme", True)
before = await memory.sync("acme", filters={"metadata.doc": "returns"}, sources=[
    SourceIn(name="Returns policy", text="Returns are accepted within 30 days.", metadata={"doc": "returns"})])
after = await memory.sync("acme", filters={"metadata.doc": "returns"}, sources=[
    SourceIn(name="Returns policy", text="Returns are accepted within 60 days.", metadata={"doc": "returns"})])
after.retired == [before.sources[0].source_id]                                   # True
old = await memory.get_source("acme", before.sources[0].source_id)
old.replaced_by == after.sources[0].source_id                                    # True
versions, _ = await memory.list_sources("acme", filters={"metadata.doc": "returns"}, history=True)
```

Retired versions stay until you erase them: `forget_retired(corpus_id, before=...)` erases those
retired before a moment — call it on your own schedule — and `delete_source` erases a source together
with every version before it, so deleting a document deletes its history; `delete_corpus`
erases them with everything else:

```python
await memory.forget_retired("acme", before=datetime.now(UTC))               # versions retired before now
```

### Stories

A story is a sentence or two, about a subject, true from `valid_from` until `valid_to`: its
planned end, or the moment it is superseded; `created_at` is when it was written down, and `valid_from` defaults to it.

```python
remembered = await memory.remember("acme", "The go-live is February 20.", subject="project:apollo",
                                   valid_from=datetime(2026, 1, 10, tzinfo=UTC))
story = remembered.story
current = (await memory.supersede("acme", story.story_id, "Ana moved the go-live to March 3.",
                                  valid_from=datetime(2026, 2, 1, tzinfo=UTC))).story
```

A story can also carry the end it is known to have — a promotion until a date, a contract term — as
`valid_to` at the write. It is current until then, and a `supersede` can still close it earlier:

```python
promo = (await memory.remember("acme", "The spring promotion runs.", subject="promo:spring",
                               valid_from=datetime(2026, 3, 1, tzinfo=UTC),
                               valid_to=datetime(2026, 3, 31, tzinfo=UTC))).story
promo.valid_to == datetime(2026, 3, 31, tzinfo=UTC)                           # True
```

Both return a `Remembered`: the story written, `related`, and `usage`, what its extraction, identity
and judge cost. `related` is a `Hits` of the open stories closest to the new one — found by every
pillar that is on, the vector one reusing the story's own vector and the graph one starting from its
entities as well as the names in its text on a write (on a dedup the existing story's text is
embedded as a query) — so the writer sees what it may be repeating or contradicting before it moves
on; `filters` keeps them to a view, as in `search`, where the graph pillar stands down; `related=0`
skips it. With a decision model, `judge=True` judges each related story as `search` judges a hit,
the new story's text as the query: *evidence* means it already says what the new one says,
*contradicts* that both cannot be true. Nothing is acted on; superseding is the writer's call:

```python
latest = await memory.remember("acme", "Ana prefers the Wednesday slot.", subject="user:ana", judge=True)
for hit in latest.related.by("contradicts").hits:
    print(hit.id, hit.judge.contradicts, hit.text)
```

`supersede` closes the old story at the new one's `valid_from` and opens the new one there, pointing
back:

```python
current.supersedes == (story.story_id,)                                       # True
(await memory.get_stories("acme", [story.story_id]))[0].superseded_by == current.story_id   # True
(await memory.get_stories("acme", [story.story_id]))[0].valid_to == current.valid_from   # True
```

`supersede` also takes several stories and closes them all into one, the way to merge what was
remembered piecemeal. Each must be open and have begun before the new story; the new one keeps the
union of their `source_ids`, and their subject and metadata when they all agree:

```python
tuesday = (await memory.remember("acme", "Ana prefers Tuesday.", subject="user:ana",
                                 valid_from=datetime(2026, 1, 5, tzinfo=UTC))).story
mornings = (await memory.remember("acme", "Ana prefers mornings.", subject="user:ana",
                                  valid_from=datetime(2026, 1, 5, tzinfo=UTC))).story
merged = (await memory.supersede("acme", [tuesday.story_id, mornings.story_id],
                                 "Ana prefers Tuesday mornings.",
                                 valid_from=datetime(2026, 2, 1, tzinfo=UTC))).story
merged.supersedes == (tuesday.story_id, mornings.story_id)                     # True
```

Nothing is edited in place, so the past stays answerable:

```python
await memory.recall("acme", "go-live date", subject="project:apollo")                                    # → March 3
await memory.recall("acme", "go-live date", subject="project:apollo", as_of=datetime(2026, 1, 20, tzinfo=UTC))   # → February 20
await memory.recall("acme", "go-live date", subject="project:apollo", history=True)                      # → both
stories, cursor = await memory.list_stories("acme", subject="project:apollo", history=True)
stories = await memory.get_stories("acme", [story.story_id])                  # by id, closed or not
```

`known_at` asks the other clock: what memory knew at a moment, before what it learned later. A
story written after it is absent, and one closed later shows open, with the end it was written
with; `as_of` defaults to it, so `known_at` alone is what `recall` would have answered then:

```python
before = datetime(2000, 1, 1, tzinfo=UTC)
(await memory.recall("acme", "go-live date", known_at=before)).hits == []      # True
```

`recall` searches stories with the pillars you chose plus time. By default it answers with what is
true now: stories that have begun and have not ended. `as_of` asks what was true at another moment;
`history=True` searches the whole history, closed stories included, and each hit's `story.valid_to`
says which; the two together are an `InputError`. `since` and `until` bound `valid_from`,
`subject` narrows to one thing. Every read that answers with stories follows the same rule —
`search`, `list_stories` and `neighbors` show current stories unless given
`history=True`, and each also takes `as_of` and `known_at`.

`delete_story` deletes a story outright and returns its `Counts` — the story, the mentions and
relations it evidenced, and the entities and names only it carried. Deleting a story that
superseded others leaves them closed; their `superseded_by` names a story that no longer exists. `source_ids` on a story link it to the sources it came from —
records, not constraints. A story is open until it is superseded or its end passes. Remembering text an open story already
says returns that story with its related stories (a search, judged when asked) and runs no
extraction or identity, and concurrent writers of one text leave one open story; a
story written already ended — a past interval — returns instead the story that repeats it exactly,
text and interval, so a retry writes nothing twice. Against an open story the match is by text alone,
ignoring case and extra whitespace: the second call's interval (`valid_from`, `valid_to`), subject
and metadata are not applied. The new story of a `supersede` keeps the union of the closed
stories' `source_ids`, and their subject and metadata when they all agree, unless given. Superseding an unknown story is a
`NotFoundError`; superseding one that is not open, at a `valid_from` not after it began or not before its end, or into text another
open story already says is an `InputError`. The new story is a story like any other: `supersede` takes the same `entities` and
`relations` as `remember`, and with a generation model it is read for them too, so the graph reaches
it as it reached the old one.

### Time

Memory keeps two clocks. **Valid time** is when a fact was true: a story is true from `valid_from`
until `valid_to`. **Known time** is when memory learned it: `created_at` is when a story was written
down, and `closed_at` when its supersede was. They differ whenever the world changes before memory
hears of it — the go-live moved on February 1, and memory was told on February 10 — and memory
keeps both.

Nothing true is lost. A change is a new story that supersedes the old one, never an edit; a story
can carry its planned end; several stories can become one, and `supersedes` and `superseded_by` keep
the lineage both ways.

Every read picks a moment on each clock. By default it answers with what is true now, as known now;
`as_of` asks what was true at another moment, `known_at` what memory knew at one — `as_of` follows
it unless given — and `history=True` every moment:

```python
first = (await memory.remember("acme", "The go-live is February 20.", subject="project:apollo",
                               valid_from=datetime(2026, 1, 10, tzinfo=UTC))).story
told = datetime.now(UTC)
await memory.supersede("acme", first.story_id, "Ana moved the go-live to March 3.",
                       valid_from=datetime(2026, 2, 1, tzinfo=UTC))
await memory.recall("acme", "go-live date", subject="project:apollo")  # → March 3: true now
await memory.recall("acme", "go-live date", subject="project:apollo",
                    as_of=datetime(2026, 1, 20, tzinfo=UTC))  # → February 20: true then
await memory.recall("acme", "go-live date", subject="project:apollo",
                    known_at=told)  # → February 20: known then
```

So a corpus is a temporal knowledge graph. Entities and relations live on the passages that
evidence them, and the graph answers for the same moment as the passages: "Bob owns Apollo"
replaces "Ana owns Apollo" in `neighbors` exactly when its story does.

Documents keep history when the corpus does (see Sources, Keeping history): a `known_at` read then
sees the documents memory held at that moment, as it sees the stories; `history=True` reads every
version. In a corpus that does not keep history, `sync` erases what it replaces, so a `known_at` read
sees only the documents written by then that are still held; a document synced again since shows in
neither version. Keep history where that answer matters.

### Pillars

| pillar | finds | needs |
| --- | --- | --- |
| `text` | what the query's words match: keyword search, ranked by BM25 | nothing |
| `vector` | what the query means: semantic search | `embedding=` |
| `graph` | what the query names, or the entities you name, and what they connect to: graph search | entities, from `generation=` or from you |

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

Keyword hits rank by BM25 on both storages. Its statistics — how many passages, how many hold each
word, their mean length — come from the passages the read can see that hold a query word, chunks
and stories apart, so a rare word, a part code, outweighs common ones, and a filter or a retired
version changes them as it changes what the read sees. Corpora searched together that share a
language are counted as one, so their keyword hits rank as they would in one corpus holding them
all; corpora in different languages read words differently, so their keyword hits take turns by
rank.

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
that names both its ends — the first chunk when none does — which then records both its ends too.

A mention keeps the name its passage used, and an entity lives on that evidence. When passages go —
`delete_source`, `delete_story`, a `sync` that drops or rewrites a source — an entity no remaining
passage mentions and no relation names is deleted with them, and one still mentioned keeps only the
names some remaining passage uses: an alias only the deleted text carried is dropped, and a name
only it carried gives way to the first, in name order, of those still in use. So a redaction takes
the names it alone held, and deleting every document of a corpus leaves no entity behind. The one
exception: when every name still in use already belongs to another entity of the same type, the
entity keeps its own name. A write always leaves its entities carrying the names it used, so a
delete running at the same moment cannot strand a name, and a write that names an entity such a
delete has just removed brings it back, under the names that write uses. And you can walk it:

```python
hood = await memory.neighbors("acme", "P-3", hops=2, relation_types=["CONNECTS_TO"])
hood.entities, hood.relations, hood.passages
```

The graph answers for the same moment as the passages. A relation is there while some passage that
evidences it is: a chunk of a source visible in the read, or a story true at the moment the read asks for — now by default, or
`as_of`, `known_at`, or every moment with `history=True`. When "Bob owns Apollo" supersedes "Ana
owns Apollo", `neighbors("acme", "Apollo")` walks to Bob, and `as_of` a day in January walks to Ana.
The graph pillar of `search` and `recall` walks the same way.

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

### Graph search

The graph pillar starts from the entities a question is about and walks up to two hops from them.
A model, or your code, names them in `entities` — each name found by its name or an alias, in every
type that holds it — and `relation_types` keeps the walk to those relations; without `entities`, the
pillar starts from the entity names written in the query. A passage scores by the entities the walk
reached that it mentions: each counts for its distance — 1, ½ and ¼ for zero, one and two hops —
times its rarity among the passages reached, the measure BM25 uses for words. So a passage naming
the entity outranks one a hop away, two entities reached outrank one, and a hub that every passage
mentions counts for little; ties go by passage id.

```python
hits = await memory.search("acme", "what does it feed?", entities=["V-101"], pillars={"graph"})
[h.text for h in hits.hits] == [feed_text, return_text]                        # True
missing = await memory.search("acme", "valve status", entities=["V-1O1"])
missing.unresolved == {"V-1O1": ["V-101"]}                                     # True
```

A name memory does not hold is not an error: `hits.unresolved` maps it to up to five similar names,
so a model can ask again with the right one. `recall` takes `entities` and `relation_types` too, and
`remember` and `supersede` find their related stories the same way, from the new story's entities.

### Judging and reranking

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

Reranking is yours to ask for: `by` returns a copy ordered by one judgement, highest first
(`descending=False` for lowest), ties in fused order; hits searched without `judge=True` raise
`InputError`:

```python
strongest = hits.by("evidence").hits[0]
```

A reranker — a cross-encoder scoring how well a passage answers a query — is a decision model that
answers one judgement. Ask for that one, and order by it:

```python
hits = await memory.search("acme", "is the March 3 date confirmed?", judge=("relevant",))
reranked = hits.by("relevant")
```

The others stay `None`; `by` on one that was not asked raises `InputError`.

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
answering with `{"source": ...}` less its text; parsing stays yours. The tools keep limits of
their own: `limit` defaults to 50 (at most 500), `top_k` is at most 100, `hops` is 1 to 3, and
`related` is 0 to 20. `neighbors` is offered only when the graph pillar is on and no filters are bound — the
graph is not faceted, so a view cannot walk it — and so are the `entities` and `relation_types` of
`search` and `recall`, whose result carries `unresolved` when a name was not found; `judge` and
`order_by` are offered only with a decision model; `include=` keeps a subset. Every tool's result is
bounded by `max_result_chars` (200,000): one over it loses rows from the end of its lists, then
text, and is marked `truncated`.
`search` and `recall` also take `order_by` — `relevant`, `evidence`, `contradicts` or `injection`
— which judges the hits and answers in that order, each hit showing its `judge`. Bound with `judge=("relevant",)`, the tools offer a reranker's one judgement: `judge` asks it and `order_by` orders by it. The corpora are fixed when the tools are built — `corpus_id="acme"`, or `corpus_id=["hr", "ops"]` —
and a model cannot name another. Bound to several, `search` and `recall` read them all, or the ones
the model names in their `corpus_id`, every hit saying its corpus; `read_source`, `list_sources` and
`list_stories` take the `corpus_id` a hit gives; the tools that write, and `neighbors`, are offered
for one corpus only. A `corpus_id` outside the bound ones is an `InputError`.
A bound filter takes one value or several, as any filter does: the model can narrow it, and a value
outside the bound ones is an `InputError`, never a wider view. The bound filters are stamped on what
`ingest`, `remember` and `supersede` write, so the model finds its own sources and stories; under a
filter bound to several values, what the model writes names one of them. A story or source outside
the view is not found, exactly as a missing one, and text already open outside it, or a past story
outside it repeated exactly, is refused without a word about that story. `remember` and `supersede`
answer with `{"story": ..., "related": [...]}`, the story written and the related open stories
inside the view; like `search` and `recall`, a tool result carries no `usage` — the bill is the
host's, not the model's to read.
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

In `MemoryTools`, `list_sources` takes `history` and `known_at`, and a chunk hit from a retired
version says `retired_at`.

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
`PostgresStorage(pool)` for everything else, taking an asyncpg pool you built. The in-memory text
pillar reads words as Postgres does in the corpus's language — lowercased, stop words dropped,
Snowball-stemmed — except what Postgres' parser keeps whole or drops: emails, URLs, file paths,
versions, hyphenated compounds and markup are read there as plain words. A storage of your own
implements the same protocol: its methods take and return the stored types and `SourceWrite`,
`GraphWrite`, `Window`, `Scope`, `Kind`, `Language`, `Ref` and `Leg`, all exported.

```python
memory = Memory(InMemoryStorage(), embedding=HashEmbedding())          # a complete memory, no services
memory = Memory(PostgresStorage(await asyncpg.create_pool(dsn)), embedding=emb, generation=llm, decision=dm,
                relation_types=["CONNECTS_TO", "PART_OF"])
```

`initialize()` creates the tables and their full-text indexes, and is safe to run again. It also
upgrades older tables in place — it adds the closure columns to `memory_stories`, turns `supersedes`
into a list, backfills the lineage and drops the old open-text unique index, indexes the entities'
aliases, and adds the history columns to `memory_corpora`, `memory_sources` and `memory_chunks`, the
chunks' `context` column and a keyword column that covers each chunk's source name and context.
Rebuilding that column rewrites `memory_chunks` once, so on a large table run the first
`initialize()` of this version in a maintenance window. The upgrade runs once and holds a
database-wide lock while it does, so `initialize()` on other schemas of the same database
waits for it.
The upgrade is one-way: every service using a schema must run this version together. Vector
search is exact within a corpus — no approximate index, so a filter never costs a hit. The
tables are prefixed `memory_` so one database can also hold rfnry-engine's and rfnry-thread's. Each `PostgresStorage`
reads its vector dimension from the tables when initialized, so a write with vectors before
`initialize()` is a `StorageError`, never a row without its vector. The vector dimension
is fixed when the tables are first created — a different embedding dimension later is a
re-provision, not a migration, and `initialize()` says so with a `ConfigurationError` rather than
letting the first write fail: `postgres: tables were created for dimension 8; this memory embeds
1024 — re-provision or use another schema`. Tables created without an embedding model have no
vector column and are refused the same way. Before 1.0 the table layout may change between
versions; what `initialize()` cannot upgrade in place is a re-provision. Vectors record which
embedding model produced them; after changing the model, `await memory.reembed("acme")` brings a
corpus up to date — sources could be re-ingested, stories cannot.

Each corpus reads its text in one language: `english` by default, `spanish`, `portuguese`, or
`simple` — no stemming and no stop words, for a corpus that mixes languages:

```python
await memory.set_language("acme-es", "spanish")       # re-indexes what the corpus holds; returns how many passages
```

The change is one transaction, and writes to that corpus wait for it, so none lands under the old
language. `spanish`, `portuguese` and `simple` fold accents, so a question typed without them still
finds the text: `informacion` finds `información`. The language goes with the corpus —
`fingerprint` reports it, `verify_absent` counts it as `corpora`, `delete_corpus` removes it. In
every language a code is read by its parts: `M13_Z3_TEMP` is `m13`, `z3` and `temp`, so a search
for it also finds `M13_Z3_HEAT`, ranked below it, while `10.0.0.1` and `20x25x1` stay whole.

`fingerprint` says what a corpus holds, so two measurements over it can show they read the same
thing. It reads the corpus when you call it, never on a search:

```python
fp = await memory.fingerprint("acme")
fp.sources, fp.chunks, fp.content, fp.stories, fp.story_content, fp.chunking, fp.language
fp.history, fp.retired
fp.embedding, fp.embedded, fp.stale
```

`sources` and `chunks` count the current ones, `history` is the corpus's history setting and
`retired` counts the source versions it keeps retired. `content` is a short hash over the current
sources' name, metadata and text, so a corpus that keeps history compares equal on identical
current content; `story_content` is one over
the stories — their text, subject, metadata and when each was true — kept apart because stories
change daily and sources do not. Neither reads an id or a write time, so re-ingesting or
re-syncing identical content leaves them as they were. `chunking` is a `Chunking`: the chunk
sizes and the pillars; `language` is the corpus's.

`embedding` is the model this memory's vector pillar embeds with, `None` without it; `embedded` is
what the corpus holds: how many chunks and stories, together, carry a vector from each model.
`stale` counts the chunks and stories the vector pillar would not find correctly — a vector from
another model, or none at all. A corpus half-moved to a new model, or written while the vector
pillar was off, says so here; `await memory.reembed("acme")` embeds those again and brings `stale`
to 0. Without the vector pillar nothing is searched by vector, so `stale` is 0 whatever `embedded`
holds. In a corpus that keeps history, `embedded` and `stale` count every vector, retired versions
included.

## Tenancy

`corpus_id` is required on every verb and is the first column of every table. It is a reserved
filter key, so a query can never smuggle another corpus in. A tenant can hold several corpora, and
`search` and `recall` read the ones the call names:

```python
hits = await memory.search(["hr", "ops"], "who approves overtime?")   # one embedding, every hit's corpus_id
```

The query is embedded once. Keyword search reads the named corpora that share a language as one
corpus, one search per language, and languages take turns by rank; vector and graph search each
named corpus and merge their lists by score. The pillars are fused as for one corpus, and a pillar
that fails in one corpus fails for the search. The searches run at the same time, each on a
connection of its own.
A corpus the call does not name is never read, and every other verb reads one. To offboard:

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
- **What reads return:** `Hit`, `Hits`, `Judgement`, `JudgementName`, `PillarReport`, `Neighborhood`; `fingerprint` returns a `CorpusFingerprint` with its `Chunking`; writes return `Ingested`, `Synced`, `Remembered`; deletes and `verify_absent` return `Counts`, rows by table.
- **Speak to models:** `EmbeddingModel`, `DocumentEmbeddingModel`, `GenerationModel`, `DecisionModel`; the shapes a question and a result are read by, `QuestionLike`, `ReplyLike`, `DecidedLike`, `AnswerLike`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; `confidence`.
- **Store:** `Storage`, `InMemoryStorage`, `PostgresStorage`; for a storage of your own, `SourceWrite`, `GraphWrite`, `Window`, `Scope`, `Kind`, `Language`, `Ref`, `Leg`.
- **Fail loudly:** `MemoryException`, `ConfigurationError`, `InputError`, `ModelError`, `StorageError`, `NotFoundError`.

## Errors

`ConfigurationError` for a setup that cannot work: an unknown or unconfigured pillar, the vector
pillar without `embedding=`, an embedding dimension the storage was not created for, extraction
without `relation_types`, a `judge` that asks for judgements without a decision model, graph facts,
`neighbors`, `entities` or `relation_types` with the graph pillar off, `reembed` without the vector
pillar, a storage without its extension, `weights` naming an unknown pillar or not positive,
`chunk_chars` below 1 or `overlap_chars` negative or not below it, a `MemoryTools` without a
`corpus_id` or with `max_result_chars` below 1000, a `MemoryTools(judge=)` that names no judgement,
an unknown one or one twice, two tools of the same name in one `Tools`, or a `Tool` called without a
`body`. `InputError` for a bad argument — an unknown kind or judgement, `Hits.by` on a judgement
that was not asked, a `judge` that is not `True`, `False` or a tuple naming each judgement at most
once, the tools' `order_by` outside the judgements they are bound to, `top_k` or `limit` below 1, a
`cursor` no listing returned — a reserved or malformed filter key, text containing a NUL character
or a number that is NaN or infinite (on every storage, since Postgres cannot store either), a story
with blank text, a `valid_to` not after `valid_from` (`remember` and `supersede`), `related` below
0, `history=True` with `as_of`, a `neighbors` name or a relation endpoint shared by several types
without `entity_type` or `from_type`/`to_type`, a refused `supersede`, a `SourceIn` with both or
neither of `text` and `sections`, an `Image`, `Audio` or `Document` without exactly one of `data`
and `url`, a `sync` without filters or with a source they do not match, an unknown language, a
`corpus_id` that names no corpus, a tool name `dispatch` or `only` does not know, tool arguments
that fail validation, or, through `MemoryTools`, a filter or `corpus_id` outside what the tools are
bound to, a graph argument they do not offer and written metadata that names none of a filter's
bound values. `ModelError` when an adapter breaks the contract. `StorageError` for the database.
`NotFoundError` when `read_source`, `delete_source`, `delete_story`, `supersede` or `neighbors`
names something the corpus does not hold. All under `MemoryException` — the root takes the
`Exception` suffix so it never shadows Python's own `MemoryError`; every child ends in `Error`.

Every message starts with the name of the module that raised it, then a lowercase sentence;
identifiers inside it keep their case: `memory: story 'abc' was not found`,
`filters: filter key 'corpus_id' is reserved — pass it as a call argument`. The library logs
nothing.