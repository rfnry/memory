# rfnry-memory

rfnry-memory is a hybrid-retrieval memory system for AI agents.

It holds both halves of what an agent knows: the knowledge it is given — documents, the job RAG does
— and the memory it keeps — facts and events that change over time. One search runs keyword,
semantic and graph search together and returns one ranked list, and every memory knows when it was
true and when it was learned.

Every public name, with its signature, is in [the API reference](api.md).

A **corpus** holds two kinds of thing. **Sources** are written down: documents you ingest as text,
chunked and indexed. **Stories** are told: small slices of experience a model or your code
remembers — a fact, an event, a decision — each true from a moment until it is superseded or
ended. Both are indexed through the same three **pillars**: text (keyword search), vector (semantic
search) and graph (graph search). You choose one, two or all; every search runs the pillars you
chose and fuses their results into one ranked list — hybrid search — whose entries say which pillar
found them.

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

**Technique is the model's; the database is yours.** Storage decides only what the contract
needs — tenancy, both clocks, provenance, erasure, one-transaction writes. Every way to retrieve is
the model's to choose, through a tool argument, or a default you can change. Every change to your
database is a call you make.

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

A model that embeds a chunk in light of its document — late chunking, context-aware embedding — is a
`DocumentEmbeddingModel` as well: `embed_in_document` takes one document's chunks, in order. Memory
then embeds each source's chunks in one call and reuses no vector across documents, since the same
paragraph in two documents is two vectors. Give it a `model` of its own — staleness is by model name
— and switching to one is a model change: the fingerprint shows the old vectors stale until
`reembed`, which re-embeds a source's chunks together.

Models are matched by shape, statically and at runtime: `ReplyLike`, `DecidedLike` and `AnswerLike`
are protocols over the fields of `Reply`, `Decided` and the answers, and a result is read by those
fields, so an adapter typed against another library's classes with the same fields fits here
unchanged. `QuestionLike` is the same over the questions a decision model is handed; an adapter
branches on `question.type` and reads `options`, `levels` or `yes` and `no`. The instructions and
each of those are a string, or a JSON object or list when a question has parts of its own — memory's
own are strings — and an option may mean `None`; the adapter passes the structure to its model as it
is, or renders it where its model reads a prompt. Annotated with `QuestionLike` from either library,
rfnry-memory's or rfnry-engine's, one adapter serves both, statically and at runtime.

A `Message`'s `content` is a string or a list of parts — `Text`, `Image`, `Audio` and `Document`,
any of which is a `Part`, with `Content` for `str | list[Part]` — and an assistant turn carries
its calls in `tool_calls`. They are rfnry-engine's, copied byte for byte, so a message means the
same thing in every library of the family and one adapter translates both. Memory itself stays
text-in: no verb takes media, and `MemoryTools` answers with text and JSON.

The generation and decision contracts are the same ones rfnry-engine uses, so one adapter serves
both libraries: what a model returns is read by its fields, not its class, so a `Reply` or a
`Decided` built from rfnry-engine's types is accepted as it is, and its values are validated as
strictly as ours. The library asks a generation model for structured output through a forced tool
call — arguments that fail validation go back as a `Message(role="tool", is_error=True)` — and asks
a decision model `Choice`, `Score` and `Check` questions; it never streams and never holds a
conversation. `decide` may return the answers alone, or a `Decided(answers, usage, model)` when the
adapter can report what the call cost; a `Reply` carries its own `usage`. `Hits.usage`,
`Ingested.usage`, `Synced.usage` and `Remembered.usage` add up every generation and decision call
the read or write made — extraction, retries included, and identity on write; the judge on read. A
fake embedding is enough to run everything in tests:

```python
class HashEmbedding:
    model, dimension = "hash-8", 8
    async def embed(self, texts, *, purpose):
        return [[b / 255 for b in sha256(t.encode()).digest()[:8]] for t in texts]
```

## Getting started

Memory keeps its corpora in Postgres with pgvector; it is tested on Postgres 16 with pgvector 0.8.
pgvector 0.8 runs on Postgres 13 and later, where memory works the same way, untested; a vector index
needs pgvector 0.8.0 or later. For one on your machine, with memory's schema
prepared by the command the package installs:

```sh
docker run -d --name rfnry-memory-pg -p 55434:5432 -e POSTGRES_USER=memory \
  -e POSTGRES_PASSWORD=memory -e POSTGRES_DB=memory pgvector/pgvector:pg16
rfnry-memory create --url postgresql://memory:memory@127.0.0.1:55434/memory --dimension 8
```

```python
import asyncpg
from rfnry_memory import Memory, PostgresStorage

pool = await asyncpg.create_pool("postgresql://memory:memory@127.0.0.1:55434/memory",
                                 server_settings={"search_path": "memory, public"})
memory = Memory(PostgresStorage(pool), embedding=emb)
await memory.check()

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

rfnry-memory needs Python 3.14 or later.

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

A facet also takes a range, a dict of `gt`, `gte`, `lt` and `lte`: a score at least 0.8, a revision
after 2, a document signed in 2024. Its bounds are all numbers, compared as numbers (an int equals
the float of the same value), or all strings, compared by code point, which orders ISO 8601 dates
written alike (all UTC, or all plain dates) by time; memory parses no date. A value of the other
kind, or a missing one, does not match, and a list value matches when any element does, as with
exact values:

```python
hits = await memory.search("acme", "seal", filters={"metadata.confidence": {"gte": 0.8}})
signed, _ = await memory.list_sources("acme", filters={"metadata.signed": {"gte": "2024-01-01", "lt": "2025-01-01"}})
```

An exact value or a list reads the metadata index; a range reads the rows the rest of the call leaves.

**Validity.** A document can say when what it says is in force: a contract from March 2024 to
February 2027, a policy that takes effect next month. `valid_from` and `valid_to`, on `ingest` and on
each `SourceIn`, record it. Memory never guesses one, so a document given none is in force whenever
it is held. Each version also records when memory wrote it, `created_at`, and when the source of
truth last vouched for it, `confirmed_at`, and each of its chunks carries both with its validity, so
a hit says how current its document is. Every read that takes a moment honours validity: by default
it answers with the documents in force now, `as_of` with those in force then, and `history=True`
with every version. A document past its `valid_to` leaves current reads, as an ended story does; one
with no end is never hidden, and its dates are there for a reader to judge:

```python
contract = (await memory.ingest("acme", name="Supply contract", text=contract_text,
                                valid_from=datetime(2024, 3, 1, tzinfo=UTC),
                                valid_to=datetime(2027, 2, 28, tzinfo=UTC))).source
contract.valid_to == datetime(2027, 2, 28, tzinfo=UTC)                         # True
contract.confirmed_at == contract.created_at                                     # True
```

**Replacing a document.** Ingesting the same document again adds a second one. To put a new version
in the place of one you hold, name it: `ingest(..., replaces=source_id)` writes the new version and
retires the old one in a corpus that keeps history, or erases it in one that does not, as a `sync`
does with the old version of a name. The name may change with it:

```python
v1 = (await memory.ingest("acme", name="Returns policy", text="Returns within 30 days.")).source
v2 = await memory.ingest("acme", name="Returns policy", text="Returns within 60 days.",
                         replaces=v1.source_id)
(await memory.get_source("acme", v1.source_id)) is None                            # True
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
the last attempt wrote among the sources it syncs, and two runs leave one set. Syncs in one corpus
take turns, so two racing writers also leave one. Chunking, embedding, extraction and identity all
happen before the transaction opens: a model that fails leaves the old set whole. `ingest` and
`sync` take `on_event=` to report each of those steps as it happens (Progress events, below).

A source the list repeats exactly — the same name, metadata, text or sections, graph facts and
validity, under the same chunk sizes, pillars and extraction — is kept as it is: its `source_id`,
chunks, vectors and graph evidence stay, no model runs for it, its `confirmed_at` moves to the sync's
moment, and `synced.kept` names it. Every other source is
written anew with a new `source_id`, and a chunk whose exact text the corpus already holds, with a
vector from the same embedding model, takes that vector rather than a new embedding — in `ingest` as
in `sync`. So a redaction rewrites the one source it touched and embeds only the chunks it changed;
a document parsed into sections, one per page, keeps its other pages' chunks, while one long `text`
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
of the same name that replaced it. A sync's moment is later than the moment of every version it
retires, even when two hosts' clocks disagree, so each version stays current for a while at some
`known_at`. `ingest` never retires. Turning it on records from that moment.

A retired version stops being in force where its successor starts: at the successor's `valid_from`
when it gives one, so an update can be backdated to the start of the version it replaces at the
earliest, which corrects that version from its start, or at the sync's moment, where a successor
given no start then starts. With no successor it stops at the sync's moment. So an `as_of` read answers
with the version in force then. An update starts when it is written or earlier: one that would
start later while the version it replaces is in force is refused, so sync it the day it takes
effect. A version replaced before it began was never in force. When two hosts' clocks disagree, a
sync's moment can run ahead of a reader's clock, and until the reader's clock passes it a default
read shows the version the sync replaced, as a supersede's moment does for stories.

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
retired before a moment — call it on your own schedule — and `delete_source` erases a source
together with every version before it, so deleting a document deletes its history; `delete_corpus`
erases them with everything else:

```python
await memory.forget_retired("acme", before=datetime.now(UTC))               # versions retired before now
```

### Stories

A story is a sentence or two, about a subject, true from `valid_from` until `valid_to`: its
planned end, or the moment it is superseded or ended. `valid_from` defaults to the moment the call is made;
`created_at` is when memory wrote the story down, after whatever model work the write does.

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
remembered piecemeal. Each must be open and begin no later than the new story; the new one keeps
the union of their `source_ids`, and their subject and metadata when they all agree:

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

When what a story says stops being true and nothing takes its place — a preference withdrawn, an
issue resolved — `end_story` closes it with no successor, at `valid_to`: now by default, or a moment
that has passed. What the end means is yours; memory records that the story stopped being true, and
in `closed_at` when it learned so. An end still to come is a planned end, given as `valid_to` to
`remember` or `supersede`:

```python
calls = (await memory.remember("acme", "Ana takes calls only in the morning.", subject="user:ana",
                               valid_from=datetime(2026, 1, 5, tzinfo=UTC))).story
ended = await memory.end_story("acme", calls.story_id, valid_to=datetime(2026, 3, 1, tzinfo=UTC))
ended.valid_to == datetime(2026, 3, 1, tzinfo=UTC)                            # True
ended.superseded_by is None                                                   # True
```

A story can also turn out never to have been true — told wrong, misheard, a plan called off before
it began. A closing at the story's own `valid_from` says so: `end_story` with that `valid_to`
retracts it, and `supersede` with that `valid_from` corrects it from the start, the new story true
from the same moment. Either way the old story keeps an empty interval, `valid_to` equal to
`valid_from`, so no read at any moment finds it, while a `known_at` before the closing still shows
what memory believed and `history=True` shows what happened. A story that has not begun can be
retracted, or corrected from its start. Retracting a story that superseded others leaves them closed
at its start: what replaced them was never true, but they still ended then. When they held on,
correct the story from its start with what stayed true instead, or remember that again from its
start. The names a retracted story brought into the graph stay, as a closed story's do, so a
`known_at` read can replay them. Deleting is for what memory should not keep at all:

```python
slot = (await memory.remember("acme", "Ana prefers the Tuesday slot.", subject="user:ana",
                              valid_from=datetime(2026, 1, 5, tzinfo=UTC))).story
fixed = (await memory.supersede("acme", slot.story_id, "Ana prefers the Thursday slot.",
                                valid_from=slot.valid_from)).story
(await memory.get_stories("acme", [slot.story_id]))[0].valid_to == slot.valid_from   # True
fixed.valid_from == slot.valid_from                                                  # True
callback = (await memory.remember("acme", "Ana asked for a callback.", subject="user:ana")).story
retracted = await memory.end_story("acme", callback.story_id, valid_to=callback.valid_from)
retracted.valid_to == retracted.valid_from                                           # True
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

`delete_story` deletes a story outright and returns its `Counts` — the story with its vector, in the
corpus's index table too when it has one, the mentions and relations it evidenced, and the entities
and names only it carried. Deleting a story that superseded others leaves them closed; their
`superseded_by` names a story that no longer exists. `source_ids` on a story link it to the sources
it came from — records, not constraints. A story is open until it is superseded, ended, or its
planned end passes.

Two stories say the same thing when they have the same text, ignoring case and extra whitespace,
about the same `subject`, with the same `metadata`. Remembering what an open story already says
returns that story with its related stories (a search, judged when asked) and runs no extraction or
identity, and concurrent writers of it leave one open story; a story written already ended — a past
interval — returns instead the story that repeats it exactly, interval too, so a retry writes
nothing twice. Against an open story the second call's interval (`valid_from`, `valid_to`) and
`source_ids` are not applied. Metadata is compared as JSON values: key order does not matter and `1`
is `1.0`, while `true` is not `1`. The same text about another subject, or with other metadata, is
another story, so in a corpus whose users are views two users' identical sentences stay two
(Per-user memory, under Tenancy, has both ways to keep memory per user):

```python
ana = (await memory.remember("acme", "Prefers mornings.", subject="user:ana")).story
bo = (await memory.remember("acme", "prefers mornings.", subject="user:bo")).story
ana.story_id != bo.story_id                                                    # True
(await memory.remember("acme", "Prefers  MORNINGS.", subject="user:ana")).story == ana   # True
```

The new story of a `supersede` keeps the union of the closed stories' `source_ids`, and their
subject and metadata when they all agree, unless given. Superseding an unknown story is a
`NotFoundError`; superseding one that is not open, at a `valid_from` before it began or not
before its end, or into what another open story already says — its text, about the new story's
subject, with its metadata — is an `InputError`. The new story is a story like any other:
`supersede` takes the same `entities` and `relations` as `remember`, and with a generation model it
is read for them too, so the graph reaches it as it reached the old one. Ending an unknown story is
a `NotFoundError`; ending one that is not open, at a `valid_to` before it began or still to come,
or one that has not begun at any moment but its start, is an `InputError`. A story closes once:
superseded, ended or retracted, it is never closed again.

### Time

Memory keeps two clocks. **Valid time** is when a fact was true: a story is true from `valid_from`
until `valid_to`. **Known time** is when memory learned it: `created_at` is when a story was written
down, and `closed_at` when it was closed, by a supersede or an end. Both are taken as the write
lands, after its model calls, and a closing's moment is later than the moment each story it closes
was written down, even when two hosts' clocks disagree — so memory never claims to have known
something before it could have, nor closes a story before it was written. They differ whenever the
world changes before memory hears of it — the go-live moved on February 1, and memory was told on
February 10 — and memory keeps both.

Nothing true is lost. A change is a new story that supersedes the old one, never an edit; a story
can carry its planned end, or be ended with nothing in its place; one found never true keeps an empty
interval, so what memory believed stays answerable; several stories can become one, and `supersedes`
and `superseded_by` keep the lineage both ways.

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

Documents have both clocks too: their validity, when given, is valid time, and when memory wrote and
retired each version is known time. Documents keep history when the corpus does (see Sources,
Keeping history): a `known_at` read then sees the documents memory held at that moment, as it sees
the stories, and an `as_of` read the version of each in force then; `history=True` reads every
version. In a corpus that does not keep history, `sync` erases what it replaces, so a `known_at`
read sees only the documents written by then that are still held; a document synced again since
shows in neither version. Keep history where that answer matters.

### Pillars

| pillar | finds | needs |
| --- | --- | --- |
| `text` | what the query's words match: keyword search, ranked by BM25 or cover density | nothing |
| `vector` | what the query means: semantic search | `embedding=` |
| `graph` | what the query names, or the entities you name, and what they connect to: graph search | entities, from `generation=` or from you |

Choose at construction, narrow per query:

```python
memory = Memory(storage, embedding=emb, generation=llm, pillars={"text", "vector", "graph"},
                relation_types=["CONNECTS_TO", "PART_OF", "OWNS"], fusion=RRF(weights={"graph": 2.0}))
hits = await memory.search("acme", query, pillars={"vector"}, kinds=("chunk",), top_k=20)
```

`kinds` picks chunks, stories or both (the default); `top_k` is 10 unless given. The pillars' lists
are fused into one (reciprocal rank by default, below), and each hit's `provenance` says what each
pillar thought of it — a pillar that did not return the hit is absent, not zero. `hits.pillars`
reports what each pillar returned and how long it took; a pillar that fails is dropped and named
there, and the rest still answer. Under `filters` the graph pillar stands down rather than answer
with passages it did not filter, and is absent from `hits.pillars`.

`match` decides which passages keyword search considers, the rest ranking among them: `"any"` of
the query's words (the default), `"all"` of them, a `"phrase"` — these words in this order, side by
side, for exact names and codes — or `"web"`, the query read as a web search, with `"quoted
phrases"`, `or` and `-excluded` words; a query that only excludes looks for nothing, and a query in
`"web"` is never an error. It narrows keyword search only: a strict match that finds nothing leaves
the other pillars to answer.

```python
hits = await memory.search("acme", "feed line", match="phrase", pillars={"text"})
[h.text for h in hits.hits] == [feed_text]                                     # True
hits = await memory.search("acme", "line -tank", match="web", pillars={"text"})
[h.text for h in hits.hits] == [feed_text]                                     # True
```

### Ranking and fusion

How the text pillar ranks what it matches, and how the pillars' lists become one, are settings,
chosen at construction and changeable per call. The defaults are the standard baselines, and the
memory shows what it reads with:

| `text=` | rewards | fits |
| --- | --- | --- |
| `BM25(k1=1.2, b=0.75)`, the default | rare words over common ones; a repeated word helps less each time (`k1`, 0 counts presence only); a long passage held back (`b`, 0 never, 1 fully) | most collections; the parameters are Lucene's |
| `CoverDensity(length="ignore")` | the query's words sitting close together | names, codes and phrases of several words |

```python
memory = Memory(storage, embedding=emb, text=BM25(k1=0.9, b=0.4))
memory.text == BM25(k1=0.9, b=0.4)  # True
hits = await memory.search("acme", "feed line", text=CoverDensity())  # this call only
```

BM25's statistics — how many passages, how many hold each word, their mean length — come from the
passages the read can see that hold a query word, not from the whole corpus, chunks and stories
apart, so a rare word, a part code, outweighs common ones, and a filter or a retired version changes
them as it changes what the read sees. Under `"all"` or `"phrase"` every passage matched holds every
word, so the words count as equally rare and BM25 ranks by how often they appear and how long the
passage is. Cover density ranks by every word of the query: a passage that holds only some of them
matches but ranks after those holding all of them, closest together first, and among themselves by
how often they hold the query's words. Its `length` is Postgres's own choice of how much a long
passage is discounted: `"ignore"`, `"log"` or `"linear"`. Corpora searched together that share a
text search configuration are counted as one, so their keyword hits rank as they would in one corpus
holding them all; corpora read by different configurations read words differently, so their keyword
hits take turns by rank. Hits that rank equal rank the same way every time: by passage id, by code
point, then by corpus. Postgres records at most 256 positions of a word in a passage, and records
every position past 16383 as 16383, so a word's count stops at 256, and far into a very long passage
phrases and cover density can no longer tell positions apart.

| `fusion=` | combines | trade-off |
| --- | --- | --- |
| `RRF(k=60)`, the default | ranks only: each pillar adds `weight / (k + rank)` to every passage it returned | robust: keyword scores and vector similarities never need to agree on a scale, but how far ahead the first hit was is lost; a smaller `k` lets the top of each list count more. k = 60 is from the paper that introduced reciprocal rank fusion |
| `RelativeScore()` | each pillar's scores scaled to 0–1 by their own range, then summed by weight | keeps how far apart the hits are; one outlying score squashes the rest |

Both take `weights` — `{"vector": 2.0}` trusts the vector pillar twice as much; a pillar left out
weighs 1 — and `window`: each pillar hands in its first `top_k × window` candidates, 4 by default.
Hits of equal fused score keep the order in which the pillars, taken by name, first returned them,
so a search answers in the same order every time.

```python
memory = Memory(storage, embedding=emb, fusion=RRF(weights={"vector": 2.0}))
memory.fusion == RRF(weights={"vector": 2.0})  # True
hits = await memory.search("acme", "feed line", fusion=RelativeScore(window=8))  # this call only
```

`fingerprint` reports the ranking and the fusion a memory reads with.

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
evidences it is: a chunk of a source visible in the read, or a story true at the moment the read
asks for — now by default, or `as_of`, `known_at`, or every moment with `history=True`. When "Bob
owns Apollo" supersedes "Ana owns Apollo", `neighbors("acme", "Apollo")` walks to Bob, and `as_of` a
day in January walks to Ana. The graph pillar of `search` and `recall` walks the same way.

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

The graph pillar starts from the entities a question is about and walks `depth` hops from them: 0
the entities themselves, 1 what they connect to, 2 by default, 3 at most — deeper reaches more and
costs more. A model, or your code, names them in `entities` — each name found by its name or an
alias, in every type that holds it — and `relation_types` keeps the walk to those relations; without
`entities` or `around`, the pillar starts from the names it spots in the query, read as `spot` says
(below). Passages rank nearest first: those naming an entity you gave (or the query names), then
those naming one a hop away, then two — in every corpus searched. Within a distance, the passages
whose reached entities are rarer and more numerous come first, rarity among the passages reached
being the measure BM25 uses for words: two entities reached outrank one, and a hub that every
passage mentions counts for little; ties go by passage id.

```python
hits = await memory.search("acme", "what does it feed?", entities=["V-101"], pillars={"graph"})
[h.text for h in hits.hits] == [feed_text, return_text]                        # True
missing = await memory.search("acme", "valve status", entities=["V-1O1"])
missing.unresolved == {"V-1O1": ["V-101"]}                                     # True
named = await memory.search("acme", "what does it feed?", entities=["V-101"], depth=0)
[h.text for h in named.hits if "graph" in h.provenance] == [feed_text]          # True
```

A name memory does not hold is not an error: `hits.unresolved` maps it to up to five similar names,
so a model can ask again with the right one. `recall` takes `entities` and `relation_types` too, and
`remember` and `supersede` find their related stories the same way, from the new story's entities.

How the pillar finds names in a query is `spot`, the asker's choice, as `match` is for keyword
search. Each reading compares every entity's name and aliases with the query, case and spacing
aside:

| `spot` | A name is found when | `valve`, "the valves leak" | `P-3`, "P-30 tripped" | `Run`, "Motor1_Run stopped" | `Dana`, "Dana's shift" |
|---|---|---|---|---|---|
| `"words"` | its words appear in order, side by side, read in the corpus's language as keyword search reads text | found | — | found | found |
| `"exact"` | it appears whole: the text's edge, or a character other than a letter, a digit, `_` or `-`, on each side | — | — | — | found |
| `"substring"` | it appears anywhere, inside longer words too | found | found | found | found |
| `"off"` | never: the walk starts only from `entities` and `around` | — | — | — | — |

`"words"` suits prose, where a name comes back in its other forms; `"exact"` suits codes and tags,
where `P-3` is not `P-30` (it finds `Motor1_Run` in `Line3_PLC:Motor1_Run`, `:` being an edge);
`"substring"` reads names inside anything. Left out, `spot` is the memory's own when the call names
neither `entities` nor `around`, and `"off"` when it does; a `spot` you give always applies, and the
names it finds join the ones you named. The memory's own is `"words"` unless you give `Memory`
another — `Memory(..., spot="exact")` for a corpus of codes — and the tools tell the model which it
is. Each corpus reads its own language:

```python
quiet = await memory.search("acme", "what does P-30 feed?", pillars={"graph"})
quiet.hits == []                                                               # True
codes = Memory(storage, pillars={"graph"}, spot="exact")
codes.spot == "exact"                                                          # True
[h.text for h in (await codes.search("acme", "is P-3 running?")).hits] == [feed_text, return_text]   # True
loose = await memory.search("acme", "what does P-30 feed?", pillars={"graph"}, spot="substring")
[h.text for h in loose.hits] == [feed_text, return_text]                       # True
```

`"words"` drops stop words from names as keyword search drops them from text. In English a name made
only of them — `IT`, `Will`, `The Who` — has no words to find, and one partly made of them reads
wider: `AND gate` is found in "the OR gate output", `The Doors` in "close the door",
`Bank of America` in "a bank in America". Codes too: `a`, `i`, `s` and `t` are English stop words,
so `A-203` reads as `-203` and is found in "is B-203 open?", and `T-1` in "P-1 tripped" — read codes
and tags with `"exact"`. `"exact"` and `"substring"` keep every word, and so does a name given in
`entities`. A corpus whose names must keep their stop words can read with a configuration that has
none — the stemmer without its stop-word list — which `set_language` names like any other; keyword
search then keeps them too:

```sql
CREATE TEXT SEARCH DICTIONARY english_stem_all (TEMPLATE = snowball, Language = english);
CREATE TEXT SEARCH CONFIGURATION english_all (COPY = pg_catalog.english);
ALTER TEXT SEARCH CONFIGURATION english_all
  ALTER MAPPING FOR asciiword, asciihword, hword_asciipart, word, hword, hword_part WITH english_stem_all;
```

```python
await memory.set_language("staff", "english_all")
await memory.ingest("staff", name="roster", text="IT owns the laptops.",
                    entities=[EntityIn(name="IT", type="team")])
found = await memory.search("staff", "who in IT signs off?", pillars={"graph"})
[h.text for h in found.hits] == ["IT owns the laptops."]                       # True
```

With the graph pillar on, every hit carries the entities its passage mentions, so a model sees what
a passage is about and can name it next:

```python
[e.name for e in hits.hits[0].entities] == ["P-3", "V-101"]                    # True
```

### Scope

A search can look only where you, or a model, say. `sources` keeps it to the chunks of those
sources and the stories that cite them; `around` keeps it to the passages about the entities it
names or, within `hops` — 0, 1 by default, or 2 — what they connect to, along `relation_types` when
given:

```python
manual = await memory.search("acme", "return line", sources=[source_id])
[h.chunk.source_id for h in manual.hits] == [source_id, source_id]             # True
feed = await memory.search("acme", "where does it go?", around=["V-101"], hops=0)
[h.text for h in feed.hits] == [feed_text]                                     # True
```

Inside the scope every pillar runs as usual: keyword search ranks with its statistics taken from
inside it, as if the corpus held only the scope; vector search ranks inside it; graph search starts
from `around`, or from `entities` when given, and keeps to it. Both together keep what both keep,
and filters, the window, `kinds` and `subject` apply on top — a scope narrows, never widens. A
scope that holds nothing — an unknown source, a name memory does not hold, reported in
`unresolved` — finds nothing rather than everything. `sources` reads across every corpus named;
`around` finds its names in each, and needs the graph pillar. `recall` takes `sources`, `around` and
`hops` too.

### Expand

From a hit, look around it. `by="document"` reads the chunks beside a chunk in its source — `width`
on each side, 2 by default, in document order — or the `top_k` other stories about a story's subject
nearest it in time, in time order; `by="vector"` searches with the hit's own stored vector, more
like this, and calls no model; `by="graph"` walks from the entities the hit mentions and ranks as
graph search does:

```python
first = (await memory.search("acme", "feed line", pillars={"text"})).hits[0]
beside = await memory.expand("acme", (first.kind, first.id), by="document")
[h.text for h in beside.hits] == [return_text]                                 # True
like = await memory.expand("acme", (first.kind, first.id), by="vector", top_k=3)
linked = await memory.expand("acme", (first.kind, first.id), by="graph")
[h.text for h in linked.hits] == [return_text]                                 # True
```

The hit itself is never among the results, and each result's `provenance` names the mode that
found it. Filters, the window and `judge` work as on `search` — the judge reads the starting
passage's text as the query — and expand stays in the hit's corpus. `by="vector"` needs the vector
pillar and `by="graph"` the graph pillar; `kinds` picks what they return, `top_k` how many, and
`relation_types` keeps the walk to those relations.

### Describe

`describe` says what a corpus holds and what works on it, for a model about to plan a search of a
corpus it does not know. It returns a `CorpusDescription`:

```python
shape = await memory.describe("acme")
(shape.sources, shape.chunks, shape.entities, shape.relations) == (1, 2, 2, 1)   # True
shape.entity_types == {"pump": 1, "valve": 1}                                    # True
shape.relation_types, shape.language, shape.history, shape.pillars, shape.judge, shape.expand
```

It counts the latest version of each source and their chunks, the stories true now, the retired source versions,
the entities and the relation rows — one for each passage that says a relation, so a relation said
again counts again, as does one a changed sync wrote again in a corpus that keeps history — and
gives each entity and relation type with how many have it; it names the corpus's language, whether
it keeps history, the pillars on, whether a decision model can judge, and the ways `expand` can
look. It reads the corpus when you call it, never on a search.

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

A cascade reads wide and keeps the best: search with a large `top_k`, judge each hit for relevance,
order by it and keep the head. Memory annotates; the caller cuts:

```python
wide = await memory.search("acme", "is the March 3 date confirmed?", top_k=50, judge=("relevant",))
best = wide.by("relevant").hits[:10]
scores = [h.judge.relevant for h in best]
len(best) <= 10 and scores == sorted(scores, reverse=True)                       # True
```

Through `MemoryTools` the tools do it with `order_by`: `{"query": ..., "top_k": 50, "order_by":
"relevant"}` answers in that order, and the model reads the first ten.

For a strong reader that judges for itself the annotations cost nothing; for a decision agent
consuming memory — a small, literal state budget — they are the difference between a clean state
and a noisy one.

### Progress events

A long call can say how it goes. `ingest`, `sync`, `reembed`, `remember`, `supersede`, `search` and
`recall` take `on_event=`, an `EventHandler`, an async function awaited with each `Event` as it
happens: the call waits for it, buffers nothing and keeps nothing, and the
events of one call come one at a time, in the order they happen. What the handler does with them —
a progress bar, a message to a browser, a log line — is yours:

```python
seen = []

async def keep(event):
    seen.append(event)

await memory.ingest("acme", name="seal notes", text=manual_text, on_event=keep)
[type(e).__name__ for e in seen] == ["Chunked", "Embedded", "Extracted"]        # True
(seen[0].chunks, seen[-1].chunk, seen[-1].chunks) == (1, 1, 1)                  # True
```

A source sends `Chunked` once it is cut — `source` of `sources` and its `chunks` — so the work ahead
is known before any model call; then `Embedded` once its vectors are ready, `computed` of them sent
to the embedding model and the rest reusing stored vectors of identical text, with the time it took;
then `Extracted` after each chunk's graph step, with the entities and relations it staged and the
`usage` and `duration_ms` of its model calls. With a generation model the graph step is usually the
slow part, the one a progress bar follows. `sync` sends a `Chunked` for every source it writes, all
of them first; a source it keeps sends nothing. `reembed` sends an `Embedded` per batch, `done`
climbing to `total`, the stale passages counted once at the start. `remember` and `supersede` send
their story's `Embedded` and `Extracted`, `name=None`. A stage whose pillar is off sends nothing,
and the call returning is the end.

A search sends a `Retrieved` per pillar as it finishes — its `PillarReport`, a failed one with its
error — then `Ranked`, the hits before any judging, so they can be shown at once, then, with
`judge`, a `Judged` per hit as each decision returns, the hit with its judgement and that call's
`usage` and `duration_ms`. The pillars and the judge work at the same time; their events still come
one at a time. `recall` sends the same, and so does the related-stories search of `remember` and
`supersede`:

```python
seen.clear()
found = await memory.search("acme", "feed line", judge=True, on_event=keep)
sorted(e.pillar for e in seen if isinstance(e, Retrieved)) == ["graph", "text", "vector"]   # True
[h.id for h in next(e for e in seen if isinstance(e, Ranked)).hits] == [h.id for h in found.hits]   # True
[e.done for e in seen if isinstance(e, Judged)] == list(range(1, len(found.hits) + 1))   # True
```

A handler that raises fails the call with its own exception, and the work still running beside it —
the other pillars, the other judgements — is cancelled. Ingestion writes once, at the end, so
nothing is written: raising is how a handler stops a call, as cancelling the call's task is. The
hits in `Ranked` and `Judged` are copies: a handler that changes them changes nothing the call
returns.

## Tools

```python
tools = MemoryTools(memory, corpus_id="acme", filters={"metadata.line": "A"})
tools.definitions()                                   # search recall expand read_source list_sources ingest
await tools.dispatch("recall", {"query": "go-live"})  # remember supersede end_story delete_story list_stories
await tools.dispatch("ingest", {"name": "seal notes", "text": manual_text})
```

Thirteen tools, each the method of the same name with its arguments less the corpus, `sections` and
graph facts — so `recall` answers with current facts unless the model passes `history`, and `ingest`
takes the text the model has read, a document it was shown, under a name, answering with
`{"source": ...}` less its text; parsing stays yours. `expand` takes the `kind` and `id` of a hit.
`search` and `recall` take `pillars`, offering the configured ones — `["text"]` alone finds an exact
code or identifier — and `describe` says what a corpus holds and what the tools can do with it. The
tools keep limits of their own: `limit` defaults to 50 (at most 500), `top_k` is at most 100, the
`hops` of `neighbors` are 1 to 3, `depth` is 0 to 3, the `width` of `expand` is 1 to 10, and
`related` is 0 to 20. `search` and `recall` take `match` too, when the text pillar is on; ranking
and fusion are not the model's, and no tool takes them. `neighbors` is offered only when the graph
pillar is on and no filters are bound — the graph is not faceted, so a view cannot walk it — and so
are the `entities`, `relation_types`, `around`, `hops`, `depth` and `spot` of `search` and `recall`,
whose result carries `unresolved` when a name was not found, `expand` by graph and the entities on
each hit, while `sources` is always offered; `describe` is offered whenever no filters are bound,
graph or not; `expand` offers the modes the pillars support; a `pillars` the tools do not offer,
`graph` under bound filters or none at all, is an `InputError`; `judge` and `order_by` are offered
only with a decision model; `include=` keeps a subset. Every tool's result is bounded by
`max_result_chars` (200,000): one over it loses rows from the end of its lists, then text, and is
marked `truncated`. `search` and `recall` also take `order_by` — `relevant`, `evidence`,
`contradicts` or `injection` — which judges the hits and answers in that order, each hit showing its
`judge`. Bound with `judge=("relevant",)`, the tools offer a reranker's one judgement: `judge` asks
it and `order_by` orders by it. The corpora are fixed when the tools are built — `corpus_id="acme"`,
or `corpus_id=["hr", "ops"]` — and a model cannot name another. Bound to several, `search` and
`recall` read them all, or the ones the model names in their `corpus_id`, every hit saying its
corpus; `expand`, `read_source`, `list_sources` and `list_stories` take the `corpus_id` a hit gives,
and `describe` the one to describe; the tools that write, and `neighbors`, are offered for one
corpus only. A `corpus_id` outside the bound ones is an `InputError`. A bound filter takes one value,
several or a range, as any filter does: the model can narrow it, with values or a range, and a value
outside the bound ones is an `InputError`, never a wider view. A range the model gives keeps the
bound values inside it, and is refused when it keeps none; a range given over a bound range keeps
the tighter of each side, and is refused when the two do not overlap. The bound filters are stamped on what `ingest`, `remember` and
`supersede` write, so the model finds its own sources and stories; under a filter bound to several
values, what the model writes names one of them, and under a bound range it gives a value inside it. A story, a source or a passage to expand from
outside the view is not found, exactly as a missing one. What the tools write carries the view, so
it repeats only a story inside it: the same text outside the view is another story, and the model is
never told it is there. `remember` and `supersede` answer with `{"story": ..., "related": [...]}`,
the story written and the related open stories inside the view, each with its `entities` under the
same rule as a hit's, and `end_story` with `{"story": ...}`, the story it ended; like `search` and
`recall`, a tool result carries no `usage` — the bill is the host's, not the model's to read. A call
that fails raises, as any `Tools` does — the library's own exception, with its message — and
whatever runs the loop decides what the model reads; an engine run marks the step failed and shows
the model the message:

```python
await tools.dispatch("remember", {"text": "Ana prefers the Tuesday slot.", "subject": "user:ana"})
try:
    await tools.dispatch("delete_story", {"story_id": "missing"})
except NotFoundError as error:
    print(error)                                      # memory: story 'missing' was not found
```

In `MemoryTools`, `list_sources` takes `as_of`, `history` and `known_at`. A chunk row carries its
document's dates, `created_at`, `confirmed_at`, and `valid_from`, `valid_to` and `retired_at` when
set, so the model judges how current a document is; the search description tells it so. `ingest`
takes `valid_from` and `valid_to`, and `replaces`, the id of a document this one is a new version of,
which under bound filters must be inside them; it answers with `retired` and `deleted` when it
replaced one. In a corpus that keeps no history, a model's `replaces` erases the version it
replaces: keep history, or leave `ingest` out with `only`, where that matters.

`MemoryTools(..., on_event=handler)` passes the handler to every `search`, `recall`, `ingest`,
`remember` and `supersede` its tools call, so the host follows what an agent's memory calls are
doing (Progress events); the model never sees an event.

`MemoryTools` is a `Tools` — the same two-method shape rfnry-engine uses — so it mounts on an
engine run as it is, and combines with tools of your own or with an agent's:

```python
run = await engine.run("support", scope="acme", input=text,
                       tools=MemoryTools(memory, corpus_id="acme"))

mine = Tools([Tool("now", "The current UTC time", body=now)])
tools = MemoryTools(memory, corpus_id="acme").only("search", "recall") + mine
```

## Storage

Memory stores everything in Postgres: `PostgresStorage(pool)`, over an asyncpg pool you built, on
a database with pgvector, in a schema of its own with tables prefixed `memory_`, so one database
can also hold rfnry-engine's and rfnry-thread's.

### Preparing the database

Your app never creates or changes memory's tables. `rfnry-memory`, the command the package
installs, prepares the schema before the app runs:

```sh
rfnry-memory create --schema memory --dimension 1536          # or --no-vector
rfnry-memory create --schema memory --dimension 1536 --sql    # print every statement, run nothing
```

It takes the connection string with `--url`, or reads Postgres' standard `PG*` variables, and never
prints it. `create` makes the schema, installs the extensions its settings need into `public` —
`vector` with a dimension, `pg_trgm` for the graph unless `--no-graph` — and records the settings:
the embedding dimension, and the language new corpora take, `--language` or else the database's
`default_text_search_config` as it is then — and prints what it recorded. `--sql` starts with the
`search_path` the statements run under, so they land in the same schema when you run them yourself.
A role that cannot install an extension is told the statement to run as one that can. Memory's
tables never go in `public`; the extensions do, when the database has none yet, so dropping memory's
schema never drops them. It never drops an extension either: extensions belong to the whole
database, and a schema of another service shares them when its `search_path` reaches them.

Your pool points at the schema, and the app checks it at start:

```python
pool = await asyncpg.create_pool(dsn, server_settings={"search_path": "memory, public"})
memory = Memory(PostgresStorage(pool), embedding=emb, generation=llm, decision=dm,
                relation_types=["CONNECTS_TO", "PART_OF"])
await memory.check()
```

`check()` only reads. It raises one `ConfigurationError` naming every way the schema does not fit
this memory, each with what fixes it: no memory schema, a schema behind or ahead of this version, a
dimension the schema does not hold, an extension the pool's `search_path` cannot reach. Storage
learns its vector dimension from it, so a write with vectors before `check()` is a `StorageError`,
never a row without its vector.

A version of memory that changes its tables ships a migration, and the schema follows it in place,
its data kept:

```sh
rfnry-memory upgrade --schema memory --sql    # what would run
rfnry-memory upgrade --schema memory          # lists the steps, asks, runs them in one transaction
rfnry-memory check --schema memory            # exits 0 when the schema fits this version
```

Take a snapshot and stop the app instances running the old version first: `upgrade` reminds you,
asks before it runs unless `--yes`, and runs every pending step in one transaction under a lock. A
setting changes the same way. `rfnry-memory set --dimension 3072` deletes every stored vector and
vector index — vectors of another size cannot be kept — and `reembed` then fills each corpus again,
after which `set_vector_index` builds a corpus's index again;
`set --graph` installs `pg_trgm`; `set --language spanish` changes the language of corpora not yet
written.

Version 2 keeps each entity's names as a text-search query in its corpus's language, which the graph
pillar reads a query with (see Graph search). A schema made by rfnry-memory 0.1.0 is at version 1:
`check()` says so, and `rfnry-memory upgrade` brings it to 2 once, reading the names of every entity
already stored.

Version 3 gives documents their validity (see Sources, Validity): `valid_from`, `valid_to` and the
end each version was written with, on sources and chunks, and `confirmed_at` on sources. A schema made
by rfnry-memory 0.1.4 or earlier needs `rfnry-memory upgrade`, which fills them in from what the
schema already holds: each version's `confirmed_at` is its `created_at`, each retired version is in
force until its retirement, and the version that replaced it from then on. So every read answers on
existing data as before, except an `as_of` read that reaches documents in a corpus holding retired
versions, which now answers with the version in force at that moment.

Vector search is exact within a corpus, so a filter never costs a hit, unless the corpus has a
vector index: then it is approximate and can miss a close passage, it runs again exactly when the
index returns fewer than `top_k`, and a scoped search still reads every passage in its scope.
Keyword and graph search never read the index. Vectors record which embedding model produced them;
after changing the model, `await memory.reembed("acme")` — with `on_event=` to follow its batches —
brings a corpus up to date — sources could be re-ingested, stories cannot.

Each corpus reads its text in one language: the name of any text search configuration the database
has — one of Postgres' built-in languages, `english`, `french`, `german`, `russian`, `turkish` and
the rest, or `simple`, which neither stems nor drops stop words, for a corpus that mixes languages —
or a configuration you created, schema-qualified if you like. A corpus takes the schema's default
language the first time anything writes to it, and keeps it — the default `rfnry-memory create`
recorded, `--language` or else the database's `default_text_search_config` at that moment, and
`rfnry-memory set --language` changes for corpora not yet written; a corpus never written reads in
the default and keeps nothing:

```python
await memory.set_language("acme-es", "spanish")       # re-indexes what the corpus holds; returns how many passages
```

The change is one transaction, and writes to that corpus wait for it, so none lands under the old
language. The names of the corpus's entities are read again with it, so the graph pillar spots them
in the new language; the count is of passages only. Every configuration is read exactly as it is — a
built-in one as Postgres ships it, so in English `cafe` does not find `café`. To find words typed
without their accents, make a configuration that folds them and name it:

```sql
CREATE EXTENSION IF NOT EXISTS unaccent;
CREATE TEXT SEARCH CONFIGURATION pt_folded (COPY = pg_catalog.portuguese);
ALTER TEXT SEARCH CONFIGURATION pt_folded
  ALTER MAPPING FOR hword, hword_part, word WITH unaccent, portuguese_stem;
```

```python
await memory.set_language("acme-pt", "pt_folded")   # acao now finds ação
```

A name the database does not have is an `InputError`. `set_language` re-reads the corpus with the
configuration as it is now, so after you alter yours, naming it again brings the corpus's keyword
index up to date. A configuration dropped, or its schema renamed, after a corpus names it makes the
corpus's writes raise a `StorageError`, and its keyword pillar and the graph pillar's `"words"`
reading fail with one — `"exact"` and `"substring"` still read — until `set_language` names another
configuration. The language is kept as Postgres writes it, so `French` reads back as `french`, and
it goes with the corpus — `fingerprint` reports it, `verify_absent` counts it as `corpora`,
`delete_corpus` removes it, and a corpus written again after that takes the default of that moment.
In every built-in language a code is read by its parts: `M13_Z3_TEMP` is `m13`, `z3` and `temp`, so
a search for it also finds `M13_Z3_HEAT`, ranked below it, while `10.0.0.1` and `20x25x1` stay
whole.

`fingerprint` says what a corpus holds, so two measurements over it can show they read the same
thing. It reads the corpus when you call it, never on a search:

```python
fp = await memory.fingerprint("acme")
fp.sources, fp.chunks, fp.content, fp.stories, fp.story_content, fp.chunking, fp.language
fp.history, fp.retired, fp.vector_index
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

### Vector index

A corpus is searched exactly until you give it an index on its vectors — `HNSW` or `IVFFlat`, kept
in a table of its own. One call builds it, naming the method; another removes it:

```python
await memory.set_vector_index("manuals", HNSW())        # builds; returns how many passages the index holds
await memory.set_vector_index("manuals", HNSW(m=32, ef_construction=128, precision="half"))
fp = await memory.fingerprint("manuals")
fp.vector_index == HNSW(m=32, ef_construction=128, precision="half")    # True
await memory.set_vector_index("manuals", IVFFlat(lists=100))             # the new index, then the old one dropped
await memory.set_vector_index("manuals", IVFFlat(lists=100), rebuild=True)
await memory.set_vector_index("manuals", None)          # removes it; returns 0
```

A setting kept as JSON — a flag from a UI, say — reads back as one:

```python
from pydantic import TypeAdapter

flag = '{"method": "ivfflat", "lists": 100}'
await memory.set_vector_index("manuals", TypeAdapter(VectorIndex).validate_json(flag))
await memory.set_vector_index("manuals", None)
```

Building copies the vector of every chunk and story of the corpus into the table, while writes to
that corpus wait, then builds one index for chunks and one for stories with
`CREATE INDEX CONCURRENTLY`, while writes go on, and returns when it is done. A corpus that holds
nothing yet can be switched on. Called again with the same settings it builds nothing, unless
`rebuild=True`, which builds them again in place with `REINDEX INDEX CONCURRENTLY`; with other
settings, or the other method, it builds the new indexes before it drops the old, and searches use
the old until then. `None` drops the table. What exists in the
database is the setting, so every service sharing it sees the same, and two services switching one
corpus run one after the other.

Every write keeps the table in step, in the write's own transaction: a chunk or story written or
re-embedded reaches it, and one deleted leaves it, while a retired chunk or a closed story keeps
its vector there. `delete_corpus` drops the table once the corpus's writes in flight and a build
of it running elsewhere are done; `Counts.vectors` counts its rows, and `verify_absent` counts the
table itself under `corpora`, so one left behind is not clean. Switch a corpus on only once every
service writing to the schema runs this version: an older service's writes do not reach the
table, and its `delete_corpus` leaves the table behind.

A vector search of an indexed corpus — `search`, `recall`, `expand` by vector — reads the index,
in a read-only transaction of its own. Filters, the window, `subject` and `kinds` apply as they
do anywhere, and each passage is scored from its own vector, as an exact search scores it. The
search is approximate: it can miss a close passage. When the index returns fewer than `top_k`,
the search runs again exactly, so a filter that leaves few passages still returns them all. A
search scoped by `sources` or `around` never reads the index: it reads every passage in its
scope. Until the index of chunks or of stories is built and valid — while the first build runs,
or after one that failed — searches of that kind run exactly as before. Memory sets one setting
for each search, `hnsw.iterative_scan` or `ivfflat.iterative_scan` to `relaxed_order`, for the
method the index uses, so the index reads on past the passages the filters and the window hide,
retired chunks and closed stories among them.

What you leave out is left to pgvector, and reads back as left out: `HNSW`'s `m` and
`ef_construction`, `IVFFlat`'s `lists`. A value pgvector refuses is an `InputError` that carries its
reason, and nothing is built. `precision="half"` indexes the vectors at half precision, which halves
the index; unset, it is `"full"` up to 2000 dimensions and `"half"` above, up to the 4000 pgvector
indexes, and it reads back as built. `IVFFlat` learns its lists from the vectors there when it is
built: on an empty or tiny corpus it learns almost nothing, so load the corpus first, or build it
again with `rebuild=True` once it has grown. How a search reads the index is Postgres' to tune, on
connections you own: `hnsw.ef_search` and `hnsw.max_scan_tuples`, or `ivfflat.probes` — one list
unless you raise it — and `ivfflat.max_probes`, through the pool's `server_settings`, `ALTER ROLE`
or `ALTER DATABASE`.

```python
pool = await asyncpg.create_pool(dsn, server_settings={"hnsw.ef_search": "100", "ivfflat.probes": "10"})
```

`fingerprint` reports the index as built, in `vector_index`, an `HNSW` or an `IVFFlat` read from the
database, so two measurements can show they read the corpus the same way; it is `None` when the
corpus has no index, or one whose build did not finish. A build that fails raises a
`StorageError`, and the next call builds it again. The index needs pgvector 0.8.0 or later.

Exact search grows with the corpus, not with the table: at 1024 dimensions it costs about 57 ms
at 10,000 passages, 290 ms at 50,000 and 510 ms at 100,000. Most corpora never need an index; a
corpus of tens of thousands of passages is where one pays. Each indexed corpus has a table of its
own, so a search plans and locks the same whether the schema holds one indexed corpus or thousands.

On a large corpus, the copy makes writes to that corpus wait, for seconds at 100,000 passages;
the index build does not. `CREATE INDEX CONCURRENTLY` waits for every transaction older than it
anywhere in the database, so one long transaction delays it. Build from a `Memory` whose
pool has no `statement_timeout` and no `lock_timeout` — either fails the build, or the drop of the
old index, while it waits: a
100,000-passage corpus builds in minutes, as fast as `maintenance_work_mem` on that pool lets
it. A second call for the same corpus waits for the first build, whatever its `lock_timeout` or
`statement_timeout`; bound it with `asyncio.wait_for` if needed. A parallel build uses shared
memory: in Docker, start Postgres with `--shm-size` of 1 GB or more, or set
`max_parallel_maintenance_workers = 0` on the building pool. Postgres plans each search
from the table's statistics, which autovacuum keeps: a corpus switched on while empty has its
table read whole until autovacuum first analyzes it, so switch a corpus on once it holds its
passages.

## Tenancy

`corpus_id` is required on every verb and is the first column of every table. It is a reserved
filter key, so a query can never smuggle another corpus in. A tenant can hold several corpora, and
`search` and `recall` read the ones the call names:

```python
hits = await memory.search(["hr", "ops"], "who approves overtime?")   # one embedding, every hit's corpus_id
```

The query is embedded once. Keyword search reads the named corpora that share a text search
configuration as one corpus, one search per configuration, and configurations take turns by rank;
vector and graph search each
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

### Per-user memory

The corpus is the boundary memory enforces. Inside one, `subject`, `metadata` and `filters` are
views, as strong as every caller that passes them; `MemoryTools` binds them for a model. An app that
keeps memory per user chooses between two shapes.

A corpus per user, read together with the corpora everyone shares, when what each user tells memory
is theirs alone. Stories never meet across users, and a user leaves with their corpus:

```python
await memory.remember("user-dana", "Dana prefers the night shift.", subject="user:dana")
hits = await memory.search(["acme", "user-dana"], "who prefers nights?")
reads = MemoryTools(memory, corpus_id=["acme", "user-dana"]).only("search", "recall")
writes = MemoryTools(memory, corpus_id="user-dana").only("remember", "supersede", "list_stories")
tools = reads + writes
await memory.delete_corpus("user-dana")
(await memory.verify_absent("user-dana")).clean                                # True
```

One corpus with users as views, when sharing inside it is the point. What each user writes names
them in `subject` or `metadata`, and `filters` keeps each read to them. A story repeats another only
with the same subject and metadata, so two users' identical sentences stay two stories, and the
tools never tell one user what another wrote. The graph pillar stands down under filters, and
`verify_absent` proves a corpus gone, not a view:

```python
dana = MemoryTools(memory, corpus_id="acme", filters={"metadata.user": "dana"})
bo = MemoryTools(memory, corpus_id="acme", filters={"metadata.user": "bo"})
await dana.dispatch("remember", {"text": "Prefers the night shift."})
await bo.dispatch("remember", {"text": "Prefers the night shift."})
stories, _ = await memory.list_stories("acme", filters={"metadata.user": ["dana", "bo"]})
len(stories) == 2                                                              # True
```

## At a glance

Everything the package exports, by role.

- **The system:** `Memory`; its model-facing door `MemoryTools`, a `Tools` of `Tool`s that combines with `+` and narrows with `only`/`without`; how it ranks keyword matches, a `TextRanking` — `BM25` or `CoverDensity` — and how it fuses the pillars, a `Fusion` — `RRF` or `RelativeScore`.
- **What is stored:** `Source`, `Section`, `Chunk`, `Story`, `Entity`, `Relation`; inputs `SourceIn`, `EntityIn`, `RelationIn`; the names verbs take for them, `Kind`, `Ref` and `Language`; how a query's words match, `Match`; how the graph finds names in it, `Spot`.
- **What reads return:** `Hit`, `Hits`, `Judgement`, `JudgementName`, `PillarReport`, `Neighborhood`; `fingerprint` returns a `CorpusFingerprint` with its `Chunking` and `VectorIndex` — an `HNSW` or an `IVFFlat`, the settings `set_vector_index` takes — `describe` a `CorpusDescription`; writes return `Ingested`, `Synced`, `Remembered`; deletes and `verify_absent` return `Counts`, rows by table.
- **Speak to models:** `EmbeddingModel`, `DocumentEmbeddingModel`, `GenerationModel`, `DecisionModel`; the shapes a question and a result are read by, `QuestionLike`, `ReplyLike`, `DecidedLike`, `AnswerLike`; `Message`, `ToolSpec`, `ToolCall`, `Reply`, `Usage`, `State`; parts `Text`, `Image`, `Audio`, `Document`, `Part`, `Content`; `Choice`, `Score`, `Check`, `Question`; `ChoiceAnswer`, `ScoreAnswer`, `CheckAnswer`, `Answer`, `Decided`; `confidence`.
- **Follow a call:** `on_event` takes an `EventHandler`, awaited with each `Event`: `Chunked`, `Embedded`, `Extracted`, `Retrieved`, `Ranked`, `Judged`.
- **Store:** `PostgresStorage`.
- **Fail loudly:** `MemoryException`, `ConfigurationError`, `InputError`, `ModelError`, `StorageError`, `NotFoundError`.

## Errors

`ConfigurationError` for a setup that cannot work: an unknown or unconfigured pillar, the vector
pillar without `embedding=`, an embedding dimension the storage was not created for, extraction
without `relation_types`, a `judge` that asks for judgements without a decision model, graph facts,
`neighbors`, `entities`, `relation_types` or `around` with the graph pillar off, `expand` by vector
or by graph without that pillar, `reembed` without the vector pillar, a vector index without an
embedding model, above 4000 dimensions or on pgvector older than 0.8.0, a storage without its
extension, a `text` or `fusion` that is not one of the settings, a `spot` that is not one of the
four readings, `chunk_chars` below 1 or `overlap_chars` negative or not below it, a `MemoryTools`
without a `corpus_id` or with `max_result_chars` below 1000, a `MemoryTools(judge=)` that names no
judgement, an unknown one or one twice, two tools of the same name in one `Tools`, or a `Tool`
called without a `body`. `InputError` for a bad argument — an unknown kind or judgement, `Hits.by`
on a judgement that was not asked, a `judge` that is not `True`, `False` or a tuple naming each
judgement at most once, the tools' `order_by` outside the judgements they are bound to, `top_k` or
`limit` below 1, `hops` outside 0 to 2, `depth` outside 0 to 3, a `match` other than any, all,
phrase or web, a `spot` other than words, exact, substring or off, `width` below 1, an unknown
`expand` mode, a passage that is not a `(kind, id)` pair, `expand` by document from a story without
a subject, a `cursor` no listing returned — a reserved or malformed filter key, text containing a
NUL character or a number that is NaN or infinite (Postgres cannot store either), a story with blank
text, a `search` or `recall` with a blank query, an entity without a name or a type, a `valid_to`
not after `valid_from` (`remember` and `supersede`), `related` below 0, `history=True` with `as_of`,
a `neighbors` name or a relation endpoint shared by several types without `entity_type` or
`from_type`/`to_type`, a refused `supersede` or `end_story`, a `SourceIn` with both or neither of `text` and
`sections` or a `valid_to` not after its `valid_from`, a new version of a document that would take effect
after its write while the one it replaces is in force, or before the one it replaces took effect, an
`ingest` that `replaces` a source already replaced or retired, an `Image`, `Audio` or `Document` without exactly one of `data` and `url`, a `sync`
without filters or with a source they do not match, a language that is not a text search
configuration in the database, a `set_vector_index` whose index is not an `HNSW`, an `IVFFlat` or
`None`, whose `rebuild` is not a bool or asks to rebuild `None`, `"full"` above 2000 dimensions, a
value pgvector refuses, a `corpus_id` that names no corpus, a tool name `dispatch` or `only` does
not know, tool arguments that fail validation, or, through `MemoryTools`, a filter or `corpus_id`
outside what the tools are bound to, a graph argument, a `pillars` or an `expand` mode they do not
offer and written metadata that names none of a filter's bound values. `ModelError` when an adapter
breaks the contract. `StorageError` for the database. `NotFoundError` when `read_source`,
`delete_source`, `delete_story`, `supersede`, `end_story`, `neighbors`, `expand` or an `ingest` that
`replaces` names something the corpus does not hold. All under `MemoryException` — the root takes the `Exception` suffix so it never
shadows Python's own `MemoryError`; every child ends in `Error`.

Every message starts with the name of the module that raised it, then a lowercase sentence;
identifiers inside it keep their case: `memory: story 'abc' was not found`,
`filters: filter key 'corpus_id' is reserved — pass it as a call argument`. The library logs
nothing.