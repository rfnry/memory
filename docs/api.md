# rfnry-memory API

The reference: every public name, its signature, what it returns and what it raises. How and why to use them is in [the guide](memory.md).

## Models

### `EmbeddingModel`

```python
@runtime_checkable
class EmbeddingModel(Protocol):
    model: str
    dimension: int

    async def embed(
        self, texts: list[str], *, purpose: Literal["query", "document"]
    ) -> list[list[float]]: ...
```

The model that indexes the vector pillar: any object with these members. `embed` returns one vector of `dimension` floats per text, in order, and `model` is recorded on every vector it produces. `purpose` says what the texts are: `"query"` for a search's query, `"document"` for the chunks and stories it indexes — so a model that embeds the two sides differently gets each right, and one that does not ignores it. A verb that gets another number of vectors, or a vector of another length, raises `ModelError`.

### `DocumentEmbeddingModel`

```python
@runtime_checkable
class DocumentEmbeddingModel(EmbeddingModel, Protocol):
    async def embed_in_document(self, texts: list[str]) -> list[list[float]]: ...
```

An `EmbeddingModel` that embeds a document's chunks in light of each other — late chunking, context-aware embedding: `embed_in_document` takes one document's chunks in order and returns one vector each. With such a model, `ingest` and `sync` embed each source's chunks in one call and reuse no vector across documents, and `reembed` re-embeds a source's chunks together; queries and stories go through `embed`. Give it a `model` of its own: staleness is by model name, so the fingerprint shows the old vectors stale until `reembed`. A wrong number of vectors, or a vector of another length, raises `ModelError`.

### `GenerationModel`

```python
@runtime_checkable
class GenerationModel(Protocol):
    async def generate(
        self, system: str, messages: list[Message], tools: list[ToolSpec]
    ) -> ReplyLike: ...
```

The model that extracts entities and relations on write: any object with this method. Memory asks it for structured output through a forced call to one tool, sends invalid arguments back as a `Message(role="tool", is_error=True)`, and never streams. A verb raises `ModelError` when a reply does not fit `Reply`, or when three attempts bring no valid call.

### `DecisionModel`

```python
@runtime_checkable
class DecisionModel(Protocol):
    async def decide(
        self, state: State, questions: Mapping[str, QuestionLike]
    ) -> Mapping[str, AnswerLike] | DecidedLike: ...
```

The model that resolves entity identity on write and judges hits on read: any object with this method. It answers every question by its id, returning the answers alone or a `DecidedLike` that also reports usage. A verb raises `ModelError` when the answers do not fit: one missing or of another type than its question, a choice or a score its question does not have, or probabilities that do not cover its options or levels, lie outside 0 to 1, or do not sum to 1.

### `Message`

```python
class Message(BaseModel):
    role: Literal["user", "assistant", "tool"]
    content: Content = ""
    tool_calls: list[ToolCall] = []
    tool_call_id: str | None = None
    reasoning: Any = None
    is_error: bool = False
```

One turn handed to a generation model. An assistant turn carries its calls in `tool_calls` and its `reasoning`; a tool turn answers the call `tool_call_id` names, with `is_error` set when that call failed.

### `Text`

```python
class Text(BaseModel):
    type: Literal["text"] = "text"
    text: str
```

A text part of a message's content.

### `Image`

```python
class Image(BaseModel):
    type: Literal["image"] = "image"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An image part, as its bytes or a URL; `data` is base64 in JSON. Raises `InputError` when it carries both `data` and `url`, or neither.

### `Audio`

```python
class Audio(BaseModel):
    type: Literal["audio"] = "audio"
    media_type: str
    data: bytes | None = None
    url: str | None = None
```

An audio part, as its bytes or a URL; `data` is base64 in JSON. Raises `InputError` when it carries both `data` and `url`, or neither.

### `Document`

```python
class Document(BaseModel):
    type: Literal["document"] = "document"
    media_type: str
    data: bytes | None = None
    url: str | None = None
    name: str | None = None
```

A document part, as its bytes or a URL, with an optional file name; `data` is base64 in JSON. Raises `InputError` when it carries both `data` and `url`, or neither.

### `Part`

```python
Part = Annotated[Text | Image | Audio | Document, Field(discriminator="type")]
```

Any one content part, told apart by `type`.

### `Content`

```python
Content = str | list[Part]
```

A message's content: a string or a list of parts.

### `ToolSpec`

```python
class ToolSpec(BaseModel):
    name: str
    description: str
    input_schema: dict[str, Any]
```

A tool offered to a generation model, with the JSON schema of its arguments.

### `ToolCall`

```python
class ToolCall(BaseModel):
    id: str
    name: str
    args: dict[str, Any] = {}
```

A call a generation model made to a tool, with its arguments.

### `Reply`

```python
class Reply(BaseModel):
    text: str = ""
    tool_calls: list[ToolCall] = []
    usage: Usage | None = None
    stop_reason: StopReason = "end_turn"
    model: str | None = None
    reasoning: Any = None
```

What a generation model returns: its text, its tool calls, what the call cost and why it stopped. `StopReason` is `Literal["end_turn", "tool_use", "max_tokens", "error"]`.

### `Usage`

```python
class Usage(BaseModel):
    input: int = 0
    output: int = 0
    cache_creation: int = 0
    cache_read: int = 0

    def __add__(self, other: Usage) -> Usage: ...
```

Token counts of model calls, none negative; frozen, and unknown fields are refused. `a + b` returns the sum, field by field.

### `ReplyLike`

```python
class ReplyLike(Protocol):
    @property
    def text(self) -> str: ...
    @property
    def tool_calls(self) -> Sequence[ToolCallLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def stop_reason(self) -> StopReason: ...
    @property
    def model(self) -> str | None: ...
    @property
    def reasoning(self) -> Any: ...
```

What a generation model's reply is read by: the fields of `Reply`, so any object that has them fits, whatever its class. `ToolCallLike` and `UsageLike` are the same over the fields of `ToolCall` and `Usage`. The values are validated as strictly as a `Reply`'s.

### `State`

```python
State = str | dict[str, Any] | list[Any]
```

What a decision model is shown along with its questions.

### `Choice`

```python
class Choice(BaseModel):
    type: Literal["choice"] = "choice"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    options: dict[str, str | dict[str, JsonValue] | list[JsonValue] | None]
```

A question answered by picking one of its options, keyed by id, each with what it means or `None` when the id says enough. `instructions` and what an option means are a string, or a JSON object or list; `JsonValue` is pydantic's, any value JSON can hold. Raises pydantic's `ValidationError` with fewer than two options.

### `Score`

```python
class Score(BaseModel):
    type: Literal["score"] = "score"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    levels: list[str | dict[str, JsonValue] | list[JsonValue]]
```

A question answered on its `levels`, in order; `instructions` and each level are a string, or a JSON object or list. Raises pydantic's `ValidationError` with fewer than two levels.

### `Check`

```python
class Check(BaseModel):
    type: Literal["check"] = "check"
    instructions: str | dict[str, JsonValue] | list[JsonValue]
    yes: str | dict[str, JsonValue] | list[JsonValue] | None = None
    no: str | dict[str, JsonValue] | list[JsonValue] | None = None
```

A yes-or-no question; `yes` and `no` may say what each answer means. `instructions`, `yes` and `no` are a string, or a JSON object or list.

### `Question`

```python
Question = Choice | Score | Check
```

The three kinds of question. Memory asks a decision model four `Check`s to judge a hit and one `Score` to resolve an entity's identity.

### `QuestionLike`

```python
QuestionLike = ChoiceLike | ScoreLike | CheckLike
```

What a question is read by: `type`, `instructions` and `options` of a `Choice`; `type`, `instructions` and `levels` of a `Score`; `type`, `instructions`, `yes` and `no` of a `Check`, each of the same type as the model's field, with `options` a `Mapping` and `levels` a `Sequence`. Each is a protocol, so an adapter annotated with another library's `QuestionLike` of the same fields fits.

### `ChoiceAnswer`

```python
class ChoiceAnswer(BaseModel):
    type: Literal["choice"] = "choice"
    choice: str
    probabilities: dict[str, float]
    confidence: float = 0.0
```

The answer to a `Choice`: the option chosen and a probability for every option. Memory sets `confidence` from the probabilities.

### `ScoreAnswer`

```python
class ScoreAnswer(BaseModel):
    type: Literal["score"] = "score"
    score: float
    probabilities: list[float]
    confidence: float = 0.0
```

The answer to a `Score`: a position from 0 to the last level and a probability for every level. Memory sets `confidence` from the probabilities.

### `CheckAnswer`

```python
class CheckAnswer(BaseModel):
    type: Literal["check"] = "check"
    probability: float
    confidence: float = 0.0
```

The answer to a `Check`: the probability of yes, from 0 to 1. Memory sets `confidence` from it.

### `Answer`

```python
Answer = ChoiceAnswer | ScoreAnswer | CheckAnswer
```

The answer to any one question.

### `AnswerLike`

```python
AnswerLike = ChoiceAnswerLike | ScoreAnswerLike | CheckAnswerLike
```

What an answer is read by: `type`, `choice` and `probabilities` of a `ChoiceAnswer`; `type`, `score` and `probabilities` of a `ScoreAnswer`; `type` and `probability` of a `CheckAnswer`. Each is a protocol, so an answer of another class with the same fields fits.

### `Decided`

```python
class Decided(BaseModel):
    answers: dict[str, Answer]
    usage: Usage | None = None
    model: str | None = None
```

A decision model's answers by question id, with what the call cost and which model answered.

### `DecidedLike`

```python
class DecidedLike(Protocol):
    @property
    def answers(self) -> Mapping[str, AnswerLike]: ...
    @property
    def usage(self) -> UsageLike | None: ...
    @property
    def model(self) -> str | None: ...
```

What a decision model's result is read by when it reports its cost: the fields of `Decided`, so any object that has them fits.

### `confidence`

```python
def confidence(probabilities: list[float]) -> float: ...
```

Returns how far a distribution is from uniform: `max(0, (n · max(p) − 1) / (n − 1))`, 0.0 when every probability is equal and 1.0 when one is certain. Returns 1.0 for fewer than two probabilities.

## Memory

### `Memory`

```python
class Memory:
    def __init__(
        self,
        storage: Storage,
        *,
        embedding: EmbeddingModel | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        pillars: set[str] | None = None,
        relation_types: list[str] | None = None,
        weights: dict[str, float] | None = None,
        chunk_chars: int = 2000,
        overlap_chars: int = 200,
    ) -> None: ...
```

Memory over one storage and the models you give it. Unless `pillars` names them, the text pillar is on, the vector pillar with `embedding` and the graph pillar with `generation`; `relation_types` is the vocabulary extraction may use, `weights` scales each pillar's share of the fused rank (1.0 each by default), and text is chunked at `chunk_chars` with `overlap_chars` of overlap. Raises `ConfigurationError` for an unknown pillar, the vector pillar without `embedding`, the graph pillar with `generation` but no `relation_types`, a weight naming an unknown pillar or not positive, `chunk_chars` below 1, or `overlap_chars` negative or not below `chunk_chars`.

Every verb that takes a `corpus_id` raises `InputError` when it is empty, and for text containing a NUL character or a number that is NaN or infinite. A datetime without a timezone is read as UTC. A verb raises `StorageError` when the storage fails, except that a pillar failing inside a search is reported in `Hits.pillars` instead.

### `Memory.initialize`

```python
async def initialize(self) -> None: ...
```

Prepares the storage for this memory's embedding dimension, or for no vectors without `embedding`; safe to run again. Upgrades an older `memory_stories` table in place — adds the closure columns, turns `supersedes` into a list, backfills the lineage and drops the old open-text unique index, indexes the entities' aliases, adds the history columns of corpora, sources and chunks and the chunks' `context`, and records on each chunk and story the text search configuration its keyword column reads, rebuilding the keyword columns — a one-time rewrite of `memory_chunks` and `memory_stories`. The upgrade runs once, under a database-wide lock that makes `initialize()` on other schemas of the same database wait for it; it is one-way, so every service using a schema must run this version together. Raises `ConfigurationError` when the storage was created for another dimension or without vectors, or when Postgres cannot create its extensions.

### `Memory.ingest`

```python
async def ingest(
    self,
    corpus_id: str,
    *,
    name: str,
    text: str | None = None,
    sections: list[Section] | None = None,
    metadata: dict[str, Any] | None = None,
    entities: list[EntityIn] | None = None,
    relations: list[RelationIn] | None = None,
) -> Ingested: ...
```

Writes one source: `text` chunked, or each of `sections` chunked on its own with its locator; every chunk is indexed by every pillar that is on — a chunk whose text the corpus holds with a vector from the same embedding model reuses that vector —, read for entities and relations when there is a generation model, and given the graph facts you supply that it names. Returns an `Ingested`: the source and the usage of every model call the write made. Raises `InputError` when given both or neither of `text` and `sections`, or for a relation end that names no known entity or one of several types without `from_type`/`to_type`; `ConfigurationError` when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

### `Memory.sync`

```python
async def sync(
    self, corpus_id: str, *, filters: dict[str, Any], sources: list[SourceIn]
) -> Synced: ...
```

Makes the sources `filters` match exactly `sources` in one storage transaction, so a reader sees the old set or the new one, never both and never neither: a current matching source whose `digest` a new one repeats is kept as it is, every other current matching source is deleted — or, in a corpus that keeps history, retired: kept out of current reads, with `retired_at` the moment of the sync and `replaced_by` the new source of the same `name` when exactly one has it —, the rest of `sources` is written, and an empty `sources` removes the set; every model call runs before the transaction opens, and a chunk whose text the corpus holds with a vector from the same model reuses it. Returns a `Synced`: the sources in list order, the ids kept, the `Counts` deleted, the ids retired and the usage. The new sources' `created_at` is the moment the sync writes them, never before a current source of the set. Raises `StorageError` when the set moved under it twice; sync again. Raises `InputError` when `filters` are empty or malformed or a new source does not match them, and otherwise what `ingest` raises.

### `Memory.remember`

```python
async def remember(
    self,
    corpus_id: str,
    text: str,
    *,
    subject: str | None = None,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    source_ids: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    entities: list[EntityIn] | None = None,
    relations: list[RelationIn] | None = None,
    related: int = 5,
    filters: dict[str, Any] | None = None,
    judge: bool | tuple[JudgementName, ...] = False,
) -> Remembered: ...
```

Writes one story, true from `valid_from` (now by default) until `valid_to` when given, or until it is superseded; when an open story already says the same text, ignoring case and extra whitespace, returns that one and writes nothing, and for a story written already ended returns the story that repeats it exactly, text and interval. Returns a `Remembered`: the story and the usage of its extraction, identity and judge. `related` holds the open stories closest to the new one, at most `related` of them, among those `filters` match when given — under them the graph pillar stands down, as in `search` — and judged when `judge` asks, the new story's text as the query. With `judge=True` a decision model judges every hit (every related story for the writes) on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises `InputError` for blank text, `related` below 0, a `judge` that is not `True`, `False` or a tuple naming each judgement at most once, a reserved or malformed filter, a `valid_to` not after `valid_from`, or a relation end as `ingest` does; `ConfigurationError` for a `judge` that asks for judgements without a decision model or when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

### `Memory.supersede`

```python
async def supersede(
    self,
    corpus_id: str,
    story_id: str | list[str],
    text: str,
    *,
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    subject: str | None = None,
    metadata: dict[str, Any] | None = None,
    entities: list[EntityIn] | None = None,
    relations: list[RelationIn] | None = None,
    related: int = 5,
    filters: dict[str, Any] | None = None,
    judge: bool | tuple[JudgementName, ...] = False,
) -> Remembered: ...
```

Closes the story, or each of several, at `valid_from` (now by default) and opens a new one there that says `text`, true until `valid_to` when given, and points back through `supersedes` in the order given; each closed story names it in `superseded_by` and records when the supersede was written in `closed_at`. The new story keeps the union of their `source_ids`, and their `subject` and `metadata` when they all agree, unless given. Returns a `Remembered`: the new story and the usage of its extraction, identity and judge. `related` holds the open stories closest to the new one, at most `related` of them, among those `filters` match when given — under them the graph pillar stands down, as in `search` — and judged when `judge` asks, the new story's text as the query. With `judge=True` a decision model judges every hit (every related story for the writes) on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises `NotFoundError` when a story does not exist; `InputError` when the list is empty or names a story twice, a story is not open (superseded, or its end passed), `valid_from` is not after one began or not before its end, `valid_to` is not after `valid_from`, another open story already says `text`, the text is blank, `related` is below 0, a `judge` is not `True`, `False` or a tuple naming each judgement at most once, or a filter is reserved or malformed; otherwise what `remember` raises, including `ConfigurationError` for a `judge` that asks for judgements without a decision model.

### `Memory.delete_story`

```python
async def delete_story(self, corpus_id: str, story_id: str) -> Counts: ...
```

Deletes a story outright, with the mentions and relations it evidenced and the entities and names only it carried: an entity nothing references any more is deleted, and one still mentioned keeps only the names its remaining passages use. Returns the `Counts` deleted. Raises `NotFoundError` when the story does not exist.

### `Memory.delete_source`

```python
async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...
```

Erases a source and every earlier version of it — the versions a sync retired on the way to it, through their `replaced_by` — with their chunks, their vectors and full-text rows, the mentions and relations their chunks evidenced, and the entities and names only they carried: an entity nothing references any more is deleted, and one still mentioned keeps only the names its remaining passages use. Returns the `Counts` erased. A later version, if any, stays. Raises `NotFoundError` when the source does not exist.

### `Memory.forget_retired`

```python
async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts: ...
```

Erases the source versions retired before `before`, as `delete_source` erases a source — chunks, vectors, full-text rows, mentions, relations and the entities and names only they carried. A naive `before` is read as UTC. When to call it is yours. Returns the `Counts` erased.

### `Memory.get_source`

```python
async def get_source(self, corpus_id: str, source_id: str) -> Source | None: ...
```

Returns the source, or `None` when the corpus holds no such source. It finds any version by id, retired or not, until it is erased.

### `Memory.list_sources`

```python
async def list_sources(
    self,
    corpus_id: str,
    *,
    filters: dict[str, Any] | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Source], str | None]: ...
```

Lists the current sources `filters` match, newest first, `limit` at a time; `known_at` lists those memory held then, each retired only if that was so by then (in a corpus without history a sync erased the versions it replaced, so only sources written by then and still held are seen), and `history=True` every version, retired ones included. Returns the page and the cursor of the next one, `None` on the last. Raises `InputError` for a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.read_source`

```python
async def read_source(
    self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None
) -> str: ...
```

Returns a source's whole text, of any version by id, retired or not, or, with `locator`, the text of the chunks whose locator holds every key and value given, joined by blank lines and empty when none does. Raises `NotFoundError` when the source does not exist.

### `Memory.get_stories`

```python
async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]: ...
```

Returns the stories with these ids, closed or not, in the order asked; an id the corpus does not hold is skipped.

### `Memory.list_stories`

```python
async def list_stories(
    self,
    corpus_id: str,
    *,
    subject: str | None = None,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    since: datetime | None = None,
    until: datetime | None = None,
    filters: dict[str, Any] | None = None,
    history: bool = False,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Story], str | None]: ...
```

Lists stories, latest `valid_from` first, `limit` at a time: those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from`, and `subject` and `filters` narrow. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its supersede was written by then — and `as_of` defaults to it. Returns the page and the cursor of the next one, `None` on the last. Raises `InputError` for `history=True` with `as_of`, a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.reembed`

```python
async def reembed(self, corpus_id: str) -> int: ...
```

Embeds again every chunk and story whose vector was not made by the current embedding model, 256 at a time — those `fingerprint` counts as `stale`. With a `DocumentEmbeddingModel` it re-embeds every chunk of a source with a stale one, in one call per source. Returns how many it embedded. Raises `ConfigurationError` without the vector pillar; `ModelError` when the embedding model breaks its contract.

### `Memory.set_language`

```python
async def set_language(self, corpus_id: str, language: Language) -> int: ...
```

Sets the language the corpus's text pillar reads in — the name of any text search configuration the database has, `english` until set — and re-indexes its chunks and stories with the configuration as it is now, in one transaction, while writes to the corpus wait; naming a configuration you created again re-reads the corpus with it, so a change you made to it takes effect. A built-in configuration is read through memory's copy of it that folds accents — `memory_english`, made by `initialize`, or one created in the schema the first time a corpus uses it; one you created is read exactly as you made it. A configuration dropped, or its schema renamed, after a corpus names it makes the corpus's writes raise `StorageError` and its keyword pillar fail with one, until `set_language` names another configuration. The name is kept as Postgres writes it: `French` reads back as `french`. Returns how many passages it re-indexed. Raises `InputError` for a name that is not a text search configuration in the database.

### `Memory.set_history`

```python
async def set_history(self, corpus_id: str, keep: bool) -> None: ...
```

Sets whether the corpus keeps the history of its sources — `False` until set. While it is `True`, `sync` retires what it replaces or drops instead of deleting it; from the moment it is set, never before. Waits for a running `sync` of the corpus. Raises `InputError` when `keep` is not a bool.

### `Memory.set_vector_index`

```python
async def set_vector_index(
    self,
    corpus_id: str,
    on: bool,
    *,
    m: int = 16,
    ef_construction: int = 64,
    precision: Literal["full", "half"] | None = None,
) -> int: ...
```

Gives the corpus an HNSW index on its vectors, in a table of its own, or takes it away. `True` copies the vector of every chunk and story of the corpus into the table, while writes to the corpus wait, then builds one index for chunks and one for stories with `CREATE INDEX CONCURRENTLY`, while writes go on, and returns when done; a corpus that holds nothing yet can be switched on. Called again with the same values it builds nothing; with others it builds the new indexes, then drops the old ones. `False` drops the table. Every write to the corpus updates the table in the write's own transaction, and `delete_corpus` drops it. What exists in the database is the setting, so every service sharing it sees the same; two calls for one corpus run one after the other. `m` and `ef_construction` go to pgvector as given; `precision` is `"half"` to index the vectors at half precision, and unset it is `"full"` up to 2000 dimensions and `"half"` above. A build that fails leaves its index unfinished, and the next `True` builds it again. Returns how many passages the table holds, `0` for `False`. Raises `InputError` when `on` is not a bool, `m` or `ef_construction` is not an int, `precision` is not `"full"`, `"half"` or `None`, `"full"` is asked above 2000 dimensions, or pgvector refuses a value; `ConfigurationError` without an embedding model, above 4000 dimensions, or on pgvector older than 0.8.0; `StorageError` when a build fails.

### `Memory.delete_corpus`

```python
async def delete_corpus(self, corpus_id: str) -> Counts: ...
```

Deletes every row of the corpus, in every table, and drops the table of its vector index, once the corpus's writes in flight and a `set_vector_index` of it running elsewhere are done; writes that come later wait for it. Returns the `Counts` deleted.

### `Memory.verify_absent`

```python
async def verify_absent(self, corpus_id: str) -> Counts: ...
```

Counts the corpus's rows, every table independently, the table of its vector index included. Returns the `Counts`; `clean` is `True` when all are zero.

### `Memory.fingerprint`

```python
async def fingerprint(self, corpus_id: str) -> CorpusFingerprint: ...
```

Reads the corpus and says what it holds: its current sources and their chunks counted, and its stories, a hash over each current source's name, metadata and text, another over every story's text, subject, metadata and interval, the models its vectors came from, how many passages the vector pillar would miss, how this memory embeds and chunks, and the corpus's language; `history` is the corpus's history setting, `retired` counts the source versions it keeps retired and `vector_index` is its vector index as built. Ids and write times are not read, so identical content under new ids gives the same fingerprint. No search computes it. Returns a `CorpusFingerprint`.

### `Memory.describe`

```python
async def describe(self, corpus_id: str) -> CorpusDescription: ...
```

Reads the corpus and says what it holds and what works on it, for a model planning a search: its current sources and their chunks, the stories true now, the retired source versions, the entities, the relation rows (one for each passage that says a relation, so the same relation said again or re-synced counts again), each entity and relation type with how many entities or relation rows have it, its language and history setting, this memory's pillars, whether it has a decision model, and the `expand` modes those pillars support. No search computes it. Returns a `CorpusDescription`.

### `Memory.search`

```python
async def search(
    self,
    corpus_id: str | list[str],
    query: str,
    *,
    kinds: tuple[Kind, ...] = ("chunk", "story"),
    pillars: set[str] | None = None,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    judge: bool | tuple[JudgementName, ...] = False,
    entities: list[str] | None = None,
    relation_types: list[str] | None = None,
    sources: list[str] | None = None,
    around: list[str] | None = None,
    hops: int = 1,
) -> Hits: ...
```

Searches chunks and stories in the corpus `corpus_id` names, or in each of several, with `pillars` (every configured pillar by default), fuses their lists by reciprocal rank and returns the first `top_k` as `Hits`. Over several corpora the query is embedded once; keyword search counts the corpora that share a language as one corpus and lets groups of different languages take turns by rank, the vector and graph pillars merge the corpora's lists by score before the fusion, every hit carries its `corpus_id`, and a pillar that fails in one corpus fails for the search; stories are those true now unless `as_of` or `history=True`; `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its supersede was written by then — and `as_of` defaults to it. Chunks are those of current sources; with `known_at`, of the sources memory held then (in a corpus without history a sync erased the versions it replaced, so only sources written by then and still held are seen); with `history=True`, of every version, retired ones included, each hit's `Chunk.retired_at` saying which. The graph pillar starts from the entities `entities` names — each name found in each corpus by its name or an alias, in every type that holds it — or, without `entities`, from the entity names written in the query, walks up to two hops along `relation_types` (read upper-case, every type by default), and ranks the passages that mention an entity it reached nearest first — by `h`, the fewest hops to an entity the passage mentions — then by `s`, the sum, over the reached entities it mentions, of `2^-hops × ln(1 + (N − n + 0.5) / (n + 0.5))` — `N` the passages that mention a reached entity, `n` those that mention this one — highest first, then by passage id, by code point; its graph score is `2^-h × (1 + s / (1 + s)) / 2`, so a passage nearer the start scores higher whichever corpus holds it; a name no corpus holds is not an error: `Hits.unresolved` maps it to up to five similar names. `sources` keeps the search to the chunks of those sources and the stories whose `source_ids` cite one of them, in every corpus named, and `around` to the passages that mention the entities it names — found as `entities` are, a missing one reported in `Hits.unresolved` the same way — or, within `hops` (0 to 2), their neighbours along `relation_types`; with both, to what both keep. Every pillar runs inside that scope — keyword search with its statistics taken from inside it, the graph pillar starting from `around` unless `entities` is given — and filters, the window and `kinds` apply on top; a scope that holds nothing gives no hits. The graph pillar stands down under `filters`, and a blank query returns no hits. With `judge=True` a decision model judges every hit on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises `ConfigurationError` for a pillar not configured, a `judge` that asks for judgements without a decision model, or `entities`, `relation_types` or `around` with the graph pillar off; `InputError` for a `corpus_id` that names no corpus, an unknown kind, `top_k` below 1, `hops` outside 0 to 2, a reserved or malformed filter, `history=True` with `as_of`, or a `judge` that is not `True`, `False` or a tuple naming each judgement at most once; `ModelError` when the judge breaks its contract.

### `Memory.recall`

```python
async def recall(
    self,
    corpus_id: str | list[str],
    query: str,
    *,
    top_k: int = 10,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    subject: str | None = None,
    filters: dict[str, Any] | None = None,
    pillars: set[str] | None = None,
    judge: bool | tuple[JudgementName, ...] = False,
    entities: list[str] | None = None,
    relation_types: list[str] | None = None,
    sources: list[str] | None = None,
    around: list[str] | None = None,
    hops: int = 1,
) -> Hits: ...
```

Searches stories only, as `search` does, among those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from` and `subject` narrows. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its supersede was written by then — and `as_of` defaults to it. Returns `Hits`. With `judge=True` a decision model judges every hit on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises what `search` raises, and `InputError` for `history=True` with `as_of`.

### `Memory.expand`

```python
async def expand(
    self,
    corpus_id: str,
    passage: Ref,
    *,
    by: Literal["document", "vector", "graph"],
    top_k: int = 10,
    width: int = 2,
    kinds: tuple[Kind, ...] = ("chunk", "story"),
    relation_types: list[str] | None = None,
    filters: dict[str, Any] | None = None,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    judge: bool | tuple[JudgementName, ...] = False,
) -> Hits: ...
```

Returns the passages around one passage of the corpus, never that passage itself, as `Hits` whose `provenance` names the mode with each hit's score divided by the best, and whose `pillars` are empty. `by="document"` reads, for a chunk, the chunks of its source within `width` positions on each side, in document order — `kinds` and `top_k` do not apply — and for a story the other stories with its subject, at most `top_k`, those whose `valid_from` is nearest its own, given in time order — `kinds` does not apply. `by="vector"` searches `kinds` with the vector the passage was stored with as the query, nearest first, at most `top_k`, and calls no model; a passage stored without a vector gives no hits. `by="graph"` walks up to two hops from the entities the passage mentions, along `relation_types` (read upper-case) when given, and ranks the passages of `kinds` as `search`'s graph pillar does, at most `top_k`; a passage that mentions no entity gives no hits. `filters` and the window hold as on `search` — stories true now unless `as_of` or `history=True`, `known_at` as memory knew things then — and a chunk's neighbours are seen only while its source is. With `judge`, a decision model judges each hit against the starting passage's text, as `search` judges against the query. Raises `NotFoundError` for a passage the corpus does not hold; `InputError` for an unknown `by`, a passage that is not a `(kind, id)` pair, `top_k` or `width` below 1, an unknown kind, a reserved or malformed filter, `history=True` with `as_of`, a `judge` that is not `True`, `False` or a tuple naming each judgement at most once, or `by="document"` from a story without a subject; `ConfigurationError` for `by="vector"` without the vector pillar, `by="graph"` without the graph pillar, or a `judge` that asks for judgements without a decision model.

### `Memory.neighbors`

```python
async def neighbors(
    self,
    corpus_id: str,
    entity: str,
    *,
    entity_type: str | None = None,
    hops: int = 1,
    relation_types: list[str] | None = None,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
) -> Neighborhood: ...
```

Walks the graph up to `hops` from the entity with this name or alias, along `relation_types` only when given, read upper-case. Returns a `Neighborhood`: the entities reached, the relations walked and the passages that mention them, current stories unless `as_of` or `history=True`. Relations are those some visible passage evidences: a chunk of a source visible in the read, or a story true at the moment read. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its supersede was written by then — and `as_of` defaults to it. Raises `ConfigurationError` with the graph pillar off; `NotFoundError` when no entity has the name, of `entity_type` when given; `InputError` when entities of several types have it and `entity_type` is not given, or for `history=True` with `as_of`.

## Records

### `Source`

```python
class Source(BaseModel):
    corpus_id: str
    source_id: str
    name: str
    metadata: dict[str, Any]
    text: str
    created_at: datetime
    chunks: int
    digest: str | None = None
    retired_at: datetime | None = None
    replaced_by: str | None = None
```

A stored source: its whole text, its `metadata` facets, how many chunks it was split into, and `digest`, a hash of the source as it was given with the chunk sizes, pillars and extraction it was written under, by which `sync` keeps a source the new list repeats; `None` when none was recorded. `retired_at` is when a `sync` retired this version, in a corpus that keeps history, and `replaced_by` the source that took its place when one did; both are `None` on a current source, and on a source read as memory knew it before it was retired.

### `Section`

```python
class Section(BaseModel):
    title: str | None = None
    text: str
    locator: dict[str, Any] = {}
    context: str | None = None
```

One part of a source as your parser knows it: its text, an optional title, and a `locator` its chunks carry. `context` is free text your parser gives the section — what document, what part of it — and every chunk of the section carries it; keyword search matches it.

### `Chunk`

```python
class Chunk(BaseModel):
    corpus_id: str
    chunk_id: str
    source_id: str
    source_name: str
    position: int
    title: str | None
    locator: dict[str, Any]
    text: str
    metadata: dict[str, Any]
    context: str | None = None
    retired_at: datetime | None = None
```

A piece of a source as indexed: the source's id, name and `metadata`, its position in the source, the section's title or else the headings it falls under, and the section's locator — so a hit can cite its document without reading the source. `context` is its section's. `retired_at` is its source's.

### `Story`

```python
class Story(BaseModel):
    corpus_id: str
    story_id: str
    text: str
    subject: str | None
    valid_from: datetime
    valid_to: datetime | None
    created_at: datetime
    supersedes: tuple[str, ...] = ()
    superseded_by: str | None = None
    closed_at: datetime | None = None
    source_ids: tuple[str, ...]
    metadata: dict[str, Any]
```

A story, true from `valid_from` until `valid_to` (`None` while no end is known), written at `created_at`, pointing through `supersedes` at the stories it replaced; a superseded story names its replacement in `superseded_by` and the moment that was written in `closed_at`. Raises pydantic's `ValidationError` when `valid_to` is not after `valid_from`.

### `Entity`

```python
class Entity(BaseModel):
    corpus_id: str
    entity_id: str
    name: str
    key: str
    type: str
    aliases: tuple[str, ...] = ()
```

A graph entity: its name as first written, the `key` it is matched by (the name lower-cased, spacing collapsed), its lower-cased `type`, and the other names it is known by.

### `Relation`

```python
class Relation(BaseModel):
    corpus_id: str
    from_id: str
    to_id: str
    type: str
    evidence_kind: Kind
    evidence_id: str
```

A typed relation between two entities, its type upper-case, with the passage that evidences it.

### `SourceIn`

```python
class SourceIn(BaseModel):
    name: str
    text: str | None = None
    sections: list[Section] | None = None
    metadata: dict[str, Any] = {}
    entities: list[EntityIn] = []
    relations: list[RelationIn] = []
```

A source to write, as `sync` takes it: what `ingest` takes, as one value. Raises `InputError` when it has both or neither of `text` and `sections`.

### `EntityIn`

```python
class EntityIn(BaseModel):
    name: str
    type: str
```

An entity you supply.

### `RelationIn`

```python
class RelationIn(BaseModel):
    from_name: str
    to_name: str
    type: str
    from_type: str | None = None
    to_type: str | None = None
```

A relation you supply between two entities by name; `from_type` and `to_type` choose when a name belongs to entities of several types.

### `Hit`

```python
class Hit(BaseModel):
    corpus_id: str
    kind: Kind
    id: str
    text: str
    score: float
    provenance: dict[str, float]
    chunk: Chunk | None = None
    story: Story | None = None
    judge: Judgement | None = None
    entities: list[Entity] = []
```

One search result: a chunk or a story of the corpus `corpus_id`, its fused `score`, and in `provenance` each pillar that returned it with its score divided by that pillar's best — for keyword hits from corpora in different languages, by the best keyword hit of the same language; for `expand`, the mode that found it. The passage itself is in `chunk` or `story`, `judge` is set when the search was judged, and `entities` are the entities the passage mentions as memory holds them now, whatever `known_at` the search asked, ordered by name key, read in one storage call per corpus after the search — empty with the graph pillar off.

### `Hits`

```python
class Hits(BaseModel):
    hits: list[Hit]
    pillars: dict[str, PillarReport]
    usage: Usage
    unresolved: dict[str, list[str]] = {}

    def by(self, check: JudgementName, *, descending: bool = True) -> Hits: ...
```

A search's hits in fused order, a report for each pillar that ran, the usage of the judge, and in `unresolved` each name of `entities` or `around` that no corpus searched holds, with up to five similar names. `by` returns a copy ordered by one judgement, a `JudgementName`, highest first unless `descending=False`, ties in fused order. `by` raises `InputError` for an unknown judgement, hits that were not judged, or a judgement they were not judged for.

### `Judgement`

```python
class Judgement(BaseModel):
    relevant: float | None = None
    evidence: float | None = None
    contradicts: float | None = None
    injection: float | None = None
```

A decision model's four probabilities for one hit against the query: it is about what the query asks, it contains what answers the query, it contradicts a premise the query states, it holds instructions addressed to an AI. A judgement the judge was not asked for is `None`.

### `PillarReport`

```python
class PillarReport(BaseModel):
    returned: int
    duration_ms: int
    error: str | None = None
```

What one pillar did in a search: how many passages it returned, how long it took, and its error when it failed.

### `Neighborhood`

```python
class Neighborhood(BaseModel):
    entities: list[Entity]
    relations: list[Relation]
    passages: list[Hit]
```

What `neighbors` returns: the entities reached, the one named first, the relations walked, and the passages that mention the entities, each with the `entities` it mentions.

### `Ingested`

```python
class Ingested(BaseModel):
    source: Source
    usage: Usage
```

What `ingest` returns: the source written and what its model calls cost.

### `Synced`

```python
class Synced(BaseModel):
    sources: list[Source]
    kept: list[str]
    deleted: Counts
    retired: list[str] = []
    usage: Usage
```

What `sync` returns: the list's sources as stored, in its order, kept or written; the ids of those kept as they were; the rows deleted; the ids of the sources retired, in a corpus that keeps history; and what the model calls cost.

### `Remembered`

```python
class Remembered(BaseModel):
    story: Story
    related: Hits
    usage: Usage
```

What `remember` and `supersede` return: the story written, or the open story that already said it, or the story a write into the past repeats exactly; the open stories closest to it, judged when asked; and what the model calls cost, the judge included.

### `Counts`

```python
class Counts(BaseModel):
    sources: int = 0
    chunks: int = 0
    stories: int = 0
    entities: int = 0
    relations: int = 0
    mentions: int = 0
    vectors: int = 0
    corpora: int = 0

    @property
    def clean(self) -> bool: ...
```

Rows by table, as deletes and `verify_absent` return them; `vectors` counts the rows of the table of the corpus's vector index, and `corpora` the corpus's own row, kept once its language or its history is set, and that table itself, so one left behind is not clean. `clean` is `True` when every count is zero.

### `CorpusFingerprint`

```python
class CorpusFingerprint(BaseModel):
    sources: int
    chunks: int
    content: str
    stories: int
    story_content: str
    embedding: str | None
    embedded: dict[str, int]
    stale: int
    chunking: Chunking
    language: Language = "english"
    history: bool = False
    retired: int = 0
    vector_index: VectorIndex | None = None
```

What `fingerprint` returns. `content` is the first 16 hex characters of a sha256 over each current source's name, metadata and text, sorted; `story_content` the same over each story's text, subject, metadata, `valid_from` and `valid_to`, in UTC. `embedding` is the model of the vector pillar, `None` without it. `embedded` counts the chunks and stories, together, that carry a vector, by the model that made it, sorted by model; `stale` counts those whose vector is from another model than `embedding` or missing — what `reembed` embeds again — and is `0` without the vector pillar. `language` is the one the corpus's text pillar reads in. `history` is the corpus's history setting and `retired` counts the source versions it keeps retired; `sources` and `chunks` count current ones, while `embedded` and `stale` count every chunk and story vector, retired versions included. `vector_index` is the corpus's vector index as `set_vector_index` built it, read from the database: `None` when the corpus has none, or when its chunks and stories have no finished index built with the same settings.

### `VectorIndex`

```python
class VectorIndex(BaseModel):
    m: int
    ef_construction: int
    precision: Literal["full", "half"]
```

How a corpus's vector index was built, as `fingerprint` reports it: pgvector's `m` and `ef_construction`, and `precision`, `"full"` for an index on the vectors and `"half"` for one on their half-precision copies.

### `Chunking`

```python
class Chunking(BaseModel):
    chunk_chars: int
    overlap_chars: int
    pillars: list[str]
```

How a memory cuts and indexes a corpus, as its fingerprint records it: `chunk_chars`, `overlap_chars` and the pillars it was given, sorted.

### `CorpusDescription`

```python
class CorpusDescription(BaseModel):
    sources: int
    chunks: int
    stories: int
    retired: int
    entities: int
    relations: int
    entity_types: dict[str, int]
    relation_types: dict[str, int]
    language: Language
    history: bool
    pillars: list[str]
    judge: bool
    expand: list[str]
```

What `describe` returns: `sources` and `chunks` count the current sources and their chunks, `stories` the stories true now and `retired` the source versions kept retired; `entities` and `relations` count the graph's rows, and `entity_types` and `relation_types` map each type to how many entities or relation rows have it, in code-point order. `language` and `history` are the corpus's settings; `pillars` are the memory's, sorted, `judge` says whether it has a decision model, and `expand` lists the modes `expand` takes, `document` always.

## Tools

### `MemoryTools`

```python
class MemoryTools(Tools):
    def __init__(
        self,
        memory: Memory,
        *,
        corpus_id: str | list[str],
        filters: dict[str, Any] | None = None,
        include: set[str] | None = None,
        max_result_chars: int = 200_000,
        judge: tuple[JudgementName, ...] | None = None,
    ) -> None: ...
```

A `Tools` that offers memory's verbs to a model, pinned to the corpus or corpora `corpus_id` names and to the bound `filters`: `search`, `recall`, `expand`, `read_source`, `list_sources`, `ingest`, `remember`, `supersede`, `delete_story`, `list_stories`, and `neighbors` when the graph pillar is on and no filter is bound, and `describe` when no filter is bound. `list_sources` takes `history` and `known_at`, and a chunk hit from a retired version carries `retired_at`. `search` and `recall` take `sources`, and `entities`, `relation_types`, `around` and `hops` when the graph pillar is on and no filter is bound, and answer with `unresolved` when a name was not found. `search` and `recall` take `pillars`, offering only the configured ones and not `graph` under a bound filter. `expand` takes a hit's `kind` and `id` and offers `by` the modes the pillars support, `graph` only when the graph pillar is on and no filter is bound; under bound filters its passage must be inside them. A hit carries `entities`, the name and type of each entity its passage mentions, when the graph pillar is on and no filter is bound. Bound to several corpora, it offers `search` and `recall`, which read them all or the ones the model names in `corpus_id`, and `expand`, `read_source`, `list_sources` and `list_stories`, which take the one `corpus_id` a hit gives, and `describe`, which takes the one to describe; every hit then says its `corpus_id`; `include` keeps a subset, and every result over `max_result_chars` is cut and marked `truncated`. `judge` names the judgements the host's decision model answers, all four unless given: the tools' `judge` asks those, `order_by` chooses among them, and `order_by` alone asks only that one. The tools keep limits of their own: `limit` defaults to 50 (at most 500), `top_k` is at most 100, the `hops` of `neighbors` are 1 to 3, the `width` of `expand` is 1 to 10, and `related` is 0 to 20. A bound filter takes one value or several; the model's filters may narrow it and never widen it, and what the tools write is stamped with it, naming one value of a filter bound to several. A call raises what its verb raises, `NotFoundError` for a story, a source or an `expand` passage outside the bound filters, and `InputError` for text an open story outside them already says or a past story outside them repeats exactly, a filter outside the bound values, written metadata that does not name one of them, an `order_by` outside the judgements, a graph argument, a `pillars` or an `expand` mode the tools do not offer, or a `corpus_id` outside the bound corpora. Raises `ConfigurationError` without a `corpus_id`, with `max_result_chars` below 1000, or with a `judge` naming no judgement, an unknown one or one twice; `InputError` for a reserved or malformed filter.

### `Tool`

```python
@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    body: Body | None = None
    args: type[BaseModel] | None = None
    input_schema: dict[str, Any] | None = None
```

One tool: a name, a description, an async `body` that takes the arguments, and optionally an `args` model that validates them or an explicit `input_schema`. `Body` is `Callable[[dict[str, Any]], Awaitable[Result]]`, and `Result` is `str | dict[str, Any] | list[Part]`.

### `Tool.schema`

```python
def schema(self) -> dict[str, Any]: ...
```

Returns `input_schema` when given, else the JSON schema of `args`, else the schema of an object with no properties.

### `Tool.definition`

```python
def definition(self) -> dict[str, Any]: ...
```

Returns what a model is shown: the tool's `name`, `description` and `input_schema`.

### `Tool.call`

```python
async def call(self, args: dict[str, Any]) -> Result: ...
```

Validates `args` against the `args` model, when there is one, and returns what `body` returns. Raises `ConfigurationError` when the tool has no body; `InputError` when the arguments fail validation.

### `Tools`

```python
class Tools:
    def __init__(self, tools: Iterable[Tool | ToolsLike] = ()) -> None: ...
    @property
    def names(self) -> list[str]: ...
    def __add__(self, other: Tools | ToolsLike) -> Tools: ...
```

Tools by name, from `Tool`s and from anything with `definitions()` and `dispatch()` (`ToolsLike`), such as another `Tools`. `names` lists them, `name in tools` and `len(tools)` work, and `a + b` returns a new `Tools` holding both. Raises `ConfigurationError` when two tools share a name.

### `Tools.definitions`

```python
def definitions(self) -> list[dict[str, Any]]: ...
```

Returns a copy of every tool's definition, in order.

### `Tools.dispatch`

```python
async def dispatch(self, name: str, args: dict[str, Any]) -> Result: ...
```

Calls the tool `name` with `args` and returns its result. Raises `InputError` for an unknown tool, and whatever the tool raises.

### `Tools.only`

```python
def only(self, *names: str) -> Tools: ...
```

Returns a new `Tools` with only the named tools, in the order named. Raises `InputError` when a name is not one of them.

### `Tools.without`

```python
def without(self, *names: str) -> Tools: ...
```

Returns a new `Tools` without the named tools; a name that is not one of them is ignored.

## Storage

### `Storage`

```python
@runtime_checkable
class Storage(Protocol):
    async def initialize(self, *, dimension: int | None) -> None: ...

    async def put_source(
        self,
        corpus_id: str,
        source: Source,
        chunks: list[Chunk],
        vectors: list[list[float]] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> None: ...

    async def get_source(self, corpus_id: str, source_id: str) -> Source | None: ...

    async def list_sources(
        self,
        corpus_id: str,
        *,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
        known_at: datetime | None = None,
        history: bool = False,
    ) -> tuple[list[Source], str | None]: ...

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]: ...

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str, int]]: ...

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...

    async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts: ...

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        at: datetime,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> tuple[Counts, list[str]] | None: ...

    async def chunk_vectors(
        self, corpus_id: str, model: str, digests: list[str]
    ) -> dict[str, list[float]]: ...

    async def put_story(
        self,
        corpus_id: str,
        story: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> Story: ...

    async def find_open_story(self, corpus_id: str, hash: str, *, at: datetime) -> Story | None: ...

    async def supersede_story(
        self,
        corpus_id: str,
        old_ids: list[str],
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[list[Story], Story]: ...

    async def get_stories(self, corpus_id: str, story_ids: list[str]) -> list[Story]: ...

    async def list_stories(
        self,
        corpus_id: str,
        *,
        subject: str | None,
        window: Window,
        filters: dict[str, Any] | None,
        limit: int,
        cursor: str | None,
    ) -> tuple[list[Story], str | None]: ...

    async def delete_story(self, corpus_id: str, story_id: str) -> Counts: ...

    async def search_text(
        self,
        corpus_ids: list[str],
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None = None,
    ) -> list[tuple[str, Leg]]: ...

    async def search_vector(
        self,
        corpus_id: str,
        vector: list[float],
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None = None,
    ) -> list[Leg]: ...

    async def get_passages(
        self, corpus_id: str, refs: list[Ref], *, known_at: datetime | None = None
    ) -> list[Chunk | Story]: ...

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]: ...

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]: ...

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]: ...

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]: ...

    async def neighbors(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        relation_types: list[str] | None,
        window: Window,
    ) -> list[Relation]: ...

    async def mentions(
        self,
        corpus_id: str,
        entity_ids: list[str],
        *,
        kinds: tuple[Kind, ...],
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
        scope: Scope | None = None,
    ) -> list[tuple[str, Ref]]: ...

    async def passage_vector(self, corpus_id: str, passage: Ref) -> list[float] | None: ...

    async def chunks_around(
        self,
        corpus_id: str,
        source_id: str,
        position: int,
        *,
        width: int,
        window: Window,
        filters: dict[str, Any] | None,
    ) -> list[Chunk]: ...

    async def passage_entities(
        self, corpus_id: str, passages: list[Ref]
    ) -> list[tuple[Ref, Entity]]: ...

    async def graph_types(self, corpus_id: str) -> tuple[dict[str, int], dict[str, int]]: ...

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]: ...

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None: ...

    async def delete_corpus(self, corpus_id: str) -> Counts: ...

    async def count_corpus(self, corpus_id: str) -> Counts: ...

    async def get_language(self, corpus_id: str) -> Language: ...

    async def set_language(self, corpus_id: str, language: Language) -> int | None: ...

    async def get_history(self, corpus_id: str) -> bool: ...

    async def set_history(self, corpus_id: str, keep: bool) -> None: ...

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]: ...

    async def get_vector_index(self, corpus_id: str) -> VectorIndex | None: ...

    async def set_vector_index(
        self,
        corpus_id: str,
        on: bool,
        *,
        m: int,
        ef_construction: int,
        precision: Literal["full", "half"] | None,
    ) -> int: ...
```

The one storage protocol, as `Memory` calls it; `PostgresStorage` implements exactly these methods, and a storage of your own implements them from exported names alone. Every method but `initialize` and `search_text` takes the corpus first and touches no other, and `search_text` takes corpora of one language that a search names, as a non-empty list without duplicates, and touches no other; a write carries its `GraphWrite` in the same transaction, the listings page by cursor, `delete_source` erases the source and every version whose `replaced_by` chain leads to it, `forget_retired` erases the versions retired before `before`, `list_sources` returns current sources unless given `known_at` (as held then: written by then, retired only if by then) or `history=True` (every version), `sync_sources` matches current sources only and, when `get_history` is true, retires the matching sources it does not keep — `retired_at = at` on the source and its chunks, `replaced_by` the one new source of the same name — instead of deleting them, returning the deleted `Counts` and the retired ids, or `None`, writing nothing, when a kept source is no longer in the matching set with its digest or, in a corpus that keeps history, a current matching source was created after `at`, `set_history`, `delete_source`, `forget_retired` and `delete_corpus` wait for a running `sync_sources` of the corpus (a storage of your own serializes them per corpus), the searches return hits best first — `search_vector` as `Leg`s, `search_text` as `(corpus_id, Leg)` pairs ordered by score, then passage id, then corpus id, by code point — and `search_text` scores the corpora it is given as one corpus by BM25 (k1 = 1.5, b = 0.75): per kind, over the passages of those corpora that the window, filters, subject and scope leave and that hold one of the query's distinct terms — a chunk matched on its source name, heading path, context and text, a story on its text — with `N`, each term's document frequency and the mean length taken from all those passages, the query read in the language of the first corpus given (`Memory` gives corpora that share one); a passage holding no query term is not a hit, `source_digests` returns each current source's name, metadata, the sha256 of its text in hex and its chunk count, in no order, the searches, `mentions` and `neighbors` see a chunk when its source is visible in the window — current while `as_of` is set and `known_at` is not, as held at `known_at`, every version when `as_of` is `None`; with `known_at` and no `as_of`, a chunk is seen when its source was written by `known_at`, retired or not — and a relation evidenced by a chunk only when that chunk is visible, `search_text`, `search_vector` and `mentions` read only inside `scope` when one is given — `Scope.sources` keeps the chunks of those sources and the stories whose `source_ids` cite one of them, `Scope.passages` keeps those passages in any corpus given, the two together keep what both keep, and an empty tuple keeps nothing — so `top_k` and BM25's statistics come from inside it, `mentions` returns one `(entity_id, Ref)` pair for each passage that mentions each entity, ordered by passage kind, passage id and entity id, by code point, `passage_vector` returns the vector a passage was stored with, or `None` for a passage the corpus does not hold or that has none, `chunks_around` returns the chunks of a source whose position is within `width` of `position`, that one included, in position order, keeping only those the searches would see under `window` and `filters`, each as `get_passages(known_at=)` shows it, `passage_entities` returns one `(Ref, Entity)` pair for each entity each of the passages given mentions, ordered as `mentions` orders them, `graph_types` returns the corpus's entity types and relation types, each with how many entities or relation rows have it, `get_passages(known_at=)` shows a chunk retired after `known_at` as current, `sync_sources` leaves the sources `keep` names (each id with its `digest`) as they are, `chunk_vectors` returns for each digest — the sha256 of a chunk's text in hex — a vector a chunk with that text holds from `model`, leaving out the digests it has none for, `count_vectors` counts the chunks and stories together by the model that made their vector, under `None` those without one, `set_vector_index` builds, re-tunes or drops the corpus's vector index as `Memory.set_vector_index` describes, choosing the precision when it is `None`, and returns how many passages the index holds, a write to a corpus with an index updates it in the write's own transaction, `get_vector_index` returns the settings the finished indexes of both kinds share, `None` without, `get_language` returns the corpus's language, `english` until one is set, and `set_language` records it and re-indexes the corpus's chunks and stories in one transaction that the corpus's writes wait for, returning how many passages it re-indexed, or `None`, writing nothing, when the database has no text search configuration of that name. A storage of your own must also guarantee the following. `put_story` serialises the writers of one text and returns the open story that already says it — or, for a story written already ended, the story that repeats it exactly, text and interval — writing nothing. `supersede_story` refuses a story that is superseded or whose end passed, closes every story it is given at `new.valid_from` with `superseded_by = new.story_id` and `closed_at = new.created_at`, and validates all of them before writing any; it takes the lock of the new text and of each closed story's text, in a fixed, sorted order so racing writers cannot deadlock, and raises `InputError` when another open story already says `new.text`, unless that story is among those being closed; it does not check `valid_from` ordering, which `Memory.supersede` does. Besides `valid_to`, the storage keeps the end each story was written with, which a supersede never changes. `find_open_story(at=...)` answers "open at that moment", `get_passages(known_at=...)` returns stories as known then, and `neighbors` returns only relations whose evidence is visible under the window. `Window.open`, meaningful only with `as_of`, reads open stories whether or not they have begun, and is used with the current view only: a `Window` refuses it combined with `known_at`. A call that removes mentions or relations, or recreates an entity of `GraphWrite.known`, then settles each entity it touched: one no mention and no relation references is deleted, counted in `Counts.entities`; one still mentioned keeps only the aliases its mentions' names carry, and when none of them is its name any more takes the first of them by name key that no other entity of its type holds — keeping its name when there is none, or when another settle takes that name first. `PostgresStorage` raises `StorageError` when a row names another corpus or graph facts name an unknown entity, `NotFoundError` and `InputError` from `supersede_story` as described above, and `ConfigurationError` from `initialize` for a dimension the storage was not created for.

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

The `Storage` that ships: Postgres with pgvector, over an asyncpg pool you built, in tables prefixed `memory_`; `initialize` creates the `vector`, `pg_trgm` and `unaccent` extensions, `memory_english` — English's text search configuration with `unaccent` ahead of its stemmer — the tables and their full-text indexes, and is safe to run again, and upgrades an older `memory_stories` table in place, under a lock, once (one-way: every service using a schema must run this version together). Raises `ConfigurationError` from `initialize` when an extension cannot be created or the tables were created for another dimension or without vectors; `StorageError` when the database rejects an operation, or vectors are written before `initialize`. `set_language` makes the same copy, `memory_<name>`, of any other built-in configuration the first time a corpus uses it. `set_vector_index` keeps an indexed corpus's vectors in `memory_vectors_` and the first 24 hex characters of the sha256 of its id, with one HNSW index per kind, which each write to the corpus updates in its own transaction and `delete_corpus` drops, after any build of the corpus; it raises `InputError` for `"full"` above 2000 dimensions or a value pgvector refuses, and `ConfigurationError` above 4000 dimensions, on pgvector older than 0.8.0, or before `initialize` gave it a dimension.

### `SourceWrite`

```python
class SourceWrite(NamedTuple):
    source: Source
    chunks: list[Chunk]
    vectors: list[list[float]] | None
```

One source as `sync_sources` writes it: the source, its chunks, and one vector per chunk or `None`.

### `GraphWrite`

```python
class GraphWrite(BaseModel):
    entities: list[Entity] = []
    aliases: list[tuple[str, str]] = []
    relations: list[Relation] = []
    mentions: list[tuple[str, Kind, str, str]] = []
    known: list[Entity] = []

    def check(self, corpus_id: str, passages: set[Ref]) -> None: ...
    def named(self, ids: dict[str, str]) -> set[str]: ...
```

The graph facts of one write: new entities, `(entity_id, alias)` pairs, relations, `(entity_id, kind, passage_id, name)` mentions — the name the passage used — and the existing entities the write names, as it read them. A storage recreates a known entity a concurrent delete has removed, under its id or, when another entity now holds its name and type, as that entity; and after writing the mentions it adds each mention's name as an alias of its entity when neither the name nor an alias has that name key. `check` raises `StorageError` when a row names another corpus or a fact is evidenced by a passage outside `passages`; `named` returns the ids of the entities the facts name, each mapped through `ids`.

### `Window`

```python
class Window(BaseModel):
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    known_at: datetime | None = None
    open: bool = False
```

The stories a read covers: those true at `as_of`, or at every moment when it is `None`, with `valid_from` from `since` through `until`; with `known_at`, as memory knew them then. `open` reads open stories instead — not superseded, end not passed — whether or not they have begun. Raises pydantic's `ValidationError` when `open` is combined with `known_at`.

### `Scope`

```python
class Scope(BaseModel):
    sources: tuple[str, ...] | None = None
    passages: tuple[Ref, ...] | None = None
```

Where a read may look: the chunks of the `sources` named and the stories whose `source_ids` cite one of them, and the `passages` named. `None` on an axis is no restriction on it, the two axes together keep what both keep, and an empty tuple keeps nothing. `Memory` builds one for a scoped search; a storage of your own applies it to `search_text`, `search_vector` and `mentions`.

### `Kind`

```python
Kind = Literal["chunk", "story"]
```

The two kinds of passage.

### `JudgementName`

```python
JudgementName = Literal["relevant", "evidence", "contradicts", "injection"]
```

The four judgements a decision model can give a hit; the names a `judge` tuple and `Hits.by` take.

### `Language`

```python
Language = str
```

The name of a text search configuration the database has, which a corpus's text pillar reads in: one of Postgres' built-in languages, which memory reads through a copy that folds accents, or one you created, schema-qualified if you like, read as you made it. `simple` neither stems nor drops stop words.

### `Ref`

```python
Ref = tuple[str, str]
```

A passage by kind and id.

### `Leg`

```python
Leg = tuple[str, str, float]
```

One pillar's result for a passage: its kind, its id and the pillar's score.

## Errors

### `MemoryException`

```python
class MemoryException(Exception): ...
```

The root of every error the library raises; it takes the `Exception` suffix so it never shadows Python's `MemoryError`. Every message starts with the name of the module that raised it: `memory: story 'abc' was not found`.

### `ConfigurationError`

```python
class ConfigurationError(MemoryException): ...
```

A setup that cannot work: an unknown or unconfigured pillar, the vector pillar without an embedding model, extraction without `relation_types`, bad `weights` or chunk sizes, a storage without its extension or created for another dimension, a `judge` that asks for judgements without a decision model, graph facts, `neighbors`, `entities`, `relation_types` or `around` with the graph pillar off, `expand` by vector or by graph without that pillar, `reembed` without the vector pillar, a `MemoryTools` without a corpus, with `max_result_chars` below 1000 or with a `judge` naming no judgement, an unknown one or one twice, two tools of the same name, or a `Tool` called without a body.

### `InputError`

```python
class InputError(MemoryException): ...
```

A bad argument: an unknown kind or judgement, hits ordered by a judgement they were not given, `top_k` or `limit` below 1, `hops` outside 0 to 2, `width` below 1, an unknown `expand` mode, a passage that is not a `(kind, id)` pair, `expand` by document from a story without a subject, a cursor no listing returned, a reserved or malformed filter, text with a NUL character or a number that is NaN or infinite, a story with blank text, a `valid_to` not after `valid_from`, `related` below 0, `history=True` with `as_of`, a name of several types without its type, a refused `supersede`, a source with both or neither of `text` and `sections`, a media part without exactly one of `data` and `url`, a `sync` without filters or with a source they do not match, a language that is not a text search configuration in the database, a `corpus_id` that names no corpus, a tool name `dispatch` or `only` does not know, tool arguments that fail validation, or, through `MemoryTools`, a filter or `corpus_id` outside what the tools are bound to, a graph argument, a `pillars` or an `expand` mode they do not offer and written metadata that names none of a filter's bound values.

### `ModelError`

```python
class ModelError(MemoryException): ...
```

A model broke its contract: a reply, an answer or a vector that does not fit.

### `StorageError`

```python
class StorageError(MemoryException): ...
```

The storage failed or refused a write: a database error, a row of another corpus, graph facts naming an unknown entity, or vectors written to Postgres before `initialize`.

### `NotFoundError`

```python
class NotFoundError(MemoryException): ...
```

A verb named a story, a source, a passage or an entity the corpus does not hold, or, through `MemoryTools`, one outside the bound filters.
