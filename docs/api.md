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
    settings: dict[str, JsonValue] | None = None
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
    def settings(self) -> Mapping[str, JsonValue] | None: ...
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
    settings: dict[str, JsonValue] | None = None
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
    @property
    def settings(self) -> Mapping[str, JsonValue] | None: ...
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
        storage: PostgresStorage,
        *,
        embedding: EmbeddingModel | None = None,
        generation: GenerationModel | None = None,
        decision: DecisionModel | None = None,
        pillars: set[str] | None = None,
        relation_types: list[str] | None = None,
        text: TextRanking = BM25(),
        fusion: Fusion = RRF(),
        spot: Spot = "words",
        chunk_chars: int = 2000,
        overlap_chars: int = 200,
        clock: Callable[[], datetime] | None = None,
        ids: Callable[[], str] | None = None,
        refinery: MemoryRefinery | None = None,
    ) -> None: ...
```

Memory over one storage and the models you give it. Unless `pillars` names them, the text pillar is on, the vector pillar with `embedding` and the graph pillar with `generation`; `relation_types` is the vocabulary extraction may use, `text` is how the text pillar ranks what it matches, `fusion` how the pillars' lists become one and `spot` how the graph pillar reads a query for names when a call leaves `spot` out (readable back as `memory.text`, `memory.fusion` and `memory.spot`), and text is chunked at `chunk_chars` with `overlap_chars` of overlap. Raises `ConfigurationError` for an unknown pillar, the vector pillar without `embedding`, the graph pillar with `generation` but no `relation_types`, a `text` that is not a `BM25` or a `CoverDensity`, a `fusion` that is not an `RRF` or a `RelativeScore`, a `spot` other than `"words"`, `"exact"`, `"substring"` or `"off"`, `chunk_chars` below 1, or `overlap_chars` negative or not below `chunk_chars`. `clock` is what memory reads as now, `datetime.now(UTC)` unless given, a naive value read as UTC; a value that is not a datetime is a `ConfigurationError` when memory reads it. `ids` names each new story, source and entity, `uuid4().hex` unless given; a value that is not a non-empty string without a NUL character is an `InputError` before anything is written, and so is an id the corpus already holds for a story, a source or an entity, or one given twice in one call. A `clock` or `ids` that is not callable is a `ConfigurationError`. `refinery` records each `search`, `recall`, `expand` and `resembling` it samples, into the storage the refinery was given (see `MemoryRefinery`).

Every verb that takes a `corpus_id` raises `InputError` when it is empty, and for text containing a NUL character or a number that is NaN or infinite. A datetime without a timezone is read as UTC. A verb raises `StorageError` when the storage fails, except that a pillar failing inside a search is reported in `Hits.pillars` instead.

### `Memory.check`

```python
async def check(self) -> None: ...
```

Reads the schema the storage's pool points to — the first schema on its `search_path` — and says whether it fits this memory; writes nothing. Raises one `ConfigurationError` that names every problem, each with what fixes it: no memory schema there — or the first schema the `search_path` names does not exist, or the path reaches only `public` — memory tables from before migrations, a schema behind this version (`rfnry-memory upgrade`) or ahead of it, a recorded dimension other than this memory's embedding dimension when the vector pillar is on, the `vector` type unreachable from the `search_path` while the schema holds vectors, or `pg_trgm` unreachable or not installed with the graph pillar on. Storage learns its vector dimension from this call, and whether the schema's current rows are indexed (`rfnry-memory index --current`); call it once at start, before any write.

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
    valid_from: datetime | None = None,
    valid_to: datetime | None = None,
    replaces: str | None = None,
    written_by: dict[str, Any] | None = None,
    on_event: EventHandler | None = None,
) -> Ingested: ...
```

Writes one source, recording `written_by` on it and its chunks, and as the `closed_by` of a source it `replaces`, in force from `valid_from` until `valid_to` when given: `text` chunked, or each of `sections` chunked on its own with its locator; every chunk is indexed by every pillar that is on — a chunk whose text the corpus holds with a vector from the same embedding model reuses that vector —, read for entities and relations when there is a generation model, and given the graph facts you supply that it names. Returns an `Ingested`: the source and the usage of every model call the write made. With `on_event`, awaits it with a `Chunked` once the source is cut, an `Embedded` once its vectors are ready (with the vector pillar) and an `Extracted` after each chunk's graph step (with the graph pillar), one at a time, before the write; a handler that raises fails the call with its exception and nothing is written. With `replaces`, the new source takes the place of that source, as one source of a `sync` takes the place of the old one of its name: in a corpus that keeps history the old one is retired, `replaced_by` the new one and out of force where the new one starts; otherwise it is erased with its chunks, vectors and graph evidence. Its name may differ. `Ingested.retired` and `Ingested.deleted` say which. Raises `NotFoundError` when `replaces` names no source; `InputError` when it names a retired one, or when the new source would take effect after this write while the one it replaces is in force. Raises `InputError` when given both or neither of `text` and `sections`, a `valid_to` not after `valid_from`, for an entity without a name or a type, or for a relation end that names no known entity or one of several types without `from_type`/`to_type`; `ConfigurationError` when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

### `Memory.sync`

```python
async def sync(
    self,
    corpus_id: str,
    *,
    filters: dict[str, Any],
    sources: list[SourceIn],
    written_by: dict[str, Any] | None = None,
    on_event: EventHandler | None = None,
) -> Synced: ...
```

Makes the sources `filters` match exactly `sources` in one storage transaction, recording `written_by` on the sources it writes and as the `closed_by` of those it retires, so a reader sees the old set or the new one, never both and never neither: a current matching source whose `digest` a new one repeats is kept as it is, its `confirmed_at` moved to the moment of the sync, every other current matching source is deleted — or, in a corpus that keeps history, retired: kept out of current reads, with `retired_at` the moment of the sync and `replaced_by` the new source of the same `name` when exactly one has it, and its `valid_to` the moment it stops being in force —, the rest of `sources` is written, and an empty `sources` removes the set; every model call runs before the transaction opens, and a chunk whose text the corpus holds with a vector from the same model reuses it. Returns a `Synced`: the sources in list order, the ids kept, the `Counts` deleted, the ids retired and the usage. The new sources' `created_at` is the moment the sync writes them, in a corpus that keeps history never before a current source of the set. A kept source's `confirmed_at` moves to that moment, and never back. With `on_event`, sends a `Chunked` for every source it writes, all of them first, then each source's `Embedded` and each chunk's `Extracted`, as `ingest` does; a kept source sends nothing, and a sync that finds the set moved and prepares again sends that preparation's events again. In a corpus that keeps history, a retired version stops being in force where the new source of its name starts, at that source's `valid_from` or, when it gives none, at the moment of the sync, where the new source then starts; with no new source of its name, at the moment of the sync; never later than its own `valid_to`; a new source that starts at the replaced version's own `valid_from` corrects it from its start, leaving it never in force. A version that had not begun at the sync is withdrawn: its `valid_to` is its `valid_from`. Raises `StorageError` when the set moved under it twice; sync again. Raises `InputError` when `filters` are empty or malformed or name anything but a `metadata.` key, a new source does not match them, a new source would take effect after the moment of the sync while it replaces a version in force — an update is synced the day it takes effect —, or before the version it replaces took effect, and otherwise what `ingest` raises.

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
    written_by: dict[str, Any] | None = None,
    on_event: EventHandler | None = None,
) -> Remembered: ...
```

Writes one story, recording `written_by` (`{}` unless given), true from `valid_from` (now by default) until `valid_to` when given, or until it is superseded or ended; when an open story already says the same text, ignoring case and extra whitespace, about the same `subject`, with the same `metadata` (compared as JSON values), returns that one and writes nothing, and for a story written already ended returns the story that repeats it exactly, interval too. Returns a `Remembered`: the story and the usage of its extraction, identity and judge. With `on_event`, sends the story's `Embedded` (with the vector pillar) and `Extracted` (with the graph pillar), `name=None`, 1 of 1 — a story that repeats an open one runs neither and sends neither — then its related-stories search's `Retrieved`, `Ranked` (the related stories, without the story itself) and `Judged` events, unless `related=0`. `related` holds the open stories closest to the new one, at most `related` of them, among those `filters` match when given — under them the graph pillar stands down, as in `search` — and judged when `judge` asks, the new story's text as the query. With `judge=True` a decision model judges every hit (every related story for the writes) on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises `InputError` for blank text, `related` below 0, a `judge` that is not `True`, `False` or a tuple naming each judgement at most once, a reserved or malformed filter, a `valid_to` not after `valid_from`, or an entity or a relation end as `ingest` does; `ConfigurationError` for a `judge` that asks for judgements without a decision model or when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

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
    written_by: dict[str, Any] | None = None,
    on_event: EventHandler | None = None,
) -> Remembered: ...
```

Closes the story, or each of several, recording `written_by` on the new one and as each closed one's `closed_by`, at `valid_from` (now by default) and opens a new one there that says `text`, true until `valid_to` when given, and points back through `supersedes` in the order given; each closed story names it in `superseded_by` and records when the supersede was written in `closed_at`. A `valid_from` equal to a story's own corrects it from the start: that story keeps an empty interval, never true, and the new one is true from the same moment. A `valid_from` left to its default that would fall at a story's own start moves one microsecond later, so only one given corrects. The new story keeps the union of their `source_ids`, and their `subject` and `metadata` when they all agree, unless given. Returns a `Remembered`: the new story and the usage of its extraction, identity and judge. With `on_event`, sends the new story's `Embedded` and `Extracted` and its related-stories search's events, as `remember` does. `related` holds the open stories closest to the new one, at most `related` of them, among those `filters` match when given — under them the graph pillar stands down, as in `search` — and judged when `judge` asks, the new story's text as the query. With `judge=True` a decision model judges every hit (every related story for the writes) on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. Raises `NotFoundError` when a story does not exist; `InputError` when the list is empty or names a story twice, a story is not open (superseded, ended, retracted, or its end passed), `valid_from` is before one began or not before its end, `valid_to` is not after `valid_from`, another open story already says `text` about the new story's subject, with its metadata, the text is blank, `related` is below 0, a `judge` is not `True`, `False` or a tuple naming each judgement at most once, or a filter is reserved or malformed; otherwise what `remember` raises, including `ConfigurationError` for a `judge` that asks for judgements without a decision model.

### `Memory.end_story`

```python
async def end_story(
    self,
    corpus_id: str,
    story_id: str,
    *,
    valid_to: datetime | None = None,
    written_by: dict[str, Any] | None = None,
) -> Story: ...
```

Ends an open story with nothing in its place, recording `written_by` as its `closed_by`: what it says stopped being true at `valid_to`, the moment of the call by default, and `closed_at` records when memory learned it, taken as the write lands and later than the moment the story was written down. `superseded_by` stays `None`. A `valid_to` left to its default that would fall at the story's own start moves one microsecond later; one given equal to the story's `valid_from` retracts it, whether or not it has begun: it was never true, and it keeps an empty interval that no read at any moment finds, while a `known_at` before the closing shows it as memory believed it; the stories a retracted story superseded stay closed at its start. Returns the ended story. Raises `NotFoundError` when the story does not exist; `InputError` when it is not open (superseded, ended, retracted, or its end passed), when it has not begun and `valid_to` is not its start, or when `valid_to` is after now or before the story's `valid_from`.

### `Memory.delete_story`

```python
async def delete_story(self, corpus_id: str, story_id: str) -> Counts: ...
```

Deletes a story outright, with its vector, in the corpus's index table too when it has one, the mentions and relations it evidenced and the entities and names only it carried: an entity nothing references any more is deleted, and one still mentioned keeps only the names its remaining passages use. Returns the `Counts` deleted. Raises `NotFoundError` when the story does not exist.

### `Memory.delete_source`

```python
async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...
```

Erases a source and every earlier version of it — the versions retired on the way to it, through their `replaced_by` — with their chunks, their vectors and full-text rows, the mentions and relations their chunks evidenced, and the entities and names only they carried: an entity nothing references any more is deleted, and one still mentioned keeps only the names its remaining passages use. Returns the `Counts` erased. A later version, if any, stays. Raises `NotFoundError` when the source does not exist.

### `Memory.forget_retired`

```python
async def forget_retired(self, corpus_id: str, *, before: datetime) -> Counts: ...
```

Erases the source versions retired before `before`, as `delete_source` erases a source — chunks, vectors, full-text rows, mentions, relations and the entities and names only they carried. A naive `before` is read as UTC. When to call it is yours. Returns the `Counts` erased.

### `Memory.retire_source`

```python
async def retire_source(
    self,
    corpus_id: str,
    source_id: str,
    *,
    replaced_by: str | None = None,
    valid_to: datetime | None = None,
    written_by: dict[str, Any] | None = None,
) -> Source: ...
```

Retires a stored version without writing a new one, in a corpus that keeps history: `retired_at` is the call's moment, later than the version's `created_at` whatever the clock says, `replaced_by` the version already filed that takes its place, if any, and `closed_by` `written_by` (`{}` when not given); its chunks are retired with it, and it stays whole for reads of the past until `forget_retired` erases it, or `delete_source` erases its successor. Its `valid_to` is the one given — not after the call, not before the version took effect, read as UTC when naive — or its successor's `valid_from` when that has one, or the call's moment, and never later than the end it already had; a version not yet in force ends where it begins, never in force. Returns the retired `Source`. Raises `InputError` in a corpus that keeps no history, for a version already retired, a successor that is the version itself or retired, a `valid_to` after the call or before the version took effect, or a successor that took effect before the version, or takes effect after the call while the version is in force; `NotFoundError` when the source or the successor does not exist.

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
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Source], str | None]: ...
```

Lists the sources `filters` match (values, lists or ranges, as in `search`) that are in force now, newest first, `limit` at a time: each in force by its validity when given, and as the version of its document in force, until the version after it took over. `as_of` lists those in force at another moment; `known_at` lists them as memory knew them then, each retired only if that was so by then (in a corpus without history a sync erased the versions it replaced, so only sources written by then and still held are seen), and `as_of` defaults to it; `history=True` lists every version, retired ones included. Returns the page and the cursor of the next one, `None` on the last. Its `filters` take a record's dates as `search`'s do, and `confirmed_at` with `known_at` is an `InputError`. Raises `InputError` for `history=True` with `as_of`, a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.read_source`

```python
async def read_source(
    self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None
) -> str: ...
```

Returns a source's whole text, of any version by id, retired or not, or, with `locator`, the text of the chunks whose locator holds every key and value given, joined by blank lines and empty when none does. Raises `NotFoundError` when the source does not exist.

### `Memory.resembling`

```python
async def resembling(
    self,
    corpus_id: str | list[str],
    *,
    text: str | None = None,
    source_id: str | None = None,
    source_corpus: str | None = None,
    filters: dict[str, Any] | None = None,
    as_of: datetime | None = None,
    known_at: datetime | None = None,
    history: bool = False,
    limit: int = 10,
    on_event: EventHandler | None = None,
) -> Resembled: ...
```

Finds the documents in the corpus or corpora `corpus_id` names that share the most text with `text`, a document not yet filed, or with `source_id`, a stored document held in `source_corpus` (the one corpus named by default), which is itself left out. The documents read are those `list_sources` reads with the same `filters`, `as_of`, `known_at` and `history`. Returns a `Resembled` holding at most `limit` matches, the highest `share` first, then by corpus and source id, by code point, and the `retrieval_id` of its recording when a refinery recorded it; `on_event` receives its `Recorded`. Every document's text is read as words, lower-cased and split at anything that is not a letter or a digit, in runs of five; `share` estimates the runs two documents have in common over their average size. Only documents whose signatures share a band with the probe's are scored: one sharing two thirds of its text or more is found 99.7% of the time, half about four in five, a third about three in ten. Its `filters` take a record's dates as `search`'s do, and `confirmed_at` with `known_at` is an `InputError`. Raises `InputError` without exactly one of `text` and `source_id`, for a text in which the database finds no letter or digit (by its own locale), a `source_corpus` without a `source_id`, a `source_id` with several corpora named and no `source_corpus`, a `source_corpus` not among them, a reserved or malformed filter, `history=True` with `as_of`, or `limit` below 1; `NotFoundError` when the source does not exist.

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

Lists stories, latest `valid_from` first, `limit` at a time: those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from`, and `subject` and `filters` (values, lists or ranges, as in `search`) narrow. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its closing was written by then — and `as_of` defaults to it. Returns the page and the cursor of the next one, `None` on the last. Its `filters` take a record's dates as `search`'s do, and `confirmed_at` with `known_at` is an `InputError`. Raises `InputError` for `history=True` with `as_of`, a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.reembed`

```python
async def reembed(self, corpus_id: str, *, on_event: EventHandler | None = None) -> int: ...
```

Embeds again every chunk and story whose vector was not made by the current embedding model, 256 at a time — those `fingerprint` counts as `stale`. With a `DocumentEmbeddingModel` it re-embeds every chunk of a source with a stale one, in one call per source. Returns how many it embedded. With `on_event`, sends an `Embedded` after each batch it writes — named by its source for a whole document, `None` otherwise — whose `done` climbs to `total`, the stale count read once at the start; when it re-embeds more than it counted, as whole documents can, `total` grows to match `done`. Raises `ConfigurationError` without the vector pillar; `ModelError` when the embedding model breaks its contract.

### `Memory.set_language`

```python
async def set_language(self, corpus_id: str, language: Language) -> int: ...
```

Sets the language the corpus's text pillar reads in, and the graph pillar's `"words"` reading of entity names — the name of any text search configuration the database has; until set, the schema's default language — the one `rfnry-memory create` recorded and `rfnry-memory set --language` changes — kept from the corpus's first write — and re-indexes its chunks and stories, and re-reads its entities' names, with the configuration as it is now, in one transaction, while writes to the corpus wait; naming a configuration you created again re-reads the corpus with it, so a change you made to it takes effect. Every configuration is read exactly as it is: a built-in one as Postgres ships it, one you created as you made it. A configuration dropped, or its schema renamed, after a corpus names it makes the corpus's writes raise `StorageError`, and its keyword pillar and the graph pillar's `"words"` reading fail with one — `"exact"` and `"substring"` still read — until `set_language` names another configuration. The name is kept as Postgres writes it: `French` reads back as `french`. Returns how many passages it re-indexed. Raises `InputError` for a name that is not a text search configuration in the database.

### `Memory.set_history`

```python
async def set_history(self, corpus_id: str, keep: bool) -> None: ...
```

Sets whether the corpus keeps the history of its sources — `False` until set. While it is `True`, `sync` retires what it replaces or drops instead of deleting it; from the moment it is set, never before. Waits for a running `sync` of the corpus. Raises `InputError` when `keep` is not a bool.

### `Memory.set_statistics`

```python
async def set_statistics(self, corpus_id: str, keep: bool) -> int: ...
```

Sets whether the corpus keeps the statistics `BM25(statistics="corpus")` ranks by — `False` until set — and returns how many passages it counted, `0` for `False`. `True` counts the corpus's current passages, chunks not retired and stories not closed, chunks and stories apart: how many there are, their total length and, for each word, how many hold it; every write to the corpus keeps them exact from then on, in its own transaction, and writes into a corpus that keeps them take turns by kind. Called again with `True` it counts again from the rows. `False` deletes the counts. It waits for the corpus's writes in flight, and they wait for it. Raises `InputError` when `keep` is not a bool.

### `Memory.set_vector_index`

```python
async def set_vector_index(
    self, corpus_id: str, index: HNSW | IVFFlat | None, *, rebuild: bool = False
) -> int: ...
```

Gives the corpus a vector index of the method `index` names, in a table of its own, or with `None` takes it away; a corpus without one is searched exactly. Building copies the vector of every chunk and story of the corpus into the table, while writes to the corpus wait, then builds one index for chunks and one for stories with `CREATE INDEX CONCURRENTLY`, while writes go on, and returns when done; a corpus that holds nothing yet can be switched on. Called again with the same settings it builds nothing, unless `rebuild=True`, which builds both indexes again in place with `REINDEX INDEX CONCURRENTLY`; with other settings or the other method it builds the new indexes, then drops the old ones, and searches use the old until then. `None` drops the table. Every write to the corpus updates the table in the write's own transaction, and `delete_corpus` drops it. While the table exists, a vector search of the corpus — `search`, `recall`, `expand` by vector — reads the index unless a scope narrows it: approximate, it can miss a close passage, and when the index returns fewer than `top_k` the search runs again exactly; a scoped search reads every passage in its scope; until the index of chunks or of stories is built and valid — while the first build runs, or after one that failed — searches of that kind run exactly as before. What exists in the database is the setting, so every service sharing it sees the same; two calls for one corpus run one after the other. An option left out is left to pgvector; `precision` unset is `"full"` up to 2000 dimensions and `"half"` above. Search options — `rerank`, `ef_search`, `probes` — are kept with the index, so a call that changes only them builds nothing and the next search reads them. A build that fails leaves its index unfinished, and the next call builds it again. Returns how many passages the table holds, `0` for `None`. Raises `InputError` when `index` is not an `HNSW`, an `IVFFlat` or `None`, `rebuild` is not a bool or is `True` with `None`, `"full"` is asked above 2000 dimensions, `rerank` is given with full precision, or pgvector refuses a value; `ConfigurationError` without an embedding model, above 4000 dimensions without `"binary"`, or on pgvector older than 0.8.0; `StorageError` when a build fails.

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

Reads the corpus and says what it holds: the latest version of each source and their chunks counted, and its stories, a hash over each current source's name, metadata and text, another over every story's text, subject, metadata and interval, the models its vectors came from, how many passages the vector pillar would miss, how this memory embeds and chunks, and the corpus's language; `history` is the corpus's history setting, `retired` counts the source versions it keeps retired and `vector_index` is its vector index as built. Ids and write times are not read, so identical content under new ids gives the same fingerprint. No search computes it. Returns a `CorpusFingerprint`.

### `Memory.describe`

```python
async def describe(self, corpus_id: str) -> CorpusDescription: ...
```

Reads the corpus and says what it holds and what works on it, for a model planning a search: the latest version of each source and their chunks, the stories true now, the retired source versions, the entities, the relation rows (one for each passage that says a relation, so the same relation said again counts again, as does one a changed sync wrote again in a corpus that keeps history), each entity and relation type with how many entities or relation rows have it, its language and history setting, this memory's pillars, whether it has a decision model, and the `expand` modes those pillars support. No search computes it. Returns a `CorpusDescription`.

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
    text: TextRanking | None = None,
    fusion: Fusion | None = None,
    match: Match = "any",
    depth: int = 2,
    spot: Spot | None = None,
    on_event: EventHandler | None = None,
) -> Hits: ...
```

Searches chunks and stories in the corpus `corpus_id` names, or in each of several, with `pillars` (every configured pillar by default), fuses their lists by the memory's `fusion` setting, or by `fusion` for this call alone, and returns the first `top_k` as `Hits`; each pillar hands in its first `top_k` × the fusion's `window` candidates. Over several corpora the query is embedded once; keyword search counts the corpora that share a text search configuration as one corpus and lets groups read by different configurations take turns by rank, the vector and graph pillars merge the corpora's lists by score before the fusion, every hit carries its `corpus_id`, and a pillar that fails in one corpus fails for the search.

`filters` keep the passages whose facets match, by `metadata.<key>`, or by `written_by.<key>` for who wrote them: a value, a list of values (any of them), or a range of `gt`, `gte`, `lt` and `lte`, all finite numbers or all strings compared by code point; a range on a key `rfnry-memory index` indexed reads its index when the call reads the present (no `as_of`, `known_at` or `history`), with the same answer. `created_at`, `confirmed_at`, `valid_from` and `valid_to` filter by a record's own dates: each a range of datetimes or ISO 8601 strings, a naive one read as UTC; a chunk reads its source's `created_at` and `confirmed_at`, a story's `confirmed_at` is its `created_at`, a record without that side never matches, and under `known_at` each row is compared by the dates it shows then. `confirmed_at` with `known_at` raises `InputError`. Stories are those true now unless `as_of` or `history=True`; `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its closing was written by then — and `as_of` defaults to it. Chunks are those of the documents in force at `as_of`, now by default: by their validity when given, and as the version of each document in force then; with `known_at`, as memory knew them then (in a corpus without history a sync erased the versions it replaced, so only sources written by then and still held are seen); with `history=True`, of every version, retired ones included, each hit's `Chunk.retired_at` saying which. Graph evidence in chunks follows the same rule.

The graph pillar starts from the entities `entities` names — each name found in each corpus by its name or an alias, in every type that holds it — and from the names it spots in the query as `spot` reads it: `"words"`, a name's words in order, side by side, read in the corpus's language as keyword search reads text, stop words dropped; `"exact"`, the whole name, with the text's edge or a character other than a letter, a digit, `_` or `-` on each side; `"substring"`, the name anywhere; `"off"`, none — each entity by its name or an alias, case and spacing aside, each corpus in its own language; left out, `spot` is the memory's (`memory.spot`, `"words"` unless `Memory` was given another) when the call names neither `entities` nor `around` and `"off"` when it does. It walks `depth` hops (0 to 3, 2 by default) along `relation_types` (read upper-case, every type by default), and ranks the passages that mention an entity it reached nearest first — by `h`, the fewest hops to an entity the passage mentions — then by `s`, the sum, over the reached entities it mentions, of `2^-hops × ln(1 + (N − n + 0.5) / (n + 0.5))` — `N` the passages that mention a reached entity, `n` those that mention this one — highest first, then by passage id, by code point; its graph score is `2^-h × (1 + s / (1 + s)) / 2`, so a passage nearer the start scores higher whichever corpus holds it; a name no corpus holds is not an error: `Hits.unresolved` maps it to up to five similar names.

`sources` keeps the search to the chunks of those sources and the stories whose `source_ids` cite one of them, in every corpus named, and `around` to the passages that mention the entities it names — found as `entities` are, a missing one reported in `Hits.unresolved` the same way — or, within `hops` (0 to 2), their neighbours along `relation_types`; with both, to what both keep. Every pillar runs inside that scope — keyword search with its statistics taken from inside it, the graph pillar starting from `around` unless `entities` is given — and filters, the window and `kinds` apply on top; a scope that holds nothing gives no hits. The graph pillar stands down under `filters`.

`match` decides which passages keyword search considers: `"any"` of the query's words, `"all"` of them, a `"phrase"` of them in order, side by side, or `"web"`, the query read by Postgres's `websearch_to_tsquery` — one that only excludes looks for nothing — never an error; it narrows keyword search only. A code in the query (`Match`) is needed whole under `"all"` and `"phrase"`, and ranks as a word under `"any"` and `"web"`; cover density ranks by the query's words. Keyword matches rank by the memory's `text` setting, or by `text` for this call alone; equal scores go by cover density's own tie-break when it ranks, then by passage id, by code point, then by corpus.

With `judge=True` a decision model judges every hit on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. With `on_event`, sends a `Retrieved` per pillar as it finishes, a failed one with its error, then `Ranked` with the hits before any judging, then, with `judge`, a `Judged` per hit as each decision returns, one at a time.

Raises `ConfigurationError` for a pillar not configured, a `judge` that asks for judgements without a decision model, or `entities`, `relation_types` or `around` with the graph pillar off; `InputError` for a blank query, a `corpus_id` that names no corpus, an unknown kind, `top_k` below 1, `hops` outside 0 to 2, `depth` outside 0 to 3, a `match` other than `"any"`, `"all"`, `"phrase"` or `"web"`, a `spot` other than `"words"`, `"exact"`, `"substring"` or `"off"`, a reserved or malformed filter, `history=True` with `as_of`, or a `judge` that is not `True`, `False` or a tuple naming each judgement at most once, a `text` that is not a `BM25` or a `CoverDensity`, or a `fusion` that is not an `RRF` or a `RelativeScore`; `ModelError` when the judge breaks its contract.

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
    text: TextRanking | None = None,
    fusion: Fusion | None = None,
    match: Match = "any",
    depth: int = 2,
    spot: Spot | None = None,
    on_event: EventHandler | None = None,
) -> Hits: ...
```

Searches stories only, as `search` does, among those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from` and `subject` narrows. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its closing was written by then — and `as_of` defaults to it. Returns `Hits`. With `judge=True` a decision model judges every hit on the four judgements; with a tuple of judgement names it is asked only those — a single-score reranker answers `("relevant",)`. With `on_event`, sends what `search` sends. Raises what `search` raises, and `InputError` for `history=True` with `as_of`.

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
    depth: int = 2,
    on_event: EventHandler | None = None,
) -> Hits: ...
```

Returns the passages around one passage of the corpus, never that passage itself, as `Hits` whose `provenance` names the mode with each hit's score divided by the best, and whose `pillars` are empty. `by="document"` reads, for a chunk, the chunks of its source within `width` positions on each side, in document order — `kinds` and `top_k` do not apply — and for a story the other stories with its subject, at most `top_k`, those whose `valid_from` is nearest its own, given in time order — `kinds` does not apply. `by="vector"` searches `kinds` with the vector the passage was stored with as the query, nearest first, at most `top_k`, and calls no model; a passage stored without a vector gives no hits. `by="graph"` walks `depth` hops (0 to 3, 2 by default) from the entities the passage mentions, along `relation_types` (read upper-case) when given, and ranks the passages of `kinds` as `search`'s graph pillar does, at most `top_k`; a passage that mentions no entity gives no hits. `filters` and the window hold as on `search` — stories true now unless `as_of` or `history=True`, `known_at` as memory knew things then — and a chunk's neighbours are seen only while its document is in force at the moment read. With `judge`, a decision model judges each hit against the starting passage's text, as `search` judges against the query. Its `filters` take a record's dates as `search`'s do, and `confirmed_at` with `known_at` is an `InputError`. With `on_event`, sends only its recording's `Recorded`, when a refinery records it. Raises `NotFoundError` for a passage the corpus does not hold; `InputError` for an unknown `by`, a passage that is not a `(kind, id)` pair, `top_k` or `width` below 1, `depth` outside 0 to 3, an unknown kind, a reserved or malformed filter, `history=True` with `as_of`, a `judge` that is not `True`, `False` or a tuple naming each judgement at most once, or `by="document"` from a story without a subject; `ConfigurationError` for `by="vector"` without the vector pillar, `by="graph"` without the graph pillar, or a `judge` that asks for judgements without a decision model.

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

Walks the graph up to `hops` from the entity with this name or alias, along `relation_types` only when given, read upper-case. Returns a `Neighborhood`: the entities reached, the relations walked and the passages that mention them, current stories unless `as_of` or `history=True`. Relations are those some visible passage evidences: a chunk of a document in force at the moment read, or a story true then. `known_at` answers as memory knew things at that moment — stories written by then, each closed only if its closing was written by then — and `as_of` defaults to it. Raises `ConfigurationError` with the graph pillar off; `NotFoundError` when no entity has the name, of `entity_type` when given; `InputError` when entities of several types have it and `entity_type` is not given, or for `history=True` with `as_of`.

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
    confirmed_at: datetime
    chunks: int
    digest: str | None = None
    retired_at: datetime | None = None
    replaced_by: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    written_by: dict[str, Any] = {}
    closed_by: dict[str, Any] | None = None
```

A stored source: its whole text, `written_by` as its write declared it, and `closed_by` as the sync, replace or `retire_source` that retired it declared it (`None` while not retired, as known), its `metadata` facets, how many chunks it was split into, and `digest`, a hash of the source as it was given with the chunk sizes, pillars and extraction it was written under, by which `sync` keeps a source the new list repeats; `None` when none was recorded. `created_at` is when memory wrote this version, and `confirmed_at` the last moment the source of truth vouched for it: its write, or a later `sync` that kept it unchanged. `valid_from` and `valid_to` are when what it says is in force, as given; `None` on a side that was not given. `retired_at` is when a `sync`, a replace or `retire_source` retired this version, in a corpus that keeps history, and `replaced_by` the source that took its place when one did; both are `None` on a current source, and on a source read as memory knew it before it was retired, whose `valid_to` then reads as it was written.

### `Resemblance`

```python
class Resemblance(BaseModel):
    source: Source
    share: float
```

A document `resembling` found, and `share`, about how much of its text it has in common with the probe, from 0 to 1.

### `Resembled`

```python
class Resembled(BaseModel):
    matches: list[Resemblance]
    retrieval_id: str | None = None
```

What `resembling` returns: the documents it found, the highest `share` first, and the `retrieval_id` of its recording when a refinery recorded it.

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
    created_at: datetime
    confirmed_at: datetime
    context: str | None = None
    retired_at: datetime | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    written_by: dict[str, Any] = {}
```

A piece of a source as indexed: the source's id, name and `metadata`, its position in the source, the section's title or else the headings it falls under, and the section's locator — so a hit can cite its document without reading the source. `context` is its section's. `created_at`, `confirmed_at`, `retired_at`, `valid_from`, `valid_to` and `written_by` are its source version's, so a hit says when its document was written, last confirmed, replaced and in force.

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
    written_by: dict[str, Any] = {}
    closed_by: dict[str, Any] | None = None
```

A story, true from `valid_from` until `valid_to` (`None` while no end is known), written at `created_at`, pointing through `supersedes` at the stories it replaced; `written_by` is what its write declared, never part of its identity. A closed story records in `closed_at` when its closing was written, and in `closed_by` what that write declared (`None` while not closed, as known): a superseded one names its replacement in `superseded_by`, and one `end_story` ended names none. An empty interval, `valid_to` equal to `valid_from` with `closed_at` set, is a story found never true: retracted by `end_story` or corrected from its start by `supersede`. Raises pydantic's `ValidationError` when `valid_to` is before `valid_from`, or equal to it with no `closed_at`.

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
    valid_from: datetime | None = None
    valid_to: datetime | None = None
```

A source to write, as `sync` takes it: what `ingest` takes, as one value. `valid_from` and `valid_to` are when what it says is in force, when known; a naive datetime reads as UTC. Raises `InputError` when it has both or neither of `text` and `sections`, or a `valid_to` not after its `valid_from`.

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

One search result: a chunk or a story of the corpus `corpus_id`, its fused `score`, and in `provenance` each pillar that returned it with its score divided by that pillar's best — for keyword hits from corpora read by different text search configurations, by the best keyword hit of the same configuration; for `expand`, the mode that found it. The passage itself is in `chunk` or `story`, `judge` is set when the search was judged, and `entities` are the entities the passage mentions as memory holds them now, whatever `known_at` the search asked, ordered by name key, read beside the passages in one storage call per corpus — empty with the graph pillar off.

### `Hits`

```python
class Hits(BaseModel):
    hits: list[Hit]
    pillars: dict[str, PillarReport]
    usage: Usage
    unresolved: dict[str, list[str]] = {}
    retrieval_id: str | None = None

    def by(self, check: JudgementName, *, descending: bool = True) -> Hits: ...
```

A search's hits in fused order, a report for each pillar that ran, the usage of the judge, and in `unresolved` each name of `entities` or `around` that no corpus searched holds, with up to five similar names. `retrieval_id` names the search's recording when a refinery recorded it, `None` otherwise. `by` returns a copy ordered by one judgement, a `JudgementName`, highest first unless `descending=False`, ties in fused order. `by` raises `InputError` for an unknown judgement, hits that were not judged, or a judgement they were not judged for.

### `Judgement`

```python
class Judgement(BaseModel):
    relevant: float | None = None
    evidence: float | None = None
    contradicts: float | None = None
    injection: float | None = None
    model: str | None = None
    settings: dict[str, JsonValue] | None = None
```

A decision model's four probabilities for one hit against the query: it is about what the query asks, it contains what answers the query, it contradicts a premise the query states, it holds instructions addressed to an AI. A judgement the judge was not asked for is `None`. `model` and `settings` name the model that judged the hit and the settings it ran with: the ones the `Decided` reported, else the adapter's `model` and `settings` attributes, `None` when there are none. They are not judgements: `by` and `order_by` refuse them, and the judgement a model reads through `MemoryTools` leaves them out. A `Reply` or a `Decided` carries its own `settings` for one call, and they win, even when empty; `settings` that are not a JSON object on an adapter are ignored.

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
    retired: list[str] = []
    deleted: Counts = Counts()
```

What `ingest` returns: the source written and what its model calls cost; with `replaces`, the id it retired, in a corpus that keeps history, or the `Counts` it erased.

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
    retrievals: int = 0
    statistics: int = 0

    @property
    def clean(self) -> bool: ...
```

Rows by table, as deletes and `verify_absent` return them; `vectors` counts the rows of the table of the corpus's vector index, and `corpora` the corpus's own row, kept from its first write until `delete_corpus`, and that table itself, so one left behind is not clean; `retrievals` counts the refinery's recordings that read the corpus, among others or alone, and `statistics` the rows of the corpus's statistics (`set_statistics`). `clean` is `True` when every count is zero.

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
    statistics: bool = False
    retired: int = 0
    vector_index: VectorIndex | None = None
    text: TextRanking = BM25()
    fusion: Fusion = RRF()
    spot: Spot = "words"
```

What `fingerprint` returns. `content` is the first 16 hex characters of a sha256 over each current source's name, metadata and text, and its `valid_from` and `valid_to` when it was given either, sorted; `story_content` the same over each story's text, subject, metadata, `valid_from` and `valid_to`, in UTC. `embedding` is the model of the vector pillar, `None` without it. `embedded` counts the chunks and stories, together, that carry a vector, by the model that made it, sorted by model; `stale` counts those whose vector is from another model than `embedding` or missing — what `reembed` embeds again — and is `0` without the vector pillar. `language` is the one the corpus's text pillar reads in. `history` is the corpus's history setting, `statistics` whether it keeps statistics (`set_statistics`), and `retired` counts the source versions it keeps retired; `sources` and `chunks` count current ones, while `embedded` and `stale` count every chunk and story vector, retired versions included. `vector_index` is the corpus's vector index as `set_vector_index` built it, an `HNSW` or an `IVFFlat` read from the database: `None` when the corpus has none, or when its chunks and stories have no finished index built with the same settings. `text`, `fusion` and `spot` are the text ranking, the fusion and the reading of names the memory uses when a call gives none.

### `HNSW`

```python
class HNSW(BaseModel):
    method: Literal["hnsw"] = "hnsw"
    m: int | None = None
    ef_construction: int | None = None
    precision: Literal["full", "half", "binary"] | None = None
    rerank: int | None = None
    ef_search: int | None = None
```

An HNSW index, for `set_vector_index`, and as `fingerprint` reports one built. `m` and `ef_construction` build the index and go to pgvector as given; left out, pgvector's own defaults apply and they read back as left out. `precision` is `"full"` for an index on the vectors, `"half"` for one on their half-precision copies and `"binary"` for one on one bit per dimension (`binary_quantize`), compared by Hamming distance, for any dimension the vectors have; left out, it is `"full"` up to 2000 dimensions and `"half"` above, never `"binary"`, and it reads back as built. `rerank` (1 or more) and `ef_search` (1 to 1000) are how searches read the index, kept with it and changed without building it again: a search of a half or binary index takes `rerank` times `top_k` candidates from it and orders them by their full vectors, 1 by default on half and 4 on binary, and a search sets `hnsw.ef_search` for its own transaction when the index carries one, the pool's value otherwise; left out, they read back as left out. Frozen, strict — `m=True` or `m="16"` is a `ValidationError` — and closed to other fields.

### `IVFFlat`

```python
class IVFFlat(BaseModel):
    method: Literal["ivfflat"] = "ivfflat"
    lists: int | None = None
    precision: Literal["full", "half", "binary"] | None = None
    rerank: int | None = None
    probes: int | None = None
```

An IVFFlat index, as `HNSW` is an HNSW one. `lists` builds the index and goes to pgvector as given, its own default when left out; the lists are learned from the vectors present when the index is built, so build it once the corpus holds its passages, or again with `rebuild=True`. `precision` and `rerank` are as on `HNSW`. `probes` (1 to 32768, pgvector's bounds, and at most `lists` when both are given) is how many lists a search reads, set as `ivfflat.probes` for the search's own transaction; left out, the pool's value applies, one list unless it is raised there.

### `VectorIndex`

```python
VectorIndex = Annotated[HNSW | IVFFlat, Field(discriminator="method")]
```

Either method, told apart by `method`, so a setting kept as JSON reads back as the right one: `TypeAdapter(VectorIndex).validate_json('{"method": "ivfflat", "lists": 100}')`.

### `BM25`

```python
class BM25(BaseModel):
    method: Literal["bm25"] = "bm25"
    k1: float = 1.2
    b: float = 0.75
    statistics: Literal["read", "corpus"] = "read"
```

Keyword matches ranked by BM25, the default `text` of a `Memory`; equal scores rank by passage id, by code point, then by corpus. `k1` (0 or more, finite) is how fast a repeated word stops adding: 0 counts presence only. `b` (0 to 1) is how much a passage longer than the mean is held back: 0 never, 1 fully. With `statistics="read"` the statistics come from the passages the read can see that hold a query word; under `match="all"` or `"phrase"` every one holds every word, so the words count as equally rare and the rank comes from how often they appear and the passage's length. With `statistics="corpus"` they come from every current passage of the corpora read (`set_statistics`), counted on write, so a word is as rare as it is in the corpus whatever the match, and a filter, a window or a scope leaves the scores as they were; a word no current passage holds counts as held by one. A read with `"corpus"` over a corpus that keeps no statistics raises `ConfigurationError`. Frozen, strict — `k1="1.2"` or `k1=True` is a `ValidationError` — and closed to other fields.

### `CoverDensity`

```python
class CoverDensity(BaseModel):
    method: Literal["cover_density"] = "cover_density"
    length: Literal["ignore", "log", "linear"] = "ignore"
```

Keyword matches ranked by how close together the query's words sit — Postgres's `ts_rank_cd` over every word of the query, so a passage holding only some of them ranks after those holding all, and among those by how often it holds the query's words, then by passage id, by code point, then by corpus. `length` is how much a long passage is discounted: `"ignore"` not at all, `"log"` by one plus the logarithm of its length, `"linear"` by its length. Frozen, strict and closed, as `BM25`.

### `TextRanking`

```python
TextRanking = Annotated[BM25 | CoverDensity, Field(discriminator="method")]
```

Either ranking, told apart by `method`, so a setting kept as JSON reads back as the right one: `TypeAdapter(TextRanking).validate_json('{"method": "cover_density", "length": "log"}')`.

### `RRF`

```python
class RRF(BaseModel):
    method: Literal["rrf"] = "rrf"
    k: int = 60
    weights: dict[Literal["text", "vector", "graph"], float] = {}
    window: int = 4
```

Reciprocal rank fusion, the default `fusion` of a `Memory`: each pillar adds `weight / (k + rank)`, rank counted from 1, to every passage it returned, and passages are ordered by the sum, equal sums in the order the pillars, by name, then their ranks first reached them. `k` (1 or more) is how much the top of each list counts — smaller counts it more. `weights` maps a pillar to a positive number, 1 for a pillar left out. `window` (1 or more): each pillar hands in its first `top_k × window` candidates. Frozen, strict — `k=1.5` is a `ValidationError` — and closed to other fields.

### `RelativeScore`

```python
class RelativeScore(BaseModel):
    method: Literal["relative_score"] = "relative_score"
    weights: dict[Literal["text", "vector", "graph"], float] = {}
    window: int = 4
```

Fusion by score: each pillar's scores are scaled to 0–1 by the lowest and highest it returned — all 1 when they are equal — and a passage scores the sum of `weight × scaled score` over the pillars that returned it; equal sums keep the order `RRF` would leave them in. `weights` and `window` as `RRF`'s. Frozen, strict and closed, as `RRF`.

### `Fusion`

```python
Fusion = Annotated[RRF | RelativeScore, Field(discriminator="method")]
```

Either fusion, told apart by `method`, so a setting kept as JSON reads back as the right one: `TypeAdapter(Fusion).validate_json('{"method": "rrf", "k": 20}')`.

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

What `describe` returns: `sources` and `chunks` count the latest version of each source and their chunks, `stories` the stories true now and `retired` the source versions kept retired; `entities` and `relations` count the graph's rows, and `entity_types` and `relation_types` map each type to how many entities or relation rows have it, in code-point order. `language` and `history` are the corpus's settings; `pillars` are the memory's, sorted, `judge` says whether it has a decision model, and `expand` lists the modes `expand` takes, `document` always.

## Refinery

### `MemoryRefinery`

```python
class MemoryRefinery:
    def __init__(self, storage: PostgresStorage, *, sample: float = 1.0) -> None: ...
```

Records retrievals so they can be read back, labeled, exported and replayed, in the storage it is given, the memory's: `Memory(storage, refinery=MemoryRefinery(storage))`. Its tables are `memory_retrievals` and `memory_retrieval_labels`, which `rfnry-memory` creates. A `Memory` given it records each `search`, `recall`, `expand` and `resembling` that `sample` keeps: a rate from 0 to 1, read against a hash of the recording's id, so the same ids are always kept. A recording holds the call's input and every argument, the window, each pillar's ranked candidates and the fused hits by id, never their text, each pillar's report, the call's duration and the memory's settings; a call that raises before its results is not recorded. Its id is a fresh `uuid4().hex`, never one the memory's `ids` gives; `at` is the memory's clock when the call began, `recorded` the wall clock when it was written. A read names its corpora, one or several, and a recording is read only when every corpus it read is among them; `delete_corpus` erases every recording that read the corpus. Its names never meet rfnry-engine's refinery: a program may hold both. Raises `ConfigurationError` for a `sample` outside 0 to 1.

### `MemoryRefinery.retrieval`

```python
async def retrieval(self, corpus_id: str | list[str], retrieval_id: str) -> Retrieval: ...
```

Returns a recording. Raises `NotFoundError` when it is not held, or read a corpus `corpus_id` does not name.

### `MemoryRefinery.retrievals`

```python
async def retrievals(
    self,
    corpus_id: str | list[str],
    *,
    since: datetime | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Retrieval], str | None]: ...
```

Lists the recordings of the corpora `corpus_id` names, the latest recorded first, `limit` at a time, those recorded after `since` when given, a naive `since` read as UTC. Returns the page and the cursor of the next one, `None` on the last. Raises `InputError` for `limit` below 1 or a cursor no listing returned.

### `MemoryRefinery.forget_recorded`

```python
async def forget_recorded(self, corpus_id: str | list[str], *, before: datetime) -> int: ...
```

Erases the recordings of the corpora `corpus_id` names recorded before `before`, with their labels, and returns how many. A naive `before` is read as UTC. When to call it is yours.

### `MemoryRefinery.is_holdout`

```python
def is_holdout(self, retrieval_id: str, *, holdout_percent: int = 20) -> bool: ...
```

Whether a recording is held out of training, by the rule rfnry-engine's refinery uses for its runs: the first four bytes of the sha256 of the id, mod 100, below `holdout_percent`. The same id always answers the same. Raises `InputError` for a percent outside 0 to 100.

### `MemoryRefinery.review`

```python
async def review(
    self,
    corpus_id: str | list[str],
    retrieval_id: str,
    review: RetrievalReview,
    *,
    label_id: str | None = None,
) -> RetrievalLabel: ...
```

Stores a person's label on a recording, source `human`: grades on the passages it returned, or on passages or documents it did not — its misses. With `label_id`, the first call stores the label and every later call with that id on the same recording returns it and writes nothing; the id names nothing beyond its recording. Raises `NotFoundError` for a recording not held or not readable by `corpus_id`, and `InputError` for a grade naming a corpus the recording did not read, a blank `label_id`, or a `label_id` the recording already holds from another source.

### `MemoryRefinery.outcome`

```python
async def outcome(
    self,
    corpus_id: str | list[str],
    retrieval_id: str,
    outcome: RetrievalOutcome,
    *,
    label_id: str | None = None,
) -> RetrievalLabel: ...
```

Stores what happened, source `world`: the passages the answer cited, each relevant, and whether it worked. Takes `label_id` and raises as `review` does.

### `MemoryRefinery.labels`

```python
async def labels(self, corpus_id: str | list[str], retrieval_id: str) -> list[RetrievalLabel]: ...
```

Returns a recording's labels, the oldest first. Raises `NotFoundError` for a recording not held or not readable by `corpus_id`.

### `MemoryRefinery.label`

```python
async def label(
    self, corpus_id: str | list[str], memory: Memory, *, limit: int = 500
) -> RetrievalLabeled: ...
```

Asks `memory`'s decision model whether each hit of up to `limit` recordings is `relevant` to its query, the oldest first, and stores one label of source `judge` per recording: grade 1 when the probability is at least 0.5, else 0, the probability kept, and counts it in `RetrievalLabeled.judged`. A recording a judge already labeled is passed over, and so is one of `resembling`; one whose call judged every hit `relevant` is labeled from those judgements, without asking again; `expand`'s are judged against the passage it looked around. A recording with no hits, or an `expand` whose passage is gone, gets an empty label, counted in `skipped`, and is not asked about again; an empty label never makes a recording labeled for `replay` or `measure`. A recording that could not be judged is named in `RetrievalLabeled.errors` and stays unlabeled, to be asked again. Raises `ConfigurationError` when `memory` has no decision model, and `InputError` for `limit` below 1.

### `MemoryRefinery.replay`

```python
def replay(
    self,
    corpus_id: str | list[str],
    candidate: Memory,
    *,
    holdout: bool | None = True,
    holdout_percent: int = 20,
    labeled: bool = True,
    moment: Literal["recorded", "now"] = "recorded",
    sources: tuple[Literal["human", "world", "judge"], ...] = ("human", "world"),
    since: datetime | None = None,
    limit: int | None = None,
) -> AsyncIterator[RetrievalReplay]: ...
```

An async iterator: runs the recordings of the corpora `corpus_id` names again on `candidate`, the oldest first, up to `limit`, those recorded after `since` when given (naive, read as UTC), yielding each beside its labels and what the candidate returned. `holdout=True` keeps the held-out recordings (`is_holdout` at `holdout_percent`), `False` the rest, `None` all; `labeled=True` those with a label from `sources`, an empty judge label not counted. Each runs the recorded verb with the recorded query, filters, window, kinds, `top_k`, scope, graph arguments, `match`, and `pillars` narrowed to the candidate's; the candidate's own `text`, `fusion` and `spot` decide, and nothing is judged. `moment="recorded"` reads as memory knew things when the call was recorded — its own `known_at`, or else its `at` — so documents written since never reach the comparison; `moment="now"` reads what the candidate holds, for a candidate filled since: another embedding model or chunking, in another schema. Nothing a replay runs is recorded. A recording the candidate cannot run is yielded with its `error`. Raises `InputError` for a `moment` other than these, `sources` outside `human`, `world` and `judge`, a `holdout_percent` outside 0 to 100, or `limit` below 1.

### `MemoryRefinery.measure`

```python
async def measure(
    self,
    corpus_id: str | list[str],
    candidate: Memory | None = None,
    *,
    k: int = 10,
    holdout: bool | None = True,
    holdout_percent: int = 20,
    labeled: bool = True,
    moment: Literal["recorded", "now"] = "recorded",
    sources: tuple[Literal["human", "world", "judge"], ...] = ("human", "world"),
    since: datetime | None = None,
    limit: int | None = None,
) -> RetrievalMetrics: ...
```

Scores the recordings as they were served, or, given a `candidate`, its replay of them (selected as `replay` selects), on their labels from `sources`: a person's grade over the world's citation over a judge's, the latest of each. For the hits the verb returned, keyed by the verb (`"search"`), and each pillar's candidates, keyed by verb and pillar (`"search.vector"`), recall@k (the share of a recording's relevant documents among its first `k`), MRR (one over the rank of its first relevant document) and nDCG@k (graded, a gain of `2^grade - 1`), each averaged over the recordings with a relevant label: at document level always — a chunk counts as its document, a story as itself, and a candidate's document as a labeled one of the same name in the same corpus when the ids differ — and at passage level, by id, over the recordings with a relevant passage, when there is no candidate or it reads the refinery's own database pool. A judge grades only what was returned, so it never counts a miss: add `"judge"` to `sources` knowingly. Raises `InputError` for `k` below 1, and as `replay` does.

### `MemoryRefinery.export`

```python
async def export(
    self,
    corpus_id: str | list[str],
    bucket: ExportBucket,
    *,
    since: datetime | None = None,
    holdout_percent: int = 20,
) -> RetrievalExport: ...
```

Writes the recordings of the corpora `corpus_id` names as one JSON Lines file to `bucket`, under `memory/<corpora>/retrievals/<stamp>.jsonl` — the corpora sorted, each URL-quoted, joined by `+`; the stamp the wall clock to the microsecond — so repeated exports add files, and rfnry-engine's datasets, under their own keys, can share the bucket. `since` takes the recordings recorded or labeled after it; pass the last export's `watermark`. A row holds the recording's id, `holdout` (`is_holdout` at `holdout_percent`), corpora, verb, query, arguments, fingerprint, `at` and `recorded`, its hits as `[corpus_id, kind, id]`, `positives` (each passage or document graded 1 or more, with its grade, the source of the label that set it and its text), `negatives` (the hits graded 0, with theirs), `answered` (whether a positive, or its document, was among the hits) and every label. Texts are read from the corpus as it is now: a passage or document since erased is left out of its row and counted in `missing`. Writes nothing when nothing was selected, and returns `since` as the watermark. Raises `InputError` for a `holdout_percent` outside 0 to 100.

### `Retrieval`

```python
class Retrieval(BaseModel):
    retrieval_id: str
    corpus_ids: list[str]
    verb: Literal["search", "recall", "expand", "resembling"]
    at: datetime
    recorded: datetime
    duration_ms: int
    query: str | None
    arguments: dict[str, Any]
    window: dict[str, datetime | None]
    candidates: dict[str, list[RecordedHit]]
    hits: list[RecordedHit]
    pillars: dict[str, PillarReport]
    fingerprint: dict[str, Any]
```

A recording: the corpora it read, sorted, the verb, `at` and `recorded`, the call's `duration_ms`, `query` (the query of `search` and `recall`, the text of `resembling`, `None` otherwise), every argument as given in JSON, `window` with the `as_of` and `known_at` it read at, each pillar's ranked candidates (by expansion for `expand`, `resemblance` for `resembling`), the hits as returned, each pillar's report and `fingerprint`: the memory's embedding model, `chunk_chars`, `overlap_chars`, pillars, `text`, `fusion` and `spot`.

### `RecordedHit`

```python
class RecordedHit(BaseModel):
    corpus_id: str
    kind: Literal["chunk", "story", "source"]
    id: str
    score: float
    source_id: str | None = None
    source_name: str | None = None
    judge: Judgement | None = None
```

One ranked passage of a recording, by id: a `chunk` or `story`, or the `source` `resembling` found; its score; for a chunk, its source's id, and among the hits its source's name too; and its judgement when the call judged.

### `RetrievalReplay`

```python
class RetrievalReplay(BaseModel):
    recorded: Retrieval
    labels: list[RetrievalLabel]
    candidate: Retrieval | None = None
    error: str | None = None
```

One recording beside what the candidate returned for it, recorded the same way but never stored, or the `error` that kept the candidate from running it.

### `RetrievalScores`

```python
class RetrievalScores(BaseModel):
    recall: float
    mrr: float
    ndcg: float
    measured: int
```

recall@k, MRR and nDCG@k averaged over the `measured` recordings.

### `RetrievalMetrics`

```python
class RetrievalMetrics(BaseModel):
    k: int
    retrievals: int
    labeled: int
    errors: int
    documents: dict[str, RetrievalScores]
    passages: dict[str, RetrievalScores] | None
    latency_ms: dict[str, float]
```

What `measure` found: `retrievals` measured, `labeled` of them scored, `errors` a candidate could not run, the scores by document and by passage (`None` when a candidate reads another database pool) under the verb, `"search"`, and under the verb and each pillar, `"search.vector"`, and the `p50` and `p95` of their durations.

### `ExportBucket`

```python
class ExportBucket(Protocol):
    async def put(self, key: str, data: bytes) -> None: ...
```

Anything `export` can write a file to: a folder, an object store, a test's dict.

### `RetrievalExport`

```python
class RetrievalExport(BaseModel):
    keys: list[str]
    rows: int
    missing: int
    watermark: datetime | None
```

What `export` wrote: the keys, how many rows, how many passages or documents it left out because they were erased, and the latest moment it covered, recorded or labeled, to pass as the next `since`.

### `Relevance`

```python
class Relevance(BaseModel):
    corpus_id: str
    kind: Literal["chunk", "story", "source"]
    id: str
    grade: int = 1
    probability: float | None = None
```

A grade on a passage, a `chunk` or `story`, or on a document as a whole, its `source`: 0 not relevant, 1 relevant, 2 highly relevant, 3 answers it; `probability` is a judge's. Raises a validation error for a grade outside 0 to 3.

### `RetrievalReview`

```python
class RetrievalReview(BaseModel):
    relevance: list[Relevance] = []
    comment: str | None = None
```

A person's label: grades, a comment, or both; a review with neither is a validation error.

### `RetrievalOutcome`

```python
class RetrievalOutcome(BaseModel):
    status: Literal["succeeded", "failed"] | None = None
    cited: list[Relevance] = []
    detail: str | None = None
```

What happened: the passages the answer cited, whether it worked, and a detail; one with none of them is a validation error.

### `RetrievalLabel`

```python
class RetrievalLabel(BaseModel):
    label_id: str
    retrieval_id: str
    at: datetime
    source: Literal["human", "world", "judge"]
    review: RetrievalReview | None = None
    outcome: RetrievalOutcome | None = None
    judged: list[Relevance] | None = None
```

A stored label: a person's `review`, the world's `outcome`, or a judge's grades, with when it was stored, by the wall clock. The sources are rfnry-engine's own words, and a judge's grade never passes for a person's.

### `RetrievalLabeled`

```python
class RetrievalLabeled(BaseModel):
    judged: int = 0
    skipped: int = 0
    errors: dict[str, str] = {}
```

What `label` did: how many recordings it graded, how many it labeled empty because there was nothing to grade, and why each one it could not label was left.

## Events

### `Chunked`

```python
class Chunked(BaseModel):
    corpus_id: str
    name: str
    source: int
    sources: int
    chunks: int
```

Sent when a source is cut into chunks, before any model call: `source` counts from 1 among the `sources` the call prepares, and `chunks` is how many the source has. A source with no chunks sends this and nothing else.

### `Embedded`

```python
class Embedded(BaseModel):
    corpus_id: str
    name: str | None
    done: int
    total: int
    computed: int
    duration_ms: int
```

Sent when a source's chunks have their vectors: `done` of `total` passages, `computed` of them sent to the embedding model — the others reuse a stored vector of identical text — and `duration_ms` the time the embedding took, 0 when nothing was sent.

### `Extracted`

```python
class Extracted(BaseModel):
    corpus_id: str
    name: str | None
    chunk: int
    chunks: int
    entities: int
    relations: int
    usage: Usage
    duration_ms: int
```

Sent after one chunk's graph step: extraction with the generation model, when there is one, and identity with the decision model, when there is one. `chunk` counts from 1 of the source's `chunks`; `entities` is how many entities the chunk mentions and `relations` how many relations its step staged; `usage` and `duration_ms` are that step's model calls and time.

### `Retrieved`

```python
class Retrieved(BaseModel):
    corpus_ids: list[str]
    pillar: Literal["text", "vector", "graph"]
    report: PillarReport
```

Sent when a pillar has its candidates, across every corpus searched: `report` is the `PillarReport` `Hits.pillars` holds for it — `returned`, `duration_ms`, the query's embedding included for the vector pillar, and `error` when it failed and the search went on.

### `Ranked`

```python
class Ranked(BaseModel):
    corpus_ids: list[str]
    hits: list[Hit]
```

Sent once the pillars are fused: the hits the call returns, before any judging, so they can be shown at once.

### `Judged`

```python
class Judged(BaseModel):
    corpus_ids: list[str]
    hit: Hit
    done: int
    total: int
    usage: Usage
    duration_ms: int
```

Sent as each hit's judgement returns: `hit` with its `judge` filled in, `done` of `total` judged so far, and that decision-model call's `usage` and `duration_ms`. The judge asks about every hit at once, so the hits come in the order their decisions return; `done` climbs in the order they are sent.

### `Recorded`

```python
class Recorded(BaseModel):
    corpus_ids: list[str]
    retrieval_id: str | None
    error: str | None = None
```

Sent once a refinery's recording of a `search`, `recall`, `expand` or `resembling` is written, with its `retrieval_id`, or when the write failed, with `retrieval_id` `None` and the failure in `error`: a recording that fails never fails the call.

### `Event`

```python
Event = Chunked | Embedded | Extracted | Retrieved | Ranked | Judged | Recorded
```

What `on_event` receives. Every event is frozen and takes no other field; the hits in `Ranked` and `Judged` are copies, so a handler that changes them changes nothing the call returns.

### `EventHandler`

```python
EventHandler = Callable[[Event], Awaitable[None]]
```

The `on_event` a verb takes: awaited where each event happens, so the call waits for it, with the events of one call one at a time, in the order they happen. A handler that raises fails the call with its own exception, and the call's work still running beside it — the other pillars, the other judgements — is cancelled.

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
        on_event: EventHandler | None = None,
        written_by: dict[str, Any] | None = None,
    ) -> None: ...
```

A `Tools` that offers memory's verbs to a model, pinned to the corpus or corpora `corpus_id` names and to the bound `filters`: `search`, `recall`, `expand`, `read_source`, `list_sources`, `ingest`, `remember`, `supersede`, `end_story`, `delete_story`, `list_stories`, `resembling`, `retire_source`, and `neighbors` when the graph pillar is on and no filter is bound, and `describe` when no filter is bound. `list_sources` takes `as_of`, `history` and `known_at`, and a chunk hit carries its document's `created_at` and `confirmed_at`, and its `valid_from`, `valid_to` and `retired_at` when set. `ingest` takes `valid_from`, `valid_to` and `replaces`, which under bound filters must name a source inside them, and answers with `retired` and `deleted` when it replaced one. `search` and `recall` take `sources`, and `entities`, `relation_types`, `around`, `hops`, `depth` and `spot` — its description naming the memory's default — when the graph pillar is on and no filter is bound, and answer with `unresolved` when a name was not found. `search` and `recall` take `pillars`, offering only the configured ones and not `graph` under a bound filter. `expand` takes a hit's `kind` and `id` and offers `by` the modes the pillars support, `graph` only when the graph pillar is on and no filter is bound; under bound filters its passage must be inside them. A hit carries `entities`, the name and type of each entity its passage mentions, when the graph pillar is on and no filter is bound. Bound to several corpora, it offers `search` and `recall`, which read them all or the ones the model names in `corpus_id`, and `expand`, `read_source`, `list_sources` and `list_stories`, which take the one `corpus_id` a hit gives, `resembling`, which reads them all or the ones the model names, with `source_corpus` for a `source_id`, and `describe`, which takes the one to describe; every hit then says its `corpus_id`; `include` keeps a subset, and every result over `max_result_chars` is cut and marked `truncated`. `written_by` is passed to every write and closing a tool makes, and the model never sets it; a story row carries its `written_by` and `closed_by`, a chunk row and a related row their `written_by`; a bound filter naming `written_by.` or a date key is a `ConfigurationError`. `on_event` is passed to every `search`, `recall`, `expand`, `resembling`, `ingest`, `remember` and `supersede` a tool calls, so the host follows them; the model never sees an event. With a refinery, the results of `search`, `recall`, `expand` and `resembling` carry the `retrieval_id` of their recording. `judge` names the judgements the host's decision model answers, all four unless given: the tools' `judge` asks those, `order_by` chooses among them, and `order_by` alone asks only that one. The tools keep limits of their own: `limit` defaults to 50 (at most 500), `top_k` is at most 100, the `hops` of `neighbors` are 1 to 3, the `width` of `expand` is 1 to 10, `related` is 0 to 20, and the `limit` of `resembling` is 1 to 50. `resembling` answers with `matches`, each a document's id, name, `share` rounded to two places, metadata, `written_by` and dates, without its text; `retire_source` takes `source_id`, `replaced_by` and `valid_to` and answers with the retired source less its text. A bound filter takes one value, several or a range; the model's filters, values or ranges, may narrow it and never widen it — a range keeps the bound values inside it, or the tighter of each side of a bound range, and refuses when it keeps nothing —, and what the tools write is stamped with it, naming one value of a filter bound to several, or must give a value inside a bound range. A call raises what its verb raises, `NotFoundError` for a story, a source or an `expand` passage outside the bound filters, and `InputError` for a filter outside the bound values, a range that keeps nothing of a bound filter or is of another kind than a bound range, written metadata that does not name one of them or falls outside a bound range, an `order_by` outside the judgements, a graph argument, a `pillars` or an `expand` mode the tools do not offer, or a `corpus_id` outside the bound corpora. Raises `ConfigurationError` without a `corpus_id`, with `max_result_chars` below 1000, or with a `judge` naming no judgement, an unknown one or one twice; `InputError` for a reserved or malformed filter.

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

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

Memory's storage: Postgres with pgvector, over an asyncpg pool you built, in tables prefixed `memory_`. Its methods are what `Memory` calls, not a surface of their own, and change between versions without notice. Keyword search scores with the `TextRanking` it is given: `ranking/bm25.py` and `ranking/cover_density.py` each hold the SQL of theirs. It never creates or alters the schema: `rfnry-memory` does, below. `check` reads the recorded version and settings — what `Memory.check` reports; raises `StorageError` when the database rejects an operation, or vectors are written before `check`. A corpus is pinned to the recorded default language by its first write. `set_vector_index` keeps an indexed corpus's vectors in `memory_vectors_` and the first 24 hex characters of the sha256 of its id, with one index per kind of the method it was given — `index/hnsw.py` and `index/ivfflat.py` each say how theirs is built, read back and searched — which each write to the corpus updates in its own transaction and `delete_corpus` drops, after any build of the corpus. `search_vector` without a `scope` reads that table for each kind whose index is built and valid, in a read-only transaction of its own that sets the method's iterative scan, `hnsw.iterative_scan` or `ivfflat.iterative_scan`, to `relaxed_order` and nothing else, scores what the index finds from `memory_chunks` and `memory_stories`, and runs the exact search instead when the index returns fewer than `top_k` or the table was dropped meanwhile; a kind whose index is not built and valid is searched exactly, as before. `set_vector_index` raises `InputError` for `"full"` above 2000 dimensions or a value pgvector refuses, and `ConfigurationError` above 4000 dimensions, on pgvector older than 0.8.0, or before `check` gave it a dimension.

## The `rfnry-memory` command

```sh
rfnry-memory create  [--url URL] [--schema memory] (--dimension N | --no-vector) [--no-graph] [--language NAME] [--sql]
rfnry-memory check   [--url URL] [--schema memory]
rfnry-memory upgrade [--url URL] [--schema memory] [--sql] [--yes]
rfnry-memory set     [--url URL] [--schema memory] (--dimension N | --graph | --language NAME) [--sql] [--yes]
rfnry-memory index   [--url URL] [--schema memory] --key KEY (--type number|text | --drop) [--sql] [--yes]
rfnry-memory index   [--url URL] [--schema memory] --current [--drop] [--sql] [--yes]
```

Installed with the package; it needs `rfnry-memory[postgres]`. Without `--url` it connects with Postgres' `PG*` variables, and it never prints the connection string. Every command that changes a schema — `create`, `upgrade`, `set` and `index` — refuses a server older than Postgres 17. `--schema` defaults to `memory` and refuses `public`, `information_schema` and names starting with `pg_`. `create` makes the schema — refusing one that already holds memory tables — installs `vector` with a dimension and `pg_trgm` unless `--no-graph` into `public` (a database that already has one keeps it where it is), runs every migration, and records the dimension and the default language of new corpora: `--language`, or the database's `default_text_search_config` as it is then — printing the settings it recorded. `upgrade` lists the pending migrations, reminds you to take a snapshot and stop the old version's app instances, asks unless `--yes`, and runs them in one transaction under a lock, recording each in `memory_migrations`. `set` changes one setting the same way: `--dimension` deletes every stored vector and per-corpus vector index and retypes the vector columns — `reembed`, then `set_vector_index`, builds each corpus back, `--graph` installs `pg_trgm`, `--language` changes the default of corpora not yet written. `--sql` prints every statement, starting with the `SET search_path` they run under, and runs nothing. `index` indexes a filter key, `metadata.<key>` or `written_by.<key>` — its parts object keys, none of them a number — for range filters on chunks, sources and stories, one btree per table over every corpus, `corpus_id` first, built with `CREATE INDEX CONCURRENTLY` outside a transaction under the migration lock, records it in `memory_indexed_keys` once the three are valid, and analyzes the three tables. Adding the rule takes each table's lock for a moment, behind any long-running transaction. The key then holds one type: a number, or a string of at most 2000 bytes; a write of a current row — a chunk or source not retired, a story not closed — holding another type, an array or a longer string there is refused with an `InputError`, from the command's first step, and a retired or closed row keeps what it held. It first counts the rows that break the rule among those a read of the present can see — current ones, which a fix lets through, and retired or closed ones still in force or yet to be (`valid_to` unset, or later than now with a non-empty interval), which keep their values until they end — and prints each table's counts with one corpus and id and builds nothing while any does. It records when the key was built, and a read at a moment before then keeps testing each row. Run again after an interruption, it finishes; with the key built it does nothing; asked for the other type, it refuses until `--drop`, which drops the indexes and lifts the rule. `index --current` builds four indexes the same way, over every corpus of the schema, and analyzes both tables: a GIN over `fts` of chunks where `retired_at IS NULL` and of stories where `closed_at IS NULL`, and a btree over `corpus_id` and each retired chunk's or closed story's end (`valid_to`, or infinity when unset; NULL for current rows); `--drop` drops the four, and what exists is the setting. Once `check()` finds the four built and valid, a keyword read of the present (no `as_of`, `known_at` or `history`) reads its rows as the current ones plus the retired or closed ones whose `valid_to` is unset or later than its moment, the same rows it would read otherwise. `check` prints what `Memory.check` would report about the version and the vector extension, then each indexed key with its type, and a key left not ready by an interrupted build with the command that finishes it, then whether the current rows are indexed, or not ready with the command that finishes them. Every command exits 0 on success and 1 otherwise, printing why on stderr; it never drops an extension.

### `Kind`

```python
Kind = Literal["chunk", "story"]
```

The two kinds of passage.

### `Match`

```python
Match = Literal["any", "all", "phrase", "web"]
```

How a query's words must appear in what keyword search returns; the `match` that `search` and `recall` take. A code — letters and digits joined by `-`, `_`, `:` or `#`, with a digit or one of the last three — is kept whole as one more word, lowercased and unstemmed: needed whole under `"all"` and `"phrase"`, ranked as a word under `"any"` and `"web"`.

### `Spot`

```python
Spot = Literal["words", "exact", "substring", "off"]
```

How the graph pillar finds the entities a query names; the `spot` that `search` and `recall` take.

### `JudgementName`

```python
JudgementName = Literal["relevant", "evidence", "contradicts", "injection"]
```

The four judgements a decision model can give a hit; the names a `judge` tuple and `Hits.by` take.

### `Language`

```python
Language = str
```

The name of a text search configuration the database has, which a corpus's text pillar reads in: one of Postgres' built-in languages, read as Postgres ships it, or one you created, schema-qualified if you like, read as you made it. `simple` neither stems nor drops stop words.

### `Ref`

```python
Ref = tuple[str, str]
```

A passage by kind and id.

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

A setup that cannot work: `BM25(statistics="corpus")` over a corpus that keeps no statistics, a refinery `sample` outside 0 to 1, a refinery's `label` given a memory without a decision model, an unknown or unconfigured pillar, the vector pillar without an embedding model, extraction without `relation_types`, a `text` or `fusion` that is not one of the settings, bad chunk sizes, a storage without its extension or created for another dimension, a `judge` that asks for judgements without a decision model, graph facts, `neighbors`, `entities`, `relation_types` or `around` with the graph pillar off, `expand` by vector or by graph without that pillar, `reembed` without the vector pillar, a `MemoryTools` without a corpus, with `max_result_chars` below 1000 or with a `judge` naming no judgement, an unknown one or one twice, two tools of the same name, or a `Tool` called without a body.

### `InputError`

```python
class InputError(MemoryException): ...
```

A bad argument: through a refinery, a grade naming a corpus its recording did not read, a blank `label_id` or one its recording holds from another source, `limit` or `k` below 1, a `moment` other than `"recorded"` or `"now"`; a write of a current row holding what an indexed key's rule refuses (`rfnry-memory index`); `sources` outside `human`, `world` and `judge`, or a `holdout_percent` outside 0 to 100; an unknown kind or judgement, hits ordered by a judgement they were not given, `top_k` or `limit` below 1, `hops` outside 0 to 2, `depth` outside 0 to 3, a `match` other than `"any"`, `"all"`, `"phrase"` or `"web"`, a `spot` other than `"words"`, `"exact"`, `"substring"` or `"off"`, `width` below 1, an unknown `expand` mode, a passage that is not a `(kind, id)` pair, `expand` by document from a story without a subject, a cursor no listing returned, a reserved or malformed filter, a date filter that is not a range of datetimes or ISO 8601 strings, a `confirmed_at` filter with `known_at`, a `sync` whose filters name anything but metadata, text with a NUL character or a number that is NaN or infinite, a story with blank text, a blank query to `search` or `recall`, a `valid_to` not after `valid_from`, `related` below 0, `history=True` with `as_of`, a name of several types without its type, a refused `supersede` or `end_story`, a source with both or neither of `text` and `sections`, a media part without exactly one of `data` and `url`, a `sync` without filters or with a source they do not match, a language that is not a text search configuration in the database, a `corpus_id` that names no corpus, a tool name `dispatch` or `only` does not know, tool arguments that fail validation, or, through `MemoryTools`, a filter or `corpus_id` outside what the tools are bound to, a graph argument, a `pillars` or an `expand` mode they do not offer and written metadata that names none of a filter's bound values.

### `ModelError`

```python
class ModelError(MemoryException): ...
```

A model broke its contract: a reply, an answer or a vector that does not fit.

### `StorageError`

```python
class StorageError(MemoryException): ...
```

The storage failed or refused a write: a database error, a row of another corpus, graph facts naming an unknown entity, or vectors written to Postgres before `check`.

### `NotFoundError`

```python
class NotFoundError(MemoryException): ...
```

A verb named a story, a source, a passage or an entity the corpus does not hold, or, through `MemoryTools`, one outside the bound filters; or a refinery was asked for a recording it does not hold, or one that read a corpus the call does not name.
