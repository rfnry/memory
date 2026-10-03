# rfnry-memory API

The reference: every public name, its signature, what it returns and what it raises. How and why to use them is in [the guide](memory.md).

## Models

### `EmbeddingModel`

```python
@runtime_checkable
class EmbeddingModel(Protocol):
    model: str
    dimension: int

    async def embed(self, texts: list[str]) -> list[list[float]]: ...
```

The model that indexes the vector pillar: any object with these members. `embed` returns one vector of `dimension` floats per text, in order, and `model` is recorded on every vector it produces. A verb that gets another number of vectors, or a vector of another length, raises `ModelError`.

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

Prepares the storage for this memory's embedding dimension, or for no vectors without `embedding`; safe to run again. Raises `ConfigurationError` when the storage was created for another dimension or without vectors, or when Postgres cannot create its extensions.

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

Writes one source: `text` chunked, or each of `sections` chunked on its own with its locator; every chunk is indexed by every pillar that is on, read for entities and relations when there is a generation model, and given the graph facts you supply that it names. Returns an `Ingested`: the source and the usage of every model call the write made. Raises `InputError` when given both or neither of `text` and `sections`, or for a relation end that names no known entity or one of several types without `from_type`/`to_type`; `ConfigurationError` when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

### `Memory.sync`

```python
async def sync(
    self, corpus_id: str, *, filters: dict[str, Any], sources: list[SourceIn]
) -> Synced: ...
```

Makes the sources `filters` match exactly `sources` in one storage transaction, so a reader sees the old set or the new one, never both and never neither: a stored source whose `digest` a new one repeats is kept as it is, every other matching source is deleted, the rest of `sources` is written, and an empty `sources` removes the set; every model call runs before the transaction opens, and a chunk whose text the corpus holds with a vector from the same model reuses it. Returns a `Synced`: the sources in list order, the ids kept, the `Counts` deleted and the usage. Raises `InputError` when `filters` are empty or malformed or a new source does not match them, and otherwise what `ingest` raises.

### `Memory.remember`

```python
async def remember(
    self,
    corpus_id: str,
    text: str,
    *,
    subject: str | None = None,
    valid_from: datetime | None = None,
    source_ids: tuple[str, ...] = (),
    metadata: dict[str, Any] | None = None,
    entities: list[EntityIn] | None = None,
    relations: list[RelationIn] | None = None,
) -> Remembered: ...
```

Writes one story, true from `valid_from` (now by default) until it is superseded; when an open story already says the same text, ignoring case and extra whitespace, returns that one and writes nothing. Returns a `Remembered`: the story and the usage of its extraction and identity. Raises `InputError` for blank text, or for a relation end as `ingest` does; `ConfigurationError` when graph facts are given with the graph pillar off; `ModelError` when a model breaks its contract.

### `Memory.supersede`

```python
async def supersede(
    self,
    corpus_id: str,
    story_id: str,
    text: str,
    *,
    valid_from: datetime | None = None,
    subject: str | None = None,
    metadata: dict[str, Any] | None = None,
    entities: list[EntityIn] | None = None,
    relations: list[RelationIn] | None = None,
) -> Remembered: ...
```

Closes the story at `valid_from` (now by default) and opens a new one there that says `text` and points back through `supersedes`; the new story keeps the old one's `source_ids`, and its `subject` and `metadata` unless given. Returns a `Remembered`: the new story and the usage. Raises `NotFoundError` when the story does not exist; `InputError` when it is already closed, `valid_from` is not after it began, another open story already says `text`, or the text is blank; otherwise what `remember` raises.

### `Memory.delete_story`

```python
async def delete_story(self, corpus_id: str, story_id: str) -> Counts: ...
```

Deletes a story outright, with the mentions and relations it evidenced; entities stay. Returns the `Counts` deleted. Raises `NotFoundError` when the story does not exist.

### `Memory.delete_source`

```python
async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...
```

Deletes a source with its chunks, their vectors and full-text rows, and the mentions and relations its chunks evidenced; entities stay. Returns the `Counts` deleted. Raises `NotFoundError` when the source does not exist.

### `Memory.get_source`

```python
async def get_source(self, corpus_id: str, source_id: str) -> Source | None: ...
```

Returns the source, or `None` when the corpus holds no such source.

### `Memory.list_sources`

```python
async def list_sources(
    self,
    corpus_id: str,
    *,
    filters: dict[str, Any] | None = None,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Source], str | None]: ...
```

Lists the sources `filters` match, newest first, `limit` at a time. Returns the page and the cursor of the next one, `None` on the last. Raises `InputError` for a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.read_source`

```python
async def read_source(
    self, corpus_id: str, source_id: str, *, locator: dict[str, Any] | None = None
) -> str: ...
```

Returns a source's whole text or, with `locator`, the text of the chunks whose locator holds every key and value given, joined by blank lines and empty when none does. Raises `NotFoundError` when the source does not exist.

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
    since: datetime | None = None,
    until: datetime | None = None,
    filters: dict[str, Any] | None = None,
    history: bool = False,
    limit: int = 100,
    cursor: str | None = None,
) -> tuple[list[Story], str | None]: ...
```

Lists stories, latest `valid_from` first, `limit` at a time: those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from`, and `subject` and `filters` narrow. Returns the page and the cursor of the next one, `None` on the last. Raises `InputError` for `history=True` with `as_of`, a reserved or malformed filter, `limit` below 1, or a cursor no listing returned.

### `Memory.reembed`

```python
async def reembed(self, corpus_id: str) -> int: ...
```

Embeds again every chunk and story whose vector was not made by the current embedding model, 256 at a time — those `fingerprint` counts as `stale`. Returns how many it embedded. Raises `ConfigurationError` without the vector pillar; `ModelError` when the embedding model breaks its contract.

### `Memory.delete_corpus`

```python
async def delete_corpus(self, corpus_id: str) -> Counts: ...
```

Deletes every row of the corpus, in every table. Returns the `Counts` deleted.

### `Memory.verify_absent`

```python
async def verify_absent(self, corpus_id: str) -> Counts: ...
```

Counts the corpus's rows, every table independently. Returns the `Counts`; `clean` is `True` when all are zero.

### `Memory.fingerprint`

```python
async def fingerprint(self, corpus_id: str) -> CorpusFingerprint: ...
```

Reads the corpus and says what it holds: its sources, chunks and stories counted, a hash over every source's name, metadata and text, another over every story's text, subject, metadata and interval, the models its vectors came from, how many passages the vector pillar would miss, and how this memory embeds and chunks. Ids and write times are not read, so identical content under new ids gives the same fingerprint. No search computes it. Returns a `CorpusFingerprint`.

### `Memory.search`

```python
async def search(
    self,
    corpus_id: str,
    query: str,
    *,
    kinds: tuple[Kind, ...] = ("chunk", "story"),
    pillars: set[str] | None = None,
    top_k: int = 10,
    filters: dict[str, Any] | None = None,
    history: bool = False,
    judge: bool = False,
) -> Hits: ...
```

Searches chunks and stories with `pillars` (every configured pillar by default), fuses their lists by reciprocal rank and returns the first `top_k` as `Hits`; stories are those true now unless `history=True`, the graph pillar stands down under `filters`, and a blank query returns no hits. With `judge=True` a decision model judges every hit. Raises `ConfigurationError` for a pillar not configured or `judge=True` without a decision model; `InputError` for an unknown kind, `top_k` below 1 or a reserved or malformed filter; `ModelError` when the judge breaks its contract.

### `Memory.recall`

```python
async def recall(
    self,
    corpus_id: str,
    query: str,
    *,
    top_k: int = 10,
    as_of: datetime | None = None,
    history: bool = False,
    since: datetime | None = None,
    until: datetime | None = None,
    subject: str | None = None,
    filters: dict[str, Any] | None = None,
    pillars: set[str] | None = None,
    judge: bool = False,
) -> Hits: ...
```

Searches stories only, as `search` does, among those true now, those true at `as_of`, or with `history=True` all of them; `since` and `until` bound `valid_from` and `subject` narrows. Returns `Hits`. Raises what `search` raises, and `InputError` for `history=True` with `as_of`.

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
    history: bool = False,
) -> Neighborhood: ...
```

Walks the graph up to `hops` from the entity with this name or alias, along `relation_types` only when given, read upper-case. Returns a `Neighborhood`: the entities reached, the relations walked and the passages that mention them, current stories unless `history=True`. Raises `ConfigurationError` with the graph pillar off; `NotFoundError` when no entity has the name, of `entity_type` when given; `InputError` when entities of several types have it and `entity_type` is not given.

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
```

A stored source: its whole text, its `metadata` facets, how many chunks it was split into, and `digest`, a hash of the source as it was given with the chunk sizes, pillars and extraction it was written under, by which `sync` keeps a source the new list repeats; `None` when none was recorded.

### `Section`

```python
class Section(BaseModel):
    title: str | None = None
    text: str
    locator: dict[str, Any] = {}
```

One part of a source as your parser knows it: its text, an optional title, and a `locator` its chunks carry.

### `Chunk`

```python
class Chunk(BaseModel):
    corpus_id: str
    chunk_id: str
    source_id: str
    position: int
    title: str | None
    locator: dict[str, Any]
    text: str
```

A piece of a source as indexed: its position in the source, the section's title or else the headings it falls under, and the section's locator.

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
    supersedes: str | None
    source_ids: tuple[str, ...]
    metadata: dict[str, Any]
```

A story, true from `valid_from` until `valid_to` (`None` while it is open), written at `created_at`, pointing through `supersedes` at the story it replaced. Raises pydantic's `ValidationError` when `valid_to` is not after `valid_from`.

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
    kind: Kind
    id: str
    text: str
    score: float
    provenance: dict[str, float]
    chunk: Chunk | None = None
    story: Story | None = None
    judge: Judgement | None = None
```

One search result: a chunk or a story, its fused `score`, and in `provenance` each pillar that returned it with its score divided by that pillar's best. The passage itself is in `chunk` or `story`, and `judge` is set when the search was judged.

### `Hits`

```python
class Hits(BaseModel):
    hits: list[Hit]
    pillars: dict[str, PillarReport]
    usage: Usage

    def by(self, check: JudgementName, *, descending: bool = True) -> Hits: ...
```

A search's hits in fused order, a report for each pillar that ran, and the usage of the judge. `by` returns a copy ordered by one judgement (`JudgementName` is `Literal["relevant", "evidence", "contradicts", "injection"]`), highest first unless `descending=False`, ties in fused order. `by` raises `InputError` for an unknown judgement or hits that were not judged.

### `Judgement`

```python
class Judgement(BaseModel):
    relevant: float
    evidence: float
    contradicts: float
    injection: float
```

A decision model's four probabilities for one hit against the query: it is about what the query asks, it contains what answers the query, it contradicts a premise the query states, it holds instructions addressed to an AI.

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

What `neighbors` returns: the entities reached, the one named first, the relations walked, and the passages that mention the entities.

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
    usage: Usage
```

What `sync` returns: the list's sources as stored, in its order, kept or written; the ids of those kept as they were; the rows deleted; and what the model calls cost.

### `Remembered`

```python
class Remembered(BaseModel):
    story: Story
    usage: Usage
```

What `remember` and `supersede` return: the story written, or the open story that already said it, and what the model calls cost.

### `Counts`

```python
class Counts(BaseModel):
    sources: int = 0
    chunks: int = 0
    stories: int = 0
    entities: int = 0
    relations: int = 0
    mentions: int = 0

    @property
    def clean(self) -> bool: ...
```

Rows by table, as deletes and `verify_absent` return them. `clean` is `True` when every count is zero.

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
```

What `fingerprint` returns. `content` is the first 16 hex characters of a sha256 over each source's name, metadata and text, sorted; `story_content` the same over each story's text, subject, metadata, `valid_from` and `valid_to`, in UTC. `embedding` is the model of the vector pillar, `None` without it. `embedded` counts the chunks and stories, together, that carry a vector, by the model that made it, sorted by model; `stale` counts those whose vector is from another model than `embedding` or missing — what `reembed` embeds again — and is `0` without the vector pillar.

### `Chunking`

```python
class Chunking(BaseModel):
    chunk_chars: int
    overlap_chars: int
    pillars: list[str]
```

How a memory cuts and indexes a corpus, as its fingerprint records it: `chunk_chars`, `overlap_chars` and the pillars it was given, sorted.

## Tools

### `MemoryTools`

```python
class MemoryTools(Tools):
    def __init__(
        self,
        memory: Memory,
        *,
        corpus_id: str,
        filters: dict[str, Any] | None = None,
        include: set[str] | None = None,
        max_result_chars: int = 200_000,
    ) -> None: ...
```

A `Tools` that offers memory's verbs to a model, pinned to `corpus_id` and to the bound `filters`: `search`, `recall`, `read_source`, `list_sources`, `ingest`, `remember`, `supersede`, `delete_story`, `list_stories`, and `neighbors` when the graph pillar is on and no filter is bound; `include` keeps a subset, and every result over `max_result_chars` is cut and marked `truncated`. The tools keep limits of their own: `limit` defaults to 50 (at most 500), `top_k` is at most 100, and `hops` is 1 to 3. A bound filter takes one value or several; the model's filters may narrow it and never widen it, and what the tools write is stamped with it, naming one value of a filter bound to several. A call raises what its verb raises, `NotFoundError` for a story or source outside the bound filters, and `InputError` for text an open story outside them already says, a filter outside the bound values, or written metadata that does not name one of them. Raises `ConfigurationError` without a `corpus_id` or with `max_result_chars` below 1000; `InputError` for a reserved or malformed filter.

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
        self, corpus_id: str, *, filters: dict[str, Any] | None, limit: int, cursor: str | None
    ) -> tuple[list[Source], str | None]: ...

    async def source_chunks(self, corpus_id: str, source_id: str) -> list[Chunk]: ...

    async def source_digests(self, corpus_id: str) -> list[tuple[str, dict[str, Any], str]]: ...

    async def delete_source(self, corpus_id: str, source_id: str) -> Counts: ...

    async def sync_sources(
        self,
        corpus_id: str,
        filters: dict[str, Any],
        writes: list[SourceWrite],
        model: str | None,
        *,
        graph: GraphWrite | None = None,
        keep: dict[str, str] | None = None,
    ) -> Counts | None: ...

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

    async def find_open_story(self, corpus_id: str, hash: str) -> Story | None: ...

    async def supersede_story(
        self,
        corpus_id: str,
        old_id: str,
        valid_to: datetime,
        new: Story,
        vector: list[float] | None,
        model: str | None,
        *,
        graph: GraphWrite | None = None,
    ) -> tuple[Story, Story]: ...

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
        corpus_id: str,
        query: str,
        *,
        kinds: tuple[Kind, ...],
        top_k: int,
        filters: dict[str, Any] | None,
        window: Window,
        subject: str | None,
    ) -> list[Leg]: ...

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
    ) -> list[Leg]: ...

    async def get_passages(self, corpus_id: str, refs: list[Ref]) -> list[Chunk | Story]: ...

    async def similar_entities(self, corpus_id: str, name: str, *, limit: int) -> list[Entity]: ...

    async def entities_named(self, corpus_id: str, name_or_key: str) -> list[Entity]: ...

    async def entities_by_id(self, corpus_id: str, entity_ids: list[str]) -> list[Entity]: ...

    async def entities_named_in(self, corpus_id: str, text: str) -> list[Entity]: ...

    async def neighbors(
        self, corpus_id: str, entity_ids: list[str], *, relation_types: list[str] | None
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
    ) -> list[Ref]: ...

    async def passages_embedded_with(
        self, corpus_id: str, *, not_model: str, limit: int
    ) -> list[Chunk | Story]: ...

    async def set_vectors(
        self, corpus_id: str, refs: list[Ref], vectors: list[list[float]], model: str
    ) -> None: ...

    async def delete_corpus(self, corpus_id: str) -> Counts: ...

    async def count_corpus(self, corpus_id: str) -> Counts: ...

    async def count_vectors(self, corpus_id: str) -> dict[str | None, int]: ...
```

The one storage protocol, as `Memory` calls it; `InMemoryStorage` and `PostgresStorage` implement exactly these methods, and a storage of your own implements them from exported names alone. Every method but `initialize` takes the corpus first and touches no other; a write carries its `GraphWrite` in the same transaction, the listings page by cursor, the searches return `Leg`s best first, `source_digests` returns each source's name, metadata and the sha256 of its text in hex, in no order, `sync_sources` leaves the sources `keep` names (each id with its `digest`) as they are and returns `None`, writing nothing, when one of them is no longer in the matching set with that digest, `chunk_vectors` returns for each digest — the sha256 of a chunk's text in hex — a vector a chunk with that text holds from `model`, leaving out the digests it has none for, and `count_vectors` counts the chunks and stories together by the model that made their vector, under `None` those without one. Both storages raise `StorageError` when a row names another corpus or graph facts name an unknown entity, `NotFoundError` and `InputError` from `supersede_story` as `Memory.supersede` does, and `ConfigurationError` from `initialize` for a dimension the storage was not created for.

### `InMemoryStorage`

```python
class InMemoryStorage:
    def __init__(self) -> None: ...
```

A `Storage` in process memory, for tests and small things; nothing outlives the object. Its text pillar reads words as Postgres' `english` configuration does, except that emails, URLs, file paths, versions, hyphenated compounds and markup are read as plain words.

### `PostgresStorage`

```python
class PostgresStorage:
    def __init__(self, pool: Any) -> None: ...
```

A `Storage` in Postgres, over an asyncpg pool you built, in tables prefixed `memory_`; `initialize` creates the `vector` and `pg_trgm` extensions, the tables and their full-text indexes, and is safe to run again. Raises `ConfigurationError` from `initialize` when an extension cannot be created or the tables were created for another dimension or without vectors; `StorageError` when the database rejects an operation, or vectors are written before `initialize`.

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
    mentions: list[tuple[str, Kind, str]] = []

    def check(self, corpus_id: str, passages: set[Ref]) -> None: ...
    def named(self, ids: dict[str, str]) -> set[str]: ...
```

The graph facts of one write: new entities, `(entity_id, alias)` pairs, relations, and `(entity_id, kind, passage_id)` mentions. `check` raises `StorageError` when a row names another corpus or a fact is evidenced by a passage outside `passages`; `named` returns the ids of the entities the facts name, each mapped through `ids`.

### `Window`

```python
class Window(BaseModel):
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
```

The stories a read covers: those true at `as_of`, or at every moment when it is `None`, with `valid_from` from `since` through `until`.

### `Kind`

```python
Kind = Literal["chunk", "story"]
```

The two kinds of passage.

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

A setup that cannot work: an unknown or unconfigured pillar, the vector pillar without an embedding model, extraction without `relation_types`, bad `weights` or chunk sizes, a storage without its extension or created for another dimension, `judge=True` without a decision model, graph facts or `neighbors` with the graph pillar off, `reembed` without the vector pillar, a `MemoryTools` without a corpus or with `max_result_chars` below 1000, two tools of the same name, or a `Tool` called without a body.

### `InputError`

```python
class InputError(MemoryException): ...
```

A bad argument: an unknown kind or judgement, hits ordered by a judgement they were not given, `top_k` or `limit` below 1, a cursor no listing returned, a reserved or malformed filter, text with a NUL character or a number that is NaN or infinite, a story with blank text, `history=True` with `as_of`, a name of several types without its type, a refused `supersede`, a source with both or neither of `text` and `sections`, a media part without exactly one of `data` and `url`, a `sync` without filters or with a source they do not match, a tool name `dispatch` or `only` does not know, or tool arguments that fail validation.

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

A verb named a story, a source or an entity the corpus does not hold, or, through `MemoryTools`, one outside the bound filters.
