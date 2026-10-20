import json
from collections.abc import Awaitable, Callable, Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Literal, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from rfnry_memory.errors import (
    ConfigurationError,
    InputError,
    NotFoundError,
    StorageError,
)
from rfnry_memory.events import EventHandler
from rfnry_memory.filters import matches_filters, stamp, validate_filters, within
from rfnry_memory.judge import JUDGEMENTS
from rfnry_memory.memory import Memory, storable
from rfnry_memory.models import Hit, Hits, JudgementName, Kind, Match, Part, Ref, Spot, Story

Result = str | dict[str, Any] | list[Part]
Body = Callable[[dict[str, Any]], Awaitable[Result]]
Dispatcher = Callable[[dict[str, Any]], Awaitable[Result]]


@runtime_checkable
class ToolsLike(Protocol):
    def definitions(self) -> list[dict[str, Any]]: ...

    async def dispatch(
        self, name: str, args: dict[str, Any]
    ) -> str | dict[str, Any] | list[Any]: ...


@dataclass(frozen=True)
class Tool:
    name: str
    description: str = ""
    body: Body | None = None
    args: type[BaseModel] | None = None
    input_schema: dict[str, Any] | None = None

    def schema(self) -> dict[str, Any]:
        if self.input_schema is not None:
            return self.input_schema
        if self.args is not None:
            return self.args.model_json_schema()
        return {"type": "object", "properties": {}}

    def definition(self) -> dict[str, Any]:
        return {"name": self.name, "description": self.description, "input_schema": self.schema()}

    async def call(self, args: dict[str, Any]) -> Result:
        if self.body is None:
            raise ConfigurationError(f"tools: tool {self.name!r} has no body")
        if self.args is not None:
            try:
                args = self.args.model_validate(args).model_dump(exclude_unset=True)
            except ValidationError as exc:
                raise InputError(
                    f"tools: invalid arguments for {self.name}: {exc.errors(include_url=False)}"
                ) from exc
        return await self.body(args)


class Tools:
    def __init__(self, tools: Iterable[Tool | ToolsLike] = ()) -> None:
        self._entries: dict[str, tuple[dict[str, Any], Dispatcher]] = {}
        for item in tools:
            if isinstance(item, Tool):
                self._put(item.definition(), item.call)
            else:
                for definition in item.definitions():
                    self._put(dict(definition), partial(item.dispatch, definition["name"]))

    def _put(self, definition: dict[str, Any], dispatcher: Dispatcher) -> None:
        name = definition["name"]
        if name in self._entries:
            raise ConfigurationError(f"tools: duplicate tool name {name!r}")
        self._entries[name] = (definition, dispatcher)

    @classmethod
    def _from(cls, entries: dict[str, tuple[dict[str, Any], Dispatcher]]) -> Tools:
        tools = cls()
        tools._entries = dict(entries)
        return tools

    @property
    def names(self) -> list[str]:
        return list(self._entries)

    def __contains__(self, name: str) -> bool:
        return name in self._entries

    def __len__(self) -> int:
        return len(self._entries)

    def definitions(self) -> list[dict[str, Any]]:
        return [dict(definition) for definition, _ in self._entries.values()]

    async def dispatch(self, name: str, args: dict[str, Any]) -> Result:
        entry = self._entries.get(name)
        if entry is None:
            raise InputError(f"tools: unknown tool {name!r}")
        return await entry[1](args)

    def __add__(self, other: Tools | ToolsLike) -> Tools:
        return Tools([self, other])

    def only(self, *names: str) -> Tools:
        missing = [n for n in names if n not in self._entries]
        if missing:
            raise InputError(f"tools: unknown tools {missing}")
        return Tools._from({n: self._entries[n] for n in names})

    def without(self, *names: str) -> Tools:
        return Tools._from({n: e for n, e in self._entries.items() if n not in names})


_PILLARS = (
    "The searches to run, all of them by default; text alone matches the words exactly, the way "
    "to find a code or an identifier. Your filters switch graph search off."
)

_ENTITIES = (
    "The names of the things the question is about. Graph search starts from them instead of "
    "the names it spots in the query; a name it cannot find comes back in unresolved with "
    "similar names to retry with. Your filters switch graph search off, and these then start "
    "nothing."
)

_RELATION_TYPES = "Walk the graph only along these relation types; every type by default."

_SOURCES = (
    "Search only inside these sources, by the source_id a hit or list_sources gives, and the "
    "stories that cite them; a scoped search reads every passage in its scope."
)

_AROUND = (
    "Search only the passages about these named things or, within hops, what they connect to; a "
    "scoped search reads every passage in its scope, and a name it cannot find comes back in "
    "unresolved with similar names to retry with."
)

_HOPS = "How far around reaches: 0 the named things only, 1 (the default) or 2 steps out."

_MATCH = (
    "How the query's words must appear in what keyword search returns: any (the default) of "
    "them, ranked; all of them; phrase, these words in this order side by side, for exact "
    'names and codes; web, the query read as a web search, with "quoted phrases", or, and '
    "-excluded words. Only keyword search narrows; the other searches still answer."
)

_DEPTH = (
    "How far graph search walks from the things the query names before it collects the "
    "passages about what it reached: 0 those things only, 1 what they connect to, 2 (the "
    "default) or 3 steps out; deeper reaches more and costs more."
)

_SPOT = (
    "How graph search finds the things the query names: words finds a name in its other forms "
    "(valve finds valves), as keyword search reads words; exact only whole names, for codes and "
    "tags (P-3, not P-30); substring a name anywhere in the text, inside longer words too; off "
    "only the entities you name. Left out: {default}, or off when you name entities or around."
)

_EXPAND_DEPTH = (
    "By graph: how many steps to walk from the passage's entities, 0 to 3; 2 by default."
)

_WIDTH = "By document: how many chunks to read on each side of a chunk."

_EXPAND_TOP_K = "How many passages to return; not by document from a chunk, where width decides."

_EXPAND_KINDS = (
    "What to return by vector or by graph; not by document, which returns chunks from a chunk "
    "and stories from a story."
)

_EXPAND_HISTORY = (
    "Include superseded and ended stories and retired document versions; a hit found with history=true, "
    "whose row shows retired_at, needs history=true to expand."
)

_EXPAND_AS_OF = "Look around as things were true at a past moment."

_EXPAND_KNOWN_AT = (
    "Look around with stories and documents as memory knew them then; documents a sync replaced "
    "are there only where the corpus keeps history."
)


class SearchArgs(BaseModel):
    query: str
    kinds: list[Kind] = ["chunk", "story"]
    top_k: int = Field(default=10, ge=1, le=100)
    pillars: list[Literal["text", "vector", "graph"]] | None = Field(
        default=None, description=_PILLARS
    )
    filters: dict[str, Any] | None = None
    entities: list[str] | None = Field(default=None, description=_ENTITIES)
    relation_types: list[str] | None = Field(default=None, description=_RELATION_TYPES)
    sources: list[str] | None = Field(default=None, description=_SOURCES)
    around: list[str] | None = Field(default=None, description=_AROUND)
    hops: int = Field(default=1, ge=0, le=2, description=_HOPS)
    match: Match = Field(default="any", description=_MATCH)
    depth: int = Field(default=2, ge=0, le=3, description=_DEPTH)
    spot: Spot | None = None
    history: bool = False
    as_of: datetime | None = None
    known_at: datetime | None = None
    judge: bool = False
    order_by: JudgementName | None = None
    corpus_id: str | list[str] | None = None


class RecallArgs(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    pillars: list[Literal["text", "vector", "graph"]] | None = Field(
        default=None, description=_PILLARS
    )
    history: bool = False
    as_of: datetime | None = None
    known_at: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    subject: str | None = None
    filters: dict[str, Any] | None = None
    entities: list[str] | None = Field(default=None, description=_ENTITIES)
    relation_types: list[str] | None = Field(default=None, description=_RELATION_TYPES)
    sources: list[str] | None = Field(default=None, description=_SOURCES)
    around: list[str] | None = Field(default=None, description=_AROUND)
    hops: int = Field(default=1, ge=0, le=2, description=_HOPS)
    match: Match = Field(default="any", description=_MATCH)
    depth: int = Field(default=2, ge=0, le=3, description=_DEPTH)
    spot: Spot | None = None
    judge: bool = False
    order_by: JudgementName | None = None
    corpus_id: str | list[str] | None = None


class ExpandArgs(BaseModel):
    kind: Kind
    id: str
    by: Literal["document", "vector", "graph"]
    top_k: int = Field(default=10, ge=1, le=100, description=_EXPAND_TOP_K)
    width: int = Field(default=2, ge=1, le=10, description=_WIDTH)
    depth: int = Field(default=2, ge=0, le=3, description=_EXPAND_DEPTH)
    kinds: list[Kind] = Field(default=["chunk", "story"], description=_EXPAND_KINDS)
    relation_types: list[str] | None = Field(default=None, description=_RELATION_TYPES)
    history: bool = Field(default=False, description=_EXPAND_HISTORY)
    as_of: datetime | None = Field(default=None, description=_EXPAND_AS_OF)
    known_at: datetime | None = Field(default=None, description=_EXPAND_KNOWN_AT)
    judge: bool = False
    order_by: JudgementName | None = None
    corpus_id: str | None = None


class DescribeArgs(BaseModel):
    corpus_id: str | None = None


class ReadSourceArgs(BaseModel):
    source_id: str
    locator: dict[str, Any] | None = None
    corpus_id: str | None = None


class ListSourcesArgs(BaseModel):
    filters: dict[str, Any] | None = None
    history: bool = False
    known_at: datetime | None = None
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None
    corpus_id: str | None = None


class IngestArgs(BaseModel):
    name: str
    text: str
    metadata: dict[str, Any] | None = None


class RememberArgs(BaseModel):
    text: str
    subject: str | None = None
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None
    related: int = Field(default=5, ge=0, le=20)
    judge: bool = False


class SupersedeArgs(BaseModel):
    story_id: str | list[str]
    text: str
    valid_from: datetime | None = None
    valid_to: datetime | None = None
    subject: str | None = None
    metadata: dict[str, Any] | None = None
    related: int = Field(default=5, ge=0, le=20)
    judge: bool = False


class EndStoryArgs(BaseModel):
    story_id: str
    valid_to: datetime | None = None


class DeleteStoryArgs(BaseModel):
    story_id: str


class ListStoriesArgs(BaseModel):
    subject: str | None = None
    history: bool = False
    as_of: datetime | None = None
    known_at: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    filters: dict[str, Any] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None
    corpus_id: str | None = None


class NeighborsArgs(BaseModel):
    entity: str
    entity_type: str | None = None
    hops: int = Field(default=1, ge=1, le=3)
    relation_types: list[str] | None = None
    history: bool = False
    as_of: datetime | None = None
    known_at: datetime | None = None


_DESCRIPTIONS = {
    "search": "Search both sources (ingested documents) and stories in this corpus: current stories only; history=true includes superseded and ended stories and retired document versions; as_of asks what was true at a past moment. "
    "Returns ranked hits with the pillar that found each."
    " known_at answers with stories and documents as memory knew them then; documents a sync replaced are there only where the corpus keeps history.",
    "recall": "Search stories only, not sources; use search for documents. Returns current facts; history=true includes superseded and ended stories, each with the valid_to "
    "when it stopped being true. as_of asks what was true at a past moment; since/until bound when it happened; subject narrows."
    " known_at answers as memory knew things then: what it had been told by that moment.",
    "read_source": "Read a source's full text, or the chunks whose locator matches; any version by id, retired or not.",
    "list_sources": "List sources, optionally by facet filters, with a cursor: current versions; history=true includes retired ones; known_at lists what memory held then; documents a sync replaced are there only where the corpus keeps history.",
    "ingest": "File a source: a document's full text under a name, with optional metadata facets. search and read_source find it; "
    "use remember for a short fact.",
    "remember": "Store a story: a short fact, event or decision, about a subject, true from valid_from (default now) "
    "until valid_to when its end is known."
    " Answers with the related open stories the new one may repeat or contradict; supersede one when it should give way.",
    "supersede": "Close one story, or several merged into one, and open the replacement at valid_from (default now), true until valid_to when known. "
    "History is kept."
    " Answers with the related open stories the new one may repeat or contradict; supersede one when it should give way.",
    "end_story": "End a story with nothing in its place: what it says stopped being true at valid_to (default now), "
    "a moment that has come. Supersede it instead when something new holds; an end still to come is the valid_to of "
    "remember or supersede. History is kept: as_of before the end, and history=true, still find it.",
    "delete_story": "Delete a story outright, its history too; when it only stopped being true, end_story keeps the history.",
    "list_stories": "List stories by subject, time and facets, with a cursor: current stories; history=true includes superseded and ended "
    "ones; as_of lists what was true then."
    " known_at answers as memory knew things then: what it had been told by that moment.",
    "neighbors": "Walk the graph from an entity: neighbors within hops, their relations, and the passages that mention them "
    "(current stories; history=true includes superseded and ended ones; as_of reads the stories true then). entity_type picks one when a name has several types."
    " known_at answers with stories and documents as memory knew them then; documents a sync replaced are there only where the corpus keeps history.",
    "expand": "Look around a hit you found: {modes}. Takes the hit's kind and id, keeps to its corpus and never returns the hit itself.",
    "describe": "Describe a corpus before planning a search of one you do not know: how many sources, chunks, stories, retired versions and entities it holds and how many relation rows (one for each passage that says a relation, so the same relation said again counts again, as does one a changed sync wrote again where the corpus keeps history), its entity and relation types with their counts, its language, whether it keeps history, which searches run, whether judging works, and how expand can look.",
}
_MODES = {
    "document": "by document for the chunks next to it in its source, or, from a story, the top_k other stories about its subject nearest it in time, in time order, when a hit looks cut off",
    "vector": "by vector for passages that read like it, more like this",
    "graph": "by graph for passages about what the entities it mentions connect to",
}


_GRAPH_ARGS = ("entities", "relation_types", "around", "hops", "depth", "spot")


def _hit(hit: Hit, several: bool, *, entities: bool) -> dict[str, Any]:
    row: dict[str, Any] = {"corpus_id": hit.corpus_id} if several else {}
    row |= {
        "kind": hit.kind,
        "id": hit.id,
        "text": hit.text,
        "score": hit.score,
        "provenance": hit.provenance,
    }
    if hit.chunk is not None:
        row |= {
            "source_id": hit.chunk.source_id,
            "source_name": hit.chunk.source_name,
            "title": hit.chunk.title,
            "locator": hit.chunk.locator,
            "metadata": hit.chunk.metadata,
        }
        if hit.chunk.context is not None:
            row["context"] = hit.chunk.context
        if hit.chunk.retired_at is not None:
            row["retired_at"] = hit.chunk.model_dump(mode="json")["retired_at"]
    if hit.story is not None:
        row["story"] = hit.story.model_dump(mode="json", exclude={"text", "corpus_id"})
    if hit.judge is not None:
        row["judge"] = hit.judge.model_dump(exclude_none=True)
    if entities and hit.entities:
        row["entities"] = [{"name": e.name, "type": e.type} for e in hit.entities]
    return row


def _offer(schema: dict[str, Any], judgements: tuple[JudgementName, ...]) -> None:
    for option in schema["properties"].get("order_by", {}).get("anyOf", []):
        if "enum" in option:
            option["enum"] = list(judgements)


def _hits(
    hits: Hits, order_by: JudgementName | None, several: bool, *, entities: bool
) -> dict[str, Any]:
    if order_by is not None:
        hits = hits.by(order_by)
    result: dict[str, Any] = {
        "hits": [_hit(h, several, entities=entities) for h in hits.hits],
        "pillars": {n: r.model_dump() for n, r in hits.pillars.items()},
    }
    if hits.unresolved:
        result["unresolved"] = hits.unresolved
    return result


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
    ) -> None:
        named = list(corpus_id) if isinstance(corpus_id, list | tuple) else [corpus_id]
        if not named or not all(isinstance(c, str) and c.strip() for c in named):
            raise ConfigurationError(
                "tools: a corpus_id is required: the model-facing surface has no global scope"
            )
        validate_filters(filters)
        if max_result_chars < 1000:
            raise ConfigurationError("tools: max_result_chars must be at least 1000")
        if judge is not None and (
            not judge
            or not all(isinstance(name, str) for name in judge)
            or len(set(judge)) != len(judge)
            or not set(judge) <= set(JUDGEMENTS)
        ):
            raise ConfigurationError(
                "tools: judge names relevant, evidence, contradicts or injection, each once"
            )
        self._judge: tuple[JudgementName, ...] = tuple(judge) if judge is not None else JUDGEMENTS
        self.memory = memory
        self._on_event = on_event
        self.corpus_ids = list(dict.fromkeys(named))
        self.corpus_id = self.corpus_ids[0]
        self._several = len(self.corpus_ids) > 1
        self._bound = dict(filters or {})
        self._graph = "graph" in memory.pillars and not self._bound
        self._text = "text" in memory.pillars
        self._modes = [
            mode
            for mode in _MODES
            if mode == "document"
            or (mode == "vector" and "vector" in memory.pillars)
            or (mode == "graph" and self._graph)
        ]
        self._pillars = [
            pillar
            for pillar in ("text", "vector", "graph")
            if pillar in memory.pillars and (pillar != "graph" or self._graph)
        ]
        self._max = max_result_chars
        handlers: dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[dict[str, Any]]]]] = {
            "search": (SearchArgs, self._search),
            "recall": (RecallArgs, self._recall),
            "expand": (ExpandArgs, self._expand),
            "read_source": (ReadSourceArgs, self._read_source),
            "list_sources": (ListSourcesArgs, self._list_sources),
            "ingest": (IngestArgs, self._ingest),
            "remember": (RememberArgs, self._remember),
            "supersede": (SupersedeArgs, self._supersede),
            "end_story": (EndStoryArgs, self._end_story),
            "delete_story": (DeleteStoryArgs, self._delete_story),
            "list_stories": (ListStoriesArgs, self._list_stories),
            "neighbors": (NeighborsArgs, self._neighbors),
            "describe": (DescribeArgs, self._describe_corpus),
        }
        if not self._graph:
            del handlers["neighbors"]
        if self._bound:
            del handlers["describe"]
        if self._several:
            for write in (
                "ingest",
                "remember",
                "supersede",
                "end_story",
                "delete_story",
                "neighbors",
            ):
                handlers.pop(write, None)
        if include is not None:
            handlers = {name: h for name, h in handlers.items() if name in include}
        tools = []
        for name, (model, handler) in handlers.items():
            schema = model.model_json_schema()
            if memory.decision is None:
                schema["properties"].pop("judge", None)
                schema["properties"].pop("order_by", None)
            else:
                _offer(schema, self._judge)
            self._shape(schema)
            self._scope(name, schema)
            tools.append(
                Tool(
                    name,
                    self._describe(name),
                    body=self._body(model, handler),
                    args=model,
                    input_schema=schema,
                )
            )
        super().__init__(tools)

    def _body(
        self, model: type[BaseModel], handler: Callable[[Any], Awaitable[dict[str, Any]]]
    ) -> Body:
        async def body(args: dict[str, Any]) -> str | dict[str, Any]:
            return self._clip(await handler(model.model_validate(args)))

        return body

    def _shape(self, schema: dict[str, Any]) -> None:
        properties = schema["properties"]
        if not self._graph:
            for name in _GRAPH_ARGS:
                properties.pop(name, None)
        if not self._text:
            properties.pop("match", None)
        if "spot" in properties:
            properties["spot"]["description"] = _SPOT.format(default=self.memory.spot)
        if "by" in properties:
            properties["by"]["enum"] = list(self._modes)
        for option in properties.get("pillars", {}).get("anyOf", []):
            if "items" in option:
                option["items"]["enum"] = list(self._pillars)

    def _chosen(self, pillars: Sequence[str] | None) -> set[str] | None:
        if pillars is None:
            return None
        if not pillars or not set(pillars) <= set(self._pillars):
            raise InputError(f"tools: pillars are among {', '.join(self._pillars)}")
        return set(pillars)

    def _offered(self, args: BaseModel) -> None:
        given = [name for name in _GRAPH_ARGS if name in args.model_fields_set]
        if given and not self._graph:
            raise InputError(f"tools: {given[0]} is not offered here")
        if "match" in args.model_fields_set and not self._text:
            raise InputError("tools: match is not offered here")

    def _scope(self, name: str, schema: dict[str, Any]) -> None:
        properties = schema["properties"]
        if "corpus_id" not in properties:
            return
        if not self._several:
            del properties["corpus_id"]
            return
        ids = {"enum": self.corpus_ids}
        if name in ("search", "recall"):
            properties["corpus_id"] = {
                "anyOf": [ids, {"type": "array", "items": ids, "minItems": 1}],
                "description": "Some of the corpora you may read; every one of them when omitted.",
            }
            return
        reading = (
            "The corpus to describe."
            if name == "describe"
            else "The corpus to read, as a hit names it."
        )
        properties["corpus_id"] = {**ids, "description": reading}
        schema["required"] = [*schema.get("required", []), "corpus_id"]

    def _describe(self, name: str) -> str:
        text = _DESCRIPTIONS[name]
        if name == "expand":
            text = text.format(modes="; ".join(_MODES[mode] for mode in self._modes))
        if not self._several:
            return text
        return text.replace("in this corpus", "in the corpora you may read")

    def _corpora(self, named: str | list[str] | None) -> list[str]:
        if named is None:
            return self.corpus_ids
        wanted = [named] if isinstance(named, str) else list(named)
        if not wanted or any(c not in self.corpus_ids for c in wanted):
            raise InputError(
                f"tools: corpus_id can name only the corpora you may read: {', '.join(self.corpus_ids)}"
            )
        return list(dict.fromkeys(wanted))

    def _one(self, named: str | None) -> str:
        if named is None and self._several:
            raise InputError(
                f"tools: corpus_id is required: name one of {', '.join(self.corpus_ids)}"
            )
        [corpus_id] = self._corpora(named) if named is not None else [self.corpus_id]
        return corpus_id

    def _filters(self, supplied: dict[str, Any] | None) -> dict[str, Any] | None:
        validate_filters(supplied)
        merged = dict(self._bound)
        for key, value in (supplied or {}).items():
            if key in self._bound and not within(value, self._bound[key]):
                raise InputError(
                    f"tools: filter {key!r} can narrow its bound value {self._bound[key]!r}, never widen it"
                )
            merged[key] = value
        return merged or None

    async def _in_view(self, story_id: str) -> Story | None:
        if not self._bound:
            return None
        stories = await self.memory.get_stories(self.corpus_id, [story_id])
        if not stories or not matches_filters(stories[0].metadata, self._bound):
            raise NotFoundError(f"tools: story {story_id!r} was not found")
        return stories[0]

    def _own(self, rows: list[Any], corpora: list[str] | None = None) -> None:
        allowed = corpora or [self.corpus_id]
        for row in rows:
            if getattr(row, "corpus_id", allowed[0]) not in allowed:
                raise StorageError("tools: a row from another corpus reached the tool surface")

    def _clip(self, result: dict[str, Any]) -> dict[str, Any]:
        for key in ("hits", "stories", "sources", "passages", "relations", "entities", "related"):
            rows = result.get(key)
            if not isinstance(rows, list):
                continue
            while rows and len(json.dumps(result)) > self._max:
                rows.pop()
                result["truncated"] = True
        if len(json.dumps(result)) > self._max and "text" in result:
            result["text"] = result["text"][: self._max // 2]
            result["truncated"] = True
        return result

    def _asked(
        self, judge: bool, order_by: JudgementName | None
    ) -> tuple[JudgementName, ...] | bool:
        if order_by is not None and order_by not in self._judge:
            raise InputError(f"tools: order_by is one of {', '.join(self._judge)}")
        if judge:
            return self._judge
        return (order_by,) if order_by is not None else False

    async def _search(self, args: SearchArgs) -> dict[str, Any]:
        self._offered(args)
        corpora = self._corpora(args.corpus_id)
        hits = await self.memory.search(
            corpora,
            args.query,
            kinds=tuple(args.kinds),
            pillars=self._chosen(args.pillars),
            top_k=args.top_k,
            filters=self._filters(args.filters),
            as_of=args.as_of,
            known_at=args.known_at,
            history=args.history,
            judge=self._asked(args.judge, args.order_by),
            entities=args.entities,
            relation_types=args.relation_types,
            sources=args.sources,
            around=args.around,
            hops=args.hops,
            match=args.match,
            depth=args.depth,
            spot=args.spot,
            on_event=self._on_event,
        )
        self._own([h.chunk or h.story for h in hits.hits], corpora)
        return _hits(hits, args.order_by, self._several, entities=self._graph)

    async def _recall(self, args: RecallArgs) -> dict[str, Any]:
        self._offered(args)
        corpora = self._corpora(args.corpus_id)
        hits = await self.memory.recall(
            corpora,
            args.query,
            top_k=args.top_k,
            as_of=args.as_of,
            known_at=args.known_at,
            history=args.history,
            since=args.since,
            until=args.until,
            subject=args.subject,
            filters=self._filters(args.filters),
            pillars=self._chosen(args.pillars),
            judge=self._asked(args.judge, args.order_by),
            entities=args.entities,
            relation_types=args.relation_types,
            sources=args.sources,
            around=args.around,
            hops=args.hops,
            match=args.match,
            depth=args.depth,
            spot=args.spot,
            on_event=self._on_event,
        )
        self._own([h.story for h in hits.hits], corpora)
        return _hits(hits, args.order_by, self._several, entities=self._graph)

    async def _expand(self, args: ExpandArgs) -> dict[str, Any]:
        self._offered(args)
        if args.by not in self._modes:
            raise InputError(f"tools: by is one of {', '.join(self._modes)}")
        corpus_id = self._one(args.corpus_id)
        ref: Ref = (args.kind, args.id)
        if self._bound:
            storable(args.id)
            found = await self.memory.storage.get_passages(corpus_id, [ref])
            if not found or not matches_filters(found[0].metadata, self._bound):
                raise NotFoundError(f"tools: {args.kind} {args.id!r} was not found")
        hits = await self.memory.expand(
            corpus_id,
            ref,
            by=args.by,
            top_k=args.top_k,
            width=args.width,
            depth=args.depth,
            kinds=tuple(args.kinds),
            relation_types=args.relation_types,
            filters=self._filters(None),
            as_of=args.as_of,
            known_at=args.known_at,
            history=args.history,
            judge=self._asked(args.judge, args.order_by),
        )
        self._own([h.chunk or h.story for h in hits.hits], [corpus_id])
        return _hits(hits, args.order_by, self._several, entities=self._graph)

    async def _read_source(self, args: ReadSourceArgs) -> dict[str, Any]:
        corpus_id = self._one(args.corpus_id)
        if self._bound:
            source = await self.memory.get_source(corpus_id, args.source_id)
            if source is None or not matches_filters(source.metadata, self._bound):
                raise NotFoundError(f"tools: source {args.source_id!r} was not found")
        text = await self.memory.read_source(corpus_id, args.source_id, locator=args.locator)
        return {"source_id": args.source_id, "text": text}

    async def _list_sources(self, args: ListSourcesArgs) -> dict[str, Any]:
        corpus_id = self._one(args.corpus_id)
        sources, cursor = await self.memory.list_sources(
            corpus_id,
            filters=self._filters(args.filters),
            known_at=args.known_at,
            history=args.history,
            limit=args.limit,
            cursor=args.cursor,
        )
        self._own(sources, [corpus_id])
        return {
            "sources": [s.model_dump(mode="json", exclude={"text", "digest"}) for s in sources],
            "cursor": cursor,
        }

    async def _ingest(self, args: IngestArgs) -> dict[str, Any]:
        source = (
            await self.memory.ingest(
                self.corpus_id,
                name=args.name,
                text=args.text,
                metadata=stamp(args.metadata, self._bound),
                on_event=self._on_event,
            )
        ).source
        self._own([source])
        return {"source": source.model_dump(mode="json", exclude={"text", "digest"})}

    async def _remember(self, args: RememberArgs) -> dict[str, Any]:
        remembered = await self.memory.remember(
            self.corpus_id,
            args.text,
            subject=args.subject,
            valid_from=args.valid_from,
            valid_to=args.valid_to,
            source_ids=tuple(args.source_ids),
            metadata=stamp(args.metadata, self._bound),
            related=args.related,
            filters=self._bound or None,
            judge=self._judge if args.judge else False,
            on_event=self._on_event,
        )
        story = remembered.story
        return {
            "story": story.model_dump(mode="json"),
            "related": self._related(remembered.related),
        }

    def _related(self, hits: Hits) -> list[dict[str, Any]]:
        self._own([hit.story for hit in hits.hits])
        rows: list[dict[str, Any]] = []
        for hit in hits.hits:
            if hit.story is None:
                continue
            story = hit.story.model_dump(mode="json")
            row: dict[str, Any] = {
                "story_id": hit.id,
                "text": hit.text,
                "subject": story["subject"],
                "valid_from": story["valid_from"],
                "valid_to": story["valid_to"],
            }
            if hit.judge is not None:
                row["judge"] = hit.judge.model_dump(exclude_none=True)
            if self._graph and hit.entities:
                row["entities"] = [{"name": e.name, "type": e.type} for e in hit.entities]
            rows.append(row)
        return rows

    async def _supersede(self, args: SupersedeArgs) -> dict[str, Any]:
        ids = [args.story_id] if isinstance(args.story_id, str) else list(args.story_id)
        if len(set(ids)) != len(ids):
            raise InputError("tools: story_id names a story more than once")
        olds = [await self._in_view(story_id) for story_id in ids]
        metadata = args.metadata
        if self._bound and metadata is not None:
            metadata = stamp(metadata, self._bound)
        elif self._bound:
            held = [old.metadata for old in olds if old is not None]
            metadata = (
                held[0] if held and all(m == held[0] for m in held) else stamp(None, self._bound)
            )
        superseded = await self.memory.supersede(
            self.corpus_id,
            args.story_id,
            args.text,
            valid_from=args.valid_from,
            valid_to=args.valid_to,
            subject=args.subject,
            metadata=metadata,
            related=args.related,
            filters=self._bound or None,
            judge=self._judge if args.judge else False,
            on_event=self._on_event,
        )
        return {
            "story": superseded.story.model_dump(mode="json"),
            "related": self._related(superseded.related),
        }

    async def _end_story(self, args: EndStoryArgs) -> dict[str, Any]:
        await self._in_view(args.story_id)
        ended = await self.memory.end_story(self.corpus_id, args.story_id, valid_to=args.valid_to)
        return {"story": ended.model_dump(mode="json")}

    async def _delete_story(self, args: DeleteStoryArgs) -> dict[str, Any]:
        await self._in_view(args.story_id)
        await self.memory.delete_story(self.corpus_id, args.story_id)
        return {"deleted": args.story_id}

    async def _list_stories(self, args: ListStoriesArgs) -> dict[str, Any]:
        corpus_id = self._one(args.corpus_id)
        stories, cursor = await self.memory.list_stories(
            corpus_id,
            subject=args.subject,
            as_of=args.as_of,
            known_at=args.known_at,
            since=args.since,
            until=args.until,
            filters=self._filters(args.filters),
            history=args.history,
            limit=args.limit,
            cursor=args.cursor,
        )
        self._own(stories, [corpus_id])
        return {"stories": [s.model_dump(mode="json") for s in stories], "cursor": cursor}

    async def _neighbors(self, args: NeighborsArgs) -> dict[str, Any]:
        hood = await self.memory.neighbors(
            self.corpus_id,
            args.entity,
            entity_type=args.entity_type,
            hops=args.hops,
            relation_types=args.relation_types,
            as_of=args.as_of,
            known_at=args.known_at,
            history=args.history,
        )
        self._own([*hood.entities, *hood.relations])
        return {
            "entities": [e.model_dump() for e in hood.entities],
            "relations": [r.model_dump() for r in hood.relations],
            "passages": [_hit(h, False, entities=True) for h in hood.passages],
        }

    async def _describe_corpus(self, args: DescribeArgs) -> dict[str, Any]:
        corpus_id = self._one(args.corpus_id)
        return (await self.memory.describe(corpus_id)).model_dump()
