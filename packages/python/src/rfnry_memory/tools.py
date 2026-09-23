from __future__ import annotations

import json
from collections.abc import Awaitable, Callable, Iterable
from dataclasses import dataclass
from datetime import datetime
from functools import partial
from typing import Any, Protocol, runtime_checkable

from pydantic import BaseModel, Field, ValidationError

from rfnry_memory.errors import (
    ConfigurationError,
    InputError,
    MemoryException,
    NotFoundError,
    StorageError,
)
from rfnry_memory.filters import matches_filters, stamp, validate_filters
from rfnry_memory.memory import Memory
from rfnry_memory.models import Hit, Kind, Story
from rfnry_memory.storage.storage import text_hash

Body = Callable[[dict[str, Any]], Awaitable[str | dict[str, Any]]]
Dispatcher = Callable[[dict[str, Any]], Awaitable[str | dict[str, Any]]]


@runtime_checkable
class ToolsLike(Protocol):
    def definitions(self) -> list[dict[str, Any]]: ...

    async def dispatch(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]: ...


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

    async def call(self, args: dict[str, Any]) -> str | dict[str, Any]:
        if self.body is None:
            raise ConfigurationError(f"tools: tool {self.name!r} has no body")
        if self.args is not None:
            try:
                args = self.args.model_validate(args).model_dump(exclude_unset=True)
            except ValidationError as exc:
                raise InputError(f"tools: invalid arguments for {self.name}: {exc.errors(include_url=False)}") from exc
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

    async def dispatch(self, name: str, args: dict[str, Any]) -> str | dict[str, Any]:
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


class SearchArgs(BaseModel):
    query: str
    kinds: list[Kind] = ["chunk", "story"]
    top_k: int = Field(default=10, ge=1, le=100)
    filters: dict[str, Any] | None = None
    history: bool = False
    judge: bool = False


class RecallArgs(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    history: bool = False
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    subject: str | None = None
    filters: dict[str, Any] | None = None
    judge: bool = False


class ReadSourceArgs(BaseModel):
    source_id: str
    locator: dict[str, Any] | None = None


class ListSourcesArgs(BaseModel):
    filters: dict[str, Any] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None


class RememberArgs(BaseModel):
    text: str
    subject: str | None = None
    occurred_at: datetime | None = None
    source_ids: list[str] = Field(default_factory=list)
    metadata: dict[str, Any] | None = None


class SupersedeArgs(BaseModel):
    story_id: str
    text: str
    effective_at: datetime | None = None
    subject: str | None = None
    metadata: dict[str, Any] | None = None


class ForgetArgs(BaseModel):
    story_id: str


class ListStoriesArgs(BaseModel):
    subject: str | None = None
    history: bool = False
    as_of: datetime | None = None
    since: datetime | None = None
    until: datetime | None = None
    filters: dict[str, Any] | None = None
    limit: int = Field(default=50, ge=1, le=500)
    cursor: str | None = None


class NeighborsArgs(BaseModel):
    entity: str
    entity_type: str | None = None
    hops: int = Field(default=1, ge=1, le=3)
    relation_types: list[str] | None = None
    history: bool = False


_DESCRIPTIONS = {
    "search": "Search sources and stories in this corpus: current stories only; history=true includes superseded stories. "
              "Returns ranked hits with the pillar that found each.",
    "recall": "Search stories. Returns current facts; history=true includes superseded stories, each with the valid_to "
              "when it stopped being true. as_of asks what was true at a past moment; since/until bound when it happened; subject narrows.",
    "read_source": "Read a source's full text, or the chunks whose locator matches.",
    "list_sources": "List sources, optionally by facet filters, with a cursor.",
    "remember": "Store a story: a short fact, event or decision, about a subject, true from occurred_at.",
    "supersede": "Close a story and open its replacement at effective_at. History is kept.",
    "forget": "Delete a story outright.",
    "list_stories": "List stories by subject, time and facets, with a cursor: current stories; history=true includes superseded "
                    "ones; as_of lists what was true then.",
    "neighbors": "Walk the graph from an entity: neighbors within hops, their relations, and the passages that mention them "
                 "(current stories; history=true includes superseded ones). entity_type picks one when a name has several types.",
}


def _hit(hit: Hit) -> dict[str, Any]:
    row: dict[str, Any] = {"kind": hit.kind, "id": hit.id, "text": hit.text, "score": hit.score, "provenance": hit.provenance}
    if hit.chunk is not None:
        row["source_id"], row["title"], row["locator"] = hit.chunk.source_id, hit.chunk.title, hit.chunk.locator
    if hit.story is not None:
        row["story"] = hit.story.model_dump(mode="json", exclude={"text", "corpus_id"})
    if hit.judge is not None:
        row["judge"] = hit.judge.model_dump()
    return row


def _error(kind: str, message: str) -> dict[str, Any]:
    return {"error": {"type": kind, "message": message}}


class MemoryTools(Tools):
    def __init__(
        self,
        memory: Memory,
        *,
        corpus_id: str,
        filters: dict[str, Any] | None = None,
        include: set[str] | None = None,
        max_result_chars: int = 200_000,
    ) -> None:
        if not isinstance(corpus_id, str) or not corpus_id.strip():
            raise ConfigurationError("tools: a corpus_id is required: the model-facing surface has no global scope")
        validate_filters(filters)
        if max_result_chars < 1000:
            raise ConfigurationError("tools: max_result_chars must be at least 1000")
        self.memory = memory
        self.corpus_id = corpus_id
        self._bound = dict(filters or {})
        self._max = max_result_chars
        handlers: dict[str, tuple[type[BaseModel], Callable[[Any], Awaitable[dict[str, Any]]]]] = {
            "search": (SearchArgs, self._search),
            "recall": (RecallArgs, self._recall),
            "read_source": (ReadSourceArgs, self._read_source),
            "list_sources": (ListSourcesArgs, self._list_sources),
            "remember": (RememberArgs, self._remember),
            "supersede": (SupersedeArgs, self._supersede),
            "forget": (ForgetArgs, self._forget),
            "list_stories": (ListStoriesArgs, self._list_stories),
            "neighbors": (NeighborsArgs, self._neighbors),
        }
        if "graph" not in memory.pillars:
            del handlers["neighbors"]
        if include is not None:
            handlers = {name: h for name, h in handlers.items() if name in include}
        self._loose = next((key for key, value in self._bound.items() if isinstance(value, list)), None)
        self._withheld = {name for name in ("remember", "supersede") if name in handlers} if self._loose else set()
        handlers = {name: h for name, h in handlers.items() if name not in self._withheld}
        tools = []
        for name, (model, handler) in handlers.items():
            schema = model.model_json_schema()
            if memory.decision is None:
                schema["properties"].pop("judge", None)
            tools.append(Tool(name, _DESCRIPTIONS[name], body=self._guarded(name, model, handler), input_schema=schema))
        super().__init__(tools)

    def _guarded(self, name: str, model: type[BaseModel], handler: Callable[[Any], Awaitable[dict[str, Any]]]) -> Body:
        async def body(args: dict[str, Any]) -> str | dict[str, Any]:
            try:
                parsed = model.model_validate(args)
            except ValidationError as exc:
                return _error("InputError", f"tools: invalid arguments for {name}: {exc.errors(include_url=False)}")
            try:
                return self._clip(await handler(parsed))
            except MemoryException as exc:
                return _error(type(exc).__name__, str(exc))

        return body

    async def dispatch(self, name: str, args: dict[str, Any]) -> dict[str, Any]:
        if name in self._withheld:
            return _error("InputError", f"tools: {name} is withheld: the bound filter {self._loose!r} matches any of several "
                                        "values, so what it writes could not be found through these tools")
        if name not in self:
            return _error("UnknownTool", f"tools: unknown tool {name!r}; available: {self.names}")
        result = await super().dispatch(name, args)
        return result if isinstance(result, dict) else {"text": result}

    def _filters(self, supplied: dict[str, Any] | None) -> dict[str, Any] | None:
        validate_filters(supplied)
        merged = {**(supplied or {}), **self._bound}
        return merged or None

    async def _in_view(self, story_id: str) -> Story | None:
        if not self._bound:
            return None
        stories = await self.memory.get_stories(self.corpus_id, [story_id])
        if stories and not matches_filters(stories[0].metadata, self._bound):
            raise NotFoundError(f"tools: story {story_id!r} was not found")
        return stories[0] if stories else None

    def _own(self, rows: list[Any]) -> None:
        for row in rows:
            if getattr(row, "corpus_id", self.corpus_id) != self.corpus_id:
                raise StorageError("tools: a row from another corpus reached the tool surface")

    def _clip(self, result: dict[str, Any]) -> dict[str, Any]:
        for key in ("hits", "stories", "sources", "passages"):
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

    async def _search(self, args: SearchArgs) -> dict[str, Any]:
        hits = await self.memory.search(self.corpus_id, args.query, kinds=tuple(args.kinds), top_k=args.top_k,
                                        filters=self._filters(args.filters), history=args.history, judge=args.judge)
        self._own([h.chunk or h.story for h in hits.hits])
        return {"hits": [_hit(h) for h in hits.hits], "pillars": {n: r.model_dump() for n, r in hits.pillars.items()}}

    async def _recall(self, args: RecallArgs) -> dict[str, Any]:
        hits = await self.memory.recall(self.corpus_id, args.query, top_k=args.top_k, as_of=args.as_of, history=args.history, since=args.since,
                                        until=args.until, subject=args.subject, filters=self._filters(args.filters), judge=args.judge)
        self._own([h.story for h in hits.hits])
        return {"hits": [_hit(h) for h in hits.hits], "pillars": {n: r.model_dump() for n, r in hits.pillars.items()}}

    async def _read_source(self, args: ReadSourceArgs) -> dict[str, Any]:
        text = await self.memory.read(self.corpus_id, args.source_id, locator=args.locator)
        return {"source_id": args.source_id, "text": text}

    async def _list_sources(self, args: ListSourcesArgs) -> dict[str, Any]:
        sources, cursor = await self.memory.list_sources(self.corpus_id, filters=self._filters(args.filters), limit=args.limit, cursor=args.cursor)
        self._own(sources)
        return {"sources": [s.model_dump(mode="json", exclude={"text"}) for s in sources], "cursor": cursor}

    async def _remember(self, args: RememberArgs) -> dict[str, Any]:
        story = (await self.memory.remember(self.corpus_id, args.text, subject=args.subject, occurred_at=args.occurred_at,
                                            source_ids=tuple(args.source_ids), metadata=stamp(args.metadata, self._bound))).story
        if not matches_filters(story.metadata, self._bound):
            raise InputError("tools: this text already exists outside your view")
        return {"story": story.model_dump(mode="json")}

    async def _supersede(self, args: SupersedeArgs) -> dict[str, Any]:
        old = await self._in_view(args.story_id)
        holder = await self.memory.storage.find_open_story(self.corpus_id, text_hash(args.text)) if self._bound else None
        if holder is not None and holder.story_id != args.story_id and not matches_filters(holder.metadata, self._bound):
            raise InputError("tools: this text already exists outside your view")
        metadata = args.metadata
        if self._bound:
            metadata = stamp(metadata if metadata is not None or old is None else old.metadata, self._bound)
        superseded = await self.memory.supersede(self.corpus_id, args.story_id, args.text, effective_at=args.effective_at,
                                                 subject=args.subject, metadata=metadata)
        return {"story": superseded.story.model_dump(mode="json")}

    async def _forget(self, args: ForgetArgs) -> dict[str, Any]:
        await self._in_view(args.story_id)
        await self.memory.forget(self.corpus_id, args.story_id)
        return {"forgotten": args.story_id}

    async def _list_stories(self, args: ListStoriesArgs) -> dict[str, Any]:
        stories, cursor = await self.memory.list_stories(self.corpus_id, subject=args.subject, as_of=args.as_of, since=args.since,
                                                         until=args.until, filters=self._filters(args.filters), history=args.history, limit=args.limit, cursor=args.cursor)
        self._own(stories)
        return {"stories": [s.model_dump(mode="json") for s in stories], "cursor": cursor}

    async def _neighbors(self, args: NeighborsArgs) -> dict[str, Any]:
        hood = await self.memory.neighbors(self.corpus_id, args.entity, entity_type=args.entity_type, hops=args.hops, relation_types=args.relation_types,
                                           history=args.history)
        self._own([*hood.entities, *hood.relations])
        return {"entities": [e.model_dump() for e in hood.entities], "relations": [r.model_dump() for r in hood.relations],
                "passages": [_hit(h) for h in hood.passages]}
