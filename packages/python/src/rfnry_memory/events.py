import asyncio
import time
from collections.abc import Awaitable, Callable

from pydantic import BaseModel

from rfnry_memory.models import Usage


class Chunked(BaseModel):
    corpus_id: str
    name: str
    source: int
    sources: int
    chunks: int

    model_config = {"extra": "forbid", "frozen": True}


class Embedded(BaseModel):
    corpus_id: str
    name: str | None
    done: int
    total: int
    computed: int
    duration_ms: int

    model_config = {"extra": "forbid", "frozen": True}


class Extracted(BaseModel):
    corpus_id: str
    name: str | None
    chunk: int
    chunks: int
    entities: int
    relations: int
    usage: Usage
    duration_ms: int

    model_config = {"extra": "forbid", "frozen": True}


Event = Chunked | Embedded | Extracted
EventHandler = Callable[[Event], Awaitable[None]]


class Events:
    def __init__(self, handler: EventHandler | None) -> None:
        self._handler = handler
        self._lock = asyncio.Lock()

    async def send(self, event: Event) -> None:
        if self._handler is None:
            return
        async with self._lock:
            await self._handler(event)


def elapsed_ms(began: float) -> int:
    return int((time.monotonic() - began) * 1000)
