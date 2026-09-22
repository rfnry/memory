from collections.abc import AsyncIterator
from typing import Any

import pytest

from rfnry_memory.storage.inmemory import InMemoryStorage
from rfnry_memory.storage.storage import Storage


@pytest.fixture(params=["inmemory"])
async def storage(request: Any) -> AsyncIterator[Storage]:
    if request.param == "inmemory":
        memory = InMemoryStorage()
        await memory.initialize(dimension=8)
        yield memory
        return
    raise AssertionError(request.param)
