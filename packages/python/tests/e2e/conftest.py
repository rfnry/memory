import importlib
import os
from typing import Any

import pytest
from database import Schemas, provision

from rfnry_memory import PostgresStorage


@pytest.fixture(scope="session")
def adapters() -> Any:
    path = os.environ.get("RFNRY_LIVE_ADAPTERS")
    if not path:
        pytest.skip("RFNRY_LIVE_ADAPTERS is not set")
    return importlib.import_module(path)


@pytest.fixture
async def storage(schemas: Schemas, adapters: Any) -> PostgresStorage:
    schema = await schemas.create()
    return await provision(await schemas.pool(schema), dimension=adapters.embedding().dimension)
