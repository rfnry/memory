import importlib
import os
from typing import Any

import pytest
from database import Schemas, provision

from rfnry_memory import PostgresStorage


def live_adapters() -> Any:
    path = os.environ.get("RFNRY_LIVE_ADAPTERS")
    if not path:
        pytest.fail(
            "tests: the live tier needs RFNRY_LIVE_ADAPTERS, the module path of your live adapters"
        )
    return importlib.import_module(path)


@pytest.fixture(scope="session")
def adapters() -> Any:
    return live_adapters()


@pytest.fixture
async def storage(schemas: Schemas, adapters: Any) -> PostgresStorage:
    schema = await schemas.create()
    return await provision(await schemas.pool(schema), dimension=adapters.embedding().dimension)
