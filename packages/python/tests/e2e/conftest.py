import importlib
import os
from typing import Any

import pytest


@pytest.fixture(scope="session")
def adapters() -> Any:
    path = os.environ.get("RFNRY_LIVE_ADAPTERS")
    if not path:
        pytest.skip("RFNRY_LIVE_ADAPTERS is not set")
    return importlib.import_module(path)
