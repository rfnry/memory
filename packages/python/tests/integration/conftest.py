import pytest
from database import Database

from rfnry_memory.postgres import PostgresStorage


@pytest.fixture
def schema(database: Database, storage: PostgresStorage) -> str:
    return database.run
