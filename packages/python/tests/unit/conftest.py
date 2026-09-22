import pytest

from .fakes import HashEmbedding


@pytest.fixture
def embedding() -> HashEmbedding:
    return HashEmbedding()
