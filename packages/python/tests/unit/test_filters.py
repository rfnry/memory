import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.filters import matches_filters, validate_filters


def test_valid_filters() -> None:
    validate_filters(None)
    validate_filters({"metadata.line": "A", "metadata.rev": [2, 3], "metadata.tags": ["x", "y"], "metadata.ok": True})


@pytest.mark.parametrize(
    "filters",
    [
        {"corpus_id": "acme"},
        {"line": "A"},
        {"metadata.rev": 1.5},
        {"metadata.rev": []},
        {"metadata.rev": [1, "x"]},
        {"": "x"},
        {"metadata.rev": {"a": 1}},
    ],
)
def test_invalid_filters(filters: dict[str, object]) -> None:
    with pytest.raises(InputError):
        validate_filters(filters)


def test_matches_filters_dotted_any_of_and_strict_bool() -> None:
    metadata = {"line": "A", "rev": 3, "flag": True, "tags": ["x", "y"], "nested": {"k": "v"}}
    assert matches_filters(metadata, None)
    assert matches_filters(metadata, {"metadata.line": "A", "metadata.rev": [2, 3]})
    assert matches_filters(metadata, {"metadata.tags": "y", "metadata.nested.k": "v"})
    assert not matches_filters(metadata, {"metadata.line": "B"})
    assert not matches_filters(metadata, {"metadata.missing": "x"})
    assert not matches_filters(metadata, {"metadata.rev": True})
    assert not matches_filters({"flag": 1}, {"metadata.flag": True})
