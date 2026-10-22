import pytest

from rfnry_memory.errors import InputError
from rfnry_memory.filters import matches_filters, validate_filters


def test_valid_filters() -> None:
    validate_filters(None)
    validate_filters(
        {
            "metadata.line": "A",
            "metadata.rev": [2, 3],
            "metadata.tags": ["x", "y"],
            "metadata.ok": True,
            "metadata.weight": 0.5,
            "metadata.scores": [0.5, 2],
            "metadata.confidence": {"gte": 0.8},
            "metadata.signed": {"gte": "2024-01-01", "lt": "2025-01-01"},
            "metadata.revision": {"gt": 2, "lte": 10.5},
        }
    )


@pytest.mark.parametrize(
    "filters",
    [
        {"corpus_id": "acme"},
        {"line": "A"},
        {"metadata.rev": float("nan")},
        {"metadata.rev": float("inf")},
        {"metadata.rev": [1.5, "x"]},
        {"metadata.rev": [True, 1]},
        {"metadata.rev": {}},
        {"metadata.rev": {"ne": 1}},
        {"metadata.rev": {"gt": 1, "lt": "x"}},
        {"metadata.rev": {"gt": True}},
        {"metadata.rev": {"gt": float("nan")}},
        {"metadata.rev": {"gt": None}},
        {"metadata.rev": {"gt": [1]}},
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


def test_a_range_matches_numbers_by_value_and_strings_by_code_point() -> None:
    metadata = {
        "confidence": 0.85,
        "rev": 3,
        "scores": [1, 5, 9],
        "signed": "2024-03-01",
        "owner": {"ext": 318},
        "flag": True,
        "nothing": None,
        "word": "é",
        "rows": [{"n": 5}],
    }
    assert matches_filters(metadata, {"metadata.confidence": {"gte": 0.8}})
    assert not matches_filters(metadata, {"metadata.confidence": {"gt": 0.85}})
    assert matches_filters(metadata, {"metadata.rev": {"gt": 2.5, "lte": 3}})
    assert matches_filters(metadata, {"metadata.scores": {"gt": 4, "lt": 6}})
    assert not matches_filters(metadata, {"metadata.scores": {"gt": 9}})
    assert matches_filters(metadata, {"metadata.signed": {"gte": "2024-01-01", "lt": "2025"}})
    assert matches_filters(metadata, {"metadata.owner.ext": {"gte": 300}})
    assert matches_filters(metadata, {"metadata.word": {"gt": "z"}})
    for key in ("flag", "nothing", "missing", "signed", "rows.n"):
        assert not matches_filters(metadata, {f"metadata.{key}": {"gte": 0}})
    assert not matches_filters(metadata, {"metadata.rev": {"gte": "0"}})
    assert not matches_filters(metadata, {"metadata.rev": {"gt": 5, "lt": 1}})
    assert matches_filters(metadata, {"metadata.confidence": 0.85, "metadata.rev": 3.0})
