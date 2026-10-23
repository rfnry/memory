import pytest
from pydantic import ValidationError

from rfnry_memory.index import HNSW, hnsw


def test_the_clause_names_the_settings_the_precision_and_the_kind() -> None:
    assert hnsw.clause("chunk", HNSW(m=16, ef_construction=64, precision="full"), 8) == (
        "USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64) "
        "WHERE passage_kind = 'chunk'"
    )
    assert hnsw.clause("story", HNSW(m=32, precision="half"), 8) == (
        "USING hnsw ((vector::halfvec(8)) halfvec_cosine_ops) WITH (m = 32) "
        "WHERE passage_kind = 'story'"
    )
    assert hnsw.clause("chunk", HNSW(precision="binary", rerank=8, ef_search=100), 8) == (
        "USING hnsw ((binary_quantize(vector)::bit(8)) bit_hamming_ops) WHERE passage_kind = 'chunk'"
    )


def test_an_option_left_out_is_left_to_pgvector() -> None:
    assert hnsw.clause("chunk", HNSW(precision="full"), 8) == (
        "USING hnsw (vector vector_cosine_ops) WHERE passage_kind = 'chunk'"
    )
    assert hnsw.session({}) == []


def test_the_settings_read_back_from_an_index_s_options_and_its_search_options() -> None:
    assert hnsw.read(["m=32", "ef_construction=128"], "half", {}) == HNSW(
        m=32, ef_construction=128, precision="half"
    )
    assert hnsw.read(["ef_construction=64"], "full", {}) == HNSW(
        ef_construction=64, precision="full"
    )
    assert hnsw.read([], "binary", {"rerank": 8, "ef_search": 100}) == HNSW(
        precision="binary", rerank=8, ef_search=100
    )


def test_a_search_sets_only_the_options_the_index_was_given() -> None:
    assert hnsw.session({"rerank": 8, "ef_search": 100}) == [("hnsw.ef_search", "100")]
    assert hnsw.session({"rerank": 8}) == []


def test_the_search_options_are_bounded_as_pgvector_bounds_them() -> None:
    for wrong in ({"ef_search": 0}, {"ef_search": 1001}, {"rerank": 0}, {"ef_search": "40"}):
        with pytest.raises(ValidationError):
            HNSW.model_validate(wrong)
    assert HNSW(ef_search=1000, rerank=1).ef_search == 1000


def test_a_search_reads_on_past_what_its_filters_hide() -> None:
    assert hnsw.SEARCH == "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
