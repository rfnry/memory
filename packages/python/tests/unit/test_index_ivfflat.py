import pytest
from pydantic import ValidationError

from rfnry_memory.index import IVFFlat, ivfflat


def test_the_clause_names_the_lists_the_precision_and_the_kind() -> None:
    assert ivfflat.clause("chunk", IVFFlat(lists=200, precision="full"), 8) == (
        "USING ivfflat (vector vector_cosine_ops) WITH (lists = 200) WHERE passage_kind = 'chunk'"
    )
    assert ivfflat.clause("story", IVFFlat(precision="half"), 8) == (
        "USING ivfflat ((vector::halfvec(8)) halfvec_cosine_ops) WHERE passage_kind = 'story'"
    )
    assert ivfflat.clause("chunk", IVFFlat(lists=10, precision="binary", probes=3), 8) == (
        "USING ivfflat ((binary_quantize(vector)::bit(8)) bit_hamming_ops) WITH (lists = 10) "
        "WHERE passage_kind = 'chunk'"
    )


def test_the_settings_read_back_from_an_index_s_options_and_its_search_options() -> None:
    assert ivfflat.read(["lists=200"], "half", {}) == IVFFlat(lists=200, precision="half")
    assert ivfflat.read([], "full", {}) == IVFFlat(precision="full")
    assert ivfflat.read(["lists=10"], "binary", {"rerank": 2, "probes": 3}) == IVFFlat(
        lists=10, precision="binary", rerank=2, probes=3
    )


def test_a_search_sets_only_the_options_the_index_was_given() -> None:
    assert ivfflat.session({"probes": 3}) == [("ivfflat.probes", "3")]
    assert ivfflat.session({"rerank": 2}) == []


def test_probes_read_at_most_the_lists_the_index_has() -> None:
    with pytest.raises(ValidationError, match="index: probes reads at most as many lists"):
        IVFFlat(lists=10, probes=11)
    for wrong in ({"probes": 0}, {"rerank": 0}, {"probes": "3"}):
        with pytest.raises(ValidationError):
            IVFFlat.model_validate(wrong)
    assert IVFFlat(probes=50).probes == 50


def test_a_search_reads_on_past_what_its_filters_hide() -> None:
    assert ivfflat.SEARCH == "SELECT set_config('ivfflat.iterative_scan', 'relaxed_order', true)"
