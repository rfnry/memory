from rfnry_memory.index import hnsw
from rfnry_memory.models import VectorIndex


def test_the_clause_names_the_settings_the_precision_and_the_kind() -> None:
    full = VectorIndex(m=16, ef_construction=64, precision="full")
    half = VectorIndex(m=32, ef_construction=128, precision="half")
    assert hnsw.clause("chunk", full, 8) == (
        "USING hnsw (vector vector_cosine_ops) WITH (m = 16, ef_construction = 64) "
        "WHERE passage_kind = 'chunk'"
    )
    assert hnsw.clause("story", half, 8) == (
        "USING hnsw ((vector::halfvec(8)) halfvec_cosine_ops) WITH (m = 32, ef_construction = 128) "
        "WHERE passage_kind = 'story'"
    )


def test_the_settings_read_back_from_an_index_s_options() -> None:
    assert hnsw.read(["m=32", "ef_construction=128"], True) == VectorIndex(
        m=32, ef_construction=128, precision="half"
    )
    assert hnsw.read(["ef_construction=64", "m=16"], False) == VectorIndex(
        m=16, ef_construction=64, precision="full"
    )


def test_a_search_reads_on_past_what_its_filters_hide() -> None:
    assert hnsw.SEARCH == "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
