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


def test_an_option_left_out_is_left_to_pgvector() -> None:
    assert hnsw.clause("chunk", HNSW(precision="full"), 8) == (
        "USING hnsw (vector vector_cosine_ops) WHERE passage_kind = 'chunk'"
    )


def test_the_settings_read_back_from_an_index_s_options() -> None:
    assert hnsw.read(["m=32", "ef_construction=128"], True) == HNSW(
        m=32, ef_construction=128, precision="half"
    )
    assert hnsw.read(["ef_construction=64"], False) == HNSW(ef_construction=64, precision="full")
    assert hnsw.read([], False) == HNSW(precision="full")


def test_a_search_reads_on_past_what_its_filters_hide() -> None:
    assert hnsw.SEARCH == "SELECT set_config('hnsw.iterative_scan', 'relaxed_order', true)"
