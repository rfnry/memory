from rfnry_memory.index import IVFFlat, ivfflat


def test_the_clause_names_the_lists_the_precision_and_the_kind() -> None:
    assert ivfflat.clause("chunk", IVFFlat(lists=200, precision="full"), 8) == (
        "USING ivfflat (vector vector_cosine_ops) WITH (lists = 200) WHERE passage_kind = 'chunk'"
    )
    assert ivfflat.clause("story", IVFFlat(precision="half"), 8) == (
        "USING ivfflat ((vector::halfvec(8)) halfvec_cosine_ops) WHERE passage_kind = 'story'"
    )


def test_the_settings_read_back_from_an_index_s_options() -> None:
    assert ivfflat.read(["lists=200"], True) == IVFFlat(lists=200, precision="half")
    assert ivfflat.read([], False) == IVFFlat(precision="full")


def test_a_search_reads_on_past_what_its_filters_hide() -> None:
    assert ivfflat.SEARCH == "SELECT set_config('ivfflat.iterative_scan', 'relaxed_order', true)"
