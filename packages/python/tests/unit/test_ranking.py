import pytest
from pydantic import TypeAdapter, ValidationError

from rfnry_memory import BM25, CoverDensity, Memory, TextRanking
from rfnry_memory.errors import ConfigurationError, InputError
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import cover_density

_FILLER = " ".join(f"note{i} about the line" for i in range(60))


async def _texts(memory: Memory, query: str, **options: object) -> list[str]:
    hits = await memory.search("a", query, kinds=("chunk",), pillars={"text"}, **options)  # type: ignore[arg-type]
    return [hit.text for hit in hits.hits]


async def _scores(memory: Memory, query: str, **options: object) -> dict[str, float]:
    hits = await memory.search("a", query, kinds=("chunk",), pillars={"text"}, **options)  # type: ignore[arg-type]
    return {hit.text: hit.provenance["text"] for hit in hits.hits}


async def test_the_default_text_ranking_is_bm25_at_the_standard_baseline(
    storage: PostgresStorage,
) -> None:
    memory = Memory(storage)
    assert memory.text == BM25(k1=1.2, b=0.75)
    assert repr(memory.text) == "BM25(method='bm25', k1=1.2, b=0.75)"
    assert repr(CoverDensity()) == "CoverDensity(method='cover_density', length='ignore')"


def test_text_ranking_settings_refuse_what_they_cannot_mean() -> None:
    for bad in (
        {"k1": -0.1},
        {"b": 1.5},
        {"b": -0.1},
        {"k1": float("inf")},
        {"k1": "1.2"},
        {"k1": True},
        {"boost": 2},
    ):
        with pytest.raises(ValidationError):
            BM25(**bad)
    with pytest.raises(ValidationError):
        CoverDensity(length="half")  # type: ignore[arg-type]
    with pytest.raises(ValidationError):
        BM25().k1 = 2.0


def test_a_text_ranking_round_trips_through_json() -> None:
    adapter: TypeAdapter[TextRanking] = TypeAdapter(TextRanking)
    for setting in (BM25(k1=0.9, b=0.4), CoverDensity(length="log")):
        assert adapter.validate_json(setting.model_dump_json()) == setting


async def test_memory_refuses_a_text_ranking_it_does_not_know(storage: PostgresStorage) -> None:
    with pytest.raises(
        ConfigurationError, match=r"^memory: text is BM25\(\.\.\.\) or CoverDensity\(\.\.\.\)$"
    ):
        Memory(storage, text={"k1": 1.2})  # type: ignore[arg-type]
    memory = Memory(storage)
    with pytest.raises(
        InputError, match=r"^memory: text is BM25\(\.\.\.\) or CoverDensity\(\.\.\.\)$"
    ):
        await memory.search("a", "valve", text="bm25")  # type: ignore[arg-type]


async def test_b_sets_how_much_a_long_passage_is_held_back(storage: PostgresStorage) -> None:
    short, long = "Check the valve.", f"Check the valve. {_FILLER}"
    for name, text in (("short", short), ("long", long)):
        await Memory(storage).ingest("a", name=name, text=text)
    flat = await _scores(Memory(storage, text=BM25(b=0)), "valve")
    assert flat[short] == flat[long]
    full = Memory(storage, text=BM25(b=1))
    assert await _texts(full, "valve") == [short, long]
    scaled = await _scores(full, "valve")
    assert scaled[long] < scaled[short]


async def test_k1_sets_how_fast_repeating_a_word_stops_helping(storage: PostgresStorage) -> None:
    once, thrice = "valve gasket pump", "valve valve valve"
    for name, text in (("once", once), ("thrice", thrice)):
        await Memory(storage).ingest("a", name=name, text=text)
    presence = await _scores(Memory(storage, text=BM25(k1=0, b=0)), "valve")
    assert presence[once] == presence[thrice]
    counted = Memory(storage, text=BM25(k1=1.2, b=0))
    assert await _texts(counted, "valve") == [thrice, once]


async def test_cover_density_ranks_words_that_sit_together_first(storage: PostgresStorage) -> None:
    apart = "pump alpha bravo charlie delta echo foxtrot golf hotel india seal"
    together = "alpha bravo charlie delta pump seal echo foxtrot golf hotel india"
    alone = "alpha bravo pump charlie"
    for name, text in (("apart", apart), ("together", together), ("alone", alone)):
        await Memory(storage).ingest("a", name=name, text=text)
    ranked = await _texts(Memory(storage, text=CoverDensity()), "pump seal")
    assert ranked == [together, apart, alone]


async def test_a_call_can_rank_text_its_own_way_and_leaves_the_memory_as_it_was(
    storage: PostgresStorage,
) -> None:
    apart = "pump alpha bravo charlie delta echo foxtrot golf hotel india seal"
    together = "alpha bravo charlie delta pump seal echo foxtrot golf hotel india"
    for name, text in (("apart", apart), ("together", together)):
        await Memory(storage).ingest("a", name=name, text=text)
    memory = Memory(storage, text=BM25(b=0))
    assert await _texts(memory, "pump seal", text=CoverDensity()) == [together, apart]
    assert memory.text == BM25(b=0)
    recalled = await memory.recall("a", "pump seal", text=CoverDensity())
    assert recalled.hits == []


async def test_the_fingerprint_reports_the_text_ranking(storage: PostgresStorage) -> None:
    await Memory(storage).ingest("a", name="manual", text="Check the valve.")
    assert (await Memory(storage).fingerprint("a")).text == BM25()
    chosen = Memory(storage, text=CoverDensity(length="log"))
    assert (await chosen.fingerprint("a")).text == CoverDensity(length="log")


def test_cover_density_names_postgres_s_length_normalization() -> None:
    for length, flag in (("ignore", 0), ("log", 1), ("linear", 2)):
        sql = cover_density.sql(
            CoverDensity(length=length),  # type: ignore[arg-type]
            table="memory_chunks",
            id_column="chunk_id",
            matches="M",
            ranks="R",
            terms="T",
            where="",
            limit="$9",
        )
        assert f"ts_rank_cd(fts, R, {flag})" in sql and "fts @@ M" in sql


async def test_cover_density_reads_inside_filters_and_a_scope(storage: PostgresStorage) -> None:
    memory = Memory(storage, text=CoverDensity())
    first = await memory.ingest(
        "a", name="line a", text="pump seal on line a", metadata={"line": "A"}
    )
    await memory.ingest("a", name="line b", text="pump seal on line b", metadata={"line": "B"})
    filtered = await memory.search(
        "a", "pump seal", kinds=("chunk",), pillars={"text"}, filters={"metadata.line": "B"}
    )
    assert [hit.text for hit in filtered.hits] == ["pump seal on line b"]
    scoped = await memory.search(
        "a", "pump seal", kinds=("chunk",), pillars={"text"}, sources=[first.source.source_id]
    )
    assert [hit.text for hit in scoped.hits] == ["pump seal on line a"]


async def test_cover_density_orders_partial_matches_by_how_often_they_hold_the_words(
    storage: PostgresStorage,
) -> None:
    full = "alpha pump seal bravo"
    partials = [" ".join(["seal"] * n + ["charlie"]) for n in (1, 2, 3, 4)]
    for i, text in enumerate([full, *partials]):
        await Memory(storage).ingest("a", name=f"p{i}", text=text)
    ranked = await _texts(Memory(storage, text=CoverDensity()), "pump seal")
    assert ranked == [full, partials[3], partials[2], partials[1], partials[0]]
