import asyncio
import itertools
import os
from datetime import UTC, datetime, timedelta

from database import URL, Schemas, provision

from rfnry_memory import cli
from rfnry_memory.memory import Memory, _window
from rfnry_memory.models import Kind
from rfnry_memory.postgres import PostgresStorage
from rfnry_memory.ranking import BM25

WORDS = ("alpha", "bravo", "charlie", "delta")
SAME = "alpha bravo bravo charlie charlie charlie delta delta delta delta"
KINDS: tuple[Kind, ...] = ("chunk", "story")
PLANS = (("on", "on"), ("off", "on"), ("on", "off"), ("off", "off"))


async def _read(
    schemas: Schemas, name: str, kind: Kind, corpora: list[str], plan: tuple[str, str]
) -> list[tuple[str, str, float]]:
    hashagg, hashjoin = plan
    pool = await schemas.pool(name, enable_hashagg=hashagg, enable_hashjoin=hashjoin)
    storage = PostgresStorage(pool)
    await storage.check(dimension=8, graph=True)
    window = _window(None, None, None, False, now=datetime.now(UTC) + timedelta(hours=1))
    found = await storage.search_text(
        corpora,
        " ".join(WORDS),
        kinds=(kind,),
        top_k=500,
        filters=None,
        window=window,
        subject=None,
        ranking=BM25(),
        match="any",
    )
    return [(corpus, passage_id, score) for corpus, (_, passage_id, score) in found]


async def test_passages_that_score_the_same_score_exactly_the_same_whatever_the_plan(
    schemas: Schemas,
) -> None:
    name = await schemas.create()
    memory = Memory(await provision(await schemas.pool(name)), pillars={"text"})
    for corpus in ("acme", "beta"):
        for n, counts in enumerate(itertools.permutations((1, 2, 3, 4))):
            text = " ".join(
                word for word, count in zip(WORDS, counts, strict=True) for _ in range(count)
            )
            await memory.remember(corpus, text, subject=f"perm{n:02d}")
            await memory.ingest(corpus, name=f"perm{n:02d}", text=text)
        for n in range(6):
            await memory.remember(corpus, SAME, subject=f"same{n}")
            await memory.ingest(corpus, name=f"same{n}", text=SAME)
    reads: dict[tuple[str, Kind, str], list[tuple[str, str, float]]] = {}
    for phase in ("one part", "two parts"):
        if phase == "two parts":
            argv = ["index", "--url", os.environ[URL], "--schema", name, "--current", "--yes"]
            assert await asyncio.to_thread(cli.main, argv) == 0
        for kind in KINDS:
            for corpora in (["acme"], ["acme", "beta"]):
                for plan in PLANS:
                    found = await _read(schemas, name, kind, corpora, plan)
                    assert len(found) == 30 * len(corpora)
                    assert len({score for *_, score in found}) == 1
                    assert [(i, c) for c, i, _ in found] == sorted((i, c) for c, i, _ in found)
                    key = (phase, kind, "+".join(corpora))
                    assert found == reads.setdefault(key, found)
    for kind in KINDS:
        for named in ("acme", "acme+beta"):
            assert reads[("one part", kind, named)] == reads[("two parts", kind, named)]
