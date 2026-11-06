import argparse
import asyncio
import json
import os
import signal
import sys
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import pytest
from database import URL, WAIT, Database, Ran, Schemas, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory import cli, cli_vectors
from rfnry_memory.events import Embedded
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import Section
from rfnry_memory.postgres import PostgresStorage, migrations
from rfnry_memory.postgres.switch import CorpusVectors, vector_status

NEW = "new-model"

MODEL = """
import asyncio
import os
import sys
from hashlib import sha256


class New:
    model = "new-model"
    dimension = 8

    async def embed(self, texts, *, purpose="document"):
        return [[b / 255 for b in sha256(t.encode()).digest()[:8]] for t in texts]
"""


class New(HashEmbedding):
    model = NEW


def _adapter(tmp_path: Path, monkeypatch: pytest.MonkeyPatch, body: str = "") -> str:
    name = f"adapter_{uuid4().hex[:8]}"
    (tmp_path / f"{name}.py").write_text(MODEL + body)
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(sys, "path", [*sys.path])
    return name


@pytest.fixture
def schema(database: Database, storage: PostgresStorage) -> str:
    return database.run


async def _vectors(action: str, schema: str, *argv: str) -> Ran:
    return await run_cli("vectors", action, "--schema", schema, *argv)


async def _filled(storage: PostgresStorage, corpus_id: str = "acme") -> Memory:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.ingest(
        corpus_id,
        name="manual",
        sections=[Section(text=f"Line {i} of {corpus_id} runs.") for i in range(3)],
        metadata={"line": "A", "year": 2021},
    )
    await memory.ingest(
        corpus_id,
        name="sheet",
        text=f"Valve notes of {corpus_id}.",
        metadata={"line": "B", "year": 2019},
    )
    await memory.remember(corpus_id, f"The pump of {corpus_id} runs.", metadata={"line": "A"})
    await memory.remember(corpus_id, f"The valve of {corpus_id} leaks.", metadata={"line": "B"})
    return memory


def _lines(text: str) -> list[str]:
    return text.splitlines()


async def test_status_shows_each_model_on_the_row_and_staged(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    await Memory(storage, embedding=New()).stage_vectors("acme", limit=4)
    assert await _vectors("status", schema) == 0
    assert _lines(capsys.readouterr().out)[:5] == [
        f"schema {schema}: vectors of 8 dimensions",
        "",
        "corpus  passages  on the row  staged       index",
        "acme           6  hash-8 6    new-model 4  none",
        "hr             6  hash-8 6    none         none",
    ]


def test_the_table_lines_up_at_millions() -> None:
    rows = [
        CorpusVectors(
            corpus_id="acme",
            passages=1204331,
            on_row={"old-model": 1204331},
            staged={NEW: 804000},
            staged_dimensions={NEW: 1024},
            missing={NEW: 400331, "old-model": 0},
            index="HNSW(m=32, ef_construction=128)",
            to_build=None,
        ),
        CorpusVectors(
            corpus_id="hr",
            passages=41200,
            on_row={"old-model": 41200},
            staged={NEW: 41200},
            staged_dimensions={NEW: 1024},
            missing={NEW: 0, "old-model": 0},
            index=None,
            to_build=None,
        ),
    ]
    assert cli_vectors._table(rows) == [
        "corpus  passages   on the row           staged               index",
        "acme    1,204,331  old-model 1,204,331  new-model   804,000  HNSW(m=32, ef_construction=128)",
        "hr         41,200  old-model    41,200  new-model    41,200  none",
    ]


async def test_status_says_what_to_run_next(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    new = Memory(storage, embedding=New())
    await new.stage_vectors("hr")
    await new.stage_vectors("acme", limit=4)
    assert await _vectors("status", schema) == 0
    assert _lines(capsys.readouterr().out)[-2:] == [
        "",
        f"new-model: 2 missing, in acme — run rfnry-memory vectors stage "
        f"--embedding MODULE:NAME --corpus acme --schema {schema}",
    ]
    await new.stage_vectors("acme")
    assert await _vectors("status", schema) == 0
    assert _lines(capsys.readouterr().out)[-1] == (
        f"new-model: nothing missing — run rfnry-memory vectors swap --model new-model "
        f"--schema {schema}"
    )


async def test_status_with_nothing_switching_says_nothing_is_to_run(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _vectors("status", schema) == 0
    assert _lines(capsys.readouterr().out)[-1] == "hash-8: on every row, nothing to run"


async def test_status_of_an_empty_schema_says_so(
    schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await _vectors("status", schema) == 0
    assert _lines(capsys.readouterr().out) == [
        f"schema {schema}: vectors of 8 dimensions",
        "no corpora",
    ]


async def test_status_stays_in_the_corpora_it_names(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    await Memory(storage, embedding=New()).stage_vectors("acme")
    assert await _vectors("status", schema, "--corpus", "acme") == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[3].startswith("acme ") and lines[4] == ""
    assert lines[5].endswith(f"swap --model new-model --corpus acme --schema {schema}")


def test_status_names_a_few_corpora_and_counts_many() -> None:
    rows = [
        CorpusVectors(
            corpus_id=f"c{i}",
            passages=2,
            on_row={},
            staged={},
            staged_dimensions={},
            missing={NEW: 2},
            index=None,
            to_build=None,
        )
        for i in range(6)
    ]
    args = argparse.Namespace(schema="memory", corpus=None, model=NEW)
    assert cli_vectors._next_steps(args, rows, 8) == [
        "new-model: 12 missing, in 6 corpora — run rfnry-memory vectors stage "
        "--embedding MODULE:NAME"
    ]
    assert cli_vectors._next_steps(args, rows[:2], 8) == [
        "new-model: 4 missing, in c0, c1 — run rfnry-memory vectors stage "
        "--embedding MODULE:NAME --corpus c0 --corpus c1"
    ]


async def test_status_with_a_model_exits_0_only_when_nothing_is_missing(
    storage: PostgresStorage, schema: str
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    new = Memory(storage, embedding=New())
    assert await _vectors("status", schema, "--model", NEW) == 1
    await new.stage_vectors("acme")
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert await _vectors("status", schema, "--model", NEW, "--corpus", "acme") == 0
    await new.stage_vectors("hr", limit=5)
    assert await _vectors("status", schema, "--model", NEW) == 1
    await new.stage_vectors("hr")
    assert await _vectors("status", schema, "--model", NEW) == 0
    assert await _vectors("status", schema, "--model", "hash-8") == 0
    assert await _vectors("status", schema, "--model", "never-seen") == 1


async def test_status_counts_a_passage_held_on_the_row_and_staged_once(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    new = Memory(storage, embedding=New())
    await new.stage_vectors("acme", limit=3)
    await new.reembed("acme", limit=3)
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 3 missing, in acme" in capsys.readouterr().out


async def test_status_json_holds_the_same_facts(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    memory = await _filled(storage, "acme")
    await _filled(storage, "hr")
    await memory.set_vector_index("acme", HNSW(m=32, ef_construction=128))
    await Memory(storage, embedding=New()).stage_vectors("acme", limit=4)
    assert await _vectors("status", schema, "--json", "--model", NEW) == 1
    assert json.loads(capsys.readouterr().out) == {
        "schema": schema,
        "dimension": 8,
        "missing": {"hash-8": 0, NEW: 8},
        "corpora": [
            {
                "corpus_id": "acme",
                "passages": 6,
                "on_row": {"hash-8": 6},
                "staged": {NEW: 4},
                "staged_dimensions": {NEW: 8},
                "missing": {"hash-8": 0, NEW: 2},
                "index": "HNSW(m=32, ef_construction=128)",
                "to_build": None,
            },
            {
                "corpus_id": "hr",
                "passages": 6,
                "on_row": {"hash-8": 6},
                "staged": {},
                "staged_dimensions": {},
                "missing": {"hash-8": 0, NEW: 6},
                "index": None,
                "to_build": None,
            },
        ],
    }


async def test_status_says_a_model_of_another_dimension_covers_the_whole_schema(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    adapter = _adapter(
        tmp_path,
        monkeypatch,
        """
class Wide:
    model = "wide-model"
    dimension = 16

    async def embed(self, texts, *, purpose="document"):
        return [[0.5] * 16 for _ in texts]
""",
    )
    assert (
        await _vectors("stage", schema, "--embedding", f"{adapter}:Wide", "--corpus", "acme") == 0
    )
    capsys.readouterr()
    assert await _vectors("status", schema, "--corpus", "acme") == 0
    out = capsys.readouterr().out
    assert (
        "wide-model: staged at 16 dimensions, the schema holds 8 — its swap covers the whole schema"
    ) in out
    assert (
        f"wide-model: nothing missing — run rfnry-memory vectors swap --model wide-model --schema {schema}\n"
        in out
    )


async def test_status_shows_the_index_a_swap_left_to_build(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_corpora SET vector_index = $1::text::jsonb WHERE corpus_id = 'acme'",
            json.dumps({"settings": HNSW(m=32).model_dump(), "owner": "app"}),
        )
    assert await _vectors("status", schema) == 0
    out = capsys.readouterr().out
    assert "index to build: HNSW(m=32)" in out
    assert (
        f"index to build, in acme — run rfnry-memory vectors index --corpus acme --schema {schema}"
    ) in out


async def test_every_action_refuses_a_corpus_the_schema_does_not_hold(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    refused = f"cli: schema {schema} holds no corpus 'typo'\n"
    assert await _vectors("status", schema, "--model", "hash-8", "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New", "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused
    assert await storage.staged_counts("acme") == {}


async def test_a_schema_behind_this_version_is_told_to_upgrade(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    now = migrations.LATEST
    adapter = _adapter(tmp_path, monkeypatch)
    monkeypatch.setattr(migrations, "LATEST", now + 1)
    wanted = (
        f"cli: {schema} is at version {now}, this rfnry-memory needs {now + 1} — "
        f"run rfnry-memory upgrade --schema {schema}\n"
    )
    assert await _vectors("status", schema) == 1
    assert capsys.readouterr().err == wanted
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 1
    assert capsys.readouterr().err == wanted


async def test_a_schema_without_vectors_says_what_gives_it_some(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name = await schemas.create()
    await provision(await schemas.pool(name), dimension=None)
    assert await _vectors("status", name) == 1
    assert capsys.readouterr().err == (
        f"cli: schema {name} holds no vectors — run rfnry-memory set --dimension N --schema {name}\n"
    )


async def test_stage_fills_every_corpus_from_a_module_path(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    adapter = _adapter(tmp_path, monkeypatch)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New", "--batch", "2") == 0
    printed = capsys.readouterr()
    assert await storage.staged_counts("acme") == {NEW: 6}
    assert await storage.staged_counts("hr") == {NEW: 6}
    assert _lines(printed.out) == [
        "acme: 6 vectors of new-model staged",
        "hr: 6 vectors of new-model staged",
        f"next: rfnry-memory vectors status --model new-model --schema {schema}",
    ]
    assert "acme: 2 of 6, " in printed.err and "/s, " in printed.err and " left\n" in printed.err
    assert "hr: 6 of 6, " in printed.err
    assert await _vectors("status", schema, "--model", NEW) == 0


async def test_stage_with_nothing_missing_says_so(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 0
    capsys.readouterr()
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 0
    assert "acme: nothing to stage for new-model\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("body", "attribute"),
    [
        ("held = New()\n", "held"),
        ("def made():\n    return New()\n", "made"),
        ("async def made():\n    return New()\n", "made"),
        (
            "class Called(New):\n    def __call__(self):\n        raise OSError\n\n\nheld = Called()\n",
            "held",
        ),
    ],
    ids=["an instance", "a factory", "an async factory", "a callable model"],
)
async def test_stage_takes_a_factory_plain_or_async(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    body: str,
    attribute: str,
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, body)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:{attribute}") == 0
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_stage_takes_filters_and_a_limit(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    adapter = _adapter(tmp_path, monkeypatch)
    embedding = ("--embedding", f"{adapter}:New")
    line = json.dumps({"metadata.line": "A"})
    assert await _vectors("stage", schema, *embedding, "--filters", line, "--corpus", "acme") == 0
    assert await storage.staged_counts("acme") == {NEW: 4}
    assert await storage.staged_counts("hr") == {}
    ranged = json.dumps({"metadata.year": {"lt": 2020}})
    assert await _vectors("stage", schema, *embedding, "--filters", ranged, "--corpus", "acme") == 0
    assert await storage.staged_counts("acme") == {NEW: 5}
    capsys.readouterr()
    assert await _vectors("stage", schema, *embedding, "--limit", "3") == 0
    assert await storage.staged_counts("acme") == {NEW: 6}
    assert await storage.staged_counts("hr") == {NEW: 2}
    assert "limit of 3 reached; running it again continues\n" in capsys.readouterr().out


@pytest.mark.parametrize(
    ("given", "wanted"),
    [
        ("{", "cli: --filters is not JSON: "),
        ("[1]", "cli: --filters is one JSON object\n"),
        ('{"metadata.line": {"near": 1}}', "filters: filter 'metadata.line' is a range of "),
    ],
    ids=["not json", "not an object", "not the grammar"],
)
async def test_bad_filters_stop_before_anything_is_embedded(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    given: str,
    wanted: str,
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    embedding = ("--embedding", f"{adapter}:New")
    capsys.readouterr()
    assert await _vectors("stage", schema, *embedding, "--filters", given) == 1
    printed = capsys.readouterr().err
    assert printed.startswith(wanted), printed
    assert "again" not in printed
    assert await storage.staged_counts("acme") == {}


async def test_a_batch_or_a_limit_below_one_says_so(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    embedding = ("--embedding", f"{adapter}:New")
    assert await _vectors("stage", schema, *embedding, "--batch", "0") == 1
    assert capsys.readouterr().err.endswith("memory: batch must be at least 1\n")
    assert await _vectors("stage", schema, *embedding, "--limit", "0") == 1
    assert capsys.readouterr().err.endswith("memory: limit must be at least 1\n")


REFUSING = """
class Picky(New):
    async def embed(self, texts, *, purpose="document"):
        if any("refuse" in t for t in texts):
            raise ValueError("too long")
        return await super().embed(texts, purpose=purpose)
"""


async def test_stage_lists_refused_passages_and_exits_1(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = await _filled(storage)
    told = await memory.remember("acme", "Please refuse this one.")
    adapter = _adapter(tmp_path, monkeypatch, REFUSING)
    ran = await _vectors("stage", schema, "--embedding", f"{adapter}:Picky")
    assert ran == 1
    printed = capsys.readouterr()
    assert "acme: 6 vectors of new-model staged\n" in printed.out
    assert printed.err.endswith(
        "cli: new-model refused 1 passage: missing until a later run embeds them\n"
        f"  acme story {told.story.story_id}\n"
    )
    assert await storage.staged_counts("acme") == {NEW: 6}
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 1 missing, in acme" in capsys.readouterr().out


async def test_many_refused_passages_stay_readable_and_a_file_holds_them_all(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    told = [await memory.remember("acme", f"Please refuse number {i}.") for i in range(25)]
    adapter = _adapter(tmp_path, monkeypatch, REFUSING)
    embedding = ("--embedding", f"{adapter}:Picky")
    assert await _vectors("stage", schema, *embedding) == 1
    printed = _lines(capsys.readouterr().err)
    at = printed.index("cli: new-model refused 25 passages: missing until a later run embeds them")
    assert len(printed[at + 1 :]) == 21
    assert printed[-1] == "  and 5 more — add --refused FILE to write every one down"
    kept = tmp_path / "refused.jsonl"
    assert await _vectors("stage", schema, *embedding, "--refused", str(kept)) == 1
    assert capsys.readouterr().err.endswith(f"  all of them are in {kept}\n")
    written = [json.loads(line) for line in kept.read_text().splitlines()]
    assert sorted(r["id"] for r in written) == sorted(t.story.story_id for t in told)
    assert {(r["corpus_id"], r["kind"]) for r in written} == {("acme", "story")}


def _failing_after(calls: int, error: str) -> str:
    return f"""
class Failing(New):
    calls = 0

    async def embed(self, texts, *, purpose="document"):
        Failing.calls += 1
        if Failing.calls > {calls}:
            {error}
        return await super().embed(texts, purpose=purpose)
"""


async def test_an_outage_is_one_cli_line_that_keeps_what_was_written_and_no_url(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    url = os.environ[URL]
    outage = _failing_after(
        1, "raise ConnectionError(f\"vendor down\\nwhile {os.environ['RFNRY_TEST_POSTGRES_URL']}\")"
    )
    adapter = _adapter(tmp_path, monkeypatch, outage)
    embedding = ("--embedding", f"{adapter}:Failing")
    assert await _vectors("stage", schema, *embedding, "--batch", "2") == 1
    printed = capsys.readouterr()
    assert url not in printed.out + printed.err
    assert "Traceback" not in printed.err
    assert _lines(printed.err)[-1] == (
        "cli: stopped by ConnectionError: vendor down while <url> — 2 vectors written and "
        "kept; running it again continues"
    )
    assert await storage.staged_counts("acme") == {NEW: 2}
    healthy = _adapter(tmp_path, monkeypatch)
    assert await _vectors("stage", schema, "--embedding", f"{healthy}:New") == 0
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_an_error_with_no_text_is_still_named(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(0, "raise TimeoutError"))
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:Failing") == 1
    assert _lines(capsys.readouterr().err)[-1] == (
        "cli: stopped by TimeoutError — 0 vectors written and kept; running it again continues"
    )


async def test_a_long_vendor_error_is_cut_only_after_the_url_is_gone(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    url = os.environ[URL]
    cut = cli_vectors._SAID - len(url) // 2
    outage = _failing_after(
        0, f"raise ConnectionError('x' * {cut} + os.environ['RFNRY_TEST_POSTGRES_URL'])"
    )
    adapter = _adapter(tmp_path, monkeypatch, outage)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:Failing") == 1
    printed = capsys.readouterr().err
    assert url[: len(url) // 3] not in printed


async def test_a_model_that_breaks_its_contract_stops_the_run_the_same_way(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    broken = """
class Short(New):
    async def embed(self, texts, *, purpose="document"):
        return [[0.5] * 3 for _ in texts]
"""
    adapter = _adapter(tmp_path, monkeypatch, broken)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:Short") == 1
    printed = capsys.readouterr().err
    assert "cli: stopped by ModelError: memory: the embedding model returned" in printed


async def test_an_interrupt_keeps_what_was_written_and_exits_130(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(1, "raise KeyboardInterrupt"))
    assert (
        await _vectors("stage", schema, "--embedding", f"{adapter}:Failing", "--batch", "2") == 130
    )
    assert _lines(capsys.readouterr().err)[-1] == (
        "cli: stopped — 2 vectors written and kept; running it again continues"
    )
    assert await storage.staged_counts("acme") == {NEW: 2}


async def test_a_signal_stops_a_running_stage_the_same_way(
    storage: PostgresStorage, schema: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(1, "await asyncio.sleep(3600)"))
    process = await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "rfnry_memory.cli",
        "vectors",
        "stage",
        "--url",
        os.environ[URL],
        "--schema",
        schema,
        "--embedding",
        f"{adapter}:Failing",
        "--batch",
        "2",
        cwd=tmp_path,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )
    assert process.stderr is not None
    async with asyncio.timeout(WAIT):
        while b"acme: 2 of 6" not in await process.stderr.readline():
            pass
        process.send_signal(signal.SIGINT)
        _, left = await process.communicate()
    assert process.returncode == 130
    assert b"Traceback" not in left
    assert "2 vectors written and kept; running it again continues" in left.decode()
    assert await storage.staged_counts("acme") == {NEW: 2}


def _with_a_timeout() -> str:
    parts = urlsplit(os.environ[URL])
    query = "&".join(filter(None, [parts.query, "statement_timeout=200"]))
    return urlunsplit(parts._replace(query=query))


async def _held_past_the_timeout(pool: Any, table: str, argv: list[str]) -> int:
    async with pool.acquire() as conn, conn.transaction():
        await conn.execute(f"LOCK TABLE {table} IN ACCESS EXCLUSIVE MODE")
        ran = asyncio.create_task(asyncio.to_thread(cli.main, argv))
        await asyncio.sleep(0.6)
    return await ran


async def test_a_default_statement_timeout_stops_neither_status_nor_stage(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    url = _with_a_timeout()
    status = ["vectors", "status", "--url", url, "--schema", schema]
    assert await _held_past_the_timeout(pool, "memory_chunks", status) == 0
    stage = ["vectors", "stage", "--url", url, "--schema", schema]
    stage += ["--embedding", f"{adapter}:New"]
    assert await _held_past_the_timeout(pool, "memory_staged_vectors", stage) == 0
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_without_a_url_stage_connects_with_the_pg_variables(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    parts = urlsplit(os.environ[URL])
    for name, value in (
        ("PGHOST", parts.hostname),
        ("PGPORT", parts.port),
        ("PGUSER", parts.username),
        ("PGPASSWORD", parts.password),
        ("PGDATABASE", parts.path.lstrip("/")),
    ):
        monkeypatch.setenv(name, str(value))
    argv = ["vectors", "stage", "--schema", schema, "--embedding", f"{adapter}:New"]
    assert await asyncio.to_thread(cli.main, argv) == 0
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_a_schema_name_that_needs_quoting_works_through_status_and_stage(
    schemas: Schemas,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    name = f"{schemas.run}_{uuid4().hex[:8]} Odd-Name"
    assert await run_cli("create", "--schema", name, "--dimension", "8") == 0
    quoted = '"' + name + '"'
    schemas._made.append(quoted)
    storage = PostgresStorage(await schemas.pool(quoted))
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    assert await _vectors("stage", name, "--embedding", f"{adapter}:New") == 0
    assert await storage.staged_counts("acme") == {NEW: 6}
    capsys.readouterr()
    assert await _vectors("status", name, "--model", NEW) == 0
    assert f"--schema '{name}'" in capsys.readouterr().out


IMPORT_FAILS = "\nraise RuntimeError(f\"cannot reach {os.environ['RFNRY_TEST_POSTGRES_URL']}\")\n"
EXITS = "\nsys.exit(f\"cannot reach {os.environ['RFNRY_TEST_POSTGRES_URL']}\")\n"
FACTORY_FAILS = """
def made():
    raise RuntimeError(f"cannot reach {os.environ['RFNRY_TEST_POSTGRES_URL']}")
"""
ASYNC_FACTORY_FAILS = """
async def made():
    raise RuntimeError(f"cannot reach {os.environ['RFNRY_TEST_POSTGRES_URL']}")
"""


@pytest.mark.parametrize(
    ("body", "path", "wanted"),
    [
        ("", "no_such_module_at_all:New", "cli: cannot import no_such_module_at_all: "),
        ("", "{adapter}:Missing", "cli: {adapter} has no attribute Missing\n"),
        ("", "{adapter}", "cli: --embedding is module:attribute; it was given '{adapter}'\n"),
        ("", ":New", "cli: --embedding is module:attribute; it was given ':New'\n"),
        ("", "{adapter}:", "cli: --embedding is module:attribute; it was given '{adapter}:'\n"),
        (
            "held = 42\n",
            "{adapter}:held",
            "cli: {adapter}:held is not an EmbeddingModel: it has model, dimension and embed\n",
        ),
        (
            "def made():\n    return 42\n",
            "{adapter}:made",
            "cli: {adapter}:made is not an EmbeddingModel: it has model, dimension and embed\n",
        ),
        (
            IMPORT_FAILS,
            "{adapter}:New",
            "cli: cannot import {adapter}: RuntimeError: cannot reach <url>\n",
        ),
        (EXITS, "{adapter}:New", "cli: cannot import {adapter}: SystemExit: cannot reach <url>\n"),
        (
            FACTORY_FAILS,
            "{adapter}:made",
            "cli: {adapter}:made raised RuntimeError: cannot reach <url>\n",
        ),
        (
            ASYNC_FACTORY_FAILS,
            "{adapter}:made",
            "cli: {adapter}:made raised RuntimeError: cannot reach <url>\n",
        ),
    ],
    ids=[
        "no module",
        "no attribute",
        "no colon",
        "no module name",
        "no attribute name",
        "not a model",
        "a factory of something else",
        "raises on import",
        "exits on import",
        "a factory that raises",
        "an async factory that raises",
    ],
)
async def test_a_wrong_embedding_path_is_a_cli_message_with_no_url(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    body: str,
    path: str,
    wanted: str,
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, body)
    ran = await _vectors("stage", schema, "--embedding", path.format(adapter=adapter))
    assert ran == 1
    printed = capsys.readouterr()
    assert printed.err.startswith(wanted.format(adapter=adapter)), printed.err
    assert "Traceback" not in printed.err
    assert os.environ[URL] not in printed.out + printed.err
    assert await storage.staged_counts("acme") == {}


def _embedded(done: int, total: int, computed: int, duration_ms: int) -> Embedded:
    return Embedded(
        corpus_id="acme",
        name=None,
        done=done,
        total=total,
        computed=computed,
        duration_ms=duration_ms,
    )


async def test_progress_prints_a_count_a_rate_and_an_estimate_every_few_seconds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    monkeypatch.setattr(cli_vectors, "_EVERY", 3600.0)
    progress = cli_vectors._Progress()
    await progress.show(_embedded(1000, 400000, 1000, 2000))
    await progress.show(_embedded(2000, 400000, 1000, 2000))
    monkeypatch.setattr(cli_vectors, "_EVERY", 0.0)
    await progress.show(_embedded(3000, 400000, 1000, 2000))
    monkeypatch.setattr(cli_vectors, "_EVERY", 3600.0)
    await progress.show(_embedded(400000, 400000, 397000, 794000))
    assert progress.written == 400000
    assert _lines(capsys.readouterr().err) == [
        "acme: 1,000 of 400,000, 500/s, about 13 min left",
        "acme: 3,000 of 400,000, 500/s, about 13 min left",
        "acme: 400,000 of 400,000, 500/s",
    ]


@pytest.mark.parametrize(
    ("seconds", "wanted"),
    [
        (20, "under a minute left"),
        (90, "about 2 min left"),
        (7199, "about 120 min left"),
        (7300, "about 2.0 h left"),
        (90000, "about 25.0 h left"),
    ],
)
def test_an_estimate_reads_in_the_unit_that_fits(seconds: float, wanted: str) -> None:
    assert cli_vectors._left(seconds) == wanted


async def test_vector_status_stays_in_the_corpora_it_is_given(
    storage: PostgresStorage, pool: Any
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    async with pool.acquire() as conn:
        everything = await vector_status(conn, None)
        named = await vector_status(conn, ["hr"], model="never-seen")
    assert [row.corpus_id for row in everything] == ["acme", "hr"]
    assert [row.corpus_id for row in named] == ["hr"]
    assert named[0].missing == {"hash-8": 0, "never-seen": 6}
