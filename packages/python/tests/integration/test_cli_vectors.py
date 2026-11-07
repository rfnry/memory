import argparse
import asyncio
import json
import os
import signal
import sys
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, NamedTuple
from urllib.parse import urlsplit, urlunsplit
from uuid import uuid4

import asyncpg
import pytest
from database import URL, WAIT, Database, Ran, Schemas, Spawned, provision, run_cli
from unit.fakes import HashEmbedding

from rfnry_memory import cli, cli_vectors
from rfnry_memory.errors import ConfigurationError, InputError, NotFoundError, StorageError
from rfnry_memory.events import Embedded
from rfnry_memory.index import HNSW
from rfnry_memory.memory import Memory
from rfnry_memory.models import Counts, Section
from rfnry_memory.postgres import PostgresStorage, migrations, switch
from rfnry_memory.postgres.switch import (
    CorpusVectors,
    SwapPlan,
    _staging_key,
    apply_swap,
    forget_staged,
    hold_staging,
    plan_swap,
    rename_model,
    vector_status,
)
from rfnry_memory.postgres.vectors import (
    _vector_indexes,
    _vectors_lock,
    _vectors_name,
    _vectors_table,
)

NEW = "new-model"
OLD = "hash-8"
WIDE = "wide-model"
RENAMED = "renamed-model"

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
        f"--schema {schema} to move to it, or rfnry-memory vectors forget --model new-model "
        f"--schema {schema} to drop its staged vectors"
    )


async def test_after_a_swap_status_offers_both_ways_and_guesses_neither(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await Memory(storage, embedding=New()).reembed("acme")
    await Memory(storage, embedding=HashEmbedding()).stage_vectors("acme")
    assert await _vectors("status", schema) == 0
    lines = _lines(capsys.readouterr().out)
    assert lines[3].split()[2:6] == [NEW, "6", "hash-8", "6"]
    assert lines[4:] == [
        "",
        f"hash-8: nothing missing — run rfnry-memory vectors swap --model hash-8 "
        f"--schema {schema} to move to it, or rfnry-memory vectors forget --model hash-8 "
        f"--schema {schema} to drop its staged vectors",
    ]


async def test_status_names_forget_for_what_stays_staged_of_a_model_on_every_row(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    assert await _swap(schema) == 0
    late = list(await _rows(pool))[:2]
    await storage.stage("acme", late, [[0.5] * 8] * 2, NEW, settings=None)
    capsys.readouterr()
    assert await _vectors("status", schema, "--model", NEW) == 0
    assert _lines(capsys.readouterr().out)[-1] == (
        f"new-model: on every row, 2 still staged — run rfnry-memory vectors forget "
        f"--model new-model --schema {schema} to drop its staged vectors"
    )
    assert await _vectors("status", schema, "--model", NEW, "--corpus", "acme") == 0
    assert _lines(capsys.readouterr().out)[-1] == (
        f"new-model: on every row, 2 still staged — run rfnry-memory vectors forget "
        f"--model new-model --corpus acme --schema {schema} to drop its staged vectors"
    )
    assert await _forget(schema, NEW) == 0
    capsys.readouterr()
    assert await _vectors("status", schema, "--model", NEW) == 0
    assert _lines(capsys.readouterr().out)[-1] == "new-model: on every row, nothing to run"


async def test_status_and_stage_read_retired_and_closed_passages_alike(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    source = await memory.ingest("acme", name="manual", text="The old manual.")
    await memory.retire_source("acme", source.source.source_id)
    told = await memory.remember("acme", "The pump runs.")
    await memory.end_story("acme", told.story.story_id)
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 2 missing, in acme" in capsys.readouterr().out
    assert await storage.count_unstaged("acme", model=NEW, filters=None) == 2
    adapter = _adapter(tmp_path, monkeypatch)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 0
    assert await _vectors("status", schema, "--model", NEW) == 0


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
    assert lines[5].startswith(
        f"new-model: nothing missing — run rfnry-memory vectors swap --model new-model "
        f"--corpus acme --schema {schema} to move to it, or rfnry-memory vectors forget "
        f"--model new-model --corpus acme --schema {schema} to drop"
    )


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
        "wide-model: nothing missing in the corpora named, the only ones checked — "
        f"run rfnry-memory vectors status --model wide-model --schema {schema} to check the whole "
        f"schema its swap needs, or rfnry-memory vectors forget --model wide-model --corpus acme "
        f"--schema {schema} to drop its staged vectors\n"
    ) in out
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:Wide") == 0
    capsys.readouterr()
    assert await _vectors("status", schema) == 0
    assert (
        f"wide-model: nothing missing — run rfnry-memory vectors swap --model wide-model "
        f"--schema {schema} to move to it, or rfnry-memory vectors forget --model wide-model "
        f"--schema {schema} to drop its staged vectors\n"
    ) in capsys.readouterr().out


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
    assert await _vectors("swap", schema, "--model", NEW, "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused
    assert await _vectors("index", schema, "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused
    assert await _vectors("forget", schema, "--model", OLD, "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused
    assert await _vectors("rename", schema, "--model", OLD, "--to", NEW, "--corpus", "typo") == 1
    assert capsys.readouterr().err == refused


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


async def test_a_refused_path_that_cannot_be_written_stops_before_the_run(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, REFUSING)
    kept = tmp_path / "no" / "such" / "folder" / "refused.jsonl"
    embedding = ("--embedding", f"{adapter}:Picky")
    assert await _vectors("stage", schema, *embedding, "--refused", str(kept)) == 1
    printed = capsys.readouterr().err
    assert printed.startswith("cli: --refused cannot be written: "), printed
    assert await storage.staged_counts("acme") == {}


async def test_a_run_stopped_early_leaves_the_refused_path_as_it_found_it(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    _adapter(tmp_path, monkeypatch)
    missing, held = tmp_path / "missing.jsonl", tmp_path / "held.jsonl"
    held.write_text("kept\n")
    for path in (missing, held):
        stage = ("--embedding", "no_such_module:New", "--refused", str(path))
        assert await _vectors("stage", schema, *stage) == 1
        assert capsys.readouterr().err.startswith("cli: cannot import no_such_module: ")
    assert not missing.exists()
    assert held.read_text() == "kept\n"


@pytest.mark.parametrize("shape", ["a folder", "under a file"])
def test_a_refused_path_that_cannot_be_written_is_refused_and_not_made(
    tmp_path: Path, shape: str
) -> None:
    (tmp_path / "file").write_text("")
    path = tmp_path if shape == "a folder" else tmp_path / "file" / "refused.jsonl"
    with pytest.raises(InputError, match=r"^cli: --refused cannot be written: "):
        cli_vectors._check_writable(str(path))


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


async def test_an_adapter_that_exits_mid_run_stops_like_an_outage(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    memory = await _filled(storage, "acme")
    told = await memory.remember("acme", "Please refuse this one.")
    await memory.remember("hr", "Please exit here.")
    body = (
        REFUSING
        + """
class Exiting(Picky):
    async def embed(self, texts, *, purpose="document"):
        if any("exit" in t for t in texts):
            sys.exit(f"cannot reach {os.environ['RFNRY_TEST_POSTGRES_URL']}")
        return await super().embed(texts, purpose=purpose)
"""
    )
    adapter = _adapter(tmp_path, monkeypatch, body)
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:Exiting") == 1
    printed = capsys.readouterr()
    assert os.environ[URL] not in printed.out + printed.err
    assert printed.err.endswith(
        "cli: stopped by SystemExit: cannot reach <url> — 6 vectors written and kept; "
        "running it again continues\n"
        "cli: new-model refused 1 passage: missing until a later run embeds them\n"
        f"  acme story {told.story.story_id}\n"
    )


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
    swap = ["vectors", "swap", "--url", url, "--schema", schema, "--model", NEW, "--yes"]
    assert await _held_past_the_timeout(pool, "memory_chunks", swap) == 0
    assert await storage.staged_counts("acme") == {OLD: 6}


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


LONG = 'f\'{chr(10).join(["first line", "x" * LENGTH])}{os.environ["RFNRY_TEST_POSTGRES_URL"]}\''


@pytest.mark.parametrize(
    ("body", "path"),
    [
        ("raise RuntimeError(TEXT)\n", "{adapter}:New"),
        ("def made():\n    raise RuntimeError(TEXT)\n", "{adapter}:made"),
    ],
    ids=["on import", "from a factory"],
)
async def test_a_long_adapter_error_is_one_line_cut_after_the_url_is_gone(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    body: str,
    path: str,
) -> None:
    url = os.environ[URL]
    length = cli_vectors._SAID - len(url) // 2 - len("first line ")
    adapter = _adapter(
        tmp_path, monkeypatch, "\n" + body.replace("TEXT", LONG.replace("LENGTH", str(length)))
    )
    assert await _vectors("stage", schema, "--embedding", path.format(adapter=adapter)) == 1
    printed = capsys.readouterr().err
    assert printed.count("\n") == 1
    assert "first line x" in printed
    assert url[: len(url) // 3] not in printed


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


class Turned(HashEmbedding):
    model = NEW
    settings = {"task": "retrieval"}

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[1 - x for x in vector] for vector in await super().embed(texts, purpose=purpose)]


class Wide(HashEmbedding):
    model = WIDE
    dimension = 16

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [[*vector, *vector] for vector in await super().embed(texts, purpose=purpose)]


Row = tuple[str | None, str | None, str | None]


async def _rows(pool: Any, corpus_id: str = "acme") -> dict[tuple[str, str], Row]:
    held: dict[tuple[str, str], Row] = {}
    async with pool.acquire() as conn:
        for kind, table, key in (
            ("chunk", "memory_chunks", "chunk_id"),
            ("story", "memory_stories", "story_id"),
        ):
            for r in await conn.fetch(
                f"SELECT {key} AS id, embedding_model AS model, embedding::text AS vector, "
                f"embedding_settings::text AS settings FROM {table} WHERE corpus_id = $1",
                corpus_id,
            ):
                held[(kind, r["id"])] = (r["model"], r["vector"], r["settings"])
    return held


async def _staged(
    pool: Any, corpus_id: str = "acme"
) -> dict[tuple[str, str, str], tuple[Any, Any]]:
    async with pool.acquire() as conn:
        return {
            (r["model"], r["passage_kind"], r["passage_id"]): (r["vector"], r["settings"])
            for r in await conn.fetch(
                "SELECT model, passage_kind, passage_id, vector::text AS vector, "
                "settings::text AS settings FROM memory_staged_vectors WHERE corpus_id = $1",
                corpus_id,
            )
        }


async def _filling(
    storage: PostgresStorage, corpus_id: str = "acme", *, limit: int | None = None
) -> Memory:
    new = Memory(storage, embedding=Turned())
    await new.stage_vectors(corpus_id, limit=limit)
    return new


async def _swap(schema: str, model: str = NEW, *argv: str) -> Ran:
    return await _vectors("swap", schema, "--model", model, "--yes", *argv)


async def _forget(schema: str, model: str, *argv: str) -> Ran:
    return await _vectors("forget", schema, "--model", model, "--yes", *argv)


async def _rename(schema: str, model: str, to: str, *argv: str) -> Ran:
    return await _vectors("rename", schema, "--model", model, "--to", to, "--yes", *argv)


async def _record(pool: Any, corpus_id: str = "acme") -> dict[str, Any] | None:
    async with pool.acquire() as conn:
        held = await conn.fetchval(
            "SELECT vector_index::text FROM memory_corpora WHERE corpus_id = $1", corpus_id
        )
    return None if held is None else json.loads(held)


async def _own(schemas: Schemas) -> tuple[str, Any, PostgresStorage]:
    name = await schemas.create()
    pool = await schemas.pool(name)
    return name, pool, await provision(pool)


async def _plan(conn: Any, model: str = NEW, *, allow_missing: bool = False) -> SwapPlan:
    return await plan_swap(
        conn, model=model, corpus_ids=None, allow_missing=allow_missing, no_index=False
    )


async def _waiting(admin: Any, count: int) -> None:
    async with asyncio.timeout(WAIT):
        while count > await admin.fetchval(
            "SELECT count(*) FROM pg_stat_activity WHERE wait_event_type = 'Lock' "
            "AND datname = current_database() AND pid <> pg_backend_pid()"
        ):
            await asyncio.sleep(0.01)


PUMP = "The pump of acme runs."


async def test_swap_puts_the_new_vectors_on_the_rows_and_keeps_the_old_staged(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    new = await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    assert (await old.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _swap(schema) == 0
    assert await _rows(pool) == {ref: (NEW, *filled[(NEW, *ref)]) for ref in before}
    assert await _staged(pool) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await old.search("acme", PUMP, pillars={"vector"})).hits


async def test_swap_back_restores_the_old_vectors_exactly(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(schema) == 0
    moved = await _rows(pool)
    assert moved != before
    assert await _swap(schema, OLD) == 0
    assert await _rows(pool) == before
    assert await _staged(pool) == filled
    assert await _swap(schema) == 0
    assert await _rows(pool) == moved


async def test_swap_refuses_while_a_passage_lacks_the_model_and_says_how_many(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: 2 passages lack a vector of new-model, in acme — fill them with "
        "rfnry-memory vectors stage, or pass --allow-missing to leave them as they are\n"
    )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_swap_with_allow_missing_leaves_those_passages_as_they_are(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    new = await _filling(storage, limit=4)
    before, filled = await _rows(pool), await _staged(pool)
    moving = {(kind, passage) for _, kind, passage in filled}
    assert await _swap(schema, NEW, "--allow-missing") == 0
    assert "  2 passages lack new-model and keep what they have" in capsys.readouterr().out
    after = await _rows(pool)
    assert {ref: after[ref] for ref in moving} == {
        ref: (NEW, *filled[(NEW, *ref)]) for ref in moving
    }
    assert {ref: after[ref] for ref in before.keys() - moving} == {
        ref: before[ref] for ref in before.keys() - moving
    }
    assert await _staged(pool) == {(OLD, *ref): before[ref][1:] for ref in moving}
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 2 missing, in acme" in capsys.readouterr().out
    assert len((await new.search("acme", PUMP, pillars={"vector"})).hits) == 4
    assert len((await old.search("acme", PUMP, pillars={"vector"})).hits) == 2


@pytest.mark.parametrize("way", ["allow-missing", "stage"])
async def test_rollback_after_new_writes_needs_allow_missing_or_a_stage(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    capsys: pytest.CaptureFixture[str],
    way: str,
) -> None:
    old = await _filled(storage)
    new = await _filling(storage)
    before = await _rows(pool)
    assert await _swap(schema) == 0
    told = await new.remember("acme", "The fan of acme hums.")
    written = ("story", told.story.story_id)
    assert await _swap(schema, OLD) == 1
    assert capsys.readouterr().err.startswith("cli: 1 passage lacks a vector of hash-8")
    if way == "stage":
        await old.stage_vectors("acme")
        assert await _swap(schema, OLD) == 0
        after = await _rows(pool)
        assert after.pop(written)[0] == OLD
        assert after == before
    else:
        assert await _swap(schema, OLD, "--allow-missing") == 0
        after = await _rows(pool)
        assert after.pop(written)[0] == NEW
        assert after == before


async def test_a_new_dimension_moves_the_whole_schema_and_refuses_part_of_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage, "acme")
    await old.set_vector_index("acme", HNSW(m=8))
    await _filled(storage, "hr")
    wide = Memory(storage, embedding=Wide())
    await wide.stage_vectors("acme")
    await wide.stage_vectors("hr")
    before = {corpus_id: await _rows(pool, corpus_id) for corpus_id in ("acme", "hr")}
    assert await _swap(name, WIDE, "--corpus", "acme") == 1
    assert capsys.readouterr().err == (
        "cli: wide-model is staged at 16 dimensions and the schema holds 8: a new dimension moves "
        "every corpus at once — leave out --corpus\n"
    )
    assert await _rows(pool) == before["acme"]
    assert await _swap(name, WIDE, "--corpus", "acme", "--corpus", "hr") == 0
    out = capsys.readouterr().out
    assert "  vectors of 16 dimensions: the whole schema moves" in out
    assert "  reads and writes wait until the swap ends" in out
    fresh = await schemas.pool(name)
    wide = Memory(PostgresStorage(fresh), embedding=Wide())
    await wide.check()
    with pytest.raises(ConfigurationError, match="holds vectors of 16 dimensions"):
        await Memory(PostgresStorage(fresh), embedding=HashEmbedding()).check()
    assert (await wide.search("acme", PUMP, pillars={"vector"})).hits
    async with fresh.acquire() as conn:
        assert await conn.fetchval("SELECT dimension FROM memory_settings") == 16
    assert await _record(fresh) is None
    assert await wide.storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    for corpus_id in ("acme", "hr"):
        assert {model for model, *_ in (await _rows(fresh, corpus_id)).values()} == {WIDE}
        assert await _staged(fresh, corpus_id) == {
            (OLD, *ref): (vector, settings)
            for ref, (_, vector, settings) in before[corpus_id].items()
        }


async def test_swap_back_from_a_new_dimension_restores_the_old_vectors_exactly(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(name, WIDE) == 0
    assert await _swap(name, OLD) == 0
    fresh = await schemas.pool(name)
    assert await _rows(fresh) == before
    assert await _staged(fresh) == filled
    await Memory(PostgresStorage(fresh), embedding=HashEmbedding()).check()


async def test_a_new_dimension_with_allow_missing_clears_what_a_passage_cannot_keep(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme", limit=4)
    before = await _rows(pool)
    assert await _swap(name, WIDE, "--allow-missing") == 0
    assert (
        "  2 passages lack wide-model: no vector on the row, and what they hold stays staged"
    ) in capsys.readouterr().out
    fresh = await schemas.pool(name)
    after = await _rows(fresh)
    left = [row for row in after.values() if row[0] != WIDE]
    assert left == [(None, None, None)] * 2
    assert await _staged(fresh) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert await _vectors("status", name, "--model", WIDE) == 1
    assert "wide-model: 2 missing, in acme" in capsys.readouterr().out
    assert await _vectors("status", name, "--model", OLD) == 0


async def test_a_new_dimension_leaves_the_staged_vectors_of_a_third_model(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await _filling(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    third = {key: held for key, held in (await _staged(pool)).items() if key[0] == NEW}
    assert await _swap(name, WIDE) == 0
    kept = await _staged(await schemas.pool(name))
    assert {key: held for key, held in kept.items() if key[0] == NEW} == third
    assert {model for model, *_ in kept} == {OLD, NEW}


async def test_a_new_dimension_refuses_rows_holding_the_model_at_the_old_one(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)

    class Grown(Wide):
        model = OLD

    await Memory(storage, embedding=Turned()).reembed("acme", limit=2)
    await Memory(storage, embedding=Grown()).stage_vectors("acme")
    before = await _rows(pool)
    assert await _swap(name, OLD, "--allow-missing") == 1
    assert capsys.readouterr().err == (
        "cli: 4 passages hold hash-8 at 8 dimensions and it is staged at 16 — drop what is staged "
        "with rfnry-memory vectors forget, give the rows another name with rfnry-memory vectors "
        "rename, then stage it again\n"
    )
    assert await _rows(pool) == before


async def test_a_model_staged_at_two_sizes_is_refused(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_staged_vectors SET vector = '[1,2]' WHERE passage_id = "
            "(SELECT min(passage_id) FROM memory_staged_vectors)"
        )
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: new-model is staged at 2 and 8 dimensions, and a model has one size — "
        "rfnry-memory vectors forget drops every staged vector of it, a rollback copy included, "
        "so be sure it can be embedded again, then stage it\n"
    )


async def test_a_row_that_already_holds_the_model_stays_and_its_staged_copy_goes(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    new = await _filling(storage)
    await new.reembed("acme", limit=2)
    async with pool.acquire() as conn:
        await conn.execute("UPDATE memory_staged_vectors SET vector = '[0,0,0,0,0,0,0,1]'")
    before = await _rows(pool)
    kept = {ref: row for ref, row in before.items() if row[0] == NEW}
    assert len(kept) == 2
    assert await _swap(schema) == 0
    after = await _rows(pool)
    assert {ref: after[ref] for ref in kept} == kept
    assert {key[0] for key in await _staged(pool)} == {OLD}
    assert {(kind, passage) for _, kind, passage in await _staged(pool)} == before.keys() - kept


@pytest.mark.parametrize("embedding", [Turned, Wide])
async def test_a_vector_with_no_model_named_is_refused_rather_than_lost(
    schemas: Schemas, capsys: pytest.CaptureFixture[str], embedding: type[HashEmbedding]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=embedding()).stage_vectors("acme")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET embedding_model = NULL WHERE chunk_id = "
            "(SELECT min(chunk_id) FROM memory_chunks)"
        )
    before = await _rows(pool)
    assert await _swap(name, embedding.model, "--allow-missing") == 1
    assert capsys.readouterr().err == (
        "cli: 1 passage holds a vector with no model named, which this swap would lose — "
        "set its embedding_model in SQL, or clear its embedding\n"
    )
    assert await _rows(pool) == before


async def test_a_vector_with_no_model_named_outside_the_swap_stops_nothing(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_stories SET embedding_model = NULL WHERE story_id = "
            "(SELECT min(story_id) FROM memory_stories WHERE story_id NOT IN "
            "(SELECT passage_id FROM memory_staged_vectors))"
        )
    before = await _rows(pool)
    assert await _swap(schema, NEW, "--allow-missing") == 0
    after = await _rows(pool)
    assert [row for row in after.values() if row[0] is None] == [
        row for row in before.values() if row[0] is None
    ]


async def test_a_label_with_no_vector_counts_as_missing_until_it_is_staged(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_chunks SET embedding = NULL WHERE chunk_id = "
            "(SELECT min(chunk_id) FROM memory_chunks)"
        )
    assert await _vectors("status", schema, "--model", OLD) == 1
    assert "hash-8: 1 missing, in acme" in capsys.readouterr().out
    assert await storage.count_unstaged("acme", model=OLD, filters=None) == 1
    assert (await old.stage_vectors("acme")).embedded == 1
    assert await storage.count_unstaged("acme", model=OLD, filters=None) == 0
    assert await _vectors("status", schema, "--model", OLD) == 0
    assert await _swap(schema, OLD) == 0
    assert all(vector is not None for _, vector, _ in (await _rows(pool)).values())


async def test_swap_moves_retired_and_closed_passages_alike(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.set_history("acme", True)
    source = await memory.ingest("acme", name="manual", text="The old manual.")
    await memory.retire_source("acme", source.source.source_id)
    told = await memory.remember("acme", "The pump runs.")
    await memory.end_story("acme", told.story.story_id)
    await _filling(storage)
    assert await _swap(schema) == 0
    assert [model for model, *_ in (await _rows(pool)).values()] == [NEW, NEW]


async def test_a_passage_with_no_vector_takes_the_staged_one(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await Memory(storage).ingest("acme", name="manual", text="Written with no model.")
    await _filling(storage)
    assert await _swap(schema) == 0
    [(model, vector, _)] = (await _rows(pool)).values()
    assert model == NEW and vector is not None
    assert await _staged(pool) == {}


async def test_a_passage_deleted_after_staging_does_not_stop_a_swap(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    source = await old.ingest("acme", name="gone", text="A sheet deleted before the swap.")
    story = await old.remember("acme", "A story deleted before the swap.")
    await _filling(storage)
    await old.delete_source("acme", source.source.source_id)
    await old.delete_story("acme", story.story.story_id)
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "VALUES ('acme', $1, 'chunk', 'a passage nothing holds', '[1,0,0,0,0,0,0,0]')",
            NEW,
        )
    before = await _rows(pool)
    assert await _swap(schema) == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _staged(pool) == {
        (OLD, *ref): (vector, settings) for ref, (_, vector, settings) in before.items()
    }
    await old.delete_corpus("acme")
    assert await old.verify_absent("acme") == Counts()


async def test_a_passage_deleted_after_a_swap_takes_its_old_vector_with_it(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    source = await old.ingest("acme", name="gone", text="A sheet deleted after the swap.")
    await _filling(storage)
    assert await _swap(schema) == 0
    staged = len(await _staged(pool))
    assert (await old.delete_source("acme", source.source.source_id)).staged == 1
    assert len(await _staged(pool)) == staged - 1


async def test_swap_of_one_corpus_leaves_the_others_alone(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    acme = await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    for memory, corpus_id in ((acme, "acme"), (hr, "hr")):
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    before, filled = await _rows(pool, "hr"), await _staged(pool, "hr")
    assert await _swap(schema, NEW, "--corpus", "acme", "--no-index") == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _rows(pool, "hr") == before
    assert await _staged(pool, "hr") == filled
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None
        assert await _vectors_table(conn, "hr") is not None
        assert (
            await conn.fetchval("SELECT vector_index FROM memory_corpora WHERE corpus_id = 'hr'")
            is None
        )
    assert (await hr.search("hr", "The pump of hr runs.", pillars={"vector"})).hits


async def test_every_statement_of_a_swap_of_one_corpus_names_it(
    storage: PostgresStorage, pool: Any
) -> None:
    acme = await _filled(storage, "acme")
    await acme.set_vector_index("acme", HNSW(m=8))
    await _filled(storage, "hr")
    await _filling(storage, "acme")
    await _filling(storage, "hr")
    async with pool.acquire() as conn:
        plan = await plan_swap(
            conn, model=NEW, corpus_ids=["acme"], allow_missing=False, no_index=False
        )
    moving = [s for s in plan.statements if s.startswith(("INSERT", "UPDATE", "DELETE"))]
    assert len(moving) == 6
    assert all(
        "corpus_id IN ('acme')" in statement or "corpus_id = 'acme'" in statement
        for statement in moving
    )
    assert not any("'hr'" in statement for statement in plan.statements)


async def test_swap_leaves_the_index_of_a_corpus_with_nothing_staged(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await hr.set_vector_index("hr", HNSW(m=8))
    await _filling(storage, "acme")
    assert await _swap(schema, NEW, "--allow-missing") == 0
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is not None


async def test_swap_drops_the_index_table_and_records_its_settings_and_owner(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8, ef_construction=32))
    new = await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None
        owner = await conn.fetchval("SELECT current_user")
    settings = HNSW(m=8, ef_construction=32, precision="full").model_dump()
    assert await _record(pool) == {"settings": settings, "owner": owner}
    assert (
        f"  acme: HNSW(m=8, ef_construction=32) goes, owned by {owner}\n"
    ) in capsys.readouterr().out
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _vectors("status", schema) == 0
    assert "index to build: HNSW(m=8, ef_construction=32)" in capsys.readouterr().out


async def test_a_swap_before_the_index_is_built_keeps_what_was_recorded(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    recorded = await _record(pool)
    assert recorded is not None
    assert await _swap(schema, OLD, "--no-index") == 0
    assert await _record(pool) == recorded


async def test_swap_with_no_staged_vectors_says_so(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "cli: nothing is staged for new-model — fill it with rfnry-memory vectors stage, "
        "or check the name with rfnry-memory vectors status\n"
    )
    await _filled(storage, "hr")
    await _filling(storage, "hr")
    assert await _swap(schema, NEW, "--corpus", "acme") == 1
    assert "cli: nothing is staged for new-model" in capsys.readouterr().err


async def test_swap_before_upgrade_is_refused(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    now = migrations.LATEST
    monkeypatch.setattr(migrations, "LATEST", now + 1)
    assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        f"cli: {schema} is at version {now}, this rfnry-memory needs {now + 1} — "
        f"run rfnry-memory upgrade --schema {schema}\n"
    )
    async with pool.acquire() as conn:
        plan = await _plan(conn)
    assert plan == SwapPlan(
        [],
        [],
        ["the schema is not at this rfnry-memory's version — run rfnry-memory upgrade first"],
        [],
    )
    assert await _rows(pool) == before


async def test_a_schema_with_no_vectors_has_nothing_to_swap(schemas: Schemas) -> None:
    name = await schemas.create()
    pool = await schemas.pool(name)
    await provision(pool, dimension=None)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
    assert plan.refusals == ["the schema holds no vectors — run rfnry-memory set --dimension N"]


async def test_a_refused_plan_runs_nothing(storage: PostgresStorage, pool: Any) -> None:
    await _filled(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        assert plan.refusals and not plan.statements
        with pytest.raises(InputError, match="^postgres: nothing is staged for new-model"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_swap_sql_prints_and_runs_nothing(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    assert await _vectors("swap", schema, "--model", NEW, "--sql") == 0
    printed = capsys.readouterr().out
    lines = printed.splitlines()
    assert lines[:2] == [
        f'SET search_path TO "{schema}", public;',
        "BEGIN ISOLATION LEVEL READ COMMITTED;",
    ]
    assert lines[-1] == "COMMIT;"
    assert "LOCK TABLE memory_chunks, memory_stories IN SHARE ROW EXCLUSIVE MODE;" in lines
    assert os.environ[URL] not in printed
    assert f":{urlsplit(os.environ[URL]).password}@" not in printed
    assert await _rows(pool) == before
    async with pool.acquire() as conn:
        await conn.execute(printed)
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_swap_sql_prints_refusals_rather_than_statements(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _vectors("swap", schema, "--model", NEW, "--sql") == 1
    printed = capsys.readouterr()
    assert printed.out == ""
    assert printed.err.startswith("cli: nothing is staged for new-model")


async def test_swap_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("swap", schema, "--model", NEW) == 0
    out = capsys.readouterr().out
    assert "  6 vectors of new-model go onto the rows, in acme" in out
    assert "  writes wait until the swap ends; reads go on and see the old vectors" in out
    assert out.endswith("nothing was run\n")
    assert await _rows(pool) == before
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("swap", schema, "--model", NEW) == 0
    assert capsys.readouterr().out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"next: rfnry-memory vectors status --model new-model --schema {schema}\n"
    )
    assert await _rows(pool) != before

    def unasked(prompt: str) -> str:
        raise AssertionError("tests: --yes asks nothing")

    monkeypatch.setattr("builtins.input", unasked)
    assert await _swap(schema, OLD) == 0


async def test_stale_statistics_cannot_turn_a_swap_back_into_a_crawl(
    storage: PostgresStorage, pool: Any
) -> None:
    told = await (await _filled(storage)).ingest("acme", name="bulk", text="Bulk.")
    vector = "[" + ",".join(["0.5"] * 8) + "]"
    async with pool.acquire() as conn:
        await conn.execute(
            "INSERT INTO memory_chunks (corpus_id, chunk_id, source_id, position, source_name, "
            "text, digest, config, embedding, embedding_model) SELECT 'acme', 'bulk-' || g, $1, g, "
            "'bulk', 'bulk', md5(g::text), 'english', $2::text::vector, $3 "
            "FROM generate_series(1, 30000) g",
            told.source.source_id,
            vector,
            OLD,
        )
        await conn.execute(
            "INSERT INTO memory_staged_vectors (corpus_id, model, passage_kind, passage_id, vector) "
            "SELECT 'acme', $1, 'chunk', chunk_id, embedding FROM memory_chunks "
            "WHERE corpus_id = 'acme'",
            NEW,
        )
        await conn.execute("ANALYZE memory_chunks, memory_staged_vectors")
        await conn.execute("UPDATE memory_chunks SET embedding_model = $1", NEW)
        await conn.execute("UPDATE memory_staged_vectors SET model = $1", OLD)
        async with asyncio.timeout(WAIT):
            await vector_status(conn, None)
            await apply_swap(conn, await _plan(conn, OLD))
    assert {model for model, *_ in (await _rows(pool)).values()} == {OLD}


async def test_a_swap_leaves_the_statistics_true(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        await conn.execute("ANALYZE memory_chunks, memory_stories, memory_staged_vectors")
        assert await _swap(schema) == 0
        common = (
            "SELECT most_common_vals::text FROM pg_stats WHERE schemaname = $1 "
            "AND tablename = $2 AND attname = $3"
        )
        assert NEW in await conn.fetchval(common, schema, "memory_chunks", "embedding_model")
        assert NEW in await conn.fetchval(common, schema, "memory_stories", "embedding_model")
        assert OLD in await conn.fetchval(common, schema, "memory_staged_vectors", "model")


async def test_a_write_during_a_swap_waits_for_it(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
    capsys: pytest.CaptureFixture[str],
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(old.ingest("acme", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
            assert (await old.search("acme", PUMP, pillars={"vector"})).hits
            assert await _rows(pool) == before
    async with asyncio.timeout(WAIT):
        await writing
    after = await _rows(pool)
    assert len(after) == len(before) + 1
    assert [model for model, *_ in after.values()].count(OLD) == 1
    assert await _vectors("status", schema, "--model", NEW) == 1
    assert "new-model: 1 missing, in acme" in capsys.readouterr().out


async def test_a_write_of_the_old_size_during_a_new_dimension_fails_out_loud(
    schemas: Schemas, database: Database, spawn: Spawned
) -> None:
    _, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    async with pool.acquire() as conn:
        plan = await _plan(conn, WIDE)
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(old.ingest("acme", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
    async with asyncio.timeout(WAIT):
        with pytest.raises(StorageError, match="expected 16 dimensions, not 8"):
            await writing
    assert {model for model, *_ in (await _rows(pool)).values()} == {WIDE}


async def test_a_swap_of_one_corpus_holds_the_writes_of_the_others_too(
    storage: PostgresStorage, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await _filling(storage, "acme")
    async with pool.acquire() as conn:
        plan = await plan_swap(
            conn, model=NEW, corpus_ids=["acme"], allow_missing=False, no_index=False
        )
        async with conn.transaction(isolation="read_committed"):
            await apply_swap(conn, plan)
            writing = spawn(hr.ingest("hr", name="late", text="A line written late."))
            await _waiting(database.admin, 1)
            assert not writing.done()
    async with asyncio.timeout(WAIT):
        await writing


async def test_a_write_after_the_plan_stops_the_swap_and_says_how_many(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.remember("acme", "A story written after the plan.")
        before, filled = await _rows(pool), await _staged(pool)
        with pytest.raises(ConfigurationError, match="^postgres: passages lacking new-model: 1 — "):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_write_after_the_plan_goes_through_with_allow_missing(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as conn:
        plan = await _plan(conn, allow_missing=True)
        told = await old.remember("acme", "A story written after the plan.")
        await apply_swap(conn, plan)
    assert (await _rows(pool))[("story", told.story.story_id)][0] == OLD


async def test_an_index_built_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=8))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_an_index_tuned_again_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=16))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == before


async def test_an_index_finished_after_the_plan_stops_the_swap(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as conn:
        [story] = [
            i.name for i in await _vector_indexes(conn, _vectors_name("acme")) if i.kind == "story"
        ]
        await conn.execute(f"DROP INDEX {story}")
        plan = await _plan(conn)
        await old.set_vector_index("acme", HNSW(m=8))
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
        assert await _vectors_table(conn, "acme") is not None
    assert await _rows(pool) == before


async def test_a_new_dimension_planned_twice_moves_once(schemas: Schemas) -> None:
    _, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    async with pool.acquire() as conn:
        plan = await _plan(conn, WIDE)
        await apply_swap(conn, plan)
        moved = await _rows(pool)
        with pytest.raises(ConfigurationError, match="changed since the plan was shown"):
            await apply_swap(conn, plan)
    assert await _rows(pool) == moved


async def test_a_source_being_deleted_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    source = await old.ingest("acme", name="gone", text="A sheet deleted beside the swap.")
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_sources WHERE corpus_id = 'acme' AND source_id = $1 FOR UPDATE",
            source.source.source_id,
        )
        deleting = spawn(old.delete_source("acme", source.source.source_id))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_a_story_being_deleted_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    old = await _filled(storage)
    told = await old.remember("acme", "A story deleted beside the swap.")
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            told.story.story_id,
        )
        deleting = spawn(old.delete_story("acme", told.story.story_id))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}


async def test_a_batch_being_staged_finishes_before_a_swap_starts(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    await _filled(storage)
    await _filling(storage)
    story_id = next(passage for _, kind, passage in await _staged(pool) if kind == "story")
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            story_id,
        )
        staging = spawn(
            storage.stage("acme", [("story", story_id)], [[0.5] * 8], NEW, settings=None)
        )
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await staging
        assert await swapping == 0
    assert (await _rows(pool))[("story", story_id)][1] == "[" + ",".join(["0.5"] * 8) + "]"


async def test_deleting_another_corpus_beside_a_swap_ends_with_both_done(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    database: Database,
    spawn: Spawned,
) -> None:
    await _filled(storage, "acme")
    hr = await _filled(storage, "hr")
    await _filling(storage, "acme")
    async with pool.acquire() as held, held.transaction():
        await held.execute("LOCK TABLE memory_stories IN SHARE ROW EXCLUSIVE MODE")
        deleting = spawn(hr.delete_corpus("hr"))
        await _waiting(database.admin, 1)
        swapping = spawn(_swap(schema, NEW, "--corpus", "acme"))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await deleting
        assert await swapping == 0
    assert await hr.verify_absent("hr") == Counts()


async def test_a_swap_stops_while_an_index_is_being_built(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as building, _vectors_lock(building, "acme"):
        assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "postgres: the vector index of acme is being built or dropped — "
        "run the command again when that ends\n"
    )
    assert await _rows(pool) == before
    assert await _swap(schema) == 0


async def test_a_swap_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    before = await _rows(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _swap(schema) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await _rows(pool) == before


async def test_an_interrupted_swap_changes_nothing(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as reading, reading.transaction():
        table = await _vectors_table(reading, "acme")
        await reading.fetchval(f"SELECT count(*) FROM {table}")
        process = await asyncio.create_subprocess_exec(
            sys.executable,
            "-m",
            "rfnry_memory.cli",
            "vectors",
            "swap",
            "--url",
            os.environ[URL],
            "--schema",
            schema,
            "--model",
            NEW,
            "--yes",
            env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )
        await _waiting(database.admin, 1)
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            _, left = await process.communicate()
    assert process.returncode == 130
    assert left.decode() == "cli: stopped — the swap rolled back, nothing moved\n"
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_swap_that_fails_part_way_changes_nothing(
    storage: PostgresStorage, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        plan = await _plan(conn)
        table = await _vectors_table(conn, "acme")
        await conn.execute(f"ALTER TABLE {table} RENAME TO memory_vectors_elsewhere")
        broken = plan._replace(statements=[s for s in plan.statements if not s.startswith("DO ")])
        with pytest.raises(StorageError, match="^postgres: the swap rolled back, nothing changed"):
            await apply_swap(conn, broken)
        await conn.execute(
            f"ALTER TABLE memory_vectors_elsewhere RENAME TO {_vectors_name('acme')}"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


class LostAtCommit:
    def __init__(self, lost: BaseException) -> None:
        self.lost = lost

    async def execute(self, statement: str) -> None:
        pass

    @asynccontextmanager
    async def transaction(self, isolation: str) -> AsyncIterator[None]:
        yield
        raise self.lost


@pytest.mark.parametrize(
    "lost", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_the_commit_does_not_claim_nothing_changed(
    lost: BaseException,
) -> None:
    with pytest.raises(
        StorageError,
        match="^postgres: the swap stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which",
    ):
        await apply_swap(LostAtCommit(lost), SwapPlan([], ["SELECT 1"], [], []))


async def test_a_connection_lost_before_the_commit_says_nothing_changed() -> None:
    conn = await asyncpg.connect(os.environ[URL])
    plan = SwapPlan([], ["SELECT pg_terminate_backend(pg_backend_pid())"], [], [])
    with pytest.raises(StorageError, match="^postgres: the swap rolled back, nothing changed"):
        await apply_swap(conn, plan)


@asynccontextmanager
async def _role(schemas: Schemas, schema: str, pool: Any) -> AsyncIterator[str]:
    role = f"{schemas.run}_role_{uuid4().hex[:8]}"
    async with pool.acquire() as conn:
        await conn.execute(f"CREATE ROLE {role} LOGIN PASSWORD 'role'")
        await conn.execute(f"GRANT USAGE, CREATE ON SCHEMA {schema} TO {role}")
        await conn.execute(
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {schema} TO {role}"
        )
    try:
        yield role
    finally:
        async with pool.acquire() as conn:
            await conn.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            await conn.execute(f"DROP OWNED BY {role}")
            await conn.execute(f"DROP ROLE {role}")


def _url_of(role: str) -> str:
    parts = urlsplit(os.environ[URL])
    return urlunsplit(parts._replace(netloc=f"{role}:role@{parts.hostname}:{parts.port}"))


async def _vectors_as(role: str, action: str, schema: str, *argv: str) -> int:
    given = ["vectors", action, "--url", _url_of(role), "--schema", schema, *argv]
    return await asyncio.to_thread(cli.main, given)


async def _swap_as(role: str, schema: str, model: str = NEW, *argv: str) -> int:
    return await _vectors_as(role, "swap", schema, "--model", model, "--yes", *argv)


async def _owner(conn: Any, corpus_id: str = "acme") -> str:
    return str(
        await conn.fetchval(
            "SELECT tableowner FROM pg_tables WHERE schemaname = current_schema() "
            "AND tablename = $1",
            _vectors_name(corpus_id),
        )
    )


class SplitRoles(NamedTuple):
    schema: str
    administrator: str
    tenant: str
    admin: Any
    served: Any
    storage: PostgresStorage


@asynccontextmanager
async def _split_roles(schemas: Schemas, database: Database) -> AsyncIterator[SplitRoles]:
    name = f"{schemas.run}_{uuid4().hex[:8]}"
    administrator, tenant = f"{name}_administrator", f"{name}_tenant"
    here = await database.admin.fetchval("SELECT quote_ident(current_database())")
    await database.admin.execute(f"CREATE ROLE {administrator} LOGIN CREATEROLE PASSWORD 'role'")
    await database.admin.execute(f"GRANT CREATE ON DATABASE {here} TO {administrator}")
    schemas._made.append(name)
    path = {"search_path": f"{name}, public"}
    admin = await asyncpg.connect(_url_of(administrator), server_settings=path)
    await admin.execute(f"CREATE ROLE {tenant} LOGIN PASSWORD 'role'")
    made = ["create", "--url", _url_of(administrator), "--schema", name, "--dimension", "8"]
    assert await asyncio.to_thread(cli.main, made) == 0
    await admin.execute(f"GRANT USAGE, CREATE ON SCHEMA {name} TO {tenant}")
    await admin.execute(
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON ALL TABLES IN SCHEMA {name} TO {tenant}"
    )
    await admin.execute(
        f"ALTER DEFAULT PRIVILEGES IN SCHEMA {name} "
        f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {tenant}"
    )
    served = await asyncpg.create_pool(_url_of(tenant), server_settings=path)
    try:
        yield SplitRoles(name, administrator, tenant, admin, served, PostgresStorage(served))
    finally:
        await served.close()
        await admin.close()
        for role in (tenant, administrator):
            await database.admin.execute(f"REASSIGN OWNED BY {role} TO CURRENT_USER")
            await database.admin.execute(f"DROP OWNED BY {role}")
            await database.admin.execute(f"DROP ROLE {role}")


async def _app_built(split: SplitRoles) -> bool:
    table = await _vectors_table(split.admin, "acme")
    readable = await split.admin.fetchval("SELECT has_table_privilege($1, 'SELECT')", table)
    return not readable and await _owner(split.admin) == split.tenant


async def test_a_role_that_cannot_alter_is_refused_before_anything_moves(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _filled(storage)
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        owner = await conn.fetchval("SELECT current_user")
    async with _role(schemas, name, pool) as role:
        assert await _swap_as(role, name, WIDE) == 1
        assert capsys.readouterr().err == (
            f"cli: {role} does not own memory_chunks and memory_stories, which a new dimension "
            f"retypes — run this as {owner}\n"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_a_role_that_cannot_drop_an_index_is_refused_before_anything_moves(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    before, filled = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        owner = await conn.fetchval("SELECT current_user")
    async with _role(schemas, name, pool) as role:
        assert await _swap_as(role, name) == 1
        assert capsys.readouterr().err == (
            f"cli: {role} cannot drop the vector index of acme — run this as {owner}\n"
        )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled


async def test_the_role_that_owns_an_index_may_swap_without_owning_the_schema(
    schemas: Schemas,
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    async with _role(schemas, name, pool) as role, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {role}")
        assert await _swap_as(role, name) == 0
        assert await _owner(conn) == role


async def test_the_schema_owner_may_drop_a_table_another_role_owns(
    schemas: Schemas, database: Database
) -> None:
    async with _split_roles(schemas, database) as split:
        old = await _filled(split.storage)
        await old.set_vector_index("acme", HNSW(m=8))
        await _filling(split.storage)
        assert await _app_built(split)
        assert await _swap_as(split.administrator, split.schema, NEW, "--no-index") == 0
        assert await _vectors_table(split.admin, "acme") is None
        recorded = await _record(split.served)
    assert recorded is not None and recorded["owner"] == split.tenant


async def test_a_non_superuser_administrator_switches_an_index_the_app_built(
    schemas: Schemas, database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _split_roles(schemas, database) as split:
        name, administrator, tenant = split.schema, split.administrator, split.tenant
        old = await _filled(split.storage)
        await old.set_vector_index("acme", HNSW(m=8))
        assert await _app_built(split)
        checked = ["check", "--url", _url_of(administrator), "--schema", name]
        assert await asyncio.to_thread(cli.main, checked) == 0
        assert await _vectors_as(administrator, "status", name) == 0
        assert "HNSW(m=8)" in capsys.readouterr().out
        await Memory(split.storage, embedding=Wide()).stage_vectors("acme")
        assert await _swap_as(tenant, name, WIDE, "--no-index") == 1
        assert capsys.readouterr().err == (
            f"cli: {tenant} does not own memory_chunks and memory_stories, which a new dimension "
            f"retypes — run this as {administrator}\n"
        )
        assert await _swap_as(administrator, name, WIDE, "--no-index") == 0
        assert capsys.readouterr().out.endswith(_next_as(tenant, name))
        assert await _vectors_as(tenant, "index", name) == 0
        assert await _owner(split.admin) == tenant
        await split.served.expire_connections()
        wide = Memory(PostgresStorage(split.served), embedding=Wide())
        await wide.check()
        assert (await wide.search("acme", PUMP, pillars={"vector"})).hits
        assert await wide.storage.get_vector_index("acme") == HNSW(m=8, precision="full")
        assert (await wide.delete_corpus("acme")).vectors == 6
        assert await _vectors_table(split.admin, "acme") is None
        assert (await wide.verify_absent("acme")).clean


async def test_a_schema_owner_still_hears_of_any_other_table_it_cannot_read(
    schemas: Schemas, database: Database, capsys: pytest.CaptureFixture[str]
) -> None:
    async with _split_roles(schemas, database) as split:
        await database.admin.execute(
            f"ALTER TABLE {split.schema}.memory_sources OWNER TO {split.tenant}"
        )
        assert await _vectors_as(split.administrator, "status", split.schema) == 1
        assert capsys.readouterr().err.startswith(
            f"cli: {split.administrator} lacks SELECT, INSERT, UPDATE, DELETE on memory_sources — "
        )


ODD_CORPUS = "o'brien; $$ 100% \\ drop"
ODD_MODEL = "it's $$ 100% \\ model"


class Odd(Turned):
    model = ODD_MODEL


async def test_names_that_are_no_sql_swap_like_any_other(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage, ODD_CORPUS)
    await old.set_vector_index(ODD_CORPUS, HNSW(m=8))
    await Memory(storage, embedding=Odd()).stage_vectors(ODD_CORPUS)
    before = await _rows(pool, ODD_CORPUS)
    assert await _swap(schema, ODD_MODEL, "--corpus", ODD_CORPUS) == 0
    assert {model for model, *_ in (await _rows(pool, ODD_CORPUS)).values()} == {ODD_MODEL}
    capsys.readouterr()
    assert await _vectors("swap", schema, "--model", OLD, "--corpus", ODD_CORPUS, "--sql") == 0
    async with pool.acquire() as conn:
        await conn.execute(capsys.readouterr().out)
    assert await _rows(pool, ODD_CORPUS) == before


async def test_a_schema_name_that_needs_quoting_swaps_to_a_new_dimension(
    schemas: Schemas,
) -> None:
    name = f"{schemas.run}_{uuid4().hex[:8]} Odd-Name"
    assert await run_cli("create", "--schema", name, "--dimension", "8") == 0
    quoted = '"' + name + '"'
    schemas._made.append(quoted)
    storage = PostgresStorage(await schemas.pool(quoted))
    memory = await _filled(storage)
    await memory.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Wide()).stage_vectors("acme")
    assert await _swap(name, WIDE) == 0
    fresh = PostgresStorage(await schemas.pool(quoted))
    wide = Memory(fresh, embedding=Wide())
    await wide.check()
    assert (await wide.search("acme", PUMP, pillars={"vector"})).hits


BUILDING = "an index takes real time at scale; searches run exactly until it is built"
HELD = "the swap held and new-model is on the rows; "
LEFT_TO_INDEX = "the index of acme is left to rfnry-memory vectors index"
UNTIL_BUILT = "and searches there run exactly until it is built"


class Broad(HashEmbedding):
    model = "broad-model"
    dimension = 2008

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [vector * 251 for vector in await super().embed(texts, purpose=purpose)]


class Vast(HashEmbedding):
    model = "vast-model"
    dimension = 4008

    async def embed(self, texts: list[str], *, purpose: str = "document") -> list[list[float]]:
        return [vector * 501 for vector in await super().embed(texts, purpose=purpose)]


class Renamed(HashEmbedding):
    model = RENAMED


def _only(owner: str, schema: str) -> str:
    return (
        f"cli: only {owner} builds the index in acme, so its table keeps its owner — "
        f"run rfnry-memory vectors index --url URL --corpus acme --schema {schema} as {owner}\n"
    )


def _next_as(owner: str, schema: str) -> str:
    return (
        f"next: rfnry-memory vectors index --url URL --corpus acme --schema {schema} as {owner}\n"
    )


def _unbuilt(schema: str, held: str = "") -> str:
    return (
        f"{held}the index in acme is still to build, its searches run exactly, and "
        f"rfnry-memory vectors index --corpus acme --schema {schema} finishes it\n"
    )


async def _running(*argv: str, cwd: Path | None = None) -> asyncio.subprocess.Process:
    return await asyncio.create_subprocess_exec(
        sys.executable,
        "-m",
        "rfnry_memory.cli",
        *argv[:2],
        "--url",
        os.environ[URL],
        *argv[2:],
        cwd=cwd,
        env={**os.environ, "PYTHONPATH": str(Path(cli.__file__).parents[1])},
        stdout=asyncio.subprocess.PIPE,
        stderr=asyncio.subprocess.PIPE,
    )


async def _interrupted(database: Database, *argv: str) -> tuple[int | None, str]:
    process = await _running(*argv)
    await _waiting(database.admin, 1)
    process.send_signal(signal.SIGINT)
    async with asyncio.timeout(WAIT):
        _, left = await process.communicate()
    return process.returncode, left.decode()


async def test_swap_rebuilds_the_index_it_dropped(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    tuned = HNSW(m=8, ef_construction=32, precision="full", ef_search=40)
    await old.set_vector_index("acme", tuned)
    new = await _filling(storage)
    assert await _swap(schema) == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme builds again after the swap, and searches there run exactly "
        "until it ends\n"
    ) in out
    assert out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"schema {schema}: {BUILDING}\n"
        "acme: building HNSW(m=8, ef_construction=32, ef_search=40)\n"
        "acme: built, 6 vectors\n"
        f"next: rfnry-memory vectors status --model new-model --schema {schema}\n"
    )
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == tuned
    async with pool.acquire() as conn:
        table = await _vectors_table(conn, "acme")
        indexed = {
            (r["passage_kind"], r["passage_id"]): r["vector"]
            for r in await conn.fetch(
                f"SELECT passage_kind, passage_id, vector::text AS vector FROM {table}"
            )
        }
    assert indexed == {ref: vector for ref, (_, vector, _) in (await _rows(pool)).items()}
    assert (await new.search("acme", PUMP, pillars={"vector"})).hits
    assert await _vectors("status", schema) == 0
    shown = capsys.readouterr().out
    assert "HNSW(m=8, ef_construction=32, ef_search=40)" in shown
    assert "index to build" not in shown


async def test_swap_no_index_leaves_it_to_vectors_index(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    out = capsys.readouterr().out
    assert f"  {LEFT_TO_INDEX}, {UNTIL_BUILT}\n" in out
    assert out.endswith(
        f"schema {schema}: new-model is on the rows\n"
        f"next: rfnry-memory vectors index --corpus acme --schema {schema}\n"
    )
    assert await storage.get_vector_index("acme") is None
    assert await _record(pool) is not None
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == (
        f"schema {schema}: {BUILDING}\nacme: building HNSW(m=8)\nacme: built, 6 vectors\n"
    )
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == f"schema {schema}: no index to build\n"


async def test_vectors_index_stays_in_the_corpora_it_names(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--no-index") == 0
    assert await _vectors("index", schema, "--corpus", "hr") == 0
    assert await storage.get_vector_index("hr") == HNSW(m=8, precision="full")
    assert await _record(pool, "hr") is None
    assert await storage.get_vector_index("acme") is None
    assert await _record(pool, "acme") is not None


async def test_swap_builds_what_it_moved_and_no_other_corpus(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--corpus", "hr", "--no-index") == 0
    assert await _swap(schema) == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await storage.get_vector_index("hr") is None
    assert await _record(pool, "hr") is not None


async def test_a_swap_builds_what_an_earlier_swap_left_to_build(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    capsys.readouterr()
    assert await _swap(schema, OLD) == 0
    assert (
        "  the index of acme builds again after the swap, and searches there run exactly "
        "until it ends\n"
    ) in capsys.readouterr().out
    assert await _record(pool) is None
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")


async def test_an_index_keeps_its_owner(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 0
        told = capsys.readouterr()
        assert f"  {LEFT_TO_INDEX}, run as {app}, {UNTIL_BUILT}\n" in told.out
        assert told.out.endswith(f"schema {name}: new-model is on the rows\n{_next_as(app, name)}")
        assert BUILDING not in told.out
        assert told.err == ""
        assert await _vectors_table(conn, "acme") is None
        recorded = await _record(pool)
        assert recorded is not None and recorded["owner"] == app
        assert await _vectors("index", name) == 1
        assert capsys.readouterr().err == _only(app, name)
        assert await _vectors_table(conn, "acme") is None
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
        assert await _record(pool) is None
        served = await asyncpg.create_pool(
            _url_of(app), server_settings={"search_path": f"{name}, public"}
        )
        try:
            deleted = await Memory(PostgresStorage(served), embedding=Turned()).delete_corpus(
                "acme"
            )
        finally:
            await served.close()
        assert deleted.vectors == 6
        assert await _vectors_table(conn, "acme") is None


async def test_a_role_builds_what_it_owns_and_names_the_rest(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        await conn.execute(
            f"ALTER DEFAULT PRIVILEGES IN SCHEMA {name} "
            f"GRANT SELECT, INSERT, UPDATE, DELETE ON TABLES TO {app}"
        )
        assert await _swap(name, NEW, "--no-index") == 0
        capsys.readouterr()
        assert await _vectors("index", name) == 1
        told = capsys.readouterr()
        assert told.err == _only(app, name)
        assert "hr: built, 6 vectors\n" in told.out
        assert await _vectors_table(conn, "acme") is None
        assert await _record(pool, "hr") is None
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
        assert await _owner(conn, "hr") == await conn.fetchval("SELECT current_user")


async def test_a_swap_builds_what_its_role_owns_and_names_the_owner_of_the_rest(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 0
        told = capsys.readouterr()
        assert told.err == ""
        assert told.out.endswith(f"hr: built, 6 vectors\n{_next_as(app, name)}")
        assert await _vectors_table(conn, "acme") is None


async def test_a_build_that_fails_beside_an_owners_index_names_both(
    schemas: Schemas, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)

    async def refused(*_: Any, **__: Any) -> int:
        raise StorageError("postgres: tests: the build failed")

    monkeypatch.setattr(PostgresStorage, "set_vector_index", refused)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        assert await _swap(name) == 3
        told = capsys.readouterr()
        assert told.err == (
            "cli: stopped by StorageError: postgres: tests: the build failed — "
            f"{HELD}the index in hr is still to build, its searches run exactly, and "
            f"rfnry-memory vectors index --corpus hr --schema {name} finishes it\n"
        )
        assert told.out.endswith(_next_as(app, name))


@pytest.mark.parametrize(
    "stop", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_the_swaps_commit_exits_4_and_names_status(
    storage: PostgresStorage,
    schema: str,
    stop: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)

    async def lost(_: Any, plan: SwapPlan) -> None:
        await apply_swap(LostAtCommit(stop), plan)

    monkeypatch.setattr(cli_vectors, "apply_swap", lost)
    assert await _swap(schema) == 4
    told = capsys.readouterr()
    assert told.err == (
        "postgres: the swap stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which\n"
    )
    assert "next:" not in told.out


async def _two_left_to_build(storage: PostgresStorage, schema: str) -> None:
    for corpus_id in ("acme", "hr"):
        memory = await _filled(storage, corpus_id)
        await memory.set_vector_index(corpus_id, HNSW(m=8))
        await _filling(storage, corpus_id)
    assert await _swap(schema, NEW, "--no-index") == 0


async def test_a_corpus_deleted_between_the_read_and_the_build_gets_no_table_back(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _two_left_to_build(storage, schema)
    capsys.readouterr()
    read = switch.plan_index

    async def then_deleted(conn: Any, corpus_ids: list[str], precision: Any) -> Any:
        plan = await read(conn, corpus_ids, precision)
        if corpus_ids == ["hr"]:
            await PostgresStorage(conn).delete_corpus("hr")
        return plan

    monkeypatch.setattr(cli_vectors, "plan_index", then_deleted)
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out.endswith("hr: nothing recorded to build any more\n")
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is None
        assert await _vectors_table(conn, "acme") is not None


async def test_a_build_of_what_is_recorded_takes_the_record_under_its_own_lock(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    with pytest.raises(NotFoundError, match=r"^postgres: "):
        await storage.set_vector_index("acme", HNSW(m=8), rebuild=False, only_recorded=True)
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "acme") is None


async def test_a_corpus_deleted_while_the_queue_waits_is_not_built_again(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _two_left_to_build(storage, schema)
    capsys.readouterr()
    build = PostgresStorage.set_vector_index

    async def then_deleted(self: PostgresStorage, corpus_id: str, index: Any, **given: bool) -> int:
        built = await build(self, corpus_id, index, **given)
        if corpus_id == "acme":
            await Memory(self, embedding=Turned()).delete_corpus("hr")
        return built

    monkeypatch.setattr(PostgresStorage, "set_vector_index", then_deleted)
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out.endswith(
        "acme: built, 6 vectors\nhr: nothing recorded to build any more\n"
    )
    assert (await Memory(storage, embedding=Turned()).verify_absent("hr")).clean
    async with pool.acquire() as conn:
        assert await _vectors_table(conn, "hr") is None


async def test_an_index_the_app_built_while_the_queue_waits_is_left_as_it_is(
    storage: PostgresStorage, schema: str, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _two_left_to_build(storage, schema)
    build = PostgresStorage.set_vector_index

    async def then_tuned(self: PostgresStorage, corpus_id: str, index: Any, **given: bool) -> int:
        built = await build(self, corpus_id, index, **given)
        if corpus_id == "acme":
            await build(self, "hr", HNSW(m=16), rebuild=False)
        return built

    monkeypatch.setattr(PostgresStorage, "set_vector_index", then_tuned)
    assert await _vectors("index", schema) == 0
    assert await storage.get_vector_index("hr") == HNSW(m=16, precision="full")
    assert await _record(pool, "hr") is None


async def test_a_full_index_moved_past_2000_dimensions_builds_as_half(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Broad()).stage_vectors("acme")
    assert await _swap(name, "broad-model") == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme builds at half precision: full holds 2000 dimensions, "
        "and broad-model has 2008\n"
    ) in out
    assert "acme: building HNSW(m=8, precision=half)\n" in out
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="half")
    assert await _record(pool) is None


async def test_past_4000_dimensions_swap_says_to_pass_binary_before_it_starts(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await Memory(storage, embedding=Vast()).stage_vectors("acme")
    before, filled = await _rows(pool), await _staged(pool)
    assert await _swap(name, "vast-model") == 1
    assert capsys.readouterr().err == (
        "cli: the index of acme reaches 4000 dimensions and vast-model has 4008 — pass --no-index, "
        "then rfnry-memory vectors index --precision binary builds it\n"
    )
    assert await _rows(pool) == before
    assert await _staged(pool) == filled
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _swap(name, "vast-model", "--no-index") == 0
    out = capsys.readouterr().out
    assert (
        "  the index of acme reaches 4000 dimensions: "
        "rfnry-memory vectors index --precision binary builds it\n"
    ) in out
    assert out.endswith(
        f"next: rfnry-memory vectors index --precision binary --corpus acme --schema {name}\n"
    )
    assert await _vectors("index", name) == 1
    assert capsys.readouterr().err == (
        "cli: the index in acme reaches 4000 dimensions and this schema holds 4008 — "
        f"run rfnry-memory vectors index --precision binary --corpus acme --schema {name}\n"
    )
    assert await _record(pool) is not None
    assert await _vectors("index", name, "--precision", "binary") == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="binary")
    assert await _record(pool) is None
    vast = Memory(PostgresStorage(await schemas.pool(name)), embedding=Vast())
    await vast.check()
    assert (await vast.search("acme", PUMP, pillars={"vector"})).hits


async def test_precision_overrides_what_was_recorded(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    assert await _swap(schema, NEW, "--no-index") == 0
    assert await _vectors("index", schema, "--precision", "half") == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="half")


async def test_a_failed_rebuild_says_the_swap_held_and_what_finishes_it(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    new = await _filling(storage)
    async with _role(schemas, name, pool) as app, pool.acquire() as conn:
        await conn.execute(f"ALTER TABLE {await _vectors_table(conn, 'acme')} OWNER TO {app}")
        await conn.execute(f"REVOKE CREATE ON SCHEMA {name} FROM {app}")
        assert await _swap_as(app, name) == 3
        told = capsys.readouterr()
        assert f"schema {name}: new-model is on the rows\n" in told.out
        assert "next:" not in told.out
        assert told.err.startswith("cli: stopped by StorageError: postgres: storage rejected ")
        assert told.err.endswith(f" — {_unbuilt(name, HELD)}")
        assert ":role@" not in told.out + told.err
        assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
        assert (await new.search("acme", PUMP, pillars={"vector"})).hits
        recorded = await _record(pool)
        assert recorded is not None and recorded["owner"] == app
        assert await _vectors_as(app, "index", name) == 1
        assert capsys.readouterr().err.endswith(f" — {_unbuilt(name)}")
        await conn.execute(f"GRANT CREATE ON SCHEMA {name} TO {app}")
        assert await _vectors_as(app, "index", name) == 0
        assert await _owner(conn) == app
    assert await _record(pool) is None


async def test_an_interrupted_build_says_the_swap_held_and_running_index_finishes_it(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    old = await _filled(storage)
    await old.set_vector_index("acme", HNSW(m=8))
    await _filling(storage)
    swap = ("vectors", "swap", "--schema", schema, "--model", NEW, "--yes")
    async with pool.acquire() as reading, reading.transaction(isolation="repeatable_read"):
        await reading.fetchval("SELECT count(*) FROM memory_chunks")
        code, left = await _interrupted(database, *swap)
    assert code == 3
    assert left == f"cli: stopped — {_unbuilt(schema, HELD)}"
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _record(pool) is not None
    assert await storage.get_vector_index("acme") is None
    async with pool.acquire() as reading, reading.transaction(isolation="repeatable_read"):
        await reading.fetchval("SELECT count(*) FROM memory_chunks")
        code, left = await _interrupted(database, "vectors", "index", "--schema", schema)
    assert code == 130
    assert left == f"cli: stopped — {_unbuilt(schema)}"
    assert await _record(pool) is not None
    assert await _vectors("index", schema) == 0
    assert await storage.get_vector_index("acme") == HNSW(m=8, precision="full")
    assert await _record(pool) is None


async def test_a_failed_forget_says_what_stays_deleted_and_what_finishes_it(
    storage: PostgresStorage,
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)

    async def lost(conn: Any, *, model: str, corpus_ids: list[str]) -> int:
        raise OSError(f"lost {os.environ['RFNRY_TEST_POSTGRES_URL']}")

    monkeypatch.setattr(cli_vectors, "forget_staged", lost)
    assert await _forget(schema, NEW) == 1
    assert capsys.readouterr().err == (
        "cli: stopped by OSError: lost <url> — "
        "what was deleted stays deleted; running it again finishes\n"
    )


class Addressed(HashEmbedding):
    model = "hf://org/embed-v2"


async def test_a_corpus_and_a_model_named_like_addresses_go_through_every_action(
    storage: PostgresStorage, schema: str
) -> None:
    tenant = "tenant:ops@acme.example"
    model = Addressed.model
    served = Memory(storage, embedding=HashEmbedding())
    await served.check()
    await served.remember(tenant, "A story of the tenant.")
    await Memory(storage, embedding=Addressed(), pillars={"vector"}).stage_vectors(tenant)
    scope = ("--corpus", tenant)
    assert await _vectors("status", schema, "--model", model, *scope) == 0
    assert await _swap(schema, model, *scope) == 0
    assert await _swap(schema, OLD, *scope) == 0
    assert await storage.staged_counts(tenant) == {model: 1}
    assert await _rename(schema, model, "hf://org/embed-v3", *scope) == 0
    assert await _forget(schema, "hf://org/embed-v3", *scope) == 0
    assert await storage.staged_counts(tenant) == {}


async def _one_passage(storage: PostgresStorage) -> None:
    memory = Memory(storage, embedding=HashEmbedding())
    await memory.check()
    await memory.remember("acme", "The only story.")


async def test_one_vector_reads_in_the_singular(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _one_passage(storage)
    assert await _rename(schema, OLD, RENAMED) == 0
    printed = capsys.readouterr().out
    assert "  1 vector labeled hash-8 takes the label renamed-model, in acme;" in printed
    assert f"schema {schema}: 1 vector of hash-8 is labeled renamed-model\n" in printed
    await _filling(storage)
    assert await _forget(schema, NEW) == 0
    assert "  1 staged vector of new-model goes, in acme\n" in capsys.readouterr().out
    await _filling(storage)
    assert await _swap(schema) == 0
    assert "  1 vector of new-model goes onto the rows, in acme\n" in capsys.readouterr().out


async def test_forget_deletes_a_models_staged_rows_in_scope(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    for corpus_id in ("acme", "hr"):
        await _filled(storage, corpus_id)
        await _filling(storage, corpus_id)
        await Memory(storage, embedding=Wide()).stage_vectors(corpus_id)
    rows = await _rows(pool)
    assert await _forget(schema, NEW, "--corpus", "acme") == 0
    assert capsys.readouterr().out == (
        "  6 staged vectors of new-model go, in acme\n"
        f"schema {schema}: deleting in batches of 10,000, each committed\n"
        f"schema {schema}: 6 staged vectors of new-model deleted\n"
        "a rollback to new-model now means staging it again\n"
    )
    assert await storage.staged_counts("acme") == {WIDE: 6}
    assert await storage.staged_counts("hr") == {NEW: 6, WIDE: 6}
    assert await _rows(pool) == rows
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("hr") == {WIDE: 6}


async def test_forget_with_nothing_staged_does_nothing(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    assert await _forget(schema, NEW) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for new-model\n"
    assert await _rows(pool) == rows


async def test_forget_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("forget", schema, "--model", NEW) == 0
    assert capsys.readouterr().out == (
        "  6 staged vectors of new-model go, in acme\n"
        "what goes is gone: moving to new-model after this means staging it again\n"
        "nothing was run\n"
    )
    assert await storage.staged_counts("acme") == {NEW: 6}
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("forget", schema, "--model", NEW) == 0
    assert await storage.staged_counts("acme") == {}


async def test_after_forget_a_rollback_needs_the_old_model_staged_again(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _filled(storage)
    await _filling(storage)
    assert await _swap(schema) == 0
    assert await _forget(schema, OLD) == 0
    assert await _swap(schema, OLD) == 1
    assert "nothing is staged for hash-8" in capsys.readouterr().err
    await old.stage_vectors("acme")
    assert await _swap(schema, OLD) == 0


class AfterEachDelete:
    def __init__(self, conn: Any, after: Any) -> None:
        self.conn = conn
        self.after = after

    async def fetchval(self, statement: str, *args: Any) -> Any:
        return await self.conn.fetchval(statement, *args)

    async def execute(self, statement: str, *args: Any) -> str:
        done = str(await self.conn.execute(statement, *args))
        if statement.startswith("DELETE"):
            await self.after()
        return done


async def test_forget_commits_each_batch(
    storage: PostgresStorage, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr(switch, "_FORGET_BATCH", 2)
    left: list[int] = []

    async def seen_from_another_connection() -> None:
        left.append((await storage.staged_counts("acme")).get(NEW, 0))

    async with pool.acquire() as conn:
        watched = AfterEachDelete(conn, seen_from_another_connection)
        assert await forget_staged(watched, model=NEW, corpus_ids=["acme"]) == 6
    assert left == [4, 2, 0, 0]


async def test_nothing_moves_a_model_while_forget_runs(
    storage: PostgresStorage, schema: str, pool: Any, monkeypatch: pytest.MonkeyPatch
) -> None:
    await _filled(storage)
    await _filling(storage)
    monkeypatch.setattr(switch, "_FORGET_BATCH", 2)
    told: list[Ran] = []

    async def try_to_move() -> None:
        if not told:
            told.append(await _swap(schema, NEW, "--allow-missing"))
            told.append(await _rename(schema, OLD, RENAMED))
            told.append(await _forget(schema, NEW))

    async with pool.acquire() as conn:
        watched = AfterEachDelete(conn, try_to_move)
        assert await forget_staged(watched, model=NEW, corpus_ids=["acme"]) == 6
    assert [int(ran) for ran in told] == [1, 1, 1]
    assert all("another command is changing this schema" in ran.printed for ran in told)
    assert {model for model, *_ in (await _rows(pool)).values()} == {OLD}


async def test_forget_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _forget(schema, NEW) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await storage.staged_counts("acme") == {NEW: 6}


async def test_an_interrupted_forget_says_what_stays(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    await _filled(storage)
    await _filling(storage)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        code, left = await _interrupted(
            database, "vectors", "forget", "--schema", schema, "--model", NEW, "--yes"
        )
    assert code == 130
    assert left == "cli: stopped — what was deleted stays deleted; running it again finishes\n"
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("acme") == {}


async def _held_both_ways(storage: PostgresStorage, corpus_id: str = "acme") -> Memory:
    old = await _filled(storage, corpus_id)
    await Memory(storage, embedding=Turned()).reembed(corpus_id, limit=3)
    await old.stage_vectors(corpus_id)
    return old


async def test_rename_changes_the_label_and_no_vector(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    old = await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    assert {model for model, *_ in rows.values()} == {OLD, NEW}
    assert {model for model, *_ in staged} == {OLD}
    assert await _rename(schema, OLD, RENAMED) == 0
    assert capsys.readouterr().out == (
        "  6 vectors labeled hash-8 take the label renamed-model, in acme; no vector changes\n"
        "  writes wait until the rename ends; reads go on; after it a Memory of hash-8 finds "
        "nothing by vector there\n"
        f"schema {schema}: 6 vectors of hash-8 are labeled renamed-model\n"
        f"next: rfnry-memory vectors status --model renamed-model --schema {schema}\n"
    )
    assert await _rows(pool) == {
        ref: (RENAMED if model == OLD else model, vector, settings)
        for ref, (model, vector, settings) in rows.items()
    }
    assert await _staged(pool) == {
        (RENAMED, kind, passage): held for (_, kind, passage), held in staged.items()
    }
    renamed = Memory(storage, embedding=Renamed())
    assert (await renamed.search("acme", PUMP, pillars={"vector"})).hits
    assert not (await old.search("acme", PUMP, pillars={"vector"})).hits


async def test_rename_refuses_a_name_already_in_scope(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage, "acme")
    await _filled(storage, "hr")
    await _filling(storage, "hr")
    held = {c: (await _rows(pool, c), await _staged(pool, c)) for c in ("acme", "hr")}
    assert await _rename(schema, OLD, NEW) == 1
    assert capsys.readouterr().err == (
        "cli: new-model already labels 6 staged vectors, in hr — give another name, or drop them "
        f"with rfnry-memory vectors forget --model new-model --schema {schema}\n"
    )
    assert await _rename(schema, NEW, OLD) == 1
    assert capsys.readouterr().err == _on_rows("12 vectors on rows, in acme, hr", NEW, OLD, schema)
    assert {c: (await _rows(pool, c), await _staged(pool, c)) for c in held} == held
    assert await _rename(schema, OLD, NEW, "--corpus", "acme") == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {NEW}
    assert await _rows(pool, "hr") == held["hr"][0]
    assert await _staged(pool, "hr") == held["hr"][1]


def _on_rows(held: str, model: str, to: str, schema: str, *scope: str, aside: str = "aside") -> str:
    named = "".join(f" {flag}" for flag in scope)
    return (
        f"cli: {to} already labels {held}, and vectors forget clears only what is staged — "
        f"give another name, or hold writes and run rfnry-memory vectors rename --model {to} "
        f"--to {aside}{named} --schema {schema}, then rfnry-memory vectors rename --model {model} "
        f"--to {to}{named} --schema {schema}, then reembed under the new adapter, which writes "
        "only the rows set aside\n"
    )


async def test_rename_counts_rows_apart_from_staged_and_keeps_to_its_scope(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _held_both_ways(storage)
    assert await _rename(schema, NEW, OLD, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "3 vectors on rows and 3 staged, in acme", NEW, OLD, schema, "--corpus", "acme"
    )


async def _deployed_first(storage: PostgresStorage) -> Memory:
    await _filled(storage)
    deployed = Memory(storage, embedding=Renamed())
    await deployed.remember("acme", "Written after the deploy.")
    return deployed


async def test_a_rename_after_the_deploy_goes_through_a_name_set_aside(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    before = await _rows(pool)
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows("1 vector on rows, in acme", OLD, RENAMED, schema)
    assert await _forget(schema, RENAMED) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for {RENAMED}\n"
    assert await _rename(schema, OLD, RENAMED) == 1
    assert await _rename(schema, RENAMED, "aside") == 0
    assert await _rename(schema, OLD, RENAMED) == 0
    assert await deployed.reembed("acme") == 1
    assert await _rows(pool) == {
        ref: (RENAMED, vector, settings) for ref, (_, vector, settings) in before.items()
    }
    assert (await deployed.search("acme", PUMP, pillars={"vector"})).hits


async def test_a_write_between_the_two_renames_refuses_the_second(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    assert await _rename(schema, RENAMED, "aside") == 0
    await deployed.remember("acme", "Written between the two.")
    before = await _rows(pool)
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, aside="aside-2"
    )
    assert await _rows(pool) == before
    assert await _rename(schema, RENAMED, "aside-2") == 0
    assert await _rename(schema, OLD, RENAMED) == 0


async def test_the_way_out_of_a_taken_name_passes_every_label_taken_in_its_scope(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    deployed = await _deployed_first(storage)
    await _filled(storage, "hr")
    assert await _rename(schema, RENAMED, "aside", "--corpus", "acme") == 0
    await deployed.remember("acme", "Written between the two.")
    assert await _rename(schema, RENAMED, "aside-2", "--corpus", "acme") == 0
    await deployed.remember("acme", "Written after the second.")
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, "aside", "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, "aside", schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, "aside-3", "--corpus", "hr") == 0
    capsys.readouterr()
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, "--corpus", "acme", aside="aside-3"
    )
    assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == _on_rows(
        "1 vector on rows, in acme", OLD, RENAMED, schema, aside="aside-4"
    )


async def test_rename_stays_in_the_corpora_it_names(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _held_both_ways(storage, "acme")
    await _held_both_ways(storage, "hr")
    rows, staged = await _rows(pool, "hr"), await _staged(pool, "hr")
    assert await _rename(schema, OLD, RENAMED, "--corpus", "acme") == 0
    assert await _rows(pool, "hr") == rows
    assert await _staged(pool, "hr") == staged
    assert {model for model, *_ in await _staged(pool)} == {RENAMED}
    assert await _rename(schema, OLD, RENAMED, "--corpus", "hr") == 0
    assert {model for model, *_ in await _staged(pool, "hr")} == {RENAMED}


async def test_rename_of_a_name_nothing_holds_says_so(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    assert await _rename(schema, "typo", RENAMED) == 1
    assert capsys.readouterr().err == (
        "cli: no vector here is labeled typo — check the name with "
        f"rfnry-memory vectors status --schema {schema}\n"
    )


async def test_a_schema_with_no_corpora_has_nothing_to_index_forget_or_rename(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    assert await _vectors("index", schema) == 0
    assert capsys.readouterr().out == f"schema {schema}: no index to build\n"
    assert await _forget(schema, NEW) == 0
    assert capsys.readouterr().out == f"schema {schema}: nothing staged for new-model\n"
    assert await _rename(schema, NEW, RENAMED) == 1


async def test_rename_asks_unless_yes(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    monkeypatch.setattr("builtins.input", lambda prompt: "n")
    assert await _vectors("rename", schema, "--model", OLD, "--to", RENAMED) == 0
    assert capsys.readouterr().out.endswith(
        "one transaction: every label moves or none does, and a stop changes nothing\n"
        "nothing was run\n"
    )
    assert await _rows(pool) == rows
    monkeypatch.setattr("builtins.input", lambda prompt: "y")
    assert await _vectors("rename", schema, "--model", OLD, "--to", RENAMED) == 0
    assert {model for model, *_ in (await _rows(pool)).values()} == {RENAMED}


async def test_a_running_stage_of_either_name_stops_a_rename_and_a_forget(
    storage: PostgresStorage,
    schema: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(1, "await asyncio.sleep(3600)"))
    process = await _running(
        "vectors",
        "stage",
        "--schema",
        schema,
        "--embedding",
        f"{adapter}:Failing",
        "--batch",
        "2",
        cwd=tmp_path,
    )
    assert process.stderr is not None
    refused = (
        "postgres: a stage of new-model is running, and it would go on writing under that name — "
        "stop it, then run the command again\n"
    )
    try:
        async with asyncio.timeout(WAIT):
            while b"acme: 2 of 6" not in await process.stderr.readline():
                pass
        assert await _rename(schema, NEW, RENAMED) == 1
        assert capsys.readouterr().err == refused
        assert await _forget(schema, NEW) == 1
        assert capsys.readouterr().err == refused
        assert await storage.staged_counts("acme") == {NEW: 2}
    finally:
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            await process.communicate()
    assert await _rename(schema, NEW, RENAMED) == 0
    assert await storage.staged_counts("acme") == {RENAMED: 2}


async def test_a_running_stage_of_the_model_stops_a_swap(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    adapter = _adapter(tmp_path, monkeypatch, _failing_after(1, "await asyncio.sleep(3600)"))
    process = await _running(
        "vectors",
        "stage",
        "--schema",
        schema,
        "--embedding",
        f"{adapter}:Failing",
        "--batch",
        "2",
        cwd=tmp_path,
    )
    assert process.stderr is not None
    try:
        async with asyncio.timeout(WAIT):
            while b"acme: 2 of 6" not in await process.stderr.readline():
                pass
        assert await _swap(schema, NEW, "--allow-missing") == 1
        assert capsys.readouterr().err == (
            "postgres: a stage of new-model is running, and it would go on writing under that "
            "name — stop it, then run the command again\n"
        )
        assert await _rows(pool) == rows
        assert await storage.staged_counts("acme") == {NEW: 2}
    finally:
        process.send_signal(signal.SIGINT)
        async with asyncio.timeout(WAIT):
            await process.communicate()
    assert await _swap(schema, NEW, "--allow-missing") == 0
    assert await storage.staged_counts("acme") == {OLD: 2}


async def test_a_rename_to_a_name_a_stage_has_yet_to_write_is_refused(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    async with pool.acquire() as staging:
        await hold_staging(staging, NEW)
        try:
            assert await _rename(schema, OLD, NEW) == 1
        finally:
            await staging.execute("SELECT pg_advisory_unlock_all()")
    assert capsys.readouterr().err == (
        "postgres: a stage of new-model is running, and it would go on writing under that name — "
        "stop it, then run the command again\n"
    )
    assert await _rows(pool) == rows


async def test_a_stage_of_a_name_being_swapped_renamed_or_forgotten_is_refused(
    storage: PostgresStorage,
    schema: str,
    pool: Any,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)
    adapter = _adapter(tmp_path, monkeypatch)
    async with pool.acquire() as held, held.transaction():
        await held.execute(f"SELECT pg_advisory_xact_lock({_staging_key(NEW)})")
        assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 1
    assert capsys.readouterr().err == (
        "postgres: a swap, a forget or a rename is at work on new-model — "
        "run the command again when that ends\n"
    )
    assert await storage.staged_counts("acme") == {}
    assert await _vectors("stage", schema, "--embedding", f"{adapter}:New") == 0


async def test_a_batch_being_staged_is_renamed_with_the_rest(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _filled(storage)
    await _filling(storage, limit=4)
    staged = await _staged(pool)
    chunk_id = next(passage for _, kind, passage in staged if kind == "chunk")
    story_id = next(passage for kind, passage in await _rows(pool) if kind == "story")
    batch = [("chunk", chunk_id), ("story", story_id)]
    async with pool.acquire() as held, held.transaction():
        await held.execute(
            "SELECT 1 FROM memory_staged_vectors WHERE corpus_id = 'acme' AND passage_id = $1 "
            "FOR UPDATE",
            chunk_id,
        )
        staging = spawn(storage.stage("acme", batch, [[0.5] * 8] * 2, NEW, settings=None))
        await _waiting(database.admin, 1)
        renaming = spawn(_rename(schema, NEW, RENAMED))
        await _waiting(database.admin, 2)
    async with asyncio.timeout(WAIT):
        await staging
        assert await renaming == 0
    assert await storage.staged_counts("acme") == {RENAMED: 5}


async def test_a_run_of_your_own_code_that_outlives_a_rename_is_seen_and_can_be_cleared(
    storage: PostgresStorage, schema: str, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    new = await _filling(storage, limit=4)
    assert await _rename(schema, NEW, RENAMED) == 0
    await new.stage_vectors("acme")
    assert await storage.staged_counts("acme") == {NEW: 6, RENAMED: 4}
    assert await _vectors("status", schema) == 0
    capsys.readouterr()
    assert await _rename(schema, NEW, RENAMED) == 1
    assert "rfnry-memory vectors forget" in capsys.readouterr().err
    assert await _forget(schema, NEW) == 0
    assert await storage.staged_counts("acme") == {RENAMED: 4}


async def test_a_read_during_a_rename_sees_every_old_label_and_a_write_waits(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    old = await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        renaming = spawn(_rename(schema, OLD, RENAMED))
        await _waiting(database.admin, 1)
        writing = spawn(old.ingest("acme", name="late", text="A line written late."))
        await _waiting(database.admin, 2)
        assert not writing.done()
        assert await _rows(pool) == rows
        assert await _staged(pool) == staged
    async with asyncio.timeout(WAIT):
        assert await renaming == 0
        await writing
    assert {model for model, *_ in await _staged(pool)} == {RENAMED}


async def test_a_rename_of_one_corpus_holds_the_writes_of_the_others_too(
    storage: PostgresStorage, schema: str, pool: Any, database: Database, spawn: Spawned
) -> None:
    await _held_both_ways(storage)
    hr = await _filled(storage, "hr")
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        renaming = spawn(_rename(schema, OLD, RENAMED, "--corpus", "acme"))
        await _waiting(database.admin, 1)
        writing = spawn(hr.ingest("hr", name="late", text="A line written late."))
        await _waiting(database.admin, 2)
        assert not writing.done()
    async with asyncio.timeout(WAIT):
        assert await renaming == 0
        await writing


async def test_a_label_with_no_vector_is_neither_moved_nor_a_name_taken(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    stories = sorted(passage for kind, passage in await _rows(pool) if kind == "story")
    async with pool.acquire() as conn:
        await conn.execute(
            "UPDATE memory_stories SET embedding = NULL WHERE story_id = $1", stories[0]
        )
        await conn.execute(
            "UPDATE memory_stories SET embedding = NULL, embedding_model = $2 WHERE story_id = $1",
            stories[1],
            RENAMED,
        )
    assert await _rename(schema, OLD, RENAMED) == 0
    assert "4 vectors of hash-8 are labeled renamed-model" in capsys.readouterr().out
    assert (await _rows(pool))[("story", stories[0])][0] == OLD


async def test_a_rename_that_fails_part_way_changes_nothing(
    schemas: Schemas, capsys: pytest.CaptureFixture[str]
) -> None:
    name, pool, storage = await _own(schemas)
    await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        await conn.execute(
            "CREATE FUNCTION refused() RETURNS trigger LANGUAGE plpgsql AS "
            "$$ BEGIN RAISE EXCEPTION 'tests: refused' USING ERRCODE = 'XX000'; END $$"
        )
        await conn.execute(
            "CREATE TRIGGER refusing BEFORE UPDATE ON memory_staged_vectors "
            "FOR EACH ROW EXECUTE FUNCTION refused()"
        )
    assert await _rename(name, OLD, RENAMED) == 1
    assert capsys.readouterr().err == (
        "postgres: the rename rolled back, nothing changed: tests: refused\n"
    )
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


async def test_a_name_taken_after_the_plan_stops_the_rename(
    storage: PostgresStorage, pool: Any
) -> None:
    await _filled(storage)
    await _filling(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as conn:
        with pytest.raises(
            ConfigurationError,
            match="^postgres: new-model came to label vectors after the plan — "
            "run the command again$",
        ):
            await rename_model(conn, model=OLD, to=NEW, corpus_ids=["acme"])
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


async def test_a_rename_stops_while_another_command_changes_the_schema(
    storage: PostgresStorage, schema: str, pool: Any, capsys: pytest.CaptureFixture[str]
) -> None:
    await _filled(storage)
    rows = await _rows(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute(migrations.LOCK)
        assert await _rename(schema, OLD, RENAMED) == 1
    assert capsys.readouterr().err == (
        "postgres: another command is changing this schema — run the command again when that ends\n"
    )
    assert await _rows(pool) == rows


async def test_an_interrupted_rename_changes_nothing(
    storage: PostgresStorage, schema: str, pool: Any, database: Database
) -> None:
    await _held_both_ways(storage)
    rows, staged = await _rows(pool), await _staged(pool)
    async with pool.acquire() as held, held.transaction():
        await held.execute("SELECT 1 FROM memory_staged_vectors FOR UPDATE")
        code, left = await _interrupted(
            database,
            "vectors",
            "rename",
            "--schema",
            schema,
            "--model",
            OLD,
            "--to",
            RENAMED,
            "--yes",
        )
    assert code == 130
    assert left == "cli: stopped — the rename rolled back, nothing changed\n"
    assert await _rows(pool) == rows
    assert await _staged(pool) == staged


@pytest.mark.parametrize(
    "lost", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_a_renames_commit_does_not_claim_nothing_changed(
    lost: BaseException,
) -> None:
    with pytest.raises(
        StorageError,
        match="^postgres: the rename stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which",
    ):
        await rename_model(LostAtCommit(lost), model=OLD, to=NEW, corpus_ids=["acme"])


@pytest.mark.parametrize(
    "stop", [ConnectionResetError("tests: the connection went"), asyncio.CancelledError()]
)
async def test_a_stop_during_a_renames_commit_exits_4_and_names_status(
    storage: PostgresStorage,
    schema: str,
    stop: BaseException,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    await _filled(storage)

    async def lost(_: Any, *, model: str, to: str, corpus_ids: list[str]) -> int:
        return await rename_model(LostAtCommit(stop), model=model, to=to, corpus_ids=corpus_ids)

    monkeypatch.setattr(cli_vectors, "rename_model", lost)
    assert await _rename(schema, OLD, RENAMED) == 4
    told = capsys.readouterr()
    assert told.err == (
        "postgres: the rename stopped during its commit, so whether it landed is unknown — "
        "rfnry-memory vectors status shows which\n"
    )
    assert "next:" not in told.out


async def test_a_rename_leaves_the_statistics_true(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage)
    async with pool.acquire() as conn:
        await conn.execute("ANALYZE memory_chunks, memory_stories")
    assert await _rename(schema, OLD, RENAMED) == 0
    async with pool.acquire() as conn:
        held = await conn.fetch(
            "SELECT most_common_vals::text AS vals FROM pg_stats "
            "WHERE schemaname = current_schema() AND tablename IN "
            "('memory_chunks', 'memory_stories') AND attname = 'embedding_model'"
        )
    assert len(held) == 2 and all(RENAMED in r["vals"] for r in held)


ODD_TO = "it's $$ 100% \\ other"


async def test_names_that_are_no_sql_forget_and_rename_like_any_other(
    storage: PostgresStorage, schema: str, pool: Any
) -> None:
    await _filled(storage, ODD_CORPUS)
    await Memory(storage, embedding=Odd()).stage_vectors(ODD_CORPUS)
    assert await _rename(schema, ODD_MODEL, ODD_TO, "--corpus", ODD_CORPUS) == 0
    assert await storage.staged_counts(ODD_CORPUS) == {ODD_TO: 6}
    assert await _forget(schema, ODD_TO, "--corpus", ODD_CORPUS) == 0
    assert await storage.staged_counts(ODD_CORPUS) == {}
