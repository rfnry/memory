import argparse
from pathlib import Path

import pytest

from rfnry_memory import cli_vectors
from rfnry_memory.errors import InputError
from rfnry_memory.postgres.switch import CorpusVectors


def test_the_table_lines_up_at_millions() -> None:
    rows = [
        CorpusVectors(
            corpus_id="acme",
            passages=1204331,
            on_row={"old-model": 1204331},
            staged={"new-model": 804000},
            staged_dimensions={"new-model": 1024},
            missing={"new-model": 400331, "old-model": 0},
            index="HNSW(m=32, ef_construction=128)",
            to_build=None,
        ),
        CorpusVectors(
            corpus_id="hr",
            passages=41200,
            on_row={"old-model": 41200},
            staged={"new-model": 41200},
            staged_dimensions={"new-model": 1024},
            missing={"new-model": 0, "old-model": 0},
            index=None,
            to_build=None,
        ),
    ]
    assert cli_vectors._table(rows) == [
        "corpus  passages   on the row           staged               index",
        "acme    1,204,331  old-model 1,204,331  new-model   804,000  HNSW(m=32, ef_construction=128)",
        "hr         41,200  old-model    41,200  new-model    41,200  none",
    ]


def test_status_names_a_few_corpora_and_counts_many() -> None:
    rows = [
        CorpusVectors(
            corpus_id=f"c{i}",
            passages=2,
            on_row={},
            staged={},
            staged_dimensions={},
            missing={"new-model": 2},
            index=None,
            to_build=None,
        )
        for i in range(6)
    ]
    args = argparse.Namespace(schema="memory", corpus=None, model="new-model")
    assert cli_vectors._next_steps(args, rows, 8) == [
        "new-model: 12 missing, in 6 corpora — run rfnry-memory vectors stage "
        "--embedding MODULE:NAME"
    ]
    assert cli_vectors._next_steps(args, rows[:2], 8) == [
        "new-model: 4 missing, in c0, c1 — run rfnry-memory vectors stage "
        "--embedding MODULE:NAME --corpus c0 --corpus c1"
    ]


@pytest.mark.parametrize("shape", ["a folder", "under a file"])
def test_a_refused_path_that_cannot_be_written_is_refused_and_not_made(
    tmp_path: Path, shape: str
) -> None:
    (tmp_path / "file").write_text("")
    path = tmp_path if shape == "a folder" else tmp_path / "file" / "refused.jsonl"
    with pytest.raises(InputError, match=r"^cli: --refused cannot be written: "):
        cli_vectors._check_writable(str(path))


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
