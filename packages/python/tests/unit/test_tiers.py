import re
from pathlib import Path

import pytest
from e2e import conftest as live

TESTS = Path(__file__).parents[1]
PASSING = re.compile(r"pytest\.(skip|importorskip)\(|\b(skipif|xfail)\b|mark\.skip\b")


def test_the_live_tier_selected_without_its_adapters_fails_naming_the_variable(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("RFNRY_LIVE_ADAPTERS", raising=False)
    with pytest.raises(
        pytest.fail.Exception,
        match="^tests: the live tier needs RFNRY_LIVE_ADAPTERS, the module path of your live adapters$",
    ):
        live.live_adapters()


def test_the_live_tier_is_deselected_by_default(pytestconfig: pytest.Config) -> None:
    assert pytestconfig.getini("addopts")[-2:] == ["-m", "not live"]


def test_no_test_passes_by_skipping() -> None:
    found = [
        f"{path.relative_to(TESTS)}:{number}"
        for path in sorted(TESTS.rglob("*.py"))
        if path != Path(__file__)
        for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1)
        if PASSING.search(line)
    ]
    assert found == []
