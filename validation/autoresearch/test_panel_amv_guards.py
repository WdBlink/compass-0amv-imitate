"""Regression tests for private-panel test gating and vacuous comparisons."""
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import test_amv
from panel import Panel


PRIVATE_TESTS = ["test_amv_reproduces_the_research_model", "test_no_lookahead",
                 "test_official_daily_change_accuracy"]


@pytest.mark.parametrize("name", PRIVATE_TESTS)
@pytest.mark.parametrize("flag", [None, "0", "true"])
def test_private_tests_require_exact_opt_in(monkeypatch, name, flag):
    if flag is None:
        monkeypatch.delenv("COMPASS_RUN_PRIVATE_TESTS", raising=False)
    else:
        monkeypatch.setenv("COMPASS_RUN_PRIVATE_TESTS", flag)
    def forbidden(*args, **kwargs):
        raise RuntimeError("private data must not be loaded")
    monkeypatch.setattr(test_amv, "build_panel", forbidden)
    with pytest.raises(pytest.skip.Exception, match="COMPASS_RUN_PRIVATE_TESTS=1"):
        getattr(test_amv, name)()


def test_empty_research_comparison_cannot_pass(monkeypatch):
    monkeypatch.setenv("COMPASS_RUN_PRIVATE_TESTS", "1")
    arrays = {name: np.empty((0, 0)) for name in (
        "close", "adj_factor", "vol_shares", "amount_cny", "float_shares",
        "free_shares", "circ_mv", "turnover_rate")}
    empty = Panel(dates=np.array([], dtype=str), codes=np.array([], dtype=str),
                  board=np.array([], dtype=str), **arrays)
    monkeypatch.setattr(test_amv, "build_panel", lambda **kwargs: empty)
    with pytest.raises(AssertionError, match="empty"):
        test_amv.test_amv_reproduces_the_research_model()


@pytest.mark.parametrize("name", PRIVATE_TESTS)
def test_opt_in_missing_data_is_not_skipped(tmp_path, monkeypatch, name):
    monkeypatch.setenv("COMPASS_RUN_PRIVATE_TESTS", "1")
    monkeypatch.setenv("COMPASS_SNAPSHOT_ROOT", str(tmp_path / "missing"))
    with pytest.raises(FileNotFoundError):
        getattr(test_amv, name)()
