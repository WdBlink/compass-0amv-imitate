"""Portable panel-loader regressions; all snapshots are synthetic."""
import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent))
import panel


@pytest.fixture(autouse=True)
def isolate_snapshot_environment(monkeypatch):
    monkeypatch.delenv("COMPASS_SNAPSHOT_ROOT", raising=False)


@pytest.mark.parametrize("cached", [False, True])
def test_empty_panel_is_rejected(tmp_path, monkeypatch, cached):
    raw = tmp_path / "raw"
    raw.mkdir()
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "RAW_DIR", str(raw))
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    if cached:
        arrays = {name: np.empty((0, 0)) for name in (
            "close", "adj_factor", "vol_shares", "amount_cny", "float_shares",
            "free_shares", "circ_mv", "turnover_rate")}
        np.savez(cache, dates=np.array([], dtype=str), codes=np.array([], dtype=str),
                 board=np.array([], dtype=str), **arrays)
    with pytest.raises(ValueError, match="empty"):
        panel.build_panel()
    if not cached:
        assert not cache.exists()


def write_snapshot(root, close=10.0):
    raw = root / "raw"
    raw.mkdir(parents=True)
    records = {
        "daily": {"close": close, "vol": 2.0, "amount": 3.0},
        "daily_basic": {"turnover_rate": 4.0, "float_share": 5.0,
                        "free_share": 6.0, "circ_mv": 7.0},
        "adj_factor": {"adj_factor": 1.0},
    }
    for endpoint, values in records.items():
        record = {"trade_date": "20260105", "ts_code": "600000.SH", **values}
        (raw / f"{endpoint}_20260105.json").write_text(json.dumps({
            "code": 0, "data": {"fields": list(record), "items": [list(record.values())]}
        }))
    return root


@pytest.mark.parametrize("via_env", [False, True])
def test_configured_root_never_reuses_another_snapshot(tmp_path, monkeypatch, via_env):
    root = write_snapshot(tmp_path / "first")
    other = write_snapshot(tmp_path / "second", close=20.0)
    monkeypatch.setattr(panel, "RAW_DIR", str(root / "raw"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(tmp_path / "cache.npz"))
    panel.build_panel(use_cache=False)
    if via_env:
        monkeypatch.setenv("COMPASS_SNAPSHOT_ROOT", str(other))
        loaded = panel.build_panel()
    else:
        loaded = panel.build_panel(snapshot_root=other)
    assert loaded.close.shape == (1, 1)
    assert loaded.close[0, 0] == 20.0
    assert loaded.vol_shares[0, 0] == 200.0
    assert loaded.amount_cny[0, 0] == 3000.0
    assert loaded.float_shares[0, 0] == 50000.0
    assert loaded.free_shares[0, 0] == 60000.0
    assert loaded.circ_mv[0, 0] == 70000.0
    assert loaded.board.tolist() == ["main_a"]



@pytest.mark.parametrize("cached", [False, True])
def test_required_dates_checked_before_return_or_write(tmp_path, monkeypatch, cached):
    root = write_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "RAW_DIR", str(root / "raw"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    if cached:
        panel.build_panel(use_cache=False)
    with pytest.raises(ValueError, match="required.*20260910"):
        panel.build_panel(required_dates=["20260910"])
    if not cached:
        assert not cache.exists()


@pytest.mark.parametrize("cached", [False, True])
def test_no_usable_quotes_is_rejected(tmp_path, monkeypatch, cached):
    root = write_snapshot(tmp_path / "snapshot", close=None if not cached else 10.0)
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "RAW_DIR", str(root / "raw"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    if cached:
        panel.build_panel(use_cache=False)
        with np.load(cache) as z:
            arrays = {key: z[key] for key in z.files}
        arrays["close"][:] = np.nan
        np.savez(cache, **arrays)
    with pytest.raises(ValueError, match="usable"):
        panel.build_panel()
    if not cached:
        assert not cache.exists()


def test_missing_raw_directory_does_not_write_cache(tmp_path, monkeypatch):
    monkeypatch.setattr(panel, "RAW_DIR", str(tmp_path / "missing"))
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    with pytest.raises(FileNotFoundError, match="raw|snapshot"):
        panel.build_panel(use_cache=False)
    assert not cache.exists()


def test_valid_cache_round_trip_and_required_date(tmp_path, monkeypatch):
    root = write_snapshot(tmp_path / "snapshot")
    monkeypatch.setattr(panel, "RAW_DIR", str(root / "raw"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(tmp_path / "cache.npz"))
    expected = panel.build_panel(use_cache=False, required_dates=["20260105"])
    monkeypatch.setattr(panel, "RAW_DIR", str(tmp_path / "missing"))
    got = panel.build_panel(required_dates=["20260105"])
    assert got.n_dates > 0 and got.n_stocks > 0
    for name, values in vars(expected).items():
        np.testing.assert_array_equal(getattr(got, name), values)


def test_explicit_root_overrides_environment_and_missing_root_fails(tmp_path, monkeypatch):
    root = write_snapshot(tmp_path / "snapshot")
    monkeypatch.setenv("COMPASS_SNAPSHOT_ROOT", str(tmp_path / "missing-env"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(tmp_path / "cache.npz"))
    got = panel.build_panel(snapshot_root=root, required_dates=["20260105"])
    assert got.close[0, 0] == 10.0
    assert not (tmp_path / "cache.npz").exists()
    with pytest.raises(FileNotFoundError):
        panel.build_panel(snapshot_root=tmp_path / "missing-argument")


def test_cache_with_empty_matrix_cannot_broadcast_to_valid_panel(tmp_path, monkeypatch):
    root = write_snapshot(tmp_path / "snapshot")
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "RAW_DIR", str(root / "raw"))
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    panel.build_panel(use_cache=False)
    with np.load(cache) as z:
        arrays = {key: z[key] for key in z.files}
    arrays["amount_cny"] = np.empty((0, 0))
    np.savez(cache, **arrays)
    with pytest.raises(ValueError, match="shape"):
        panel.build_panel()
