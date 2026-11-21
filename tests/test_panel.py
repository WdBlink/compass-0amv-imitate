"""Portable panel-loader regressions; all snapshots are synthetic."""
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "research" / "autoresearch"))
import panel
from model import market_mask, model_components
from spec import FINAL_SPEC, LEGACY_SPEC
from verify_model import _slice_panel


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


def test_missing_turnover_rate_field_uses_volume_fallback(tmp_path):
    root = write_snapshot(tmp_path / "snapshot")
    path = root / "raw" / "daily_basic_20260105.json"
    response = json.loads(path.read_text())
    data = response["data"]
    i = data["fields"].index("turnover_rate")
    data["fields"].pop(i)
    for row in data["items"]:
        row.pop(i)
    path.write_text(json.dumps(response))
    p = panel.build_panel(snapshot_root=root)
    assert np.isnan(p.turnover_rate).all()
    current = model_components(p, FINAL_SPEC)["levels"]
    fallback = model_components(p, dict(FINAL_SPEC, turnover_source="vol"))["levels"]
    assert current[0] > 0
    np.testing.assert_array_equal(current, fallback)


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


def test_legacy_cache_reads_matching_lifecycle_events(tmp_path, monkeypatch):
    root = write_snapshot(tmp_path / "snapshot")
    raw = root / "raw"
    cache = tmp_path / "cache.npz"
    monkeypatch.setattr(panel, "RAW_DIR", str(raw))
    monkeypatch.setattr(panel, "CACHE_PATH", str(cache))
    fields = ["ts_code", "list_date", "delist_date"]
    (raw / "stock_basic_D_000000.json").write_text(json.dumps({
        "code": 0, "data": {"fields": fields, "items": [
            ["600000.SH", "20050101", "20160101"],
            ["600000.SH", "20160101", "20260107"],
        ]},
    }))
    loaded = panel.build_panel(use_cache=False)
    assert loaded.delist_dates.tolist() == ["20260107"]
    assert panel.build_panel().delist_dates.tolist() == ["20260107"]
    np.savez(cache, **{name: values for name, values in vars(loaded).items() if name != "delist_dates"})
    assert panel.build_panel().delist_dates.tolist() == ["20260107"]
    # A future re-listing must not cancel this listing's end date.
    (raw / "stock_basic_L_000000.json").write_text(json.dumps({
        "code": 0, "data": {"fields": fields, "items": [["600000.SH", "20260108", None]]},
    }))
    assert panel.build_panel().delist_dates.tolist() == ["20260107"]
    (raw / "stock_basic_D_000000.json").write_text(json.dumps({
        "code": 0, "data": {"fields": fields, "items": [["600000.SH", "20050101", "20160101"]]},
    }))
    assert panel.build_panel().delist_dates.tolist() == ["20160101"]
    (raw / "stock_basic_L_000000.json").write_text(json.dumps({
        "code": 0, "data": {"fields": fields, "items": [["600000.SH", "20250101", None]]},
    }))
    assert panel.build_panel().delist_dates.tolist() == [""]
    monkeypatch.setattr(panel, "RAW_DIR", str(tmp_path / "detached"))
    assert panel.build_panel().delist_dates.tolist() == [""]


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


def test_research_market_preserves_archived_scope(tmp_path):
    p = panel.build_panel(snapshot_root=write_snapshot(tmp_path / "snapshot"))
    codes = np.array(["600000.SH", "302001.SZ", "302001", "200001.SZ", "900001.SH",
                      "920001.BJ", "301001.SH", "600001.SZ", "600bad.SH"])
    p = replace(p, codes=codes, board=np.array([panel._board_of(c) for c in codes]))
    assert market_mask(p, FINAL_SPEC["market"]).tolist() == [True] * 3 + [False] * 6
    assert market_mask(p, LEGACY_SPEC["market"]).tolist() == [
        True, False, False, True, True, False, True, True, True]


def test_freeze_values_during_gaps_then_resume_and_delist():
    dates = np.array([f"202601{day:02}" for day in range(1, 9)])
    close = np.array([np.nan, 10., np.nan, np.nan, 12., 12., np.nan, 20.])[:, None]
    floats = np.array([np.nan, 100., np.nan, 500., 200., 200., np.nan, 200.])[:, None]
    vol = np.array([np.nan, 10., np.nan, 20., 50., 0., np.nan, 50.])[:, None]
    p = panel.Panel(dates=dates, codes=np.array(["302001.SZ"]), close=close,
                    float_shares=floats, vol_shares=vol, adj_factor=np.ones_like(close),
                    amount_cny=close*vol, free_shares=floats, circ_mv=close*floats,
                    turnover_rate=100*vol/floats, board=np.array(["other"]),
                    delist_dates=np.array(["20260107"]))
    spec = dict(FINAL_SPEC, init="zero")
    lam = 12 / 13
    resumed_fraction = 0.25 + lam * 0.75 * 0.1
    resumed = 12 * 200 * resumed_fraction
    expected = [0, 100, 100, 100, resumed, lam * resumed, 0, 0]
    comp = model_components(p, spec)
    np.testing.assert_allclose(comp["levels"], expected)
    np.testing.assert_allclose(comp["price_only"][2:4], 0)
    np.testing.assert_allclose(comp["drift_only"][2:4], 0)
    short = _slice_panel(p, slice(None, 5))
    np.testing.assert_allclose(model_components(short, spec)["levels"], expected[:5])


@pytest.mark.parametrize("mid_float,mid_volume", [(np.nan, 50.), (100., np.nan), (np.nan, np.nan)])
def test_disclosed_turnover_survives_missing_float_or_volume(tmp_path, mid_float, mid_volume):
    from compass_0amv import from_arrays

    base = panel.build_panel(snapshot_root=write_snapshot(tmp_path / "snapshot"))
    dates = np.array(["20260105", "20260106", "20260107"])
    close = np.ones((3, 1))
    floats = np.array([100., mid_float, 100.])[:, None]
    volume = np.array([10., mid_volume, 10.])[:, None]
    reported = np.array([10., 50., 10.])[:, None]
    p = replace(base, dates=dates, close=close, float_shares=floats, vol_shares=volume,
                turnover_rate=reported, free_shares=floats, circ_mv=close*floats,
                amount_cny=close*volume, adj_factor=np.ones_like(close))
    levels = model_components(p, FINAL_SPEC)["levels"]
    _, deployed = from_arrays(dates, p.codes, close, volume, floats, turnover_rate=reported)
    np.testing.assert_allclose(deployed, levels, rtol=1e-14)
    if np.isnan(mid_float):
        assert levels[1] == 0.
    else:
        assert levels[1] > 0.
    assert levels[2] == pytest.approx(74.1958041958042, rel=1e-14)


def test_expanding_seed_and_reported_turnover_are_causal_on_every_prefix(tmp_path):
    base = panel.build_panel(snapshot_root=write_snapshot(tmp_path / "snapshot"))
    dates = np.array([f"202601{day:02}" for day in range(1, 26)])
    close = np.full((len(dates), 1), 10.)
    close[[1, 2, 17, 18, 19, 20]] = np.nan
    vol = np.full_like(close, 10.)
    floats = np.full_like(close, 100.)
    reported = np.linspace(0., 45., len(dates))[:, None]
    reported[4:8, 0] = [np.nan, -0.1, np.inf, 125.]
    p = replace(base, dates=dates, close=close, float_shares=floats, vol_shares=vol,
                turnover_rate=reported, free_shares=floats, circ_mv=close*floats,
                amount_cny=close*vol, adj_factor=np.ones_like(close))
    actual = model_components(p, FINAL_SPEC)["levels"]

    lam = 12 / 13
    q = []
    last_value = 0.
    for i in range(len(dates)):
        rate = reported[i, 0]
        valid_close = np.isfinite(close[i, 0])
        fraction = min(rate / 100., 1.) if np.isfinite(rate) and rate >= 0 else 0.1
        q.append(fraction if valid_close else 0.)
        if valid_close:
            median = np.median(q[:20])
            f = median / (1 - lam * (1 - median))
            for j, turnover in enumerate(q):
                if np.isfinite(close[j, 0]):
                    f = turnover + lam * (1 - turnover) * f
            last_value = 100 * 10 * f
        np.testing.assert_allclose(actual[i], last_value, rtol=1e-14)
        short = _slice_panel(p, slice(None, i + 1))
        np.testing.assert_array_equal(model_components(short, FINAL_SPEC)["levels"], actual[:i + 1])

    for cut in (0, 4, 18, 19):
        poisoned = replace(p, close=close.copy(), float_shares=floats.copy(),
                           vol_shares=vol.copy(), turnover_rate=reported.copy())
        poisoned.close[cut + 1:] = 50.
        poisoned.vol_shares[cut + 1:] = 1.
        poisoned.float_shares[cut + 1:] = 1000.
        poisoned.turnover_rate[cut + 1:] = 90.
        changed = model_components(poisoned, FINAL_SPEC)["levels"]
        np.testing.assert_array_equal(changed[:cut + 1], actual[:cut + 1])
        assert np.any(changed[cut + 1:] != actual[cut + 1:])
