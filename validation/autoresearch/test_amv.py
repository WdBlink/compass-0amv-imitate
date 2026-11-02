"""Tests for the deployable 0AMV implementation (`amv.py`).

Run with pytest, or directly:

    python3 validation/autoresearch/test_amv.py

Covers the recursion itself, the index construction, causality, agreement
with the research harness, and the end-to-end accuracy on the official
64-day set.
"""

from __future__ import annotations

import os
import sys

import numpy as np

try:
    import pytest
except ImportError:  # pytest is optional; the __main__ runner below works without it
    pytest = None

HERE = os.path.dirname(os.path.abspath(__file__))
if HERE not in sys.path:
    sys.path.insert(0, HERE)

import amv  # noqa: E402
from evaluate import load_official  # noqa: E402
from model import build_panel, market_mask, model_components  # noqa: E402
from spec import FINAL_SPEC  # noqa: E402
from panel import Panel  # noqa: E402

LAMBDA = amv.DECAY


class Skipped(Exception):
    """Raised by the standalone runner when a test needs pytest or a snapshot."""


# pytest's own skip exception, when pytest is installed.  The standalone
# runner must work without pytest, so the skip channel is a plain tuple.
_SKIP_EXCEPTIONS = (Skipped,) if pytest is None else (Skipped, pytest.skip.Exception)


def _require_private():
    if os.environ.get("COMPASS_RUN_PRIVATE_TESTS") != "1":
        message = ("private snapshot tests require COMPASS_RUN_PRIVATE_TESTS=1; "
                   "set COMPASS_SNAPSHOT_ROOT to your offline snapshot")
        if pytest is None:
            raise Skipped(message)
        pytest.skip(message)


def _checked_mask(panel):
    assert panel.n_dates > 0 and panel.n_stocks > 0, "empty panel"
    mask = market_mask(panel, FINAL_SPEC["market"])
    assert mask.shape == (panel.n_stocks,) and mask.any(), "empty market mask"
    for values in (panel.close, panel.float_shares, panel.vol_shares):
        assert values.shape == (panel.n_dates, panel.n_stocks)
        assert np.isfinite(values[:, mask]).any(), "no finite market observations"
    return mask


def _checked_levels(levels, n_dates):
    assert n_dates > 0 and levels.shape == (n_dates,), "empty or misaligned output"
    assert np.isfinite(levels).all(), "nonfinite output"
    assert (levels > 0).all(), "no positive market value"


def test_recursion_matches_hand_computation():
    """f_t = q_t + lambda*(1-q_t)*f_{t-1}, started from an empty pool."""
    fcap = np.array([[1000.0], [1000.0], [1000.0]])
    vol = np.array([[100.0], [100.0], [200.0]])
    close = np.array([[10.0], [10.0], [10.0]])

    f = amv.active_chip_fraction(vol, fcap, close, decay=LAMBDA, init="zero")

    q0, q1, q2 = 0.1, 0.1, 0.2
    e0 = q0
    e1 = q1 + LAMBDA * (1 - q1) * e0
    e2 = q2 + LAMBDA * (1 - q2) * e1
    np.testing.assert_allclose(f[:, 0], [e0, e1, e2], rtol=1e-12, atol=1e-15)


def test_steady_init_is_the_fixed_point():
    """A constant-turnover stock must sit at its quasi-stationary level."""
    q = 0.02
    fcap = np.full((40, 1), 1000.0)
    vol = np.full((40, 1), q * 1000.0)
    close = np.full((40, 1), 10.0)

    f = amv.active_chip_fraction(vol, fcap, close, decay=LAMBDA, init="steady")
    fixed = q / (1 - LAMBDA * (1 - q))
    np.testing.assert_allclose(f[:, 0], fixed, rtol=1e-12)


def test_active_fraction_stays_in_the_unit_interval():
    rng = np.random.default_rng(20261102)
    nd, ns = 60, 200
    fcap = rng.uniform(1e6, 1e10, size=(nd, ns))
    vol = rng.uniform(0, 0.6, size=(nd, ns)) * fcap
    close = rng.uniform(3, 80, size=(nd, ns))
    close[rng.random((nd, ns)) < 0.05] = np.nan  # suspensions

    f = amv.active_chip_fraction(vol, fcap, close)
    assert np.all(f >= 0.0)
    assert np.all(f <= 1.0)
    assert np.all(np.isfinite(f))


def test_level_is_the_sum_of_price_times_active_shares():
    fcap = np.array([[1000.0, 2000.0], [1000.0, 2000.0]])
    vol = np.array([[100.0, 200.0], [150.0, 100.0]])
    close = np.array([[10.0, 5.0], [11.0, 4.5]])

    f = amv.active_chip_fraction(vol, fcap, close, init="zero")
    level = amv.active_market_value(vol, fcap, close, init="zero")
    np.testing.assert_allclose(level, (f * fcap * close).sum(axis=1), rtol=1e-12)


def test_daily_change_pct():
    levels = np.array([100.0, 110.0, 99.0])
    chg = amv.daily_change_pct(levels)
    assert np.isnan(chg[0])
    np.testing.assert_allclose(chg[1:], [10.0, -10.0], rtol=1e-12)


def _synthetic_panel():
    """Deterministic, nonconstant public data, not reconstructed market history."""
    rng = np.random.default_rng(20261102)
    nd, ns = 60, 4
    close = rng.uniform(5.0, 30.0, (nd, ns))
    floats = rng.uniform(1e6, 2e6, (nd, ns))
    volume = floats * rng.uniform(0.001, 0.1, (nd, ns))
    dates = np.arange(np.datetime64("2026-01-01"), np.datetime64("2026-03-02"))
    return Panel(
        dates=np.char.replace(dates.astype(str), "-", ""),
        codes=np.array(["600000.SH", "300001.SZ", "688001.SH", "430001.BJ"]),
        close=close, adj_factor=np.ones((nd, ns)), vol_shares=volume,
        amount_cny=close * volume, float_shares=floats, free_shares=floats * 0.8,
        circ_mv=close * floats, turnover_rate=100 * volume / floats,
        board=np.array(["main_a", "growth_a", "star_a", "bse_a"]),
    )


def test_synthetic_panel_reproduces_research_model():
    panel = _synthetic_panel()
    mask = _checked_mask(panel)
    assert mask.tolist() == [True, True, True, False]
    for values in (panel.close, panel.float_shares, panel.vol_shares):
        assert values.size > 0 and np.isfinite(values).all()
    for init in ("zero", "steady"):
        spec = {**FINAL_SPEC, "init": init}
        got = amv.active_market_value(panel.vol_shares, panel.float_shares, panel.close,
                                      decay=spec["decay"], init=init, market_mask=mask)
        ref = model_components(panel, spec)["levels"]
        _checked_levels(got, panel.n_dates)
        _checked_levels(ref, panel.n_dates)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-6)


def test_synthetic_panel_causality():
    panel = _synthetic_panel()
    mask = _checked_mask(panel)
    # Steady initialization uses the FIRST 20 observations (early lookahead).
    # Preserve this disclosed historical policy; test it only after that window.
    for init, cuts in (("zero", (0, 7, 29)), ("steady", (19, 29))):
        spec = {**FINAL_SPEC, "init": init}
        def deployed(p):
            return amv.active_market_value(p.vol_shares, p.float_shares, p.close,
                                           decay=spec["decay"], init=init, market_mask=mask)
        for calculate in (deployed, lambda p: model_components(p, spec)["levels"]):
            full = calculate(panel)
            _checked_levels(full, panel.n_dates)
            for cut in cuts:
                short = Panel(**{name: (value.copy() if name in ("codes", "board")
                                       else value[:cut + 1].copy())
                                 for name, value in vars(panel).items()})
                prefix = calculate(short)
                _checked_levels(prefix, cut + 1)
                np.testing.assert_allclose(full[:cut + 1], prefix, rtol=1e-12)
                changed = Panel(**{name: value.copy() for name, value in vars(panel).items()})
                for name in ("close", "float_shares", "vol_shares"):
                    getattr(changed, name)[cut + 1:] *= 7
                perturbed = calculate(changed)
                _checked_levels(perturbed, panel.n_dates)
                np.testing.assert_allclose(full[:cut + 1], perturbed[:cut + 1], rtol=1e-12)
                assert not np.allclose(full[cut + 1:], perturbed[cut + 1:])


def test_amv_reproduces_the_research_model():
    """The deployable module and the research harness must agree exactly."""
    _require_private()
    panel = build_panel()
    mask = _checked_mask(panel)
    levels = amv.active_market_value(
        panel.vol_shares, panel.float_shares, panel.close,
        decay=FINAL_SPEC["decay"], init=FINAL_SPEC["init"], market_mask=mask,
    )
    ref = model_components(panel, FINAL_SPEC)["levels"]
    _checked_levels(levels, panel.n_dates)
    _checked_levels(ref, panel.n_dates)
    np.testing.assert_allclose(levels, ref, rtol=1e-12, atol=1e-6)


def test_no_lookahead():
    """Truncating the future must not change the level at the cut date."""
    _require_private()
    panel = build_panel(required_dates=["20260910"])
    mask = _checked_mask(panel)
    cut = panel.date_index("20260910")

    full = amv.active_market_value(
        panel.vol_shares, panel.float_shares, panel.close,
        decay=FINAL_SPEC["decay"], init=FINAL_SPEC["init"], market_mask=mask,
    )
    short = amv.active_market_value(
        panel.vol_shares[: cut + 1], panel.float_shares[: cut + 1], panel.close[: cut + 1],
        decay=FINAL_SPEC["decay"], init=FINAL_SPEC["init"], market_mask=mask,
    )
    _checked_levels(full, panel.n_dates)
    _checked_levels(short, cut + 1)
    np.testing.assert_allclose(full[cut], short[cut], rtol=1e-12)


def test_official_daily_change_accuracy():
    """End-to-end: the index must track the official daily change."""
    _require_private()
    off_dates, off_close, off_chg = load_official()
    assert len(off_dates) > 0, "empty official series"
    assert np.isfinite(off_close).all() and np.isfinite(off_chg).all()
    panel = build_panel(required_dates=off_dates)
    mask = _checked_mask(panel)
    levels = amv.active_market_value(
        panel.vol_shares, panel.float_shares, panel.close,
        decay=FINAL_SPEC["decay"], init=FINAL_SPEC["init"], market_mask=mask,
    )
    _checked_levels(levels, panel.n_dates)
    chg = amv.daily_change_pct(levels)
    idx = np.array([panel.date_index(d) for d in off_dates])
    pred = chg[idx]
    assert pred.size > 0 and np.isfinite(pred).all()

    mae = float(np.mean(np.abs(pred - off_chg)))
    assert mae < 0.1, f"daily-change MAE {mae:.4f} pp exceeds the 0.1 pp target"
    assert float(np.mean(np.sign(pred) == np.sign(off_chg))) == 1.0

    # Structural check: the level must be the official level times ONE
    # constant factor, not a curve fit.
    ratio = off_close / (levels[idx] / 1e8)
    cv = float(ratio.std() / ratio.mean())
    assert cv < 0.005, f"level ratio CV {cv:.5f} means the composition is wrong"


def test_a_share_mask():
    """The official universe is every A-share except the Beijing exchange."""
    codes = ["600000.SH", "601398.SH", "603259.SH", "605117.SH",
             "000001.SZ", "002594.SZ", "300750.SZ", "301269.SZ",
             "688981.SH", "689009.SH",
             "430047.BJ", "830799.BJ", "873169.BJ", "920002.BJ",
             "900901.SH", "200011.SZ"]
    m = amv.a_share_mask(codes)
    assert list(m) == [True] * 10 + [False] * 6


def test_cli_matches_the_library():
    """The command line path must produce the same index as the API."""
    import csv
    import subprocess
    import tempfile

    rows = [
        ("20260105", "600000.SH", 10.0, 1000.0, 100000.0),
        ("20260105", "000001.SZ", 12.0, 800.0, 200000.0),
        ("20260106", "600000.SH", 10.5, 1200.0, 100000.0),
        ("20260106", "000001.SZ", 11.8, 900.0, 200000.0),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "panel.csv")
        dst = os.path.join(tmp, "amv.csv")
        with open(src, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trade_date", "ts_code", "close", "vol", "float_share"])
            w.writerows(rows)
        subprocess.run(
            [sys.executable, os.path.join(HERE, "amv.py"), "--input", src, "--out", dst],
            check=True,
        )
        with open(dst, newline="", encoding="utf-8") as fh:
            out = list(csv.DictReader(fh))

    assert [r["trade_date"] for r in out] == ["20260105", "20260106"]

    vol = np.array([[1000.0 * amv.HAND, 800.0 * amv.HAND],
                     [1200.0 * amv.HAND, 900.0 * amv.HAND]])
    fsh = np.array([[100000.0 * amv.WAN, 200000.0 * amv.WAN],
                     [100000.0 * amv.WAN, 200000.0 * amv.WAN]])
    close = np.array([[10.0, 12.0], [10.5, 11.8]])
    mask = amv.a_share_mask(["600000.SH", "000001.SZ"])
    ref = amv.active_market_value(vol, fsh, close, market_mask=mask)

    got = np.array([float(r["amv_cny"]) for r in out])
    # the CSV column is written rounded to cents
    np.testing.assert_allclose(got, ref, rtol=1e-9)


def test_scale_moves_the_level_but_not_the_change():
    """--scale is a pure level calibration: daily changes must not move."""
    import csv
    import subprocess
    import tempfile

    rows = [
        ("20260105", "600000.SH", 10.0, 1000.0, 100000.0),
        ("20260105", "000001.SZ", 12.0, 800.0, 200000.0),
        ("20260106", "600000.SH", 10.5, 1200.0, 100000.0),
        ("20260106", "000001.SZ", 11.8, 900.0, 200000.0),
    ]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "panel.csv")
        raw = os.path.join(tmp, "raw.csv")
        scaled = os.path.join(tmp, "scaled.csv")
        with open(src, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            w.writerow(["trade_date", "ts_code", "close", "vol", "float_share"])
            w.writerows(rows)
        for dst, extra in ((raw, []), (scaled, ["--scale", str(amv.OFFICIAL_LEVEL_SCALE)])):
            subprocess.run(
                [sys.executable, os.path.join(HERE, "amv.py"),
                 "--input", src, "--out", dst, *extra],
                check=True,
            )
        with open(raw, newline="", encoding="utf-8") as fh:
            a = list(csv.DictReader(fh))
        with open(scaled, newline="", encoding="utf-8") as fh:
            b = list(csv.DictReader(fh))

    for r_a, r_b in zip(a, b):
        np.testing.assert_allclose(
            float(r_b["amv_cny"]), float(r_a["amv_cny"]) * amv.OFFICIAL_LEVEL_SCALE,
            rtol=1e-6,
        )
        assert r_b["change_pct"] == r_a["change_pct"]


def _main() -> int:
    tests = [v for k, v in sorted(globals().items()) if k.startswith("test_")]
    failed = 0
    skipped = 0
    for t in tests:
        try:
            t()
            print(f"PASS {t.__name__}")
        except _SKIP_EXCEPTIONS as exc:
            skipped += 1
            print(f"SKIP {t.__name__}: {exc}")
        except AssertionError as exc:
            failed += 1
            print(f"FAIL {t.__name__}: {exc}")
        except Exception as exc:  # noqa: BLE001
            failed += 1
            print(f"ERROR {t.__name__}: {type(exc).__name__}: {exc}")
    print(f"\n{len(tests) - failed - skipped} passed, {skipped} skipped, {failed} failed")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(_main())
