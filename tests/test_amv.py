"""Tests for the deployable 0AMV implementation (`compass_0amv`).

Run with pytest, or directly:

    python3 tests/test_amv.py

Covers the recursion itself, the index construction, causality, agreement
with the research harness, and the end-to-end accuracy on the official
64-day set.
"""

from __future__ import annotations

import os
import sys
from copy import deepcopy
from dataclasses import replace

import numpy as np

try:
    import pytest
except ImportError:  # pytest is optional; the __main__ runner below works without it
    pytest = None

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESEARCH = os.path.join(ROOT, "research", "autoresearch")
for _path in (ROOT, RESEARCH):
    if _path not in sys.path:
        sys.path.insert(0, _path)

from compass_0amv import amv  # noqa: E402
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


def test_disclosed_daily_turnover_is_percent_with_per_cell_fallback():
    reported = np.array([[2.0, np.nan, -1.0, np.inf, 300.0, 0.0, 7.0]])
    vol = np.full_like(reported, 10.)
    floats = np.full_like(reported, 100.)
    close = np.ones_like(reported)
    close[0, -1] = np.nan
    out = amv.active_chip_fraction(vol, floats, close, init="zero", turnover_rate=reported)
    np.testing.assert_allclose(out, [[0.02, 0.10, 0.10, 0.10, 1.0, 0.0, 0.0]])


def test_robust_seed_uses_only_observed_sessions_and_freezes_quoted_state():
    turnover = np.array([[0.1], [0.8], [0.0], [0.2], [0.3]])
    close = np.array([[1.], [1.], [np.nan], [1.], [1.]])
    got = amv.active_chip_fraction(turnover, np.ones_like(close), close, decay=0.5, init_window=3)
    for t in range(len(close)):
        if not np.isfinite(close[t, 0]):
            np.testing.assert_array_equal(got[t], got[t-1])
            continue
        median = np.median(turnover[:min(t + 1, 3)])
        expected = median / (1 - 0.5 * (1 - median))
        for day in range(t + 1):
            if np.isfinite(close[day, 0]):
                q = turnover[day, 0]
                expected = q + 0.5 * (1 - q) * expected
        np.testing.assert_allclose(got[t, 0], expected)


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
    close[3:6, 1] = np.nan
    close[:10, 2] = np.nan
    reported = np.round(100 * volume / floats, 2)
    reported[5::7] = np.nan
    return Panel(
        dates=np.char.replace(dates.astype(str), "-", ""),
        codes=np.array(["600000.SH", "300001.SZ", "688001.SH", "430001.BJ"]),
        close=close, adj_factor=np.ones((nd, ns)), vol_shares=volume,
        amount_cny=close * volume, float_shares=floats, free_shares=floats * 0.8,
        circ_mv=close * floats, turnover_rate=reported,
        board=np.array(["main_a", "growth_a", "star_a", "bse_a"]),
        delist_dates=np.array(["", "", "20260215", ""]),
    )


def _deployed(panel, reported=True, **kwargs):
    return amv.from_arrays(panel.dates, panel.codes, panel.close, panel.vol_shares, panel.float_shares,
                           delist_dates=panel.delist_dates, turnover_rate=panel.turnover_rate if reported else None,
                           **kwargs)[1]


def test_synthetic_panel_reproduces_research_model():
    panel = _synthetic_panel()
    mask = _checked_mask(panel)
    assert mask.tolist() == [True, True, True, False]
    assert amv.DECAY == FINAL_SPEC["decay"]
    assert amv.INIT_WINDOW == FINAL_SPEC["init_window"]
    for init in ("zero", "steady", "robust"):
        spec = {**FINAL_SPEC, "init": init}
        got = _deployed(panel, init=init)
        ref = model_components(panel, spec)["levels"]
        _checked_levels(got, panel.n_dates)
        _checked_levels(ref, panel.n_dates)
        np.testing.assert_allclose(got, ref, rtol=1e-12, atol=1e-6)


def test_synthetic_panel_causality():
    panel = _synthetic_panel()
    _checked_mask(panel)
    # Steady initialization uses the FIRST init_window observations (early lookahead).
    # Preserve this disclosed historical policy; test it only after that window.
    iw = FINAL_SPEC["init_window"]
    for init, cuts in (("zero", (0, 7, 29)), ("steady", (iw - 1, iw)), ("robust", (0, 3, 7, 18, 19, 29))):
        spec = {**FINAL_SPEC, "init": init}
        for calculate in (lambda p: _deployed(p, init=init), lambda p: model_components(p, spec)["levels"]):
            full = calculate(panel)
            _checked_levels(full, panel.n_dates)
            for cut in cuts:
                short = replace(panel, **{name: value[:cut + 1].copy() for name, value in vars(panel).items()
                                          if name not in ("codes", "board", "delist_dates")})
                prefix = calculate(short)
                _checked_levels(prefix, cut + 1)
                np.testing.assert_allclose(full[:cut + 1], prefix, rtol=1e-12)
                changed = deepcopy(panel)
                for name in ("close", "float_shares", "vol_shares", "turnover_rate"):
                    getattr(changed, name)[cut + 1:] *= 7
                perturbed = calculate(changed)
                _checked_levels(perturbed, panel.n_dates)
                np.testing.assert_allclose(full[:cut + 1], perturbed[:cut + 1], rtol=1e-12)
                assert not np.allclose(full[cut + 1:], perturbed[cut + 1:])


def test_amv_reproduces_the_research_model():
    """The deployable module and the research harness must agree exactly."""
    _require_private()
    panel = build_panel()
    _checked_mask(panel)
    levels = _deployed(panel)
    ref = model_components(panel, FINAL_SPEC)["levels"]
    _checked_levels(levels, panel.n_dates)
    _checked_levels(ref, panel.n_dates)
    np.testing.assert_allclose(levels, ref, rtol=1e-12, atol=1e-6)


def test_no_lookahead():
    """Truncating the future must not change the level at the cut date."""
    _require_private()
    panel = build_panel(required_dates=["20260910"])
    _checked_mask(panel)
    cut = panel.date_index("20260910")

    full = _deployed(panel)
    _, short = amv.from_arrays(panel.dates[:cut + 1], panel.codes, panel.close[:cut + 1],
                               panel.vol_shares[:cut + 1], panel.float_shares[:cut + 1],
                               delist_dates=panel.delist_dates, turnover_rate=panel.turnover_rate[:cut + 1])
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
    _checked_mask(panel)
    levels = _deployed(panel)
    _checked_levels(levels, panel.n_dates)
    chg = amv.daily_change_pct(levels)
    idx = np.array([panel.date_index(d) for d in off_dates])
    pred = chg[idx]
    assert pred.size > 0 and np.isfinite(pred).all()

    # Pinned regression on this exposed snapshot, not the <0.01 pp research target.
    mae = float(np.mean(np.abs(pred - off_chg)))
    np.testing.assert_allclose(mae, 0.01127874409018208, rtol=0, atol=1e-9)
    assert float(np.mean(np.sign(pred) == np.sign(off_chg))) == 1.0
    fallback = amv.daily_change_pct(_deployed(panel, reported=False))[idx]
    np.testing.assert_allclose(np.mean(np.abs(fallback - off_chg)), 0.011298742099918106, rtol=0, atol=1e-9)

    # Structural check: the level must be the official level times ONE
    # constant factor, not a curve fit.
    ratio = off_close / (levels[idx] / 1e8)
    np.testing.assert_allclose(ratio.mean(), amv.OFFICIAL_LEVEL_SCALE, rtol=0, atol=1e-10)
    cv = float(ratio.std() / ratio.mean())
    assert cv < 0.005, f"level ratio CV {cv:.5f} means the composition is wrong"


def test_a_share_mask():
    """The official universe is every A-share except the Beijing exchange."""
    codes = ["600000.SH", "601398.SH", "603259.SH", "605117.SH",
             "000001.SZ", "002594.SZ", "300750.SZ", "301269.SZ",
             "688981.SH", "689009.SH", "302001.SZ",
             "430047.BJ", "830799.BJ", "873169.BJ", "920002.BJ",
             "900901.SH", "200011.SZ", "000001.SH", "600000.BJ"]
    assert list(amv.a_share_mask(codes)) == [True] * 11 + [False] * 8
    assert list(amv.market_mask_for("all_a_bse", codes)) == [True] * 15 + [False] * 4


def test_pause_resume_and_quote_gaps_preserve_the_entire_value():
    vol = np.array([[np.nan], [100.0], [999.0], [np.nan], [200.0]])
    floats = np.array([[np.nan], [1000.0], [5000.0], [np.nan], [2000.0]])
    close = np.array([[np.nan], [10.0], [np.nan], [np.nan], [12.0]])
    fraction = amv.active_chip_fraction(vol, floats, close, init="zero")
    resumed = 0.1 + amv.DECAY * 0.9 * 0.1
    np.testing.assert_allclose(fraction[:, 0], [0, 0.1, 0.1, 0.1, resumed])
    levels = amv.active_market_value(vol, floats, close, init="zero")
    np.testing.assert_allclose(levels, [0, 1000, 1000, 1000, resumed * 24000])
    np.testing.assert_allclose(
        amv.active_market_value(vol, floats, close, init="zero", missing_quote="drop"),
        [0, 1000, 0, 0, (0.1 + amv.DECAY ** 3 * 0.9 * 0.1) * 24000],
    )
    for cut in range(1, len(close)):
        np.testing.assert_array_equal(
            levels[:cut], amv.active_market_value(vol[:cut], floats[:cut], close[:cut], init="zero")
        )
    for init in ("zero", "steady"):
        assert amv.active_market_value(vol, floats, close, init=init)[0] == 0


def test_quoted_zero_or_invalid_inputs_are_not_suspensions():
    for volume in (0.0, -1.0, np.nan, np.inf):
        vol = np.array([[100.0], [volume]])
        floats = np.full((2, 1), 1000.0)
        close = np.array([[10.0], [12.0]])
        np.testing.assert_allclose(
            amv.active_market_value(vol, floats, close, init="zero"),
            [1000, amv.DECAY * 0.1 * 12000],
        )
    for shares in (0.0, -1.0, np.nan, np.inf):
        vol = np.full((3, 1), 100.0)
        floats = np.array([[1000.0], [shares], [1000.0]])
        close = np.array([[10.0], [12.0], [np.nan]])
        np.testing.assert_allclose(amv.active_market_value(vol, floats, close, init="zero"), [1000, 0, 0])


def test_delisting_applies_only_from_its_effective_date_and_never_reenters():
    dates = ["20260102", "20260105", "20260106", "20260107"]
    close = np.array([[10.0], [np.nan], [np.nan], [20.0]])
    vol = np.array([[100.0], [np.nan], [np.nan], [200.0]])
    floats = np.full((4, 1), 1000.0)
    for end in ("20260106", "2026-01-06", "20260106.0", 20260106.0, np.datetime64("2026-01-06")):
        _, levels = amv.from_arrays(dates, ["600000.SH"], close, vol, floats,
                                    delist_dates=[end], init="zero")
        np.testing.assert_array_equal(levels, [1000, 1000, 0, 0])
        _, prefix = amv.from_arrays(dates[:2], ["600000.SH"], close[:2], vol[:2], floats[:2],
                                    delist_dates=[end], init="zero")
        np.testing.assert_array_equal(prefix, levels[:2])
    _, weekend_exit = amv.from_arrays(dates, ["600000.SH"], close, vol, floats,
                                      delist_dates=["20260104"], init="zero")
    np.testing.assert_array_equal(weekend_exit, [1000, 0, 0, 0])
    _, unknown = amv.from_arrays(dates, ["600000.SH"], close, vol, floats,
                                 delist_dates=[None], init="zero")
    np.testing.assert_array_equal(unknown[:3], [1000, 1000, 1000])
    assert unknown[-1] > 0


def test_listing_metadata_and_dense_array_alignment_are_validated():
    close = np.ones((2, 1))
    cases = [
        {"delist_dates": ["20260105"]},
        {"dates": ["20260102", "20260105"], "delist_dates": []},
        {"dates": ["20260105", "20260102"], "delist_dates": [None]},
        {"dates": ["20260102", "20260102"], "delist_dates": [None]},
        {"dates": ["20260102", "20260105"], "delist_dates": ["20260230"]},
        {"market_mask": [True, False]},
        {"turnover_rate": [[2.0, 2.0]]},
        {"missing_quote": "guess"},
    ]
    for kwargs in cases:
        try:
            amv.active_market_value(close, close, close, **kwargs)
        except ValueError:
            continue
        raise AssertionError(f"accepted invalid arguments {kwargs}")
    try:
        amv._delist_dates_for(["600000.SH"], [("600000.SH", "20260105"), ("600000.SH", "20260106")])
    except ValueError:
        pass
    else:
        raise AssertionError("accepted conflicting instrument delisting dates")


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
            [sys.executable, "-m", "compass_0amv", "--input", src, "--out", dst],
            check=True, cwd=ROOT,
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


def test_cli_and_frame_preserve_gaps_and_delisting_dates():
    import csv
    import subprocess
    import tempfile

    rows = [
        ("2026-01-02", "302001.SZ", 10, 1, 0.1, "2026-01-06"),
        ("2026-01-02", "600000.SH", 10, 1, 0.1, ""),
        ("2026-01-02", "430047.BJ", 10, 1, 0.1, ""),
        ("2026-01-05", "600000.SH", 11, 1, 0.1, ""),
        ("2026-01-06", "302001.SZ", "", "", 0.5, ""),
        ("2026-01-06", "600000.SH", 12, 1, 0.1, ""),
        ("2026-01-07", "302001.SZ", 20, 1, 0.1, ""),
        ("2026-01-07", "600000.SH", 13, 1, 0.1, ""),
    ]
    rows = [row + (rate,) for row, rate in zip(rows, [2, 3, 8, "", 12, 0, 16, 4])]
    columns = ["trade_date", "ts_code", "close", "vol", "float_share", "delist_date", "turnover_rate"]
    dates = ["20260102", "20260105", "20260106", "20260107"]
    codes = ["302001.SZ", "600000.SH", "430047.BJ"]
    close = np.array([[10, 10, 10], [np.nan, 11, np.nan],
                      [np.nan, 12, np.nan], [20, 13, np.nan]])
    vol = np.where(np.isfinite(close), 100.0, np.nan)
    floats = np.where(np.isfinite(close), 1000.0, np.nan)
    floats[2, 0] = 5000
    rates = np.array([[2, 3, 8], [np.nan, np.nan, np.nan], [12, 0, np.nan], [16, 4, np.nan]])
    references = {}
    with tempfile.TemporaryDirectory() as tmp:
        src, dst = os.path.join(tmp, "panel.csv"), os.path.join(tmp, "out.csv")
        with open(src, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        for policy in ("freeze", "drop"):
            _, expected = amv.from_arrays(dates, codes, close, vol, floats,
                                          delist_dates=["20260106", None, None], missing_quote=policy,
                                          turnover_rate=rates)
            references[policy] = expected
            flags = [] if policy == "freeze" else ["--missing-quote", policy]
            subprocess.run([sys.executable, "-m", "compass_0amv", "--input", src, "--out", dst, *flags],
                           check=True, cwd=ROOT)
            with open(dst, newline="", encoding="utf-8") as fh:
                out = list(csv.DictReader(fh))
            assert [r["trade_date"] for r in out] == dates
            np.testing.assert_allclose([float(r["amv_cny"]) for r in out], expected, atol=0.005)
            np.testing.assert_allclose([float(r["change_pct"]) for r in out[1:]],
                                       amv.daily_change_pct(expected)[1:], atol=0.0000005)

    try:
        import pandas as pd
    except ImportError:
        return
    frame = pd.DataFrame(rows, columns=columns)
    for policy, expected in references.items():
        got_dates, got = amv.from_tushare_frame(frame, missing_quote=policy)
        assert got_dates == dates
        np.testing.assert_allclose(got, expected, rtol=1e-12)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"])
    got_dates, got = amv.from_tushare_frame(frame)
    assert got_dates == dates
    np.testing.assert_allclose(got, references["freeze"], rtol=1e-12)


def test_long_inputs_reject_missing_trade_dates():
    import csv
    import subprocess
    import tempfile

    columns = ["trade_date", "ts_code", "close", "vol", "float_share", "delist_date"]
    rows = [("20260102", "302001.SZ", 10., 1., 0.1, ""), ("", "302001.SZ", 11., 1., 0.1, "")]
    with tempfile.TemporaryDirectory() as tmp:
        src = os.path.join(tmp, "panel.csv")
        with open(src, "w", newline="", encoding="utf-8") as fh:
            writer = csv.writer(fh)
            writer.writerow(columns)
            writer.writerows(rows)
        result = subprocess.run([sys.executable, "-m", "compass_0amv", "--input", src], cwd=ROOT,
                                text=True, capture_output=True)
        assert result.returncode != 0 and "trade_date" in result.stderr and not result.stdout
    try:
        import pandas as pd
    except ImportError:
        return
    for missing in ("", "  ", pd.NA, pd.NaT, np.nan):
        frame = pd.DataFrame(rows, columns=columns)
        frame["trade_date"] = ["20260102", missing]
        with np.testing.assert_raises_regex(ValueError, "trade_date"):
            amv.from_tushare_frame(frame)


def test_frame_explicit_turnover_and_delisting_override_optional_columns():
    try:
        import pandas as pd
    except ImportError:
        if pytest is None:
            raise Skipped("requires pandas")
        pytest.skip("requires pandas")
    frame = pd.DataFrame({
        "trade_date": ["20260102", "20260105"], "ts_code": ["302001.SZ"] * 2,
        "close": [10., 10.], "vol": [1., 1.], "float_share": [0.1, 0.1],
        "turnover_rate": [pd.NA, pd.NA], "delist_date": [pd.NA, pd.NA],
    })
    for disclosed, exit_date in ((pd.NA, pd.NA), (80.0, "20260107")):
        frame["turnover_rate"] = disclosed
        frame["delist_date"] = exit_date
        _, got = amv.from_tushare_frame(frame, turnover_rate=np.array([[2.], [3.]]),
                                        delist_dates=["20260105"], init="zero")
        np.testing.assert_allclose(got, [200., 0.])
    _, got = amv.from_tushare_frame(frame, turnover_rate=None, delist_dates=None, init="zero")
    np.testing.assert_allclose(got, [8000., 10000. * (0.8 + amv.DECAY * 0.2 * 0.8)])


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
                [sys.executable, "-m", "compass_0amv",
                 "--input", src, "--out", dst, *extra],
                check=True, cwd=ROOT,
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
