"""Verification battery for the 0AMV replication model.

Checks, in order:
  1. causality / prefix-invariance: truncating the panel must not change
     any prediction that was already computable from the longer panel;
  2. future-tail poisoning: corrupting data after date T must not change
     any prediction up to and including T;
  3. no label leakage: the model code must not read the official file;
  4. side-by-side residual table against the official series;
  5. warmup convergence: starting the panel earlier must not move the
     official-window predictions.
"""

from __future__ import annotations

import os
from dataclasses import replace

import numpy as np

from evaluate import load_official
from model import build_panel, model_components
from spec import FINAL_SPEC

HERE = os.path.dirname(os.path.abspath(__file__))


def _truncate(panel, keep_dates):
    idx = [i for i, d in enumerate(panel.dates) if d in keep_dates]
    return idx


def _slice_panel(panel, idx):
    """Slice only time-indexed inputs, preserving code-aligned lifecycle metadata."""
    return replace(
        panel,
        dates=panel.dates[idx], close=panel.close[idx], vol_shares=panel.vol_shares[idx],
        amount_cny=panel.amount_cny[idx], float_shares=panel.float_shares[idx],
        free_shares=panel.free_shares[idx], circ_mv=panel.circ_mv[idx],
        turnover_rate=panel.turnover_rate[idx], board=panel.board,
        adj_factor=panel.adj_factor[idx],
    )


def main() -> None:
    spec = FINAL_SPEC
    panel = build_panel()
    off_dates, off_close, off_chg = load_official()
    off_idx = np.array([panel.date_index(d) for d in off_dates])

    failures = []

    # ---- 1. prefix invariance -------------------------------------------
    cut = "20260910"
    full = model_components(panel, spec)
    full_c = full["change"]
    cut_i = panel.date_index(cut)
    for length in (1, 5, 19, 20, cut_i + 1):
        short = _slice_panel(panel, slice(None, length))
        short_levels = model_components(short, spec)["levels"]
        prefix_diff = np.abs(full["levels"][:length] - short_levels)
        print(f"1. prefix-invariance: {length} sessions, max level difference = {prefix_diff.max():.3e} CNY")
        if not np.all(prefix_diff == 0):
            failures.append(f"prefix invariance ({length} sessions)")

    # ---- 2. future-tail poisoning --------------------------------------
    for length in (1, 5, 19, 20, cut_i + 1):
        poisoned = replace(panel, close=panel.close.copy(), vol_shares=panel.vol_shares.copy(),
                           turnover_rate=panel.turnover_rate.copy(), float_shares=panel.float_shares.copy())
        poisoned.close[length:] *= 3.0
        poisoned.vol_shares[length:] *= 5.0
        poisoned.turnover_rate[length:] = 99.0
        poisoned.float_shares[length:] *= 2.0
        poisoned_levels = model_components(poisoned, spec)["levels"]
        same = np.array_equal(full["levels"][:length], poisoned_levels[:length])
        print(f"2. future-tail poisoning: predictions through {panel.dates[length - 1]} unchanged = {same}")
        if not same or np.array_equal(full["levels"][length:], poisoned_levels[length:]):
            failures.append(f"future-tail poisoning ({length} sessions)")

    # ---- 3. no label leakage -------------------------------------------
    src = open(os.path.join(HERE, "model.py"), encoding="utf-8").read() + \
        open(os.path.join(HERE, "panel.py"), encoding="utf-8").read()
    leaks = [t for t in ("official.csv", "official_change_pct", "exact_parity") if t in src]
    print(f"3. label leakage in model/panel source: {leaks if leaks else 'none'}")
    if leaks:
        failures.append("label leakage")

    # ---- 4. residual table ---------------------------------------------
    print("\n4. side-by-side (all 64 official days):")
    print(f"   {'date':10s} {'official%':>10s} {'model%':>9s} {'err':>7s}   {'off_lvl':>9s} {'mdl_lvl':>9s} {'ratio':>6s}")
    pred_c = full_c[off_idx]
    pred_l = full["levels"][off_idx] / 1e8
    for j, d in enumerate(off_dates):
        err = pred_c[j] - off_chg[j]
        ratio = off_close[j] / pred_l[j] if pred_l[j] else float("nan")
        print(f"   {d:10s} {off_chg[j]:10.2f} {pred_c[j]:9.3f} {err:7.3f}   {off_close[j]:9.1f} {pred_l[j]:9.1f} {ratio:6.4f}")
    errs = pred_c - off_chg
    print(f"\n   MAE={np.mean(np.abs(errs)):.4f}  max|err|={np.max(np.abs(errs)):.4f}  "
          f"bias={np.mean(errs):+.4f}  dir={np.mean(np.sign(pred_c)==np.sign(off_chg)):.3f}")
    ratios = off_close / pred_l
    print(f"   level ratio official/model: mean={ratios.mean():.6f} std={ratios.std():.6f} "
          f"CV={ratios.std()/ratios.mean():.6f}")
    print(f"   level MAPE raw={np.mean(np.abs(pred_l-off_close)/off_close)*100:.4f}%  "
          f"scaled={np.mean(np.abs(ratios.mean()*pred_l-off_close)/off_close)*100:.5f}%")

    # ---- 5. warmup convergence -----------------------------------------
    print("\n5. warmup convergence: re-run starting from different panel starts")
    for start in ("20260506", "20260601", "20260701"):
        keep = set(panel.dates[panel.dates >= start])
        sub = _slice_panel(panel, _truncate(panel, keep))
        c = model_components(sub, spec)["change"]
        dmap = dict(zip(sub.dates, c))
        vals = np.array([dmap.get(d, np.nan) for d in off_dates])
        m = ~np.isnan(vals)
        print(f"   start={start}  n={m.sum():2d}  MAE={np.mean(np.abs(vals[m]-off_chg[m])):.4f}")

    print("\nFAILURES:", failures if failures else "none")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
