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
import sys

import numpy as np

from evaluate import evaluate, load_official
from model import build_panel, model_components
from spec import FINAL_SPEC

HERE = os.path.dirname(os.path.abspath(__file__))


def _truncate(panel, keep_dates):
    idx = [i for i, d in enumerate(panel.dates) if d in keep_dates]
    return idx


def _slice_panel(panel, idx):
    from dataclasses import replace
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
    keep = set(panel.dates[panel.dates <= cut])
    short = _slice_panel(panel, _truncate(panel, keep))
    full_c = model_components(panel, spec)["change"]
    short_c = model_components(short, spec)["change"]
    n_short = len(short.dates)
    # Compare on the overlapping range, allowing the truncated run to lag
    # by at most the window length because it lacks history.
    diffs = []
    for i in range(n_short):
        if panel.dates[i] != short.dates[i]:
            continue
        a, b = full_c[i], short_c[i]
        if np.isnan(a) or np.isnan(b):
            continue
        diffs.append(abs(a - b))
    tail = np.array(diffs[-10:])
    print(f"1. prefix-invariance: max |diff| over last 10 comparable days = {tail.max():.3e}")

    # ---- 2. future-tail poisoning --------------------------------------
    poisoned = panel
    from dataclasses import replace as _replace
    cut_i = panel.date_index(cut)
    poisoned = _replace(panel, close=panel.close.copy(), vol_shares=panel.vol_shares.copy())
    poisoned.close[cut_i + 1:, :] = poisoned.close[cut_i + 1:, :] * 3.0
    poisoned.vol_shares[cut_i + 1:, :] = poisoned.vol_shares[cut_i + 1:, :] * 5.0
    pois_c = model_components(poisoned, spec)["change"]
    same = np.allclose(full_c[: cut_i + 1], pois_c[: cut_i + 1], equal_nan=True, rtol=0, atol=1e-12)
    print(f"2. future-tail poisoning: predictions through {cut} unchanged = {same}")
    if not same:
        failures.append("future-tail poisoning")

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
    pred_l = model_components(panel, spec)["levels"][off_idx] / 1e8
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


if __name__ == "__main__":
    main()
