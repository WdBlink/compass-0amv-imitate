"""Metric harness for the 0AMV replication search.

Primary metric: MAE of the predicted daily change (percentage points)
against the official 0AMV daily change on the 64-day official set.

    METRIC <value>          lower is better

Secondary diagnostics are printed to stderr so the primary metric line
stays machine-parseable.
"""

from __future__ import annotations

import csv
import json
import os
import sys

import numpy as np

from model import build_panel, daily_change_pct, model_series

HERE = os.path.dirname(os.path.abspath(__file__))
OFFICIAL_CSV = os.path.join(HERE, "..", "exact_parity_20261030", "official.csv")

# The official set is 2026-07-27 .. 2026-10-30.
# Split for honesty reporting: the first half is the "dev" slice, the
# second half the "holdout" slice. All 64 days are already exposed, so
# neither slice is a genuine blind window -- they only guard against a
# single lucky regime.
DEV_END = "20260913"


def load_official() -> tuple[list[str], np.ndarray, np.ndarray]:
    dates, close, chg = [], [], []
    with open(OFFICIAL_CSV, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            dates.append(row["date"].replace("-", ""))
            close.append(float(row["official_close"]))
            chg.append(float(row["official_change_pct"]))
    return dates, np.array(close), np.array(chg)


def evaluate(spec: dict, panel=None) -> dict:
    panel = panel or build_panel()
    off_dates, off_close, off_chg = load_official()
    levels = model_series(panel, spec)
    chg = daily_change_pct(levels)

    idx = [panel.date_index(d) for d in off_dates]
    pred_chg = chg[idx]
    pred_level = levels[idx] / 1e8  # display units of 1e8 CNY

    err = pred_chg - off_chg
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    bias = float(np.mean(err))
    dir_hit = float(np.mean(np.sign(pred_chg) == np.sign(off_chg)))

    dev = np.array([d <= DEV_END for d in off_dates])
    mae_dev = float(np.mean(np.abs(err[dev])))
    mae_hold = float(np.mean(np.abs(err[~dev])))

    lvl_err = pred_level - off_close
    lvl_mae = float(np.mean(np.abs(lvl_err)))
    lvl_mape = float(np.mean(np.abs(lvl_err) / off_close) * 100.0)

    # Structural check: the model level should be the official level times
    # ONE constant factor.  If the factor drifts, the composition is wrong.
    ratio = off_close / np.where(pred_level != 0, pred_level, np.nan)
    ratio_mean = float(np.nanmean(ratio))
    ratio_cv = float(np.nanstd(ratio) / ratio_mean)

    # Threshold-trigger agreement on the +/-4% / -2.3% rule.
    trig = lambda a: np.where(a >= 4.0, 1, np.where(a <= -2.3, -1, 0))
    trig_match = int((trig(pred_chg) == trig(off_chg)).sum())

    return {
        "metric": mae,
        "mae": mae,
        "rmse": rmse,
        "bias": bias,
        "dir_hit": dir_hit,
        "mae_dev": mae_dev,
        "mae_holdout": mae_hold,
        "level_mae": lvl_mae,
        "level_mape_pct": lvl_mape,
        "level_ratio_mean": ratio_mean,
        "level_ratio_cv": ratio_cv,
        "trigger_match": trig_match,
        "trigger_n": len(off_chg),
        "pred_level_last": float(pred_level[-1]),
        "official_level_last": float(off_close[-1]),
        "n": len(off_chg),
    }


def main() -> None:
    spec = json.loads(sys.argv[1]) if len(sys.argv) > 1 else {"window": 13}
    res = evaluate(spec)
    print(json.dumps({"spec": spec, **{k: v for k, v in res.items()}}, ensure_ascii=False))
    print(f"METRIC {res['metric']:.6f}", file=sys.stderr)


if __name__ == "__main__":
    main()
