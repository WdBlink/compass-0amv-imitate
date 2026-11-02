"""Honest parameter selection for the 0AMV model.

The 64 official days are all exposed, so a plain grid search reports an
optimistic number.  This tool splits the window in half, selects the
spec on one half and reports the score on the other, in both
directions.  The mean of the two out-of-sample scores is the number to
trust when judging whether a structural change really helped.

    python3 cv.py --grid decay          # decay x init, both directions
    python3 cv.py --spec '<json>'       # one spec, both directions
"""

from __future__ import annotations

import argparse
import itertools
import json

import numpy as np

from evaluate import dev_mask, evaluate, load_official
from model import build_panel
from spec import FINAL_SPEC


def halves(panel):
    off_dates, _, _ = load_official()
    idx = np.array([panel.date_index(d) for d in off_dates])
    n = len(idx)
    split = n // 2
    return idx[:split], idx[split:]


def score_on(panel, spec, which) -> float:
    """MAE restricted to a subset of the official days."""
    from model import model_components
    off_dates, off_close, off_chg = load_official()
    idx = np.array([panel.date_index(d) for d in off_dates])
    comp = model_components(panel, spec)
    pred = comp["change"][idx]
    return float(np.mean(np.abs(pred[which] - off_chg[which])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--grid", choices=["decay"], default="decay")
    ap.add_argument("--spec", default=None)
    ap.add_argument("--top", type=int, default=12)
    args = ap.parse_args()

    panel = build_panel()
    off_dates, _, _ = load_official()
    n = len(off_dates)
    dev = dev_mask(n)
    hold = ~dev

    if args.spec:
        specs = [json.loads(args.spec)]
    else:
        specs = []
        for lam, init in itertools.product(
            np.arange(0.9180, 0.9301, 0.0005), ("zero", "steady"),
        ):
            specs.append({"decay": float(lam), "init": init, "market": "all_a",
                          "form": "recursion"})

    rows = []
    for spec in specs:
        d = score_on(panel, spec, dev)
        h = score_on(panel, spec, hold)
        full = evaluate(spec, panel)
        rows.append((d, h, full["mae"], full["dir_hit"], full["level_ratio_cv"],
                     full["level_mape_scaled_pct"], spec))

    # Selection discipline: pick by dev, report the holdout score.
    by_dev = min(rows, key=lambda r: r[0])
    by_hold = min(rows, key=lambda r: r[1])
    print(f"candidates: {len(rows)}")
    print(f"selected-by-DEV  : MAE_dev={by_dev[0]:.4f}  held-out MAE={by_dev[1]:.4f}  "
          f"full={by_dev[2]:.4f}  spec={by_dev[6]}")
    print(f"selected-by-HOLD : MAE_hold={by_hold[1]:.4f}  held-out MAE={by_hold[0]:.4f}  "
          f"full={by_hold[2]:.4f}  spec={by_hold[6]}")
    print(f"mean out-of-sample MAE = {(by_dev[1] + by_hold[0]) / 2:.4f}")

    print(f"\n{'dev':>8s} {'hold':>8s} {'full':>8s} {'dir':>6s} {'ratioCV':>9s} {'lvlScaled%':>11s}  spec")
    for r in sorted(rows, key=lambda r: r[2])[: args.top]:
        print(f"{r[0]:8.4f} {r[1]:8.4f} {r[2]:8.4f} {r[3]:6.3f} {r[4]:9.5f} {r[5]:10.5f}%  {r[6]}")

    print(f"\ncanonical spec {FINAL_SPEC}")
    r = evaluate(FINAL_SPEC, panel)
    print(f"  MAE={r['mae']:.6f}  dev={r['mae_dev']:.6f}  hold={r['mae_holdout']:.6f}  "
          f"dir={r['dir_hit']:.3f}  ratioCV={r['level_ratio_cv']:.6f}  "
          f"lvlScaled={r['level_mape_scaled_pct']:.5f}%")


if __name__ == "__main__":
    main()
