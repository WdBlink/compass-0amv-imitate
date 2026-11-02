"""Grid scan over the model spec space.

Used for landscape diagnosis and to seed the autoresearch loop. Prints a
ranked table of the best specs by MAE of the predicted daily change.
"""

from __future__ import annotations

import itertools

import numpy as np

from evaluate import evaluate
from model import build_panel


def main() -> None:
    panel = build_panel()
    rows = []
    for lam, init, market in itertools.product(
        np.arange(0.900, 0.9601, 0.005), ("zero", "steady"),
        ("all_a", "all_a_bse", "sh_sz", "main_a", "main_growth", "main_star"),
    ):
        spec = {"decay": float(lam), "init": init, "market": market, "form": "recursion"}
        r = evaluate(spec, panel)
        rows.append((r["mae"], r["mae_dev"], r["mae_holdout"], r["dir_hit"],
                     r["level_mape_scaled_pct"], r["level_ratio_cv"], init, market, float(lam)))
    rows.sort()
    print(f"{'MAE':>8s} {'dev':>8s} {'hold':>8s} {'dir':>6s} {'lvlScaled%':>11s} {'ratioCV':>9s} "
          f"{'init':>7s} {'market':>12s} {'lambda':>7s}")
    for mae, dev, hold, dhit, lms, cv, init, market, lam in rows[:20]:
        print(f"{mae:8.4f} {dev:8.4f} {hold:8.4f} {dhit:6.3f} {lms:10.5f}% {cv:9.5f} "
              f"{init:>7s} {market:>12s} {lam:7.3f}")


if __name__ == "__main__":
    main()
