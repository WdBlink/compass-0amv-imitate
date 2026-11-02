"""Grid scan over the model spec space.

Used for landscape diagnosis and to seed the autoresearch loop. Prints a
ranked table of the best specs by MAE of the predicted daily change.
"""

from __future__ import annotations

import itertools
import sys

import numpy as np

from evaluate import evaluate, load_official
from model import build_panel


def main() -> None:
    panel = build_panel()
    rows = []
    for N, lam in itertools.product((5, 10, 13, 15, 20, 25, 30, 40, 60, 90), (None, 0.90, 0.95, 0.97, 0.98, 0.99)):
        spec = {"window": N, "decay": lam, "market": "all_a"}
        r = evaluate(spec, panel)
        rows.append((r["mae"], r["mae_dev"], r["mae_holdout"], r["dir_hit"], r["level_mape_pct"], N, lam))
    rows.sort()
    print(f"{'MAE':>8s} {'dev':>8s} {'hold':>8s} {'dir':>6s} {'lvlMAPE':>9s}  {'N':>4s} {'lambda':>7s}")
    for mae, dev, hold, dhit, lm, N, lam in rows[:20]:
        print(f"{mae:8.4f} {dev:8.4f} {hold:8.4f} {dhit:6.3f} {lm:8.2f}%  {N:4d} {str(lam):>7s}")


if __name__ == "__main__":
    main()
