"""Structural sweep for the 2026-11-21 follow-up round.

Every row is scored with the shipped harness (``research/autoresearch``) on the
same 64 official dates.  Selection discipline, unchanged from the first round:
a candidate is kept only if the full-window MAE falls, the second-half MAE does
not rise, and the level-ratio coefficient of variation does not rise.

    python3 autoresearch/loop-261121-2318/scripts/scan.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "autoresearch"))

from evaluate import evaluate  # noqa: E402
from model import build_panel  # noqa: E402
from spec import FINAL_SPEC  # noqa: E402


def main() -> None:
    panel = build_panel()
    base = dict(FINAL_SPEC)
    rows: list[tuple[str, dict]] = [
        ("baseline (round-1 winner, init_window=20)", {"init_window": 20}),
        ("init_window=30", {"init_window": 30}),
        ("init_window=32", {"init_window": 32}),
        ("init_window=40", {"init_window": 40}),
        ("decay=0.9229", {"decay": 0.9229}),
        ("decay=0.9230", {"decay": 0.9230}),
        ("decay=0.9231", {"decay": 0.9231}),
        ("decay=0.9230 + init_window=30", {"decay": 0.9230, "init_window": 30}),
        ("init=steady", {"init": "steady"}),
        ("init=zero", {"init": "zero"}),
        ("turnover_source=vol", {"turnover_source": "vol"}),
        ("survival=off", {"survival": "off"}),
        ("survival_power=0.95", {"survival_power": 0.95}),
        ("survival_power=1.05", {"survival_power": 1.05}),
        ("float_mode=lagged", {"float_mode": "lagged"}),
        ("float_field=free_share", {"float_field": "free_share"}),
        ("price_mode=adj", {"price_mode": "adj"}),
        ("price_fill=ffill + legacy quotes", {"price_fill": "ffill", "quote_policy": "legacy"}),
        ("market=sh_sz (no 302)", {"market": "sh_sz"}),
        ("market=all_a (with BSE)", {"market": "all_a"}),
    ]
    print(f"{'candidate':38s} {'full':>9s} {'dev':>9s} {'hold':>9s} {'cv':>9s} {'guard':>7s}")
    for name, override in rows:
        spec = dict(base, **override)
        res = evaluate(spec, panel)
        guard = "pass" if (res["mae_holdout"] <= 7.782204970050785e-3
                           and res["level_ratio_cv"] <= 2.4143655486525984e-4
                           and res["mae"] < 0.011320464284177417) else "fail"
        print(f"{name:38s} {res['mae']:9.6f} {res['mae_dev']:9.6f} {res['mae_holdout']:9.6f} "
              f"{res['level_ratio_cv']:9.6f} {guard:>7s}")


if __name__ == "__main__":
    main()
