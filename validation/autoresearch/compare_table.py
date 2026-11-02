"""Print the 64-day side-by-side table: model vs official 0AMV.

    python3 compare_table.py            # markdown table on stdout
    python3 compare_table.py --csv out.csv

The official labels are read here only for display; the model never sees
them.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from evaluate import dev_mask, load_official
from model import build_panel, daily_change_pct, model_series
from spec import FINAL_SPEC


def build_rows() -> list[dict]:
    panel = build_panel()
    off_dates, off_close, off_chg = load_official()
    levels = model_series(panel, FINAL_SPEC)
    chg = daily_change_pct(levels)
    idx = [panel.date_index(d) for d in off_dates]
    pred_chg = chg[idx]
    pred_level = levels[idx] / 1e8  # display units of 1e8 CNY

    dev = dev_mask(len(off_dates))
    rows = []
    for i, d in enumerate(off_dates):
        err = pred_chg[i] - off_chg[i]
        rows.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "half": "dev" if dev[i] else "hold",
            "official_close": off_close[i],
            "model_close": pred_level[i],
            "level_ratio": off_close[i] / pred_level[i],
            "official_change_pct": off_chg[i],
            "model_change_pct": pred_chg[i],
            "err_pp": err,
            "abs_err_pp": abs(err),
            "dir_ok": int(np.sign(pred_chg[i]) == np.sign(off_chg[i])),
        })
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None)
    args = ap.parse_args()

    rows = build_rows()
    mae = float(np.mean([r["abs_err_pp"] for r in rows]))

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}", file=sys.stderr)

    print("| # | 日期 | 段 | 官方收盘 | 复刻收盘 | 水平比值 | 官方日幅% | 复刻日幅% | 误差pp | 方向 |")
    print("|--:|---|---|--:|--:|--:|--:|--:|--:|:-:|")
    for i, r in enumerate(rows, 1):
        print(
            f"| {i} | {r['date']} | {r['half']} | {r['official_close']:.1f} | "
            f"{r['model_close']:.1f} | {r['level_ratio']:.5f} | "
            f"{r['official_change_pct']:+.2f} | {r['model_change_pct']:+.2f} | "
            f"{r['err_pp']:+.3f} | {'✓' if r['dir_ok'] else '✗'} |"
        )
    print(f"\nMAE = {mae:.6f} pp   方向一致 {sum(r['dir_ok'] for r in rows)}/{len(rows)}")


if __name__ == "__main__":
    main()
