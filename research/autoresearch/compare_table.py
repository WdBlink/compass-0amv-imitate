"""Print the 64-day side-by-side table: model vs official 0AMV.

    python3 compare_table.py                     # markdown table on stdout
    python3 compare_table.py --csv out.csv       # same rows as CSV
    python3 compare_table.py --series-out s.csv  # full model series, raw + aligned

The official labels are read here only for display; the model never sees
them.

Level alignment
---------------
The official aggregate equals this model's level times one constant factor
(`level_ratio`, 0.997364 on the 64-day window).  The model level is
therefore ~0.27% *above* the official one, and aligning means multiplying
the model level BY that factor, not dividing.  The daily change percentage
is invariant to the factor, so only the level columns move.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from evaluate import dev_mask, load_official
from model import build_panel, daily_change_pct, model_series
from spec import FINAL_SPEC


def build_rows() -> tuple[list[dict], float]:
    """Return (rows, level_ratio) for the 64 official days."""
    panel = build_panel()
    off_dates, off_close, off_chg = load_official()
    levels = model_series(panel, FINAL_SPEC)
    chg = daily_change_pct(levels)
    idx = [panel.date_index(d) for d in off_dates]
    pred_chg = chg[idx]
    pred_level = levels[idx] / 1e8  # display units of 1e8 CNY

    ratio = off_close / pred_level
    level_ratio = float(np.mean(ratio))

    dev = dev_mask(len(off_dates))
    rows = []
    for i, d in enumerate(off_dates):
        err = pred_chg[i] - off_chg[i]
        rows.append({
            "date": f"{d[:4]}-{d[4:6]}-{d[6:]}",
            "half": "dev" if dev[i] else "hold",
            "official_close": off_close[i],
            "model_close": pred_level[i],
            "aligned_close": pred_level[i] * level_ratio,
            "level_ratio": off_close[i] / pred_level[i],
            "official_change_pct": off_chg[i],
            "model_change_pct": pred_chg[i],
            "err_pp": err,
            "abs_err_pp": abs(err),
            "dir_ok": int(np.sign(pred_chg[i]) == np.sign(off_chg[i])),
        })
    return rows, level_ratio


def write_series(path: str) -> None:
    """Write the full panel series, raw and official-aligned.

    The alignment factor is the one fitted on the 64-day official window;
    it is a constant, so the aligned series keeps the same daily changes.
    """
    panel = build_panel()
    levels = model_series(panel, FINAL_SPEC)
    chg = daily_change_pct(levels)
    _, level_ratio = build_rows()
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["trade_date", "amv_yi", "amv_yi_aligned", "change_pct"])
        for i, d in enumerate(panel.dates):
            w.writerow([f"{d[:4]}-{d[4:6]}-{d[6:]}", f"{levels[i]/1e8:.4f}",
                        f"{levels[i]/1e8*level_ratio:.4f}",
                        "" if i == 0 else f"{chg[i]:.6f}"])
    print(f"wrote {path} (aligned by {level_ratio:.6f})", file=sys.stderr)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--csv", default=None, help="write the 64 comparison rows as CSV")
    ap.add_argument("--md", default=None, help="write the 64 comparison rows as Markdown")
    ap.add_argument("--series-out", default=None,
                    help="write the full model series (raw + official-aligned) as CSV")
    args = ap.parse_args()

    rows, level_ratio = build_rows()
    mae = float(np.mean([r["abs_err_pp"] for r in rows]))
    lvl_mape_raw = float(np.mean([abs(r["model_close"] - r["official_close"])
                                   / r["official_close"] for r in rows]) * 100.0)
    lvl_mape_aligned = float(np.mean([abs(r["aligned_close"] - r["official_close"])
                                       / r["official_close"] for r in rows]) * 100.0)

    if args.csv:
        with open(args.csv, "w", newline="", encoding="utf-8") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0]))
            w.writeheader()
            w.writerows(rows)
        print(f"wrote {args.csv}", file=sys.stderr)

    table = [
        "| # | 日期 | 段 | 官方收盘 | 复刻收盘 | 对齐后收盘 | 水平比值 | 官方日幅% | 复刻日幅% | 误差pp | 方向 |",
        "|--:|---|---|--:|--:|--:|--:|--:|--:|--:|:-:|",
    ]
    for i, r in enumerate(rows, 1):
        table.append(
            f"| {i} | {r['date']} | {r['half']} | {r['official_close']:.1f} | "
            f"{r['model_close']:.1f} | {r['aligned_close']:.1f} | "
            f"{r['level_ratio']:.5f} | "
            f"{r['official_change_pct']:+.2f} | {r['model_change_pct']:+.2f} | "
            f"{r['err_pp']:+.3f} | {'✓' if r['dir_ok'] else '✗'} |"
        )
    table.append("")
    table.append(f"MAE = {mae:.6f} pp   方向一致 {sum(r['dir_ok'] for r in rows)}/{len(rows)}   "
                 f"level ratio = {level_ratio:.6f}   level MAPE 原始 {lvl_mape_raw:.4f}%  "
                 f"对齐后 {lvl_mape_aligned:.4f}%")

    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write("# 官方 0AMV vs 复刻模型：64 日逐日对比\n\n")
            fh.write("收盘列为亿元（1e8 CNY）；「对齐后收盘」= 复刻收盘 × "
                     f"{level_ratio:.6f}。误差 = 复刻日幅 − 官方日幅，单位为百分点。\n\n")
            fh.write("\n".join(table) + "\n")
        print(f"wrote {args.md}", file=sys.stderr)

    if args.series_out:
        write_series(args.series_out)

    print("\n".join(table))


if __name__ == "__main__":
    main()
