"""Command line entry point for the 0AMV index.

    python3 -m compass_0amv --input panel.csv --out amv.csv

`panel.csv` has the columns `trade_date,ts_code,close,vol,float_share` in
Tushare units (`vol` in hands, `float_share` in 万股).  The output has
the columns `trade_date,amv_cny,amv_yi,change_pct`.
"""

from __future__ import annotations

import argparse
import csv
import sys

import numpy as np

from . import amv


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(prog="compass-0amv",
                                 description="Compute the 0AMV active-market-value index.")
    ap.add_argument("--input", required=True,
                    help="CSV with trade_date,ts_code,close,vol,float_share")
    ap.add_argument("--out", default=None, help="CSV to write (default: stdout)")
    ap.add_argument("--decay", type=float, default=amv.DECAY)
    ap.add_argument("--market", choices=["all_a", "all_a_bse"], default="all_a",
                    help="universe: all_a = A-shares ex-BSE (the official definition)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the index level by this factor; pass "
                         f"{amv.OFFICIAL_LEVEL_SCALE} to align absolute levels with "
                         "the official index (daily changes are unaffected)")
    args = ap.parse_args(argv)

    rows = _read_csv(args.input)
    dates = sorted({r["trade_date"].replace("-", "") for r in rows})
    codes = sorted({r["ts_code"] for r in rows})
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}

    close = np.full((len(dates), len(codes)), np.nan)
    vol = np.full((len(dates), len(codes)), np.nan)
    fsh = np.full((len(dates), len(codes)), np.nan)
    for r in rows:
        i = di[r["trade_date"].replace("-", "")]
        j = ci[r["ts_code"]]
        close[i, j] = float(r["close"])
        vol[i, j] = float(r["vol"]) * amv.HAND
        fsh[i, j] = float(r["float_share"]) * amv.WAN

    levels = amv.active_market_value(vol, fsh, close, decay=args.decay,
                                      market_mask=amv.market_mask_for(args.market, codes))
    if args.scale != 1.0:
        levels = levels * args.scale
    chg = amv.daily_change_pct(levels)

    fh = open(args.out, "w", newline="", encoding="utf-8") if args.out else None
    w = csv.writer(fh or sys.stdout)
    w.writerow(["trade_date", "amv_cny", "amv_yi", "change_pct"])
    for i, d in enumerate(dates):
        w.writerow([d, f"{levels[i]:.2f}", f"{levels[i]/1e8:.4f}",
                    "" if i == 0 else f"{chg[i]:.6f}"])
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
