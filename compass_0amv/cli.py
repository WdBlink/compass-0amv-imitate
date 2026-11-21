"""Command line entry point for the 0AMV index.

    python3 -m compass_0amv --input panel.csv --out amv.csv

`panel.csv` has the columns `trade_date,ts_code,close,vol,float_share` in
Tushare units (`vol` in hands, `float_share` in 万股). Optional columns:
`turnover_rate` (daily percent) and `delist_date` (effective delisting date).
The output has the columns `trade_date,amv_cny,amv_yi,change_pct`.
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
                    help="CSV with trade_date,ts_code,close,vol,float_share; optional turnover_rate (percent), delist_date")
    ap.add_argument("--out", default=None, help="CSV to write (default: stdout)")
    ap.add_argument("--decay", type=float, default=amv.DECAY)
    ap.add_argument("--init", choices=["robust", "steady", "zero"], default="robust",
                    help="initial pool: expanding warmup median (default), full warmup mean, or zero")
    ap.add_argument("--missing-quote", choices=["freeze", "drop"], default="freeze",
                    help="freeze the pool and last stock value (default), or decay the pool and drop the value")
    ap.add_argument("--market", choices=["all_a", "all_a_bse"], default="all_a",
                    help="all_a includes Shanghai and Shenzhen A-shares (default)")
    ap.add_argument("--scale", type=float, default=1.0,
                    help="multiply the index level by this factor; pass "
                         f"{amv.OFFICIAL_LEVEL_SCALE} to align absolute levels with "
                         "the official index (daily changes are unaffected)")
    args = ap.parse_args(argv)

    rows = _read_csv(args.input)
    date_keys = {d: amv._date_key(d) for d in {r["trade_date"] for r in rows}}
    dates = sorted(set(date_keys.values()))
    codes = sorted({r["ts_code"] for r in rows})
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}

    close = np.full((len(dates), len(codes)), np.nan)
    vol = np.full((len(dates), len(codes)), np.nan)
    fsh = np.full((len(dates), len(codes)), np.nan)
    rates = np.full_like(close, np.nan) if rows and "turnover_rate" in rows[0] else None
    for r in rows:
        i = di[date_keys[r["trade_date"]]]
        j = ci[r["ts_code"]]
        close[i, j] = float(r["close"] or "nan")
        vol[i, j] = float(r["vol"] or "nan") * amv.HAND
        fsh[i, j] = float(r["float_share"] or "nan") * amv.WAN
        if rates is not None:
            rates[i, j] = float(r["turnover_rate"] or "nan")

    ends = amv._delist_dates_for(codes, {(r["ts_code"], r["delist_date"]) for r in rows
                                         if r.get("delist_date")})
    _, levels = amv.from_arrays(dates, codes, close, vol, fsh, market=args.market,
                                delist_dates=ends, turnover_rate=rates, decay=args.decay,
                                missing_quote=args.missing_quote, init=args.init)
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
