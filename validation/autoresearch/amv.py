"""指南针 0AMV (活跃市值 / 活筹指数) replication — deployable implementation.

This module is the production form of the formula recovered by the
autoresearch loop in this directory.  It is deliberately free of any
dependency on the frozen snapshot, on the official labels, and on the
research harness (`model.py`): it takes market inputs and returns the
index.

The recovered formula
---------------------

For every stock, let

    q_{i,t} = traded_shares_{i,t} / float_shares_{i,t}      turnover rate

be the fraction of the free float that changed hands on day t.  The
stock's chips form a pool: every day a fresh batch of size `q_t` is
created by that day's trading, the rest of the pool (`1 - q_t` of it)
survives, and the pool additionally loses a fraction `1 - lambda` of its
weight to age:

    f_{i,t} = q_{i,t} + lambda * (1 - q_{i,t}) * f_{i,t-1}

`f` is the share of the float that is "active" (活筹).  The market's
active market value is

    0AMV_t = sum_i close_{i,t} * float_shares_{i,t} * f_{i,t}

with `lambda = 0.9235` and every A-share except the Beijing Stock
Exchange.

Reproduced on the 64 official days (2026-07-27 … 2026-10-30):
daily-change MAE 0.0217 pp, direction 64/64, and the model level equals
the official level times 0.997364 with a coefficient of variation of
0.0006 — i.e. one constant factor, not a curve fit.

Usage
-----

    import amv
    dates, levels = amv.from_tushare_frame(df)      # df: one row per stock-day

or from the command line:

    python3 amv.py --input panel.csv --out amv.csv

where `panel.csv` has the columns `trade_date,ts_code,close,vol,
float_share` in Tushare units (vol in hands, float_share in 万股).
"""

from __future__ import annotations

import argparse
import csv
from typing import Iterable

import numpy as np

# Age decay of the chip pool: the pool keeps 92.35% of its weight per
# additional day.  Selected on the first half of the official window and
# confirmed on the second half (see results.tsv, iterations 8-10).
DECAY = 0.9235

# Number of days used to estimate each stock's own turnover level when
# initialising the chip pool.  The result is insensitive to this choice
# (MAE varies in the 5th decimal over 5..60).
INIT_WINDOW = 20

# Tushare unit conversions.
HAND = 100.0          # 1 手 = 100 股
WAN = 10_000.0        # 1 万股 = 10000 股

# Board prefixes, used to select the official universe: every A-share
# except the Beijing Stock Exchange.  B-shares are not A-shares at all.
_A_MAIN = ("600", "601", "603", "605", "000", "001", "002", "003")
_A_GROWTH = ("300", "301")
_A_STAR = ("688", "689")
_BSE = ("43", "83", "87", "88", "92")
_B_SHARE = ("900", "200")


def a_share_mask(codes) -> np.ndarray:
    """Boolean mask of the A-shares the official index covers (ex-BSE).

    Including the Beijing Stock Exchange degrades the daily-change MAE
    from 0.0217 pp to 0.0256 pp, so the exclusion is part of the
    recovered definition, not a free parameter.
    """
    out = []
    for c in codes:
        num = str(c)[:6]
        if num.startswith(_B_SHARE):
            out.append(False)
        elif num.startswith(_A_MAIN) or num.startswith(_A_GROWTH) or num.startswith(_A_STAR):
            out.append(True)
        elif num.startswith(_BSE):
            out.append(False)
        else:
            out.append(False)
    return np.array(out, dtype=bool)


def active_chip_fraction(vol_shares: np.ndarray,
                         float_shares: np.ndarray,
                         close: np.ndarray,
                         *,
                         decay: float = DECAY,
                         init: str = "steady",
                         init_window: int = INIT_WINDOW) -> np.ndarray:
    """Active (活筹) share of the free float, per (date, stock).

    Parameters
    ----------
    vol_shares, float_shares, close
        (nd, ns) arrays.  NaN marks a date on which the stock had no
        quote (suspended, or not yet listed); such a day counts as zero
        turnover and the stock contributes nothing to that day's index.
    decay
        Age decay `lambda` of the chip pool.
    init
        "steady" starts each pool at its quasi-stationary level, which
        removes the warmup bias of a finite history; "zero" starts it
        empty.

    Returns
    -------
    (nd, ns) float array of active fractions, each in [0, 1].
    """
    vol_shares = np.asarray(vol_shares, dtype=float)
    float_shares = np.asarray(float_shares, dtype=float)
    close = np.asarray(close, dtype=float)

    fcap = np.where(np.isfinite(float_shares), float_shares, 0.0)
    traded = np.where(np.isfinite(vol_shares), vol_shares, 0.0)

    with np.errstate(invalid="ignore", divide="ignore"):
        q = traded / fcap
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    q = np.clip(q, 0.0, 1.0)

    # No quote means the stock did not trade that day.
    has_quote = np.isfinite(close) & (close > 0)
    q = np.where(has_quote, q, 0.0)

    nd, ns = q.shape
    if init == "steady":
        k = min(init_window, nd)
        qbar = q[:k].mean(axis=0)
        prev = qbar / np.maximum(1.0 - decay * (1.0 - qbar), 1e-6)
    elif init == "zero":
        prev = np.zeros(ns)
    else:
        raise ValueError(f"unknown init {init!r}")

    out = np.empty((nd, ns))
    for t in range(nd):
        prev = q[t] + decay * (1.0 - q[t]) * prev
        out[t] = prev
    return out


def active_market_value(vol_shares: np.ndarray,
                         float_shares: np.ndarray,
                         close: np.ndarray,
                         *,
                         decay: float = DECAY,
                         init: str = "steady",
                         init_window: int = INIT_WINDOW,
                         market_mask: np.ndarray | None = None) -> np.ndarray:
    """The 0AMV level series, in CNY.

    `market_mask` is an (ns,) boolean array selecting the stocks to
    include; by default every stock with a float is included.
    """
    f = active_chip_fraction(vol_shares, float_shares, close,
                              decay=decay, init=init, init_window=init_window)
    fcap = np.where(np.isfinite(float_shares), float_shares, 0.0)
    px = np.where(np.isfinite(close), close, 0.0)
    w = f * fcap
    if market_mask is not None:
        w = np.where(market_mask[None, :], w, 0.0)
        px = np.where(market_mask[None, :], px, 0.0)
    return (w * px).sum(axis=1)


def daily_change_pct(levels: Iterable[float]) -> np.ndarray:
    """Percent change of a level series; the first entry is NaN."""
    levels = np.asarray(levels, dtype=float)
    out = np.full(len(levels), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = (levels[1:] / levels[:-1] - 1.0) * 100.0
    return out


def from_arrays(dates, codes, close, vol_shares, float_shares,
                *, market_mask=None, **kwargs):
    """Build the index from dense (date x stock) arrays."""
    levels = active_market_value(vol_shares, float_shares, close,
                                  market_mask=market_mask, **kwargs)
    return list(dates), levels


def from_tushare_frame(df, *, market="all_a", **kwargs):
    """Build the index from a long-format DataFrame of Tushare fields.

    Required columns: `trade_date`, `ts_code`, `close`, `vol` (hands),
    `float_share` (万股).  Optional: `adj_factor` (ignored).

    `market` is "all_a" (the official universe: A-shares ex-BSE) or
    "all_a_bse".

    Returns `(dates, levels)` with `dates` as YYYYMMDD strings and
    `levels` in CNY.
    """
    df = df.copy()
    df["trade_date"] = df["trade_date"].astype(str).str.replace("-", "", regex=False)
    df = df.sort_values(["trade_date", "ts_code"], kind="stable")

    dates = sorted(df["trade_date"].unique())
    codes = sorted(df["ts_code"].unique())
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}

    nd, ns = len(dates), len(codes)
    close = np.full((nd, ns), np.nan)
    vol = np.full((nd, ns), np.nan)
    fsh = np.full((nd, ns), np.nan)

    di_idx = df["trade_date"].map(di).to_numpy()
    ci_idx = df["ts_code"].map(ci).to_numpy()
    close[di_idx, ci_idx] = df["close"].to_numpy(dtype=float)
    vol[di_idx, ci_idx] = df["vol"].to_numpy(dtype=float) * HAND
    fsh[di_idx, ci_idx] = df["float_share"].to_numpy(dtype=float) * WAN

    mask = _market_mask_for(market, codes)
    levels = active_market_value(vol, fsh, close,
                                  market_mask=mask, **kwargs)
    return dates, levels


def _market_mask_for(market: str, codes) -> np.ndarray | None:
    if market == "all_a":
        return a_share_mask(codes)
    if market == "all_a_bse":
        return np.ones(len(list(codes)), dtype=bool)
    raise ValueError(f"unknown market {market!r}")


def _read_csv(path):
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def main() -> None:
    ap = argparse.ArgumentParser(description="Compute the 0AMV active-market-value index.")
    ap.add_argument("--input", required=True,
                    help="CSV with trade_date,ts_code,close,vol,float_share")
    ap.add_argument("--out", default=None, help="CSV to write (default: stdout)")
    ap.add_argument("--decay", type=float, default=DECAY)
    ap.add_argument("--market", choices=["all_a", "all_a_bse"], default="all_a",
                    help="universe: all_a = A-shares ex-BSE (the official definition)")
    args = ap.parse_args()

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
        vol[i, j] = float(r["vol"]) * HAND
        fsh[i, j] = float(r["float_share"]) * WAN

    levels = active_market_value(vol, fsh, close, decay=args.decay,
                                 market_mask=_market_mask_for(args.market, codes))
    chg = daily_change_pct(levels)

    out = args.out
    fh = open(out, "w", newline="", encoding="utf-8") if out else None
    w = csv.writer(fh or __import__("sys").stdout)
    w.writerow(["trade_date", "amv_cny", "amv_yi", "change_pct"])
    for i, d in enumerate(dates):
        w.writerow([d, f"{levels[i]:.2f}", f"{levels[i]/1e8:.4f}",
                    "" if i == 0 else f"{chg[i]:.6f}"])
    if fh:
        fh.close()


if __name__ == "__main__":
    main()
