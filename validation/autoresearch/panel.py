"""Full-market panel loader for the frozen r2-20261030 snapshot.

Loads the frozen Tushare responses into dense (date x stock) matrices so that
active-chip models can be evaluated without touching the network.

Units are normalised to physical units on load:
    vol   (hands)      -> shares   = vol * 100
    amount(thousand)  -> CNY      = amount * 1000
    *_share (10k)      -> shares   = value * 10000
    *_mv    (10k CNY)  -> CNY      = value * 10000

The snapshot is read-only. Nothing here writes to it.
"""

from __future__ import annotations

import glob
import json
import os
from dataclasses import dataclass

import numpy as np

SNAPSHOT_ROOT = "/Users/wdblink/Data/datasets/derived/compass-0amv/r2-20261030"
RAW_DIR = os.path.join(SNAPSHOT_ROOT, "raw")
CACHE_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_panel_cache.npz")

# Board classification by numeric code prefix.
#   SH main   600/601/603/605
#   SH STAR   688/689
#   SZ main   000/001/002/003
#   SZ ChiNext 300/301
#   BJ        43/83/87/88/92
MAIN_A = ("600", "601", "603", "605", "000", "001", "002", "003")
GROWTH_A = ("300", "301")
STAR_A = ("688", "689")
BSE_A = ("43", "83", "87", "88", "92")
B_SHARE = ("900", "200")


@dataclass
class Panel:
    dates: np.ndarray          # (nd,)  YYYYMMDD strings
    codes: np.ndarray          # (ns,)  ts_code strings
    close: np.ndarray          # (nd, ns) unadjusted CNY/share, NaN where absent
    vol_shares: np.ndarray     # (nd, ns)
    amount_cny: np.ndarray     # (nd, ns)
    float_shares: np.ndarray   # (nd, ns)
    free_shares: np.ndarray    # (nd, ns)
    circ_mv: np.ndarray        # (nd, ns) CNY
    turnover_rate: np.ndarray  # (nd, ns) percent
    board: np.ndarray          # (ns,) board label string

    @property
    def n_dates(self) -> int:
        return len(self.dates)

    @property
    def n_stocks(self) -> int:
        return len(self.codes)

    def date_index(self, yyyymmdd: str) -> int:
        idx = np.where(self.dates == np.array(yyyymmdd))[0]
        if len(idx) == 0:
            raise KeyError(f"date {yyyymmdd} not in panel")
        return int(idx[0])


def _load_endpoint(prefix: str) -> dict[str, dict[str, dict]]:
    """prefix -> {trade_date -> {ts_code -> {field: value}}}"""
    out: dict[str, dict[str, dict]] = {}
    # `daily_*` would also match `daily_basic_*`; anchor on the date prefix.
    pattern = f"{prefix}_[0-9]*.json"
    for path in sorted(glob.glob(os.path.join(RAW_DIR, pattern))):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("code") != 0:
            continue
        data = payload["data"]
        names = data["fields"]
        for row in data["items"]:
            rec = dict(zip(names, row))
            day = rec.pop("trade_date")
            code = rec.pop("ts_code")
            out.setdefault(day, {})[code] = rec
    return out


def _board_of(code: str) -> str:
    num = code[:6]
    if num.startswith(B_SHARE):
        return "b_share"
    if num.startswith(MAIN_A):
        return "main_a"
    if num.startswith(GROWTH_A):
        return "growth_a"
    if num.startswith(STAR_A):
        return "star_a"
    if num.startswith(BSE_A):
        return "bse_a"
    return "other"


def build_panel(use_cache: bool = True) -> Panel:
    if use_cache and os.path.exists(CACHE_PATH):
        z = np.load(CACHE_PATH, allow_pickle=False)
        return Panel(
            dates=z["dates"], codes=z["codes"], close=z["close"], vol_shares=z["vol_shares"],
            amount_cny=z["amount_cny"], float_shares=z["float_shares"],
            free_shares=z["free_shares"], circ_mv=z["circ_mv"],
            turnover_rate=z["turnover_rate"], board=z["board"],
        )

    daily = _load_endpoint("daily")
    basic = _load_endpoint("daily_basic")

    dates = sorted(set(daily) & set(basic))
    codes = sorted({c for d in dates for c in daily[d]})
    di = {d: i for i, d in enumerate(dates)}
    ci = {c: i for i, c in enumerate(codes)}
    nd, ns = len(dates), len(codes)

    close = np.full((nd, ns), np.nan)
    vol = np.full((nd, ns), np.nan)
    amt = np.full((nd, ns), np.nan)
    fsh = np.full((nd, ns), np.nan)
    frsh = np.full((nd, ns), np.nan)
    cmv = np.full((nd, ns), np.nan)
    tor = np.full((nd, ns), np.nan)

    for d, i in di.items():
        for c, rec in daily[d].items():
            j = ci[c]
            close[i, j] = rec["close"]
            vol[i, j] = rec["vol"] * 100.0 if rec["vol"] is not None else np.nan
            amt[i, j] = rec["amount"] * 1000.0 if rec["amount"] is not None else np.nan
        for c, rec in basic.get(d, {}).items():
            j = ci.get(c)
            if j is None:
                continue
            tor[i, j] = rec["turnover_rate"]
            fsh[i, j] = rec["float_share"] * 10000.0 if rec["float_share"] is not None else np.nan
            frsh[i, j] = rec["free_share"] * 10000.0 if rec["free_share"] is not None else np.nan
            cmv[i, j] = rec["circ_mv"] * 10000.0 if rec["circ_mv"] is not None else np.nan

    board = np.array([_board_of(c) for c in codes])

    panel = Panel(dates=np.array(dates), codes=np.array(codes), close=close, vol_shares=vol,
                  amount_cny=amt, float_shares=fsh, free_shares=frsh, circ_mv=cmv,
                  turnover_rate=tor, board=board)
    np.savez_compressed(CACHE_PATH, dates=panel.dates, codes=panel.codes, close=close,
                        vol_shares=vol, amount_cny=amt, float_shares=fsh, free_shares=frsh,
                        circ_mv=cmv, turnover_rate=tor, board=board)
    return panel


if __name__ == "__main__":
    p = build_panel()
    print(f"dates {p.n_dates} {p.dates[0]}..{p.dates[-1]}")
    print(f"stocks {p.n_stocks}")
    for b in np.unique(p.board):
        print(f"  {b:10s} {int((p.board == b).sum())}")
    print("circ_mv total at last date (CNY):", np.nansum(p.circ_mv[-1]))
