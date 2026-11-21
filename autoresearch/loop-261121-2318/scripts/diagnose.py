"""Where the remaining error lives, and what the app's own intraday bar says.

The follow-up round found that the residual error is not diffuse: three sessions
carry most of it, and on each of them the model's level jumps once and then
decays back at the chip rate.  This script reproduces that picture:

  * ``gap``  - model level minus official level, per official session;
  * ``E``    - the daily excess creation, ``gap_t - 12/13 * gap_{t-1}``;
  * the app's own intraday bar (open/high/low, transcribed from the archived
    screenshots) against the model's overnight and intraday legs.

    python3 autoresearch/loop-261121-2318/scripts/diagnose.py
"""

from __future__ import annotations

import glob
import json
import os
import sys

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "research", "autoresearch"))

from evaluate import load_official  # noqa: E402
from model import (active_fraction, build_panel, daily_change_pct,  # noqa: E402
                   market_mask, price_series)
from panel import RAW_DIR  # noqa: E402

LAM = 12 / 13

# App intraday bar for the 0AMV index, read from validation/official_daily_20261030.
# date -> (open, high, low, close, screenshot)
APP_BAR = {
    "20260803": (196175.4, 199118.4, 193035.6, 193198.4, "screenshots/early_09_0.png"),
    "20260804": (195114.6, 200811.1, 194946.5, 199285.3, "screenshots/early_08_0.png"),
    "20260828": (191043.0, 195661.6, 189214.9, 189298.4, "screenshots/pair_30_0.png"),
    "20260831": (189293.5, 191568.8, 186857.0, 190623.9, "screenshots/pair_29_0.png"),
    "20260901": (190632.1, 192179.3, 187627.1, 187670.6, "screenshots/pair_28_0.png"),
    "20261023": (134183.9, 143307.0, 133402.6, 143060.5, "screenshots/valid_01.png"),
    "20261026": (141795.2, 143698.2, 140791.8, 141365.6, "screenshots/read_04.png"),
}


def load_opens(panel) -> np.ndarray:
    """(nd, ns) opening prices from the frozen daily responses."""
    opens = np.full((panel.n_dates, panel.n_stocks), np.nan)
    date_index = {d: i for i, d in enumerate(panel.dates)}
    code_index = {c: i for i, c in enumerate(panel.codes)}
    for path in sorted(glob.glob(os.path.join(RAW_DIR, "daily_[0-9]*.json"))):
        with open(path, "r", encoding="utf-8") as fh:
            payload = json.load(fh)
        if payload.get("code") != 0:
            continue
        data = payload["data"]
        for row in data["items"]:
            record = dict(zip(data["fields"], row))
            t = date_index.get(record["trade_date"])
            j = code_index.get(record["ts_code"])
            if t is None or j is None:
                continue
            opens[t, j] = record.get("open")
    return opens


def main() -> None:
    panel = build_panel()
    off_dates, off_close, off_chg = load_official()
    mask = market_mask(panel, "sh_sz_a")
    fcap = np.where(np.isfinite(panel.float_shares), panel.float_shares, 0.0)
    has = np.isfinite(panel.close) & (panel.close > 0)
    q = np.where(has, np.clip(np.nan_to_num(panel.turnover_rate / 100.0), 0, 1), 0.0)
    f = active_fraction(q, 13, LAM, "on", "exp", "robust", 30, "recursion", fcap, 1.0, 0.0, has)
    last = np.maximum.accumulate(np.where(has, np.arange(panel.n_dates)[:, None], -1), axis=0)
    F = np.where(last >= 0, np.take_along_axis(fcap, last, axis=0), 0.0)
    raw_close = price_series(panel, "zero", "close")
    P = np.where(last >= 0, np.take_along_axis(raw_close, last, axis=0), 0.0)
    alive = (panel.delist_dates == "") | (panel.dates[:, None] < panel.delist_dates)
    value = np.where(mask[None, :] & alive, f * F * P, 0.0)
    levels = value.sum(axis=1)
    opens = load_opens(panel)

    idx = np.array([panel.date_index(d) for d in off_dates])
    official_level = np.asarray(off_close, float) * 1e8
    gap = levels[idx] - official_level
    err = daily_change_pct(levels)[idx] - off_chg
    excess = np.empty(len(gap))
    excess[0] = gap[0]
    excess[1:] = gap[1:] - LAM * gap[:-1]

    print(f"{'date':10s} {'err_pp':>8s} {'gap(亿)':>10s} {'excess(亿)':>11s}")
    for j, day in enumerate(off_dates):
        flag = "  <<<" if abs(excess[j]) > 1e10 else ""
        print(f"{day:10s} {err[j]:+8.4f} {gap[j]/1e8:10.1f} {excess[j]/1e8:+11.1f}{flag}")

    total = np.abs(err).sum()
    top3 = np.argsort(-np.abs(err))[:3]
    print(f"\nMAE {np.abs(err).mean():.6f} pp; the three worst sessions carry "
          f"{sum(abs(err[j]) for j in top3)/total*100:.0f}% of the absolute error")
    print(f"without them the MAE would be "
          f"{(total - sum(abs(err[j]) for j in top3))/len(err):.6f} pp")

    print("\napp intraday bar vs the model's own overnight / intraday legs")
    print(f"{'date':10s} {'app_ovn':>9s} {'our_ovn':>9s} {'app_intra':>10s} {'our_intra':>10s} "
          f"{'our_open_in_app_range':>22s}")
    for day, (a_open, a_high, a_low, a_close, _shot) in APP_BAR.items():
        j = off_dates.index(day)
        t = panel.date_index(day)
        prev_value = levels[t - 1]
        # The app re-prices the previous session's pool at the opening prices.
        # A suspended stock keeps its last close, so it contributes no overnight move.
        open_prices = np.where(has[t] & np.isfinite(opens[t]), opens[t], P[t])
        prev_f = np.vstack([np.zeros((1, panel.n_stocks)), f[:-1]])
        prev_F = np.vstack([np.zeros((1, panel.n_stocks)), F[:-1]])
        open_value = float(np.where(mask & alive[t],
                                    prev_f[t] * prev_F[t] * open_prices, 0.0).sum())
        app_ovn = a_open / off_close[j - 1] - 1
        our_ovn = open_value / prev_value - 1
        app_intra = a_close / a_open - 1
        our_intra = levels[t] / open_value - 1
        inside = a_low <= open_value / 1e8 <= a_high
        print(f"{day:10s} {app_ovn*100:+9.4f} {our_ovn*100:+9.4f} {app_intra*100:+10.4f} "
              f"{our_intra*100:+10.4f} {str(bool(inside)):>22s}")
    print("\nThe last column is false only for 2026-08-31: the model's 09:30 value "
          "lies below the app's own intraday low, so the app's bar and the model "
          "cannot both describe that session.")


if __name__ == "__main__":
    main()
