"""Diagnose the residual drift in official/model level ratio.

The model reproduces the official level up to a scale factor that should
be constant.  It is not quite: it wanders by ~0.3% over the window.
This script correlates that wander with candidate daily aggregates to
find the missing structural piece.
"""

from __future__ import annotations

import numpy as np

from evaluate import load_official
from model import build_panel, model_components, market_mask

SPEC = {"window": 70, "decay": 0.924, "market": "all_a"}


def main() -> None:
    p = build_panel()
    off_dates, off_close, off_chg = load_official()
    idx = np.array([p.date_index(d) for d in off_dates])

    comp = model_components(p, SPEC)
    mdl_level = comp["levels"][idx] / 1e8
    ratio = off_close / mdl_level

    mask = market_mask(p, "all_a")
    px = np.where(np.isfinite(p.close), p.close, 0.0)
    amt = np.where(np.isfinite(p.amount_cny), p.amount_cny, 0.0)
    fcap = np.where(np.isfinite(p.float_shares), p.float_shares, 0.0)
    traded = np.isfinite(p.close) & (p.vol_shares > 0)

    rows = []
    for j, d in enumerate(off_dates):
        i = idx[j]
        prev = i - 1
        cap = np.nansum(np.where(mask, p.circ_mv[i], 0.0)) / 1e8
        new_listings = int(np.sum(np.isfinite(p.close[i]) & ~np.isfinite(p.close[max(i - 1, 0)])))
        n_traded = int(np.sum(traded[i] & mask))
        n_suspended = int(np.sum(np.isfinite(p.close[prev]) & ~np.isfinite(p.close[i]) & mask))
        turnover = np.nansum(np.where(mask, p.vol_shares[i], 0.0)) / np.nansum(
            np.where(mask, fcap[i], 0.0)) * 100.0
        rows.append({
            "date": d, "ratio": ratio[j],
            "cap": cap, "amount": np.nansum(np.where(mask, amt[i], 0.0)) / 1e12,
            "new_listings": new_listings, "n_traded": n_traded,
            "n_suspended": n_suspended, "turnover_pct": turnover,
        })

    print(f"{'date':10s} {'ratio':>8s} {'d_ratio':>9s} {'cap(亿)':>10s} {'amt(万亿)':>9s} "
          f"{'new':>4s} {'ntrd':>5s} {'susp':>5s} {'turn%':>6s}")
    for k, r in enumerate(rows):
        dr = ratio[k] - ratio[k - 1] if k else 0.0
        print(f"{r['date']:10s} {r['ratio']:8.5f} {dr:+9.6f} {r['cap']:10.1f} {r['amount']:9.3f} "
              f"{r['new_listings']:4d} {r['n_traded']:5d} {r['n_suspended']:5d} {r['turnover_pct']:6.3f}")

    # What explains the day-to-day change of the ratio?
    dr = np.diff(ratio)
    print("\ncorrelation of d(ratio) with candidate drivers:")
    for key in ("cap", "amount", "n_traded", "n_suspended", "turnover_pct", "new_listings"):
        v = np.array([r[key] for r in rows[1:]], dtype=float)
        if np.std(v) == 0:
            continue
        print(f"  {key:16s} corr={np.corrcoef(dr, v)[0,1]:+.4f}")

    # Same, but against the daily change error itself.
    pred = comp["change"][idx]
    err = pred - off_chg
    print("\ncorrelation of |daily-change error| with drivers:")
    for key in ("cap", "amount", "n_traded", "n_suspended", "turnover_pct"):
        v = np.array([r[key] for r in rows], dtype=float)
        print(f"  {key:16s} corr={np.corrcoef(np.abs(err), v)[0,1]:+.4f}")

    print("\nworst daily-change errors:")
    order = np.argsort(-np.abs(err))
    for j in order[:12]:
        print(f"  {off_dates[j]}  off={off_chg[j]:+6.2f}%  model={pred[j]:+6.3f}%  err={err[j]:+6.3f}  "
              f"ratio={ratio[j]:.5f}  turn%={rows[j]['turnover_pct']:.3f}  amt={rows[j]['amount']:.3f}")


if __name__ == "__main__":
    main()
