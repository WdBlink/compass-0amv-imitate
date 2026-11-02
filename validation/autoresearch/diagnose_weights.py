"""Diagnostic: which weighting scheme explains the official 0AMV daily change?

0AMV_t = sum_i W_{i,t} * P_{i,t}   with W = active shares of stock i.

    change% = [sum_i W_{i,t} P_{i,t} - sum_i W_{i,t-1} P_{i,t-1}]
              / sum_i W_{i,t-1} P_{i,t-1}

The numerator splits into a PRICE effect and a WEIGHT-DRIFT effect:

    price effect  = sum_i W_{i,t-1} (P_{i,t} - P_{i,t-1})
    drift effect  = sum_i P_{i,t-1} (W_{i,t} - W_{i,t-1})

This script scores candidate W schemes under (a) price effect only,
(b) price effect + drift (the full model), so we can see which half the
official series looks like.
"""

from __future__ import annotations

import numpy as np

from evaluate import load_official
from model import _survival, build_panel, market_mask, turnover_rates


def score(series: np.ndarray, off_idx, off_chg) -> float:
    pred = series[off_idx]
    return float(np.mean(np.abs(pred - off_chg)))


def main() -> None:
    p = build_panel()
    off_dates, off_close, off_chg = load_official()
    off_idx = np.array([p.date_index(d) for d in off_dates])

    px = np.where(np.isfinite(p.close), p.close, 0.0)
    fcap = np.where(np.isfinite(p.float_shares), p.float_shares, 0.0)
    amt = np.where(np.isfinite(p.amount_cny), p.amount_cny, 0.0)
    mask = market_mask(p, "all_a")
    mask3 = mask[None, :]

    # per-stock daily return, computed from close (NaN -> 0 contribution)
    prev = np.vstack([px[:1], px[:-1]])
    ret = np.where(prev > 0, px / np.where(prev > 0, prev, 1.0) - 1.0, 0.0)

    q = turnover_rates(p, "float_share", "vol")
    q = np.where(np.isfinite(p.close) & (p.close > 0), np.nan_to_num(q), 0.0)
    s = _survival(q)

    def active_shares(window: int, decay: float | None = None) -> np.ndarray:
        nd, ns = q.shape
        out = np.zeros((nd, ns))
        lam = 1.0 if decay is None else decay
        for k in range(min(window, nd)):
            if k == 0:
                out += np.nan_to_num(q)
                continue
            surv = np.exp(s[k:] - s[:-k])
            out[k:] += np.nan_to_num(q[:-k] * surv) * (lam ** k)
        return out * fcap

    schemes: dict[str, np.ndarray] = {}
    schemes["EW (equal weight)"] = np.ones_like(px)
    schemes["cap (float shares)"] = fcap
    schemes["amount (traded CNY)"] = amt
    schemes["active N=13"] = active_shares(13)
    schemes["active N=25"] = active_shares(25)
    schemes["active N=25 lam=0.97"] = active_shares(25, 0.97)
    schemes["active N=40 lam=0.95"] = active_shares(40, 0.95)

    print(f"{'scheme':26s} {'price-only MAE':>15s} {'full MAE':>10s} {'level last (亿)':>16s}")
    for name, W in schemes.items():
        Wm = np.where(mask3, W, 0.0)
        val = Wm * px
        total_prev = np.vstack([val[:1], val[:-1]]).sum(axis=1)
        price_eff = (Wm * (px - prev)).sum(axis=1)
        drift_eff = (prev * (Wm - np.vstack([Wm[:1], Wm[:-1]]))).sum(axis=1)
        denom = total_prev[1:]
        price_only = np.full(len(px), np.nan)
        price_only[1:] = price_eff[1:] / denom * 100.0
        full = np.full(len(px), np.nan)
        full[1:] = (price_eff[1:] + drift_eff[1:]) / denom * 100.0
        print(f"{name:26s} {score(price_only, off_idx, off_chg):15.4f} {score(full, off_idx, off_chg):10.4f} "
              f"{val[-1].sum()/1e8:16.1f}")

    print()
    # How much of the official daily change is explainable by ANY static
    # per-stock weight vector?  Fit official on the daily cross-section of
    # returns and look at the implied weight concentration.
    print("official daily change vs simple market returns:")
    for label, w in (("EW", schemes["EW (equal weight)"]),
                     ("cap", schemes["cap (float shares)"]),
                     ("amount", schemes["amount (traded CNY)"])):
        Wm = np.where(mask3, w, 0.0)
        val = Wm * px
        denom = np.vstack([val[:1], val[:-1]]).sum(axis=1)
        pe = (Wm * (px - prev)).sum(axis=1)
        ser = np.full(len(px), np.nan)
        ser[1:] = pe[1:] / denom[1:] * 100.0
        r = np.corrcoef(ser[off_idx], off_chg)[0, 1]
        print(f"  {label:8s} corr={r:+.4f}  MAE={score(ser, off_idx, off_chg):.4f}")


if __name__ == "__main__":
    main()
