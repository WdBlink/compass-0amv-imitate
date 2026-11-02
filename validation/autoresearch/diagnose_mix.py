"""Structural mix test.

If the official 0AMV is a single mechanism, one candidate model level
should reproduce it up to ONE constant factor.  This script asks a
sharper question: does a *fixed linear combination* of several candidate
model levels reproduce the official level with constant coefficients?

    official_close_t ~= sum_k c_k * level_k(t)

A fit with tiny residual and constant c_k means the official aggregate is
a fixed blend of these mechanisms.  A fit with tiny residual and a
single dominant c_k means one mechanism is the answer.
"""

from __future__ import annotations

import itertools

import numpy as np

from evaluate import load_official
from model import build_panel, float_matrix, market_mask, model_components, turnover_rates


def levels_for(p, spec):
    return model_components(p, spec)["levels"][idx_of(p)] / 1e8


def idx_of(p):
    off_dates, _, _ = load_official()
    return np.array([p.date_index(d) for d in off_dates])


def main() -> None:
    p = build_panel()
    off_dates, off_close, _ = load_official()
    idx = idx_of(p)
    y = off_close

    mask = market_mask(p, "all_a")
    px = np.where(np.isfinite(p.close), p.close, 0.0)
    fcap = float_matrix(p, "float_share")
    frcap = float_matrix(p, "free_share")
    amt = np.where(np.isfinite(p.amount_cny), p.amount_cny, 0.0)
    M = mask[None, :]

    base = {"window": 70, "decay": 0.924, "market": "all_a"}

    candidates: dict[str, np.ndarray] = {}
    candidates["A_float_decay"] = levels_for(p, base)
    candidates["B_float_hardcut13"] = levels_for(p, {"window": 13, "decay": None, "market": "all_a"})
    candidates["C_free_decay"] = levels_for(p, {"market": "all_a", "float_field": "free_share",
                                                "window": 70, "decay": 0.924})
    candidates["D_amount_decay"] = levels_for(p, {"market": "all_a", "turnover_source": "amount_vwap",
                                                  "window": 70, "decay": 0.924})
    candidates["E_fullcap"] = (np.where(M, px * fcap, 0.0).sum(axis=1)[idx]) / 1e8
    candidates["F_float_amt_w"] = (np.where(M, amt * fcap, 0.0).sum(axis=1)[idx]) / 1e8

    names = list(candidates)
    print("=== single-component fits (official ~= c * level) ===")
    for n in names:
        x = candidates[n]
        c = float((x @ y) / (x @ x))
        resid = y - c * x
        rel = np.abs(resid / y) * 100
        cv = resid.std() / abs(resid.mean() + c * x.mean())
        print(f"  {n:20s} c={c:.6f}  relMAPE={rel.mean():7.4f}%  max={rel.max():7.4f}%  "
              f"residCV={resid.std()/np.mean(np.abs(resid)):.5f}")

    print("\n=== two-component fits ===")
    for a, b in itertools.combinations(names, 2):
        A = np.column_stack([candidates[a], candidates[b]])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rel = np.abs(resid / y) * 100
        print(f"  {a:20s} + {b:20s} c=({coef[0]:.5f},{coef[1]:.5f})  relMAPE={rel.mean():8.5f}%  max={rel.max():8.5f}%")

    print("\n=== three-component fit (A + C + D) ===")
    for combo in [("A_float_decay", "C_free_decay", "D_amount_decay"),
                  ("A_float_decay", "E_fullcap", "D_amount_decay"),
                  ("A_float_decay", "C_free_decay", "E_fullcap")]:
        A = np.column_stack([candidates[n] for n in combo])
        coef, *_ = np.linalg.lstsq(A, y, rcond=None)
        resid = y - A @ coef
        rel = np.abs(resid / y) * 100
        print(f"  {'+'.join(combo):60s} c={np.round(coef,5)}  relMAPE={rel.mean():8.5f}%  max={rel.max():8.5f}%")


if __name__ == "__main__":
    main()
