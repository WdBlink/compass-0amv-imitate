"""Active-chip market-value model family for the 0AMV replication.

Core idea (standard 筹码分布 / 活筹 batch-replacement model):

    q_{i,s} = traded_shares_{i,s} / float_shares_{i,s}      daily turnover rate
    batch created on day s has size q_s * F and each later day j a
    fraction q_j of it is resold (re-entering the pool as a fresh batch
    bought that day). The batch still held on day t is

        batch_s(t) = q_s * F * prod_{j=s+1..t} (1 - q_j)

    The "active" (活筹) share of the float on day t is the sum of the
    batches created in the trailing window:

        f_i(t) = sum_{k=0..N-1} q_{t-k} * prod_{j=t-k+1..t} (1 - q_j)
               = 1 - prod_{j=t-N+1..t} (1 - q_j)          (hard cutoff)

    and the market's active market value is

        0AMV_t = sum_i close_{i,t} * float_shares_{i,t} * f_i(t)

Generalisation used by the search: replace the hard cutoff by an age
decay lambda, f_i(t) = sum_{k=0..N-1} lambda^k * q_{t-k} * prod(...).

Everything is vectorised over the (date x stock) panel.
"""

from __future__ import annotations

import numpy as np

from panel import Panel, build_panel

EPS = 1e-12


def market_mask(panel: Panel, market: str) -> np.ndarray:
    """(ns,) boolean mask of the stocks included in the market aggregate."""
    b = panel.board
    if market == "all_a":
        return (b != "bse_a") & (b != "other")
    if market == "all_a_bse":
        return b != "other"
    if market == "sh_sz":
        return np.isin(b, ["main_a", "growth_a", "star_a"])
    if market == "main_a":
        return b == "main_a"
    if market == "main_growth":
        return np.isin(b, ["main_a", "growth_a"])
    if market == "main_star":
        return np.isin(b, ["main_a", "star_a"])
    raise ValueError(f"unknown market {market!r}")


def float_matrix(panel: Panel, float_field: str) -> np.ndarray:
    if float_field == "float_share":
        fcap = panel.float_shares
    elif float_field == "free_share":
        fcap = panel.free_shares
    else:
        raise ValueError(f"unknown float_field {float_field!r}")
    return np.where(np.isfinite(fcap), fcap, 0.0)


def turnover_rates(panel: Panel, float_field: str, turnover_source: str) -> np.ndarray:
    """(nd, ns) daily turnover rate as a fraction of float."""
    fcap = float_matrix(panel, float_field)
    if turnover_source == "vol":
        traded = panel.vol_shares
    elif turnover_source == "amount_vwap":
        traded = panel.amount_cny / np.where(panel.close > 0, panel.close, np.nan)
    else:
        raise ValueError(f"unknown turnover_source {turnover_source!r}")
    with np.errstate(invalid="ignore", divide="ignore"):
        q = traded / fcap
    q = np.nan_to_num(q, nan=0.0, posinf=0.0, neginf=0.0)
    # A stock that cannot trade more than its float: cap at 1.
    return np.clip(q, 0.0, 1.0)


def _survival(q: np.ndarray) -> np.ndarray:
    """cumsum of log(1-q) along the date axis."""
    log1m = np.log(np.maximum(1.0 - q, EPS))
    return np.cumsum(log1m, axis=0)


def active_fraction(q: np.ndarray, window: int, decay: float | None = None,
                    survival: str = "on") -> np.ndarray:
    """(nd, ns) active share fraction under the batch-replacement model.

    survival="on"  : batch created at age k keeps prod_{j}(1-q_j) of itself.
    survival="off" : pure geometric age decay, f = sum_k lambda^k q_{t-k}.
    """
    nd, ns = q.shape
    s = _survival(q)
    out = np.zeros((nd, ns))
    lam = 1.0 if decay is None else decay
    for k in range(min(window, nd)):
        if k == 0:
            out += q
            continue
        if survival == "on":
            surv = np.exp(s[k:] - s[:-k])
        elif survival == "off":
            surv = 1.0
        else:
            raise ValueError(f"unknown survival {survival!r}")
        out[k:] += q[:-k] * surv * (lam ** k)
    return out


def apply_float_mode(fcap: np.ndarray, mode: str) -> np.ndarray:
    """Choose which float-share vector weights the active chips.

    current : F_t      (shares outstanding today)
    lagged  : F_{t-1}  (shares outstanding yesterday; removes the jump that a
                         same-day float change would inject into the drift term)
    """
    if mode == "current":
        return fcap
    if mode == "lagged":
        return np.vstack([fcap[:1], fcap[:-1]])
    raise ValueError(f"unknown float_mode {mode!r}")


def model_components(panel: Panel, spec: dict) -> dict:
    """Return the level series plus its price / drift decomposition."""
    mask = market_mask(panel, spec.get("market", "all_a"))
    fcap_raw = float_matrix(panel, spec.get("float_field", "float_share"))
    fcap = apply_float_mode(fcap_raw, spec.get("float_mode", "current"))

    q = turnover_rates(panel, spec.get("float_field", "float_share"),
                        spec.get("turnover_source", "vol"))
    # No quote (suspended / not yet listed) means no turnover that day.
    q = np.where(np.isfinite(panel.close) & (panel.close > 0), q, 0.0)

    f = active_fraction(q, spec.get("window", 13), spec.get("decay"),
                        spec.get("survival", "on"))
    px = np.where(np.isfinite(panel.close), panel.close, 0.0)

    W = f * fcap
    Wm = np.where(mask[None, :], W, 0.0)
    Pm = np.where(mask[None, :], px, 0.0)

    prev_px = np.vstack([Pm[:1], Pm[:-1]])
    prev_W = np.vstack([Wm[:1], Wm[:-1]])
    val = Wm * Pm
    prev_val = prev_W * prev_px
    denom = prev_val.sum(axis=1)

    price_eff = (prev_W * (Pm - prev_px)).sum(axis=1)
    drift_eff = (prev_px * (Wm - prev_W)).sum(axis=1)

    levels = val.sum(axis=1)
    change = np.full(len(levels), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        change[1:] = (levels[1:] / levels[:-1] - 1.0) * 100.0
    return {
        "levels": levels,
        "change": change,
        "price_only": np.where(denom > 0, price_eff / np.where(denom > 0, denom, 1.0) * 100.0, np.nan),
        "drift_only": np.where(denom > 0, drift_eff / np.where(denom > 0, denom, 1.0) * 100.0, np.nan),
    }


def model_series(panel: Panel, spec: dict) -> np.ndarray:
    """Return the (nd,) 0AMV level series for one model spec."""
    return model_components(panel, spec)["levels"]


def daily_change_pct(levels: np.ndarray) -> np.ndarray:
    """Percent change of a level series; index 0 is NaN."""
    out = np.full(len(levels), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = (levels[1:] / levels[:-1] - 1.0) * 100.0
    return out


if __name__ == "__main__":
    p = build_panel()
    for N in (5, 10, 13, 20, 34, 60):
        lv = model_series(p, {"window": N})
        ch = daily_change_pct(lv)
        print(f"N={N:3d}  level[last]={lv[-1]/1e8:12.1f} 亿   mean chg std={np.nanstd(ch):.3f}%")
