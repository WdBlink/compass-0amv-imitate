"""Active-chip market-value model family for the 0AMV replication.

Recovered structure (see program.md for the search that produced it):

    q_{i,s}   = traded_shares_{i,s} / float_shares_{i,s}     daily turnover rate
    f_{i,t}   = q_{i,t} + lambda * (1 - q_{i,t}) * f_{i,t-1}  active float fraction
    0AMV_t    = sum_i close_{i,t} * float_shares_{i,t} * f_{i,t}

Reading of the recursion: a stock's chips are a pool.  Every day a fresh
batch of size `q_t` of the float is created by the day's trading, and the
rest of the pool (`1 - q_t` of it) survives; on top of that the pool loses
a fraction `1 - lambda` of its weight to age.  `lambda = 0.924` means the
pool is worth ~92% of its previous day after one more day of no trading.

Two implementation paths are kept:

* `form="recursion"` (default) evaluates the recursion directly.  It is
  the exact infinite-horizon limit and costs O(nd) instead of O(nd * N).
  The finite-window batch sum below agrees with it to 2e-4 relative.
* `form="batch"` evaluates the explicit batch sum with a hard age cutoff,
  which is what the rejected `survival="off"` / `decay_shape="linear"`
  diagnostics need.

Initialisation matters: the official index has infinite history, the
frozen panel has 121 days.  `init="steady"` starts each stock's pool at
its quasi-stationary level (the fixed point of the recursion under the
stock's own mean turnover) instead of at zero, which removes the warmup
bias without needing more data.

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


def _steady_state(qbar: np.ndarray, lam: float, survival: str) -> np.ndarray:
    """Fixed point of the recursion under a constant turnover `qbar`.

    survival="on" : qbar + lam*(1-qbar)*f = f  ->  f = qbar / (1 - lam*(1-qbar))
    survival="off": qbar + lam*f = f            ->  f = qbar / (1 - lam)
    """
    if survival == "on":
        denom = 1.0 - lam * (1.0 - qbar)
    elif survival == "off":
        denom = 1.0 - lam
    else:
        raise ValueError(f"unknown survival {survival!r}")
    return qbar / np.maximum(denom, 1e-6)


def active_fraction(q: np.ndarray, window: int = 13, decay: float | None = None,
                    survival: str = "on", decay_shape: str = "exp",
                    init: str = "zero", init_window: int = 20,
                    form: str = "recursion",
                    float_shares: np.ndarray | None = None,
                    survival_power: float = 1.0,
                    decay_turnover_beta: float = 0.0) -> np.ndarray:
    """(nd, ns) active share fraction under the chip-pool model.

    form="recursion" (default, exact infinite horizon)

        f_t = q_t + lambda * (1 - q_t)^p * f_{t-1}      (survival="on", p=1)

    form="batch" (finite age cutoff, for the rejected diagnostics)

        f_t = sum_{k=0..N-1} w_k * q_{t-k} * prod_{j=t-k+1..t} (1 - q_j)

        w_k = lambda^k                    decay_shape="exp"
        w_k = (window - k) / window       decay_shape="linear"
        prod(...) replaced by 1           survival="off"

    init="zero"   : the pool starts empty at the first panel date.
    init="steady" : the pool starts at its quasi-stationary level, which
                    removes the warmup bias of a finite panel.

    form="shares" (rejected hypothesis, kept for reproducibility) tracks the
    chip pool as an absolute SHARE COUNT rather than a fraction of the
    float:

        A_t = V_t + lambda * (1 - q_t) * A_{t-1}          (V = traded shares)

    It coincides with the fraction form when the float never changes, but
    does not inflate when a float increase (解禁) lands.
    """
    nd, ns = q.shape
    lam = 1.0 if decay is None else decay

    if form == "shares":
        if init == "steady":
            raise ValueError("form='shares' only supports init='zero'")
        if float_shares is None:
            raise ValueError("form='shares' needs float_shares")
        out = np.zeros((nd, ns))
        prev = np.zeros(ns)
        for t in range(nd):
            prev = q[t] * float_shares[t] + lam * (1.0 - q[t]) * prev
            out[t] = prev
        return out

    if form == "recursion":
        if decay_shape != "exp":
            raise ValueError("form='recursion' only supports decay_shape='exp'")
        if init == "steady":
            k = min(init_window, nd)
            f0 = _steady_state(q[:k].mean(axis=0), lam, survival)
        elif init == "zero":
            f0 = np.zeros(ns)
        else:
            raise ValueError(f"unknown init {init!r}")
        # Optional: make the age decay depend on the stock's own turnover.
        # beta=0 (the default) is the constant-decay model.
        lam_vec = None
        if decay_turnover_beta:
            k = min(init_window, nd)
            qbar = q[:k].mean(axis=0)
            ref = qbar.mean()
            lam_vec = lam * (1.0 + decay_turnover_beta * (qbar / max(ref, EPS) - 1.0))
        out = np.zeros((nd, ns))
        prev = f0
        for t in range(nd):
            lm = lam if lam_vec is None else lam_vec
            if survival == "on":
                prev = q[t] + lm * (1.0 - q[t]) ** survival_power * prev
            elif survival == "off":
                prev = q[t] + lm * prev
            else:
                raise ValueError(f"unknown survival {survival!r}")
            out[t] = prev
        return out

    if form != "batch":
        raise ValueError(f"unknown form {form!r}")
    if init != "zero":
        raise ValueError("form='batch' only supports init='zero'")

    s = _survival(q)
    out = np.zeros((nd, ns))
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
        if decay_shape == "exp":
            w = lam ** k
        elif decay_shape == "linear":
            w = (window - k) / window
        else:
            raise ValueError(f"unknown decay_shape {decay_shape!r}")
        out[k:] += q[:-k] * surv * w
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


def price_series(panel: Panel, fill: str, price_mode: str = "close") -> np.ndarray:
    """(nd, ns) price used to value the active chips.

    fill="zero"  : a day with no quote contributes nothing (the stock drops
                   out of the aggregate).
    fill="ffill" : the last observed close is carried forward across a
                   suspension, so the stock keeps contributing its active
                   market value while it cannot trade.
    price_mode="close" : unadjusted traded close (a real market value).
    price_mode="adj"   : close * adj_factor anchored on the last date, i.e.
                   dividend/split adjusted.  Daily returns then differ from
                   the unadjusted ones only on ex-dates.
    """
    close = panel.close
    if price_mode == "adj":
        af = panel.adj_factor
        anchor = np.where(np.isfinite(af[-1]), af[-1], np.nan)
        with np.errstate(invalid="ignore", divide="ignore"):
            close = close * (af / anchor[None, :])
    elif price_mode != "close":
        raise ValueError(f"unknown price_mode {price_mode!r}")
    if fill == "zero":
        return np.where(np.isfinite(close), close, 0.0)
    if fill == "ffill":
        nd, ns = close.shape
        out = np.zeros((nd, ns))
        last = np.zeros(ns)
        seen = np.zeros(ns, bool)
        for t in range(nd):
            row = close[t]
            valid = np.isfinite(row)
            seen |= valid
            last = np.where(valid, row, last)
            out[t] = np.where(seen, last, 0.0)
        return out
    raise ValueError(f"unknown price fill {fill!r}")


def model_components(panel: Panel, spec: dict) -> dict:
    """Return the level series plus its price / drift decomposition."""
    mask = market_mask(panel, spec.get("market", "all_a"))
    fcap_raw = float_matrix(panel, spec.get("float_field", "float_share"))
    fcap = apply_float_mode(fcap_raw, spec.get("float_mode", "current"))

    q = turnover_rates(panel, spec.get("float_field", "float_share"),
                        spec.get("turnover_source", "vol"))
    # No quote (suspended / not yet listed) means no turnover that day.
    q = np.where(np.isfinite(panel.close) & (panel.close > 0), q, 0.0)

    form = spec.get("form", "recursion")
    f = active_fraction(q, spec.get("window", 13), spec.get("decay"),
                        spec.get("survival", "on"),
                        spec.get("decay_shape", "exp"),
                        spec.get("init", "zero"),
                        spec.get("init_window", 20),
                        form,
                        fcap_raw,
                        spec.get("survival_power", 1.0),
                        spec.get("decay_turnover_beta", 0.0))
    px = price_series(panel, spec.get("price_fill", "zero"), spec.get("price_mode", "close"))

    # form="shares" already yields a share count; the others yield a
    # fraction of the float.
    W = f if form == "shares" else f * fcap
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
    for lam in (0.918, 0.922, 0.924, 0.926):
        lv = model_series(p, {"decay": lam, "init": "steady"})
        ch = daily_change_pct(lv)
        print(f"lambda={lam:.3f}  level[last]={lv[-1]/1e8:12.1f} 亿   "
              f"mean chg std={np.nanstd(ch):.3f}%")
