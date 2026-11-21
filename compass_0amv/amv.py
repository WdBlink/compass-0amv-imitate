"""指南针 0AMV (活跃市值 / 活筹指数) replication — deployable implementation.

This module computes the per-stock active-chip model using prices, volume,
float shares and optional delisting dates. It accepts market data directly.

The active-chip model
---------------------

For every stock, let

    q_{i,t} = turnover_rate_{i,t} / 100

be the fraction of the float that changed hands on day t. When daily
reported turnover is unavailable, use traded_shares / float_shares. The
stock's chips form a pool: every day a fresh batch of size `q_t` is
created by that day's trading, the rest of the pool (`1 - q_t` of it)
survives, and the pool additionally loses a fraction `1 - lambda` of its
weight to age:

    f_{i,t} = q_{i,t} + lambda * (1 - q_{i,t}) * f_{i,t-1}

`f` is the share of the float that is "active" (活筹).  The market's
active market value is

    0AMV_t = sum_i close_{i,t} * float_shares_{i,t} * f_{i,t}

with `lambda = 12 / 13` for Shanghai and Shenzhen A-shares. On days
without a quote, the stock retains its last active fraction and its last
active value. It contributes zero before its first quote and from its
known effective delisting date. Reproducing delisting exclusions requires
instrument metadata because a missing quote also occurs during suspensions.

Usage
-----

    from compass_0amv import from_tushare_frame
    dates, levels = from_tushare_frame(df)      # df: one row per stock-day

or from the command line:

    python3 -m compass_0amv --input panel.csv --out amv.csv

where `panel.csv` has the columns `trade_date,ts_code,close,vol,
float_share` in Tushare units (vol in hands, float_share in 万股).
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np

# Age decay per quoted trading day.
DECAY = 12 / 13

# Initial panel sessions used to estimate the steady-state pool.
INIT_WINDOW = 20

# Mean official / unscaled model level over the 64 reference days.
# For level comparisons only; constant scaling cannot change daily returns.
OFFICIAL_LEVEL_SCALE = 0.999666095071546

# Tushare unit conversions.
HAND = 100.0          # 1 手 = 100 股
WAN = 10_000.0        # 1 万股 = 10000 股

_SH_A = ("600", "601", "603", "605", "688", "689")
_SZ_A = ("000", "001", "002", "003", "300", "301", "302")
_BSE = ("43", "83", "87", "88", "92")


def a_share_mask(codes) -> np.ndarray:
    """Boolean mask selecting Shanghai and Shenzhen A-shares."""
    return market_mask_for("all_a", codes)


def active_chip_fraction(vol_shares: np.ndarray,
                         float_shares: np.ndarray,
                         close: np.ndarray,
                         *,
                         decay: float = DECAY,
                         init: str = "robust",
                         turnover_rate: np.ndarray | None = None,
                         init_window: int = INIT_WINDOW,
                         missing_quote: str = "freeze") -> np.ndarray:
    """Active (活筹) share of the free float, per (date, stock).

    Parameters
    ----------
    vol_shares, float_shares, close
        (nd, ns) arrays in shares and CNY/share. A finite, positive close
        marks a quoted day. Missing volume or float gives zero derived turnover.
    turnover_rate
        Optional (nd, ns) disclosed daily turnover in percent (2 means 2%).
        Missing, nonfinite or negative rates fall back to volume / float for
        that cell. The active turnover fraction is capped at 1.
    decay
        Age decay `lambda` of the chip pool.
    init
        "robust" estimates the initial pool from median turnover over the
        first `init_window` panel days, using only days observed so far.
        "steady" uses the full initial window mean, even on earlier days.
        "zero" starts the pool empty.
    missing_quote
        "freeze" holds the fraction on unquoted days. "drop" ages the
        pool with zero turnover on these days, as in the legacy model.

    Returns
    -------
    (nd, ns) float array of active fractions, each in [0, 1].
    """
    vol_shares = np.asarray(vol_shares, dtype=float)
    float_shares = np.asarray(float_shares, dtype=float)
    close = np.asarray(close, dtype=float)

    if close.ndim != 2 or vol_shares.shape != close.shape or float_shares.shape != close.shape:
        raise ValueError("close, vol_shares and float_shares must have matching (date, stock) shapes")
    if missing_quote not in ("freeze", "drop"):
        raise ValueError(f"unknown missing_quote {missing_quote!r}")

    has_quote = np.isfinite(close) & (close > 0)
    valid = has_quote & np.isfinite(vol_shares) & np.isfinite(float_shares) & (float_shares > 0)
    q = np.divide(vol_shares, float_shares, out=np.zeros_like(close), where=valid)
    if turnover_rate is not None:
        reported = np.asarray(turnover_rate, dtype=float)
        if reported.shape != close.shape:
            raise ValueError("turnover_rate must have the same (date, stock) shape as close")
        use_reported = has_quote & np.isfinite(reported) & (reported >= 0)
        np.divide(reported, 100, out=q, where=use_reported)
    np.clip(q, 0.0, 1.0, out=q)

    nd, ns = q.shape
    prev = np.zeros(ns)
    out = np.empty((nd, ns))
    if init in ("robust", "steady") and init_window < 1:
        raise ValueError("init_window must be positive")
    if init == "robust":
        # The recursion is affine in the seed: f = intercept + retained * seed.
        intercept, retained = np.zeros(ns), np.ones(ns)
    elif init == "steady":
        if nd:
            qbar = q[:init_window].mean(axis=0)
            prev = qbar / np.maximum(1.0 - decay * (1.0 - qbar), 1e-6)
    elif init != "zero":
        raise ValueError(f"unknown init {init!r}")

    for t in range(nd):
        keep = decay * (1 - q[t])
        if missing_quote == "freeze":
            keep = np.where(has_quote[t], keep, 1.0)
        if init == "robust" and t < init_window:
            median = np.median(q[:t + 1], axis=0)
            seed = median / np.maximum(1 - decay * (1 - median), 1e-6)
            intercept, retained = q[t] + keep * intercept, keep * retained
            prev = intercept + retained * seed
        else:
            prev = q[t] + keep * prev
        if init == "robust" and missing_quote == "freeze":
            # A revised warmup seed takes effect only at the next actual quote.
            out[t] = np.where(has_quote[t], prev, out[t - 1] if t else 0.0)
        else:
            out[t] = prev
    return out


def active_market_value(vol_shares: np.ndarray,
                         float_shares: np.ndarray,
                         close: np.ndarray,
                         *,
                         decay: float = DECAY,
                         init: str = "robust",
                         turnover_rate: np.ndarray | None = None,
                         init_window: int = INIT_WINDOW,
                         market_mask: np.ndarray | None = None,
                         missing_quote: str = "freeze",
                         dates=None, delist_dates=None) -> np.ndarray:
    """Daily aggregate in CNY. All supplied columns count unless `market_mask` selects a subset.

    By default a missing quote carries the stock's last entire active value.
    A quoted day updates the pool even if turnover is zero. A quoted day with
    missing float has zero value. With `missing_quote="drop"`, unquoted stocks
    contribute zero and their pools decay.

    Supply ascending `dates` and a `delist_dates` vector aligned with the stocks
    to exclude a stock on and after its effective delisting date. Missing end
    dates mean its listing continues for this calculation. A gap in quotes
    alone cannot establish a delisting.
    """
    close = np.asarray(close, dtype=float)
    float_shares = np.asarray(float_shares, dtype=float)
    f = active_chip_fraction(vol_shares, float_shares, close, decay=decay, turnover_rate=turnover_rate,
                             init=init, init_window=init_window, missing_quote=missing_quote)
    nd, ns = f.shape
    included = np.ones(ns, dtype=bool) if market_mask is None else np.asarray(market_mask, dtype=bool)
    if included.shape != (ns,):
        raise ValueError("market_mask must contain one flag per stock")
    ends = np.full(ns, nd)
    if delist_dates is not None:
        if dates is None or len(dates) != nd or len(delist_dates) != ns:
            raise ValueError("delist_dates requires one date per row and one end date per stock")
        days = [_date_key(d) for d in dates]
        if any(a >= b for a, b in zip(days, days[1:])):
            raise ValueError("dates must be strictly increasing trading dates")
        ends = np.searchsorted(np.array(days, dtype=str), [_date_key(d, nullable=True) or "99999999" for d in delist_dates])

    levels = np.empty(nd)
    last_value = np.zeros(ns)
    for t in range(nd):
        quoted = np.isfinite(close[t]) & (close[t] > 0)
        usable = quoted & np.isfinite(float_shares[t]) & (float_shares[t] > 0)
        value = np.multiply(f[t], float_shares[t], out=np.zeros(ns), where=usable)
        np.multiply(value, close[t], out=value, where=usable)
        last_value = np.where(quoted, value, last_value) if missing_quote == "freeze" else value
        levels[t] = np.where(included & (t < ends), last_value, 0.0).sum()
    return levels


def _date_key(value, *, nullable=False) -> str:
    """Normalize a trading date, or an optional delisting date when nullable."""
    text = str(value).strip()
    if text.lower() in ("", "none", "nan", "nat", "<na>"):
        if nullable:
            return ""
        raise ValueError("trade_date is required on every row")
    if isinstance(value, datetime):
        value = value.date()
    if isinstance(value, date):
        return value.strftime("%Y%m%d")
    if isinstance(value, np.datetime64):
        text = np.datetime_as_string(value, unit="D")
    # A nullable YYYYMMDD column often round-trips through CSV as 20260105.0.
    if len(text) == 10 and text[:8].isdigit() and text.endswith(".0"):
        text = text[:8]
    return date.fromisoformat(text).strftime("%Y%m%d")


def _delist_dates_for(codes, entries):
    by_code = {}
    for code, value in entries:
        end = _date_key(value, nullable=True)
        if end and by_code.setdefault(code, end) != end:
            raise ValueError(f"conflicting delist_date values for {code}")
    return [by_code.get(c) for c in codes] if by_code else None


def daily_change_pct(levels: Iterable[float]) -> np.ndarray:
    """Percent change of a level series; the first entry is NaN."""
    levels = np.asarray(levels, dtype=float)
    out = np.full(len(levels), np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[1:] = (levels[1:] / levels[:-1] - 1.0) * 100.0
    return out


def from_arrays(dates, codes, close, vol_shares, float_shares,
                *, market="all_a", market_mask=None, delist_dates=None, turnover_rate=None, **kwargs):
    """Build the index from dense (date x stock) arrays in shares and CNY/share.

    The default market contains Shanghai and Shenzhen A-shares. An explicit
    `market_mask` overrides it. Optional `delist_dates` are effective dates in
    `codes` order, with None for unknown dates. Reproducing the reference
    uses both these end dates and the (date x stock) matrix of disclosed
    `turnover_rate` percentages. Omit the latter to use volume / float.
    """
    dates, codes = list(dates), list(codes)
    if np.shape(close) != (len(dates), len(codes)):
        raise ValueError("close must have one row per date and one column per code")
    if market_mask is None:
        market_mask = market_mask_for(market, codes)
    levels = active_market_value(vol_shares, float_shares, close, market_mask=market_mask,
                                 dates=dates, delist_dates=delist_dates, turnover_rate=turnover_rate, **kwargs)
    return dates, levels


def from_tushare_frame(df, *, market="all_a", delist_dates=None, turnover_rate=None, **kwargs):
    """Build the index from a long-format DataFrame of Tushare fields.

    Required columns: `trade_date`, `ts_code`, `close`, `vol` (hands),
    `float_share` (万股). Optional `delist_date` gives an instrument's effective
    delisting date. It may be supplied on any row for the instrument. Blank
    values mean unknown. Optional `turnover_rate` is the disclosed rate in
    percent (2 means 2%); missing cells fall back to volume / float.
    `adj_factor` is ignored.

    Non-None `delist_dates` and `turnover_rate` keyword arguments override
    their corresponding columns. Align them with sorted codes (and rows
    of `turnover_rate` with sorted trade dates), as for `from_arrays`.

    `market` is "all_a" (Shanghai and Shenzhen A-shares) or "all_a_bse".

    Returns `(dates, levels)` with `dates` as YYYYMMDD strings and
    `levels` in CNY.
    """
    df = df.copy()
    date_keys = {d: _date_key(d) for d in df["trade_date"].unique()}
    df["trade_date"] = df["trade_date"].map(date_keys)
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
    fields = [("close", close, 1.0), ("vol", vol, HAND), ("float_share", fsh, WAN)]
    if turnover_rate is None and "turnover_rate" in df:
        turnover_rate = np.full((nd, ns), np.nan)
        fields.append(("turnover_rate", turnover_rate, 1.0))
    for col, matrix, unit in fields:
        values = df[col].replace("", np.nan).to_numpy(dtype=float, na_value=np.nan)
        matrix[di_idx, ci_idx] = values * unit

    if delist_dates is None and "delist_date" in df:
        entries = df[["ts_code", "delist_date"]].dropna().drop_duplicates()
        delist_dates = _delist_dates_for(codes, entries.itertuples(index=False, name=None))
    return from_arrays(dates, codes, close, vol, fsh, market=market, delist_dates=delist_dates,
                       turnover_rate=turnover_rate, **kwargs)


def market_mask_for(market: str, codes) -> np.ndarray:
    """Select A-share code prefixes on their corresponding exchanges."""
    if market not in ("all_a", "all_a_bse"):
        raise ValueError(f"unknown market {market!r}")
    selected = []
    for code in codes:
        num, _, exchange = str(code).partition(".")
        selected.append(len(num) == 6 and num.isdigit() and (
            (exchange in ("", "SH") and num.startswith(_SH_A))
            or (exchange in ("", "SZ") and num.startswith(_SZ_A))
            or (market == "all_a_bse" and exchange in ("", "BJ") and num.startswith(_BSE))
        ))
    return np.array(selected, dtype=bool)
