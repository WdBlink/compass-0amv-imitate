"""Falsifiable stock-volume mechanism probe, NOT the official 0AMV algorithm."""
from numbers import Integral

import numpy as np
import pandas as pd


def _dates(values) -> pd.DatetimeIndex:
    if not pd.api.types.is_datetime64_any_dtype(values.dtype):
        raise ValueError("Dates must have an explicit datetime64 dtype")
    dates = pd.DatetimeIndex(values)
    if dates.hasnans or dates.tz is not None or not dates.equals(dates.normalize()):
        raise ValueError("Dates must be nonmissing, timezone-naive calendar dates")
    return dates


def calculate_panel(
    panel: pd.DataFrame, *, expected_dates: pd.DatetimeIndex, expected_symbols: list[str], n: int = 10
) -> pd.DataFrame:
    """Return H1/H2 in CNY using first-observation SMA(N,1) seeds.

    Requires a complete, positive fixed-universe panel, declared independently
    of observations. No filling, corporate-action adjustment, labels or I/O.
    Every output is available after that day's close. No burn-in is discarded.
    """
    columns = ["date", "symbol", "close", "volume", "amount"]
    if panel.empty or panel.columns.has_duplicates or set(panel.columns) != set(columns):
        raise ValueError("Panel must contain exactly date,symbol,close,volume,amount")
    if isinstance(n, bool) or not isinstance(n, Integral) or n < 1:
        raise ValueError("n must be a positive integer")
    dates = _dates(expected_dates)
    if dates.empty or dates.has_duplicates or not dates.is_monotonic_increasing:
        raise ValueError("Expected dates must be nonempty, unique and increasing")
    symbols = list(expected_symbols)
    if not symbols or any(not isinstance(s, str) or not s or s.strip() != s for s in symbols):
        raise ValueError("Expected symbols must be nonempty, unpadded strings")
    if len(set(symbols)) != len(symbols):
        raise ValueError("Expected symbols must be unique")
    data = panel[columns].copy()
    data["date"] = _dates(data.date)
    if data.symbol.map(lambda s: not isinstance(s, str) or not s or s.strip() != s).any():
        raise ValueError("Panel symbols must be nonempty, unpadded strings")
    if data.duplicated(["date", "symbol"]).any():
        raise ValueError("Duplicate date/symbol observations")
    expected = pd.MultiIndex.from_product([dates, sorted(symbols)], names=["date", "symbol"])
    data = data.set_index(["date", "symbol"])
    if len(data) != len(expected) or not data.index.isin(expected).all():
        raise ValueError("Incomplete or unexpected panel: no missing dates, stocks or suspensions allowed")
    data = data.reindex(expected)
    for dtype in data.dtypes:
        if (not pd.api.types.is_numeric_dtype(dtype) or pd.api.types.is_bool_dtype(dtype)
                or pd.api.types.is_complex_dtype(dtype)):
            raise ValueError("Price, volume and amount must be real numeric values")
    values = data.to_numpy(dtype=float)
    if not np.isfinite(values).all() or (values <= 0).any():
        raise ValueError("Price, volume and amount must be positive and finite")
    prices, volumes, amounts = np.moveaxis(values.reshape(len(dates), len(symbols), 3), 2, 0)
    inventory = volumes[0].copy()
    result = np.empty((len(dates), 2))
    alpha = 1 / n
    with np.errstate(over="ignore", invalid="ignore"):
        total_amount = amounts.sum(axis=1)
        smoothed_amount = total_amount[0]
        for t in range(len(dates)):
            if t:
                inventory = (1 - alpha) * inventory + alpha * volumes[t]
                smoothed_amount = (1 - alpha) * smoothed_amount + alpha * total_amount[t]
            result[t] = ((inventory * prices[t]).sum(), smoothed_amount)
    if not np.isfinite(result).all() or (result <= 0).any():
        raise ValueError("Calculation overflow or underflow")
    return pd.DataFrame(result, index=dates.rename("date"), columns=["h1_cny", "h2_cny"])
