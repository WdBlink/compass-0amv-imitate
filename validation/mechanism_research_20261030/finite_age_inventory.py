"""Exact single-stock finite-age hypothesis, not an official CYF/CYC algorithm."""

from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from fractions import Fraction
from typing import Sequence

Exact = int | str | Decimal | Fraction


@dataclass(frozen=True)
class StockDay:
    date: str
    symbol: str
    open: Exact
    high: Exact
    low: Exact
    close: Exact
    volume: Exact
    amount: Exact
    float_shares: Exact
    adj_factor: Exact


@dataclass(frozen=True)
class FiniteAgePoint:
    date: str
    period: int
    available: bool
    fraction: Fraction | None
    cost: Fraction | None
    initial_survival: Fraction


def _exact(value: Exact) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal, Fraction)):
        raise ValueError("Values must be exact integers, decimal strings, Decimal or Fraction")
    try:
        result = Fraction(value)
    except (ValueError, TypeError, OverflowError, ZeroDivisionError) as exc:
        raise ValueError("Values must be finite exact numbers") from exc
    return result


def _date(value: str) -> str:
    if not isinstance(value, str) or len(value) != 8 or not value.isascii() or not value.isdigit():
        raise ValueError("Dates must be YYYYMMDD strings")
    try:
        datetime.strptime(value, "%Y%m%d")
    except ValueError as exc:
        raise ValueError("Invalid calendar date") from exc
    return value


def calculate_finite_age(
    series: Sequence[StockDay], *, expected_dates: Sequence[str], periods: Sequence[int] = (5, 13, 34)
) -> tuple[FiniteAgePoint, ...]:
    """Calculate causal windows on a complete independently declared market calendar.

    Volume and float_shares are shares. Amount is CNY. Prices are unadjusted
    CNY per share. Missing dates and changing float shares or factors fail.
    Windows shorter than N are unavailable, not zero-initialized estimates.
    Zero-activity complete windows have fraction zero and unidentified cost.
    Initial survival is the remaining fraction of an unspecified initial
    inventory. It is diagnostic, never an initial price or an official reset.
    """
    dates = tuple(_date(value) for value in expected_dates)
    rows, windows = tuple(series), tuple(periods)
    if not dates or dates != tuple(sorted(set(dates))):
        raise ValueError("Expected dates must be nonempty, unique and chronological")
    if not windows or any(type(n) is not int or n < 1 for n in windows) or len(set(windows)) != len(windows):
        raise ValueError("Periods must be distinct positive integers")
    if len(rows) != len(dates) or any(not isinstance(row, StockDay) for row in rows):
        raise ValueError("Require one StockDay for every declared market date")
    if tuple(_date(row.date) for row in rows) != dates:
        raise ValueError("Rows must exactly match the complete chronological declared calendar")
    symbol = rows[0].symbol
    if not isinstance(symbol, str) or not symbol or symbol.strip() != symbol:
        raise ValueError("Require a nonempty unpadded single-stock symbol")
    turnovers, prices = [], []
    first_shares, first_factor = None, None
    for row in rows:
        if row.symbol != symbol:
            raise ValueError("A series must contain exactly one stock")
        open_, high, low, close, volume, amount, shares, factor = map(_exact, (
            row.open, row.high, row.low, row.close, row.volume, row.amount, row.float_shares, row.adj_factor
        ))
        if min(open_, high, low, close, shares, factor) <= 0 or volume < 0 or amount < 0:
            raise ValueError("Positive prices, shares and factor; nonnegative volume and amount required")
        if not low <= open_ <= high or not low <= close <= high:
            raise ValueError("OHLC outside price bounds")
        if volume == 0:
            if amount != 0:
                raise ValueError("Zero volume requires zero amount")
            price = None
        else:
            price = amount / volume
            if not low <= price <= high:
                raise ValueError("Amount/volume outside price bounds")
        if first_shares is None:
            first_shares, first_factor = shares, factor
        if shares != first_shares or factor != first_factor:
            raise ValueError("Unsupported float-share or adjustment-factor change")
        q = volume / shares
        if not 0 <= q < 1:
            raise ValueError("Turnover must satisfy 0 <= volume/float_shares < 1")
        turnovers.append(q)
        prices.append(price)
    result = []
    initial_survival = Fraction(1)
    for t, day in enumerate(dates):
        initial_survival *= 1 - turnovers[t]
        for n in windows:
            if t + 1 < n:
                result.append(FiniteAgePoint(day, n, False, None, None, initial_survival))
                continue
            survival, mass, moment = Fraction(1), Fraction(0), Fraction(0)
            for s in range(t, t - n, -1):
                weight = turnovers[s] * survival
                mass += weight
                if prices[s] is not None:
                    moment += prices[s] * weight
                survival *= 1 - turnovers[s]
            assert mass == 1 - survival
            result.append(FiniteAgePoint(day, n, True, mass, moment / mass if mass else None, initial_survival))
    return tuple(result)
