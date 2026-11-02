"""Finite-age mathematical checks. No official labels or empirical truth claims."""

from dataclasses import replace
from decimal import Decimal
from fractions import Fraction as F

import pytest

from validation.mechanism_research_20261030.finite_age_inventory import StockDay, calculate_finite_age


def rows(volumes=(10, 20, 0, 0), prices=(10, 20, 30, 40)):
    return tuple(StockDay(f"2026010{i + 1}", "A", p, p, p, p, v, v * p, 100, 1)
                 for i, (v, p) in enumerate(zip(volumes, prices)))


def run(data, periods=(2,)):
    return calculate_finite_age(data, expected_dates=[r.date for r in data], periods=periods)


def test_hand_calculated_mass_cost_and_unavailable_window():
    result = run(rows())
    assert not result[0].available and result[0].fraction is None and result[0].cost is None
    assert result[1].fraction == F(7, 25)
    assert result[1].cost == F(120, 7)
    assert result[1].initial_survival == F(18, 25)
    assert result[2].fraction == F(1, 5) and result[2].cost == 20
    assert result[3].fraction == 0 and result[3].cost is None


def test_pulse_survives_then_leaves_market_day_window():
    result = run(rows((25, 0, 0, 0)), periods=(3,))
    assert result[2].fraction == F(1, 4) and result[2].cost == 10
    assert result[3].fraction == 0 and result[3].cost is None
    assert result[3].initial_survival == F(3, 4)


def test_no_activity_and_n1_exact_truth():
    assert all(p.fraction == 0 and p.cost is None for p in run(rows((0, 0, 0, 0)), (1,)))
    result = run(rows(), (1,))
    assert result[0].fraction == F(1, 10) and result[0].cost == 10
    assert result[1].fraction == F(1, 5) and result[1].cost == 20


def test_shifted_turnover_negative_control_changes_age_and_cost():
    original = run(rows((25, 0, 0, 0)), (3,))
    shifted = run(rows((0, 25, 0, 0)), (3,))
    assert original[2].fraction == shifted[2].fraction
    assert original[2].cost == 10 and shifted[2].cost == 20
    assert original[3].fraction == 0 and shifted[3].fraction == F(1, 4)


def test_prefix_and_future_tail_poison_invariance_without_mutation():
    data = rows()
    saved = tuple(data)
    full = run(data, (1, 2, 3))
    prefix = run(data[:2], (1, 2, 3))
    poison = data[:2] + tuple(replace(r, open=100, high=100, low=100, close=100,
                                      volume=99, amount=9900) for r in data[2:])
    assert full[:6] == prefix == run(poison, (1, 2, 3))[:6]
    assert data == saved


@pytest.mark.parametrize("value", [True, 1.0, float("nan"), float("inf"), "NaN", "Infinity", None])
def test_nonexact_or_nonfinite_inputs_fail(value):
    with pytest.raises(ValueError):
        run((replace(rows()[0], close=value),))


@pytest.mark.parametrize("changes", [
    {"volume": -1}, {"volume": 100, "amount": 1000}, {"float_shares": 0}, {"adj_factor": 0},
    {"volume": 0, "amount": 1}, {"amount": 1}, {"low": 11}, {"high": 9}, {"close": 0},
    {"symbol": " B"}, {"date": "20260230"}, {"date": "2026-01-01"},
])
def test_illegal_values_fail_closed(changes):
    with pytest.raises(ValueError):
        run((replace(rows()[0], **changes),))


@pytest.mark.parametrize("changes", [{"float_shares": 101}, {"adj_factor": 2}, {"symbol": "B"}])
def test_unsupported_corporate_actions_or_mixed_symbols_fail(changes):
    data = rows()
    with pytest.raises(ValueError):
        run(data[:1] + (replace(data[1], **changes),) + data[2:])


@pytest.mark.parametrize("periods", [(), (0,), (-1,), (True,), (1.0,), (2, 2)])
def test_invalid_periods_fail(periods):
    with pytest.raises(ValueError):
        run(rows(), periods)


def test_missing_duplicate_reordered_and_extra_rows_fail():
    data = rows()
    expected = [r.date for r in data]
    for bad in (data[:-1], data[::-1], data[:1] + data, data[:1] + data[:1] + data[2:]):
        with pytest.raises(ValueError):
            calculate_finite_age(bad, expected_dates=expected)
    for dates in ((), expected[::-1], expected[:1] + expected):
        with pytest.raises(ValueError):
            calculate_finite_age(data, expected_dates=dates)


def test_decimal_context_does_not_affect_exact_result():
    from decimal import localcontext
    data = tuple(replace(r, volume=str(r.volume), float_shares=Decimal(100)) for r in rows())
    with localcontext() as context:
        context.prec = 2
        assert run(data) == run(rows())


def test_same_market_totals_do_not_identify_stock_level_inventory():
    from datetime import date, timedelta
    dates = [(date(2026, 1, 1) + timedelta(days=i)).strftime("%Y%m%d") for i in range(13)]

    def total(volumes):
        contributions = []
        for symbol, price, volume in zip(("A", "B", "C"), (1, 2, 3), volumes):
            series = [StockDay(day, symbol, price, price, price, price, volume, volume * price, 100, 1)
                      for day in dates]
            point = calculate_finite_age(series, expected_dates=dates, periods=(13,))[-1]
            contributions.append(100 * price * point.fraction)
        return sum(contributions), contributions

    uniform, parts = total((10, 10, 10))
    concentrated, other_parts = total((15, 0, 15))
    assert sum((10, 10, 10)) == sum((15, 0, 15)) == 30
    assert sum(p * v for p, v in zip((1, 2, 3), (10, 10, 10))) == 60
    assert sum(p * v for p, v in zip((1, 2, 3), (15, 0, 15))) == 60
    assert sum(100 * p for p in (1, 2, 3)) == 600
    assert uniform == 600 * (1 - F(9, 10) ** 13)
    assert concentrated == 400 * (1 - F(17, 20) ** 13)
    assert uniform != concentrated
    assert sum(parts) == sum(reversed(parts))
    assert sum(other_parts) == sum(reversed(other_parts))
