"""Synthetic engineering checks, never official 0AMV replication evidence."""
import numpy as np
import pandas as pd
import pytest

from validation.mechanism_research_20261030.stock_inventory import calculate_panel


@pytest.fixture
def panel():
    dates = pd.date_range("2020-01-06", periods=4)
    frame = pd.DataFrame({
        "date": np.repeat(dates, 2), "symbol": ["A", "B"] * 4,
        "close": [10., 20., 20., 10., 8., 25., 20., 16.],
        "volume": [100., 200., 300., 100., 80., 240., 160., 120.],
    })
    return frame.assign(amount=frame.close * frame.volume)


def run(panel, dates=None, **kwargs):
    if dates is None:
        dates = pd.date_range("2020-01-06", periods=4)
    return calculate_panel(panel, expected_dates=dates, expected_symbols=["A", "B"], **kwargs)


def test_known_synthetic_inventory_truth_exactly_recovered_but_amount_fails(panel):
    result = run(panel, n=2)
    # Hand-recursed share inventories: A=[100,200,140,150], B=[200,150,195,157.5].
    truth = [5000., 5500., 5995., 5520.]
    np.testing.assert_array_equal(result.h1_cny, truth)
    np.testing.assert_array_equal(result.h2_cny, [5000., 6000., 6320., 5720.])
    assert not np.array_equal(result.h2_cny, truth)
    smoothed_per_stock = sum(g.amount.ewm(alpha=.5, adjust=False).mean().to_numpy()
                             for _, g in panel.groupby("symbol"))
    np.testing.assert_array_equal(result.h2_cny, smoothed_per_stock)


def test_fixed_prices_commute_and_current_revaluation_is_distinct(panel):
    panel["close"] = [10., 20.] * 4
    panel["amount"] = panel.close * panel.volume
    result = run(panel, n=2)
    np.testing.assert_array_equal(result.h1_cny, result.h2_cny)
    changed = panel.copy()
    changed.loc[6, "close"] *= 2
    changed.loc[6, "amount"] *= 2
    delta = run(changed, n=2).iloc[-1] - result.iloc[-1]
    assert delta.h1_cny == 1500 and delta.h2_cny == 800


def test_order_prefix_and_poisoned_future_invariance(panel):
    result = run(panel, n=2)
    pd.testing.assert_frame_equal(result, run(panel.sample(frac=1, random_state=7), n=2))
    reversed_symbols = calculate_panel(panel, expected_dates=result.index, expected_symbols=["B", "A"], n=2)
    pd.testing.assert_frame_equal(result, reversed_symbols)
    prefix = run(panel.iloc[:4], dates=result.index[:2], n=2)
    changed = panel.copy()
    changed.loc[4:, ["close", "volume", "amount"]] *= 1e10
    pd.testing.assert_frame_equal(result.iloc[:2], prefix)
    pd.testing.assert_frame_equal(prefix, run(changed, n=2).iloc[:2])


def test_real_amount_is_not_replaced_by_close_times_volume(panel):
    expected = run(panel, n=2)
    panel["amount"] *= 1.125  # Legitimate non-close VWAP, not an input error.
    actual = run(panel, n=2)
    pd.testing.assert_series_equal(actual.h1_cny, expected.h1_cny)
    np.testing.assert_array_equal(actual.h2_cny, expected.h2_cny * 1.125)


def test_inception_n1_and_history_dependence(panel):
    result = run(panel, n=2)
    assert result.iloc[0].tolist() == [5000., 5000.]
    np.testing.assert_array_equal(run(panel, n=1).h1_cny, panel.groupby("date").amount.sum())
    np.testing.assert_array_equal(run(panel, n=1).h2_cny, panel.groupby("date").amount.sum())
    restarted = run(panel.iloc[2:], dates=result.index[1:], n=2)
    assert restarted.iloc[0].tolist() == [7000., 7000.]
    assert restarted.iloc[-1].tolist() != result.iloc[-1].tolist()


@pytest.mark.parametrize("column", ["close", "volume", "amount"])
@pytest.mark.parametrize("bad", [0., -1., np.nan, np.inf, -np.inf])
def test_bad_numeric_inputs_rejected(panel, column, bad):
    panel.loc[0, column] = bad
    with pytest.raises(ValueError, match="positive and finite"):
        run(panel)


@pytest.mark.parametrize("bad", [0, -1, 1.5, True])
def test_bad_n_rejected(panel, bad):
    with pytest.raises(ValueError, match="positive integer"):
        run(panel, n=bad)


def test_duplicate_missing_date_missing_stock_and_new_stock_rejected(panel):
    for bad in (pd.concat([panel, panel.iloc[:1]]), panel.iloc[2:], panel.iloc[1:],
                panel.assign(symbol=panel.symbol.replace("B", "C"))):
        with pytest.raises(ValueError, match="Duplicate|Incomplete"):
            run(bad)


def test_calendar_schema_overflow_and_no_mutation(panel):
    original = panel.copy(deep=True)
    run(panel)
    pd.testing.assert_frame_equal(panel, original)
    for bad in (panel.assign(official=1.), panel.iloc[:0], panel.assign(date="2020-01-06"),
                panel.assign(close=1e308, volume=1e308), panel.assign(close=True)):
        with pytest.raises(ValueError):
            run(bad)
    for dates in (pd.date_range("2020-01-06", periods=4, tz="UTC"),
                  pd.date_range("2020-01-06", periods=4)[::-1]):
        with pytest.raises(ValueError):
            run(panel, dates=dates)
