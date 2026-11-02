"""Synthetic compact-date regressions shared by the offline bounds."""
from datetime import date

import pytest

from validation.mechanism_research_20261030 import active_value_bound, acquisition_cost_bound, overlap_bound
from validation.mechanism_research_20261030 import market_snapshot


@pytest.mark.parametrize("calculate", [active_value_bound.calculate_bounds, acquisition_cost_bound.calculate_costs,
                                       overlap_bound.calculate_bounds])
def test_compact_leap_date_emits_iso_date_without_changing_denominator(calculate):
    panel = active_value_bound.Panel("20240229", [], [], [])
    rows = list(calculate([panel.date], [panel]))
    assert len(rows) == 6
    assert {row["date"] for row in rows} == {"2024-02-29"}
    assert all(row["eligible"] == row["seen_asof"] == row["excluded"] == 0 for row in rows)


@pytest.mark.parametrize("token", ["2024-02-29", "2024022", "202402290", "20230229", "20241301",
                                   "2024W091", " 20240229", "20240229\n", "２０２４０２２９", 20240229, None])
def test_compact_date_parser_rejects_noncanonical_or_invalid_tokens(token):
    with pytest.raises(ValueError):
        market_snapshot.parse_date(token)


def test_compact_date_parser_preserves_calendar_value():
    assert market_snapshot.parse_date("20240229") == date(2024, 2, 29)
