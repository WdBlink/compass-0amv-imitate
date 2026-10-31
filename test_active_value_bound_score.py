import hashlib

import pytest

from validation.mechanism_research_20261030.score_active_value_bound import score


ROWS = [
    # Exactly at, below and above the upper display interval of 100.0.
    ("SH", "5", "0", "0"), ("SH_SZ", "5", "2", "100.05"),
    ("SH", "13", "1", "100.00"), ("SH_SZ", "13", "2", "100.050000000001"),
    ("SH", "34", "1", "100.06"), ("SH_SZ", "34", "3", "101.0"),
]


def run(tmp_path, rows, digest=None, days="2020-01-01\n2020-01-02\n"):
    labels = tmp_path / "official.csv"
    labels.write_text("date,official_close,official_change_pct\n2020-01-02,100.0,0.00\n")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("date\n" + days)
    bounds = tmp_path / "bounds.csv"
    bounds.write_text("date,market,window,eligible,lower_yi\n" + "".join(
        "2020-01-02," + ",".join(row) + "\n" for row in rows))
    return score(labels, calendar, bounds, digest or hashlib.sha256(bounds.read_bytes()).hexdigest())


def test_strict_half_unit_and_unknowns(tmp_path):
    result = run(tmp_path, ROWS[::-1])
    statuses = {(r["market"], r["window"]): r["excluded_days"] for r in result["summary"]}
    assert statuses == {("SH", 5): 0, ("SH_SZ", 5): 0, ("SH", 13): 0,
                        ("SH_SZ", 13): 1, ("SH", 34): 1, ("SH_SZ", 34): 1}
    assert result["official_formula_replication"] == "not_established"
    assert result["change_and_trigger"] == "not_scored"
    assert all(r["undecided_days"] + r["excluded_days"] == 1 for r in result["summary"])


@pytest.mark.parametrize("bad", [ROWS[:-1], ROWS + ROWS[:1],
                                 [("SH", "5", "0", "100.1"), *ROWS[1:]],
                                 [("SH", "5", "1", "101"), *ROWS[1:]],
                                 [("SH", "5", "1", "NaN"), *ROWS[1:]]])
def test_invalid_bound_is_not_a_scientific_result(tmp_path, bad):
    with pytest.raises(ValueError):
        run(tmp_path, bad)


def test_reject_changed_freeze_or_missing_session(tmp_path):
    with pytest.raises(ValueError, match="freeze"):
        run(tmp_path, ROWS, digest="0" * 64)
    with pytest.raises(ValueError, match="every scored date"):
        run(tmp_path, ROWS, days="2020-01-01\n2020-01-02\n2020-01-03\n")
