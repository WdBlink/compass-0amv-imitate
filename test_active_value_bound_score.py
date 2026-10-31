import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from validation.mechanism_research_20261030.score_active_value_bound import score
from validation.mechanism_research_20261030.active_value_bound import COLUMNS, EXCLUSIONS


ROWS = [
    # Exactly at, below and above the upper display interval of 100.0.
    ("SH", "5", "2", "0"), ("SH_SZ", "5", "5", "100.05"),
    ("SH", "13", "1", "100.00"), ("SH_SZ", "13", "4", "100.050000000001"),
    ("SH", "34", "1", "100.06"), ("SH_SZ", "34", "3", "101.0"),
]


def grid(rows, day="2020-01-02"):
    for market, window, eligible, lower in rows:
        seen = 4 if market == "SH" else 8
        excluded = seen - int(eligible)
        yield dict(date=day, market=market, window=window, eligible=eligible, lower_yi=lower,
                   seen_asof=seen, excluded=excluded, **{"ex_"+r: excluded if r == "missing" else 0 for r in EXCLUSIONS})


def write_grid(path, records, columns=COLUMNS):
    with path.open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=columns, lineterminator="\n", extrasaction="ignore")
        writer.writeheader()
        writer.writerows(records)


def inputs(tmp_path, rows=ROWS, days="2020-01-01\n2020-01-02\n"):
    labels = tmp_path / "official.csv"
    labels.write_text("date,official_close,official_change_pct\n2020-01-02,100.0,0.00\n")
    calendar = tmp_path / "calendar.csv"
    calendar.write_text("date\n" + days)
    bounds = tmp_path / "bounds.csv"
    write_grid(bounds, grid(rows))
    return labels, calendar, bounds


def run(tmp_path, rows, digest=None, days="2020-01-01\n2020-01-02\n"):
    labels, calendar, bounds = inputs(tmp_path, rows, days)
    return score(labels, calendar, bounds, digest or hashlib.sha256(bounds.read_bytes()).hexdigest(), generic=True)


def cli(paths, out, *flags):
    labels, calendar, bounds = paths
    return subprocess.run([sys.executable, "-B", "-m", "validation.mechanism_research_20261030.score_active_value_bound",
                           "--official", str(labels), "--calendar", str(calendar), "--bounds", str(bounds),
                           "--out", str(out), *flags], capture_output=True, text=True, cwd=Path(__file__).resolve().parent)


def test_strict_half_unit_and_unknowns(tmp_path):
    result = run(tmp_path, ROWS[::-1])
    statuses = {(r["market"], r["window"]): r["excluded_days"] for r in result["summary"]}
    assert statuses == {("SH", 5): 0, ("SH_SZ", 5): 0, ("SH", 13): 0,
                        ("SH_SZ", 13): 1, ("SH", 34): 1, ("SH_SZ", 34): 1}
    assert result["official_formula_replication"] == "not_established"
    assert result["assessment"] == "generic_diagnostic_only_not_campaign_evidence"
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


@pytest.mark.parametrize(("mutations", "message"), [
    ([(5, {"eligible": 999999999})], "population"),
    ([(0, {"ex_missing": 1})], "reason totals"),
    ([(0, {"ex_missing": "-1"})], "coverage count"),
    ([(0, {"excluded": "1.0"})], "coverage count"),
    ([(0, {"seen_asof": ""})], "coverage count"),
    ([(4, {"seen_asof": 5, "excluded": 4, "ex_missing": 4})], "seen_asof"),
    ([(2, {"eligible": 3, "excluded": 1, "ex_missing": 1})], "nested"),
    ([(3, {"eligible": 5, "excluded": 3, "ex_missing": 3})], "nested"),
    ([(i, {"ex_action": 1, "ex_missing": 1 if i == 0 else 2}) for i in (0, 2, 4)], "SZ"),
    ([(5, {"eligible": 1, "excluded": 7, "ex_missing": 7})], "SZ"),
])
def test_population_relationships_survive_a_caller_rehash(tmp_path, mutations, message):
    labels, calendar, bounds = inputs(tmp_path)
    cells = list(grid(ROWS))
    for i, edits in mutations:
        cells[i].update(edits)
    write_grid(bounds, cells)
    with pytest.raises(ValueError, match=message):
        score(labels, calendar, bounds, hashlib.sha256(bounds.read_bytes()).hexdigest(), generic=True)


def test_seen_population_cannot_shrink_across_reported_days(tmp_path):
    labels, calendar, bounds = inputs(tmp_path, days="2020-01-01\n2020-01-02\n2020-01-03\n")
    labels.write_text(labels.read_text() + "2020-01-03,100.0,0.00\n")
    later = list(grid(ROWS, "2020-01-03"))
    for c in later:
        for key in ("seen_asof", "excluded", "ex_missing"):
            c[key] -= 1
    write_grid(bounds, [*grid(ROWS), *later])
    with pytest.raises(ValueError, match="not decrease"):
        score(labels, calendar, bounds, hashlib.sha256(bounds.read_bytes()).hexdigest(), generic=True)


def test_full_coverage_required_but_bounds_need_not_increase_with_lookback(tmp_path):
    labels, calendar, bounds = inputs(tmp_path)
    cells = list(grid(ROWS))
    cells[0]["lower_yi"], cells[1]["lower_yi"] = 180, 190
    write_grid(bounds, cells)
    assert len(score(labels, calendar, bounds, hashlib.sha256(bounds.read_bytes()).hexdigest(), generic=True)["details"]) == 6
    write_grid(bounds, cells, columns=COLUMNS[:-1])
    with pytest.raises(ValueError, match="complete native"):
        score(labels, calendar, bounds, hashlib.sha256(bounds.read_bytes()).hexdigest(), generic=True)


@pytest.mark.parametrize(("which", "message"), [("one-day", "official labels"), ("labels", "official labels"),
                         ("sunday", "calendar"), ("rehashed-bounds", "campaign prediction freeze"),
                         ("fake-receipt", "original pre-score freeze receipt")])
def test_campaign_rejects_substitution_without_writing_a_report(tmp_path, which, message):
    official = Path(__file__).resolve().parent / "validation/exact_parity_20261030"
    labels, calendar = official/"official.csv", official/"calendar.csv"
    bounds, out = tmp_path/"bounds.csv", tmp_path/"result.json"
    days = calendar.read_text().splitlines()[2:]
    options = []
    if which == "one-day":
        labels, calendar, bounds = inputs(tmp_path)
        days = ["2020-01-02"]
    elif which == "labels":
        labels = tmp_path/"fake.csv"
        labels.write_text("date,official_close,official_change_pct\n" + "".join(f"{d},900000.0,0.00\n" for d in days))
    elif which == "sunday":
        original = calendar.read_text()
        calendar = tmp_path/"fake.csv"
        calendar.write_text(original.replace("2026-07-24", "2026-07-26"))
    elif which == "fake-receipt":
        receipt = tmp_path/"receipt.json"
        receipt.write_text("{}")
        options = ["--freeze-receipt", str(receipt)]
    write_grid(bounds, (c for day in days for c in grid(ROWS, day)))
    p = cli((labels, calendar, bounds), out, "--frozen-sha256", hashlib.sha256(bounds.read_bytes()).hexdigest(), *options)
    assert p.returncode != 0 and message in p.stderr and not out.exists()


def test_cli_diagnostic_is_explicit_and_preserves_existing_output(tmp_path):
    paths = inputs(tmp_path)
    pin = hashlib.sha256(paths[2].read_bytes()).hexdigest()
    out = tmp_path / "result.json"
    assert cli(paths, out, "--generic").returncode != 0 and not out.exists()
    p = cli(paths, out, "--generic", "--frozen-sha256", pin)
    assert p.returncode == 0, p.stderr
    saved = out.read_bytes()
    assert json.loads(p.stdout)["assessment"] == json.loads(saved)["assessment"] == "generic_diagnostic_only_not_campaign_evidence"
    assert cli(paths, out, "--generic", "--frozen-sha256", pin).returncode != 0
    assert out.read_bytes() == saved
