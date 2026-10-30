"""Independent synthetic scorer fixtures, not an official-formula reference solution."""
import csv
import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from validation.verify_exact_parity import main, verify_exact_parity


DATES = ["2026-01-02", "2026-01-05", "2026-01-06", "2026-01-07"]


def write(path, header, rows):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(header.split(","))
        writer.writerows(rows)


@pytest.fixture
def inputs(tmp_path):
    paths = [tmp_path / name for name in ("official.csv", "predictions.csv", "calendar.csv")]
    write(paths[0], "date,official_close,official_change_pct",
          [(DATES[1], "105.00", "5.00"), (DATES[3], "107.2", "2.00")])
    write(paths[1], "date,computed_close", zip(DATES, ("100", "105", "105.1", "107.202")))
    write(paths[2], "date", [(day,) for day in DATES])
    return paths


def replace(path, old, new):
    path.write_text(path.read_text().replace(old, new))


def test_independent_sparse_path_and_hashes(inputs):
    report = verify_exact_parity(*inputs)
    assert report["strict_gate"] == report["observed_display_match"] == "pass"
    assert report["details"][1]["computed_change_pct"] == "2"
    assert report["details"][1]["previous_market_date"] == DATES[2]
    assert report["counts"]["numeric_pass"] == 2
    assert report["official_formula_replication"] == "not_established"
    assert "forbidden" in report["label_exposure"]["solver_use"]
    assert report["input_sha256"]["official"] == hashlib.sha256(inputs[0].read_bytes()).hexdigest()
    assert str(inputs[0].parent) not in json.dumps(report)
    assert verify_exact_parity(*inputs) == report


@pytest.mark.parametrize("wrong", ["-5.00", "0.05", "6.00"])
def test_sign_unit_and_internally_consistent_wrong_values(inputs, wrong):
    replace(inputs[0], ",5.00", "," + wrong)
    report = verify_exact_parity(*inputs)
    assert report["strict_gate"] == "fail"
    assert report["counts"]["return_fail"] >= 1


def test_uniformly_wrong_levels_even_when_returns_correct(inputs):
    write(inputs[1], "date,computed_close", zip(DATES, ("200", "210", "210.2", "214.404")))
    report = verify_exact_parity(*inputs)
    assert report["counts"]["level_fail"] == 2
    assert report["counts"]["return_pass"] == 2
    assert report["strict_gate"] == "fail"


def test_return_is_independent_of_sparse_official_levels(inputs):
    replace(inputs[1], ",105.1", ",100.0")
    report = verify_exact_parity(*inputs)
    assert report["counts"]["level_pass"] == 2
    assert report["details"][1]["computed_change_pct"] == "7.202"
    assert report["counts"]["return_fail"] == 1
    assert report["counts"]["signal_known_mismatches"] == 1
    assert report["exact_trigger_status"] == "fail"
    assert report["strict_gate"] == "fail"


@pytest.mark.parametrize("close,change,raw,status,field", [
    ("100.0", "0.05", "100.05", "ambiguous", "level_display_match"),
    ("100.0", "0.00", "100.005", "ambiguous", "return_display_match"),
    ("100.0", "-0.05", "99.95", "ambiguous", "level_display_match"),
    ("100.0", "0.00", "99.995", "ambiguous", "return_display_match"),
    ("100.0", "0.05", "100.04999999999999999999999999999999999999", "pass", "level_display_match"),
    ("100.0", "0.05", "100.05000000000000000000000000000000000001", "fail", "level_display_match"),
])
def test_exact_open_rounding_intervals(inputs, close, change, raw, status, field):
    write(inputs[0], "date,official_close,official_change_pct", [(DATES[1], close, change)])
    write(inputs[1], "date,computed_close", [(DATES[0], "100"), (DATES[1], raw)])
    report = verify_exact_parity(*inputs)
    assert report["details"][0][field] == report["strict_gate"] == status


@pytest.mark.parametrize("raw,close,change,display_match", [
    ("104", "104.0", "4.00", True), ("103.999", "104.0", "4.00", False),
    ("97.7", "97.7", "-2.30", True), ("97.701", "97.7", "-2.30", False),
])
def test_threshold_rounding_is_not_exact_trigger_proof(inputs, raw, close, change, display_match):
    write(inputs[0], "date,official_close,official_change_pct", [(DATES[1], close, change)])
    write(inputs[1], "date,computed_close", [(DATES[0], "100"), (DATES[1], raw)])
    report = verify_exact_parity(*inputs)
    assert report["observed_display_match"] == "pass"
    assert report["strict_gate"] == "ambiguous"
    assert report["exact_trigger_status"] == "not_established"
    assert report["counts"]["signal_known_mismatches"] == 0
    assert report["details"][0]["signal_display_match"] is display_match
    args = [flag for pair in zip(("--official", "--predictions", "--calendar"), map(str, inputs)) for flag in pair]
    assert main(args + ["--output", str(inputs[0].with_suffix(".report.json"))]) == 1


def test_opposite_signal_is_known_mismatch_even_if_threshold_label_ambiguous(inputs):
    write(inputs[0], "date,official_close,official_change_pct", [(DATES[1], "104", "4.00")])
    write(inputs[1], "date,computed_close", [(DATES[0], "100"), (DATES[1], "97")])
    report = verify_exact_parity(*inputs)
    assert report["counts"]["signal_rounding_ambiguous"] == 1
    assert report["counts"]["signal_known_mismatches"] == 1
    assert report["strict_gate"] == "fail"
    assert report["exact_trigger_status"] == "fail"


@pytest.mark.parametrize("index", [0, 1, 2])
@pytest.mark.parametrize("fault", ["duplicate", "unsorted", "invalid_date", "empty", "header_only", "extra_column"])
def test_invalid_dates_and_structure(inputs, index, fault):
    path = inputs[index]
    rows = list(csv.reader(path.open()))
    if fault == "duplicate":
        rows.append(rows[-1])
    elif fault == "unsorted":
        rows[1:] = rows[:0:-1]
    elif fault == "invalid_date":
        rows[1][0] = "2026-02-30"
    elif fault == "empty":
        rows[1][0] = ""
    elif fault == "header_only":
        rows = rows[:1]
    else:
        rows[0].append("claimed_return")
        for row in rows[1:]:
            row.append("5.00")
    write(path, ",".join(rows[0]), rows[1:])
    with pytest.raises(ValueError):
        verify_exact_parity(*inputs)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "", "0", "-1", "1.001"])
def test_official_levels_must_be_finite_positive_and_on_grid(inputs, bad):
    replace(inputs[0], "105.00", bad)
    with pytest.raises(ValueError):
        verify_exact_parity(*inputs)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "", "-100", "-101", "5.001"])
def test_official_returns_must_be_finite_above_minus_100_and_on_grid(inputs, bad):
    replace(inputs[0], ",5.00", "," + bad)
    with pytest.raises(ValueError):
        verify_exact_parity(*inputs)


@pytest.mark.parametrize("bad", ["NaN", "Infinity", "", "0", "-1"])
def test_predictions_require_finite_positive_levels(inputs, bad):
    replace(inputs[1], ",105\n", "," + bad + "\n")
    with pytest.raises(ValueError):
        verify_exact_parity(*inputs)


@pytest.mark.parametrize("fault", ["missing_predecessor", "missing_middle", "extra_date", "unknown_date"])
def test_prediction_calendar_coverage(inputs, fault):
    if fault == "missing_predecessor":
        replace(inputs[1], f"{DATES[0]},100\n", "")
    elif fault == "missing_middle":
        replace(inputs[1], f"{DATES[2]},105.1\n", "")
    elif fault == "extra_date":
        inputs[2].write_text(inputs[2].read_text() + "2026-01-08\n")
        inputs[1].write_text(inputs[1].read_text() + "2026-01-08,110\n")
    else:
        replace(inputs[1], DATES[2], "2026-01-04")
    with pytest.raises(ValueError):
        verify_exact_parity(*inputs)


def test_calendar_requires_first_predecessor_but_accepts_contiguous_warmup(inputs):
    replace(inputs[1], "date,computed_close\n", "date,computed_close\n2025-12-31,99\n")
    replace(inputs[2], "date\n", "date\n2025-12-31\n")
    assert verify_exact_parity(*inputs)["strict_gate"] == "pass"
    for path in inputs[1:]:
        rows = list(csv.reader(path.open()))
        write(path, ",".join(rows[0]), rows[3:])
    with pytest.raises(ValueError, match="preceding market day"):
        verify_exact_parity(*inputs)


def test_cli_from_unrelated_directory_and_exit_codes(inputs, tmp_path):
    script = Path(__file__).parent / "validation" / "verify_exact_parity.py"
    output = tmp_path / "report.json"
    args = [flag for pair in zip(("--official", "--predictions", "--calendar"), map(str, inputs)) for flag in pair]
    args += ["--output", str(output)]
    result = subprocess.run([sys.executable, str(script), *args], cwd=tmp_path, capture_output=True, text=True)
    assert result.returncode == 0, result.stderr
    assert json.loads(output.read_text())["strict_gate"] == "pass"
    replace(inputs[0], ",5.00", ",4.00")
    assert main(args) == 1
    replace(inputs[0], ",4.00", ",NaN")
    assert main(args) == 2
    assert json.loads(output.read_text())["input_status"] == "invalid"
    before = inputs[0].read_bytes()
    assert main(args[:-1] + [str(inputs[0])]) == 2
    assert inputs[0].read_bytes() == before
