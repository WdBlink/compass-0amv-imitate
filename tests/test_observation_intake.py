"""Synthetic acquisition fixtures only. No real official auxiliary readings."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from research.mechanism_research_20261030.observation_intake import plan_sha256, validate_intake


AS_OF = "2026-10-30T12:00:00+00:00"


@pytest.fixture
def synthetic(tmp_path):
    def reference(name, content):
        (tmp_path / name).write_bytes(content)
        return {"source_id": name, "path": name, "sha256": hashlib.sha256(content).hexdigest()}

    prediction = reference("synthetic-prediction.txt", b"SYNTHETIC predictions, not official results\n")
    basis = reference("synthetic-contract.txt", b"SYNTHETIC independently stipulated nearest-grid fixture\n")
    image = reference("synthetic-capture.txt", b"SYNTHETIC capture placeholder; not an actual screenshot\n")
    contract = {"status": "known", "quantum": "0.1", "origin": "0", "rounding": "nearest",
                "allows_cyc_zero": False, "basis": "independent_calibration", "evidence": basis,
                "established_at": "2026-10-30T08:00:00+00:00"}
    expected = [
        {"symbol": symbol, "date": "2026-10-30", "indicator": indicator, "period": period,
         "field_label": indicator, "unit": "percent" if indicator == "CYF" else "CNY/share",
         "display_contract": deepcopy(contract)}
        for symbol in ("SYNTHETIC-A", "SYNTHETIC-B", "SYNTHETIC-C")
        for period in (5, 13, 34) for indicator in ("CYF", "CYC")
    ]
    plan = {"planned_at": "2026-10-30T09:00:00+00:00", "prediction_frozen_at": "2026-10-30T08:30:00+00:00",
            "prediction": prediction, "purpose": "contract_calibration", "expected": expected}
    observations = [
        {**{field: case[field] for field in ("symbol", "date", "indicator", "period")}, "status": "observed",
         "value": "50.0" if case["indicator"] == "CYF" else "12.3",
         "display_text": "50.0" if case["indicator"] == "CYF" else "12.3", "source": deepcopy(image),
         "software": {"name": "SYNTHETIC fixture", "version": "fixture-1"},
         "field": {"label": case["field_label"], "period": case["period"], "unit": case["unit"]},
         "captured_at": "2026-10-30T10:00:00+00:00"} for case in expected
    ]
    packet = {"plan_sha256": plan_sha256(plan), "prediction_sha256": prediction["sha256"],
              "prediction_frozen_at": plan["prediction_frozen_at"], "purpose": plan["purpose"],
              "assembled_at": "2026-10-30T11:00:00+00:00", "observations": observations}
    return plan, packet, tmp_path


def check(fixture):
    return validate_intake(*fixture, as_of=AS_OF)


def rebind(plan, packet):
    packet["plan_sha256"] = plan_sha256(plan)


def unknown_contract(reason="Synthetic contract has not been established"):
    return {"status": "unknown", "quantum": None, "origin": None, "rounding": "unknown", "reason": reason}


def assert_invalid(result):
    assert result["status"] == "invalid"
    assert result["coverage"]["usable_fraction"] == "0/18"
    assert len(result["cases"]) == 18
    assert result["errors"]
    assert result["official_formula_replication"] == "not_established"


def test_valid_synthetic_packet_is_only_ready_for_independent_review(synthetic):
    before = deepcopy(synthetic[:2])
    result = check(synthetic)
    assert result["status"] == "ready_for_independent_review"
    assert result["coverage"]["usable_fraction"] == "18/18"
    assert result["coverage"]["observed"] == 18
    assert result["label_truth"] == "not_verified"
    assert result["official_formula_replication"] == "not_established"
    assert result["mechanism_parity"] == "not_evaluated"
    assert result["blindness_verified"] is False
    assert all(row["metadata_checked"] and row["evidence_checked"] for row in result["cases"])
    assert all(row["display_text"] == row["value"] for row in result["cases"])
    assert '"path"' not in json.dumps(result)
    assert before == synthetic[:2]


@pytest.mark.parametrize("size", [0, 262145])
def test_evidence_sha256_without_file_digest(synthetic, monkeypatch, size):
    monkeypatch.delattr(hashlib, "file_digest", raising=False)
    _, packet, root = synthetic
    source = packet["observations"][0]["source"]
    content = b"x" * size
    (root / source["path"]).write_bytes(content)
    for row in packet["observations"]:
        row["source"]["sha256"] = hashlib.sha256(content).hexdigest()
    assert check(synthetic)["status"] == "ready_for_independent_review"
    (root / source["path"]).write_bytes(content + b"changed")
    assert_invalid(check(synthetic))


def test_case_and_plan_order_are_immaterial(synthetic):
    plan, packet, _ = synthetic
    before = check(synthetic)
    plan["expected"].reverse()
    packet["observations"].reverse()
    assert check(synthetic) == before


def test_all_unavailable_and_unknown_metadata_keep_zero_of_full_denominator(synthetic):
    plan, packet, _ = synthetic
    for case in plan["expected"]:
        case.update(field_label=None, unit=None, display_contract=unknown_contract())
    packet["observations"] = [
        {**{key: row[key] for key in ("symbol", "date", "indicator", "period")},
         "status": "unavailable", "reason": "Synthetic environment cannot collect this field"}
        for row in packet["observations"]
    ]
    rebind(plan, packet)
    result = check(synthetic)
    assert result["status"] == "collection_incomplete"
    assert result["coverage"] == {"planned": 18, "observed": 0, "usable": 0, "usable_fraction": "0/18",
                                  "unavailable": 18, "ambiguous": 0, "missing": 0, "invalid": 0}
    assert all(row["reason"] and row["status"] == "unavailable" for row in result["cases"])


@pytest.mark.parametrize("status", ["unavailable", "ambiguous"])
def test_nonobserved_reason_required_and_no_value_substitution(synthetic, status):
    _, packet, _ = synthetic
    row = {key: packet["observations"][0][key] for key in ("symbol", "date", "indicator", "period")}
    packet["observations"][0] = {**row, "status": status, "reason": "Synthetic reason"}
    result = check(synthetic)
    assert result["status"] == "collection_incomplete"
    assert result["coverage"][status] == 1
    assert result["coverage"]["usable_fraction"] == "17/18"
    packet["observations"][0]["reason"] = ""
    assert_invalid(check(synthetic))
    packet["observations"][0].update(reason="Synthetic reason", value="0")
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("field,value", [("symbol", "WRONG"), ("date", "2026-10-29"),
                                         ("indicator", "CYC"), ("period", 6)])
def test_same_count_wrong_identity_is_not_coverage(synthetic, field, value):
    synthetic[1]["observations"][0][field] = value
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("mutation", ["missing", "duplicate", "unexpected", "empty"])
def test_missing_duplicate_and_unexpected_rows_are_invalid(synthetic, mutation):
    rows = synthetic[1]["observations"]
    if mutation == "missing":
        rows.pop()
    elif mutation == "duplicate":
        rows.append(deepcopy(rows[0]))
    elif mutation == "unexpected":
        rows.append({**deepcopy(rows[0]), "symbol": "UNPLANNED"})
    else:
        rows.clear()
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("bad", [1, 1.0, True, None, [], {}, float("nan"), float("inf"), "NaN", "Infinity",
                                  "", " 50", "50 ", "1/2", "1_0", "-1", "100.01",
                                  "1e999999999999999999999999999999"])
def test_exact_numeric_and_cyf_domain_failures(synthetic, bad):
    synthetic[1]["observations"][0].update(value=bad, display_text=bad)
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("field,value", [("label", "WRONG"), ("unit", "fraction"), ("period", 13),
                                         ("period", True), ("unit", None), ("label", "unknown")])
def test_complete_row_with_wrong_actual_metadata_is_invalid(synthetic, field, value):
    synthetic[1]["observations"][0]["field"][field] = value
    assert_invalid(check(synthetic))


def test_no_silent_display_rounding_or_unknown_software(synthetic):
    row = synthetic[1]["observations"][0]
    row["display_text"] = "50.00"
    assert_invalid(check(synthetic))
    row["display_text"] = row["value"]
    row["software"]["version"] = "unknown"
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("field,value", [("prediction_sha256", "f" * 64), ("prediction_sha256", ""),
                                         ("plan_sha256", "f" * 64),
                                         ("prediction_frozen_at", "2026-10-30T08:31:00+00:00"),
                                         ("purpose", "blind_validation")])
def test_packet_binding_failures(synthetic, field, value):
    synthetic[1][field] = value
    assert_invalid(check(synthetic))


@pytest.mark.parametrize("stamp", ["2026-10-30T10:00:00", "2026-10-30", "bad",
                                     "2026-10-30T08:00:00+00:00", "2026-10-30T13:00:00+00:00"])
def test_naive_early_and_future_capture_rejected(synthetic, stamp):
    synthetic[1]["observations"][0]["captured_at"] = stamp
    assert_invalid(check(synthetic))


def test_packet_time_and_caller_as_of_are_checked(synthetic):
    synthetic[1]["assembled_at"] = "2026-10-30T12:01:00+00:00"
    assert_invalid(check(synthetic))
    with pytest.raises(ValueError, match="aware"):
        validate_intake(*synthetic, as_of="2026-10-30T12:00:00")
    with pytest.raises(ValueError, match="aware"):
        validate_intake(*synthetic, as_of="")


@pytest.mark.parametrize("path", ["../escape", "/etc/passwd", "sub/../../escape", "./synthetic-capture.txt",
                                    "sub\\escape", "missing-file"])
def test_local_evidence_path_boundaries(synthetic, path):
    synthetic[1]["observations"][0]["source"]["path"] = path
    assert_invalid(check(synthetic))


def test_symlink_hash_drift_and_conflicting_source_identity(synthetic):
    _, packet, root = synthetic
    source = packet["observations"][0]["source"]
    original = deepcopy(source)
    (root / "link").symlink_to(root / source["path"])
    source["path"] = "link"
    assert_invalid(check(synthetic))
    source.update(original)
    source["sha256"] = "f" * 64
    assert_invalid(check(synthetic))
    source.update(original)
    source["source_id"] = "synthetic-prediction.txt"
    assert_invalid(check(synthetic))
    source.update(original)
    (root / source["path"]).write_bytes(b"SYNTHETIC changed capture")
    assert_invalid(check(synthetic))


def test_prediction_local_bytes_must_match_before_all_missing_can_be_valid(synthetic):
    plan, _, root = synthetic
    (root / plan["prediction"]["path"]).write_bytes(b"SYNTHETIC changed prediction")
    with pytest.raises(ValueError, match="hash"):
        check(synthetic)


@pytest.mark.parametrize("status", ["unknown", "ambiguous"])
def test_unknown_or_ambiguous_rounding_blocks_readiness(synthetic, status):
    plan, packet, _ = synthetic
    for case in plan["expected"]:
        case["display_contract"] = {**unknown_contract(), "status": status}
    rebind(plan, packet)
    result = check(synthetic)
    assert result["status"] == "needs_contract"
    assert result["coverage"]["observed"] == 18
    assert result["coverage"]["usable_fraction"] == "0/18"


def test_known_but_unsupported_contract_is_retained_not_scored(synthetic):
    plan, packet, _ = synthetic
    for case in plan["expected"]:
        case["display_contract"]["rounding"] = "floor"
    rebind(plan, packet)
    assert check(synthetic)["status"] == "needs_contract"
    for case in plan["expected"]:
        case["display_contract"].update(rounding="nearest", origin="0.1")
    rebind(plan, packet)
    assert check(synthetic)["status"] == "needs_contract"


@pytest.mark.parametrize("field", ["unit", "field_label"])
def test_unknown_planned_metadata_is_not_filled_from_observation(synthetic, field):
    plan, packet, _ = synthetic
    for case in plan["expected"]:
        case[field] = None
    rebind(plan, packet)
    result = check(synthetic)
    assert result["status"] == "needs_metadata"
    assert result["coverage"]["usable_fraction"] == "0/18"


def test_cyc_zero_needs_explicit_positive_rounding_contract(synthetic):
    plan, packet, _ = synthetic
    packet["observations"][1].update(value="0.0", display_text="0.0")
    assert_invalid(check(synthetic))
    plan["expected"][1]["display_contract"]["allows_cyc_zero"] = True
    rebind(plan, packet)
    assert check(synthetic)["status"] == "ready_for_independent_review"
    plan["expected"][1]["display_contract"] = unknown_contract()
    rebind(plan, packet)
    assert_invalid(check(synthetic))


def test_known_grid_is_exact_and_contract_cannot_be_derived_after_plan(synthetic):
    plan, packet, _ = synthetic
    packet["observations"][0].update(value="50.01", display_text="50.01")
    assert_invalid(check(synthetic))
    plan["expected"][0]["display_contract"]["established_at"] = "2026-10-30T10:00:00+00:00"
    rebind(plan, packet)
    with pytest.raises(ValueError, match="independently established"):
        check(synthetic)


@pytest.mark.parametrize("mutation", ["empty", "duplicate", "bool_period", "unknown_unit"])
def test_invalid_plan_has_no_trusted_denominator(synthetic, mutation):
    plan, _, _ = synthetic
    if mutation == "empty":
        plan["expected"].clear()
    elif mutation == "duplicate":
        plan["expected"].append(deepcopy(plan["expected"][0]))
    elif mutation == "bool_period":
        plan["expected"][0]["period"] = True
    else:
        plan["expected"][0]["unit"] = "unknown"
    with pytest.raises(ValueError):
        check(synthetic)


def test_cli_stdout_only_and_invalid_duplicate_json_keys(synthetic):
    plan, packet, root = synthetic
    plan_path, packet_path = root / "plan.json", root / "packet.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    command = [sys.executable, "-m", "research.mechanism_research_20261030.observation_intake",
               "--plan", str(plan_path), "--packet", str(packet_path), "--evidence-root", str(root), "--as-of", AS_OF]
    before = {path.name: path.read_bytes() for path in root.iterdir()}
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert result.returncode == 0 and not result.stderr
    assert json.loads(result.stdout)["status"] == "ready_for_independent_review"
    assert before == {path.name: path.read_bytes() for path in root.iterdir()}
    packet_path.write_text('{"purpose":"contract_calibration","purpose":"blind_validation"}', encoding="utf-8")
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert result.returncode == 2 and not result.stderr
    assert json.loads(result.stdout)["errors"] == ["JSON contains duplicate object keys"]


@pytest.mark.parametrize("depth", [1, 20, 100, 101, 1200])
@pytest.mark.parametrize("container", ["array", "object", "mixed"])
def test_json_nesting_limit_is_explicit_and_parser_independent(tmp_path, depth, container):
    from research.mechanism_research_20261030.observation_intake import _load_json

    raw = '"brackets [ { and escaped quote \\" are text"'
    for index in range(depth):
        raw = ('{"child":' + raw + '}') if container == "object" or (
            container == "mixed" and index % 2) else '[' + raw + ']'
    filename = tmp_path / "nested.json"
    filename.write_text(raw, encoding="utf-8")
    if depth > 100:
        with pytest.raises(ValueError, match="^JSON nesting is too deep$"):
            _load_json(filename)
    else:
        value = _load_json(filename)
        for _ in range(depth):
            value = value["child"] if isinstance(value, dict) else value[0]
        assert value == 'brackets [ { and escaped quote " are text'


def run_cli(plan_path, packet_path, root):
    command = [sys.executable, "-m", "research.mechanism_research_20261030.observation_intake",
               "--plan", str(plan_path), "--packet", str(packet_path), "--evidence-root", str(root), "--as-of", AS_OF]
    result = subprocess.run(command, cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True,
                            env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"}, check=False)
    assert result.returncode == 2 and not result.stderr
    report = json.loads(result.stdout)
    assert report["status"] == "invalid" and report["errors"]
    assert report["ready_for_independent_review"] is False
    assert report["blindness_verified"] is False
    assert report["label_truth"] == "not_verified"
    assert report["official_formula_replication"] == "not_established"
    assert report["mechanism_parity"] == "not_evaluated"
    assert str(root) not in result.stdout
    return report


@pytest.mark.parametrize("kind,cause", [
    ("loop", "evidence root cannot be resolved"),
    ("missing", "evidence root must be a local directory"),
    ("dangling", "evidence root must be a local directory"),
    ("file", "evidence root must be a local directory"),
])
def test_cli_invalid_evidence_root_has_no_trusted_denominator(synthetic, kind, cause):
    plan, packet, root = synthetic
    plan_path, packet_path = root / "plan.json", root / "packet.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    evidence_root = root / "evidence-root"
    if kind == "loop":
        evidence_root.symlink_to("loop-a")
        (root / "loop-a").symlink_to("loop-b")
        (root / "loop-b").symlink_to("loop-a")
    elif kind == "dangling":
        evidence_root.symlink_to("nonexistent-target")
    elif kind == "file":
        evidence_root.write_bytes(b"not a directory")
    report = run_cli(plan_path, packet_path, evidence_root)
    assert "coverage" not in report and "cases" not in report
    assert report["errors"] == [cause]
    assert str(root) not in json.dumps(report)


@pytest.mark.parametrize("failure,cause", [
    ("duplicate_keys", "JSON contains duplicate object keys"),
    ("truncated_json", "Expecting value"),
    ("deeply_nested_json", "JSON nesting is too deep"),
    ("missing_file", "FileNotFoundError"),
    ("unreadable_file", "IsADirectoryError"),
    ("oversize_source", "File name too long"),
])
def test_cli_packet_failures_preserve_valid_plan(synthetic, failure, cause):
    plan, packet, root = synthetic
    plan_path, packet_path = root / "plan.json", root / "packet.json"
    plan_path.write_text(json.dumps(plan), encoding="utf-8")
    if failure == "duplicate_keys":
        packet_path.write_text('{"purpose":"contract_calibration","purpose":"blind_validation"}', encoding="utf-8")
    elif failure == "truncated_json":
        packet_path.write_text('{"observations": [', encoding="utf-8")
    elif failure == "deeply_nested_json":
        packet_path.write_text('{"observations":' + "[" * 1200 + "0" + "]" * 1200 + "}", encoding="utf-8")
    elif failure == "unreadable_file":
        packet_path.mkdir()
    elif failure == "oversize_source":
        packet["observations"][0]["source"]["path"] = "x" * (os.pathconf(root, "PC_NAME_MAX") + 1)
        packet_path.write_text(json.dumps(packet), encoding="utf-8")
    report = run_cli(plan_path, packet_path, root)
    assert_invalid(report)
    assert report["coverage"]["planned"] == 18
    assert report["coverage"]["usable"] == 0
    identity = lambda row: tuple(row[field] for field in ("symbol", "date", "indicator", "period"))
    assert {identity(row) for row in report["cases"]} == {identity(row) for row in plan["expected"]}
    assert all(row["usable"] is False for row in report["cases"])
    assert any(cause in error for error in report["errors"])
    if failure == "deeply_nested_json":
        assert report["errors"] == [cause]
        assert all(row["reason"] == cause for row in report["cases"])
    if failure == "oversize_source":
        assert report["coverage"]["invalid"] == 1
        assert report["coverage"]["observed"] == 17
        failed = next(row for row in report["cases"] if identity(row) == identity(packet["observations"][0]))
        assert failed["status"] == "invalid" and cause in failed["reason"]
    else:
        assert report["coverage"]["invalid"] == 18
        assert report["coverage"]["observed"] == 0


@pytest.mark.parametrize("failure", ["duplicate_identity", "missing_file", "unreadable_file", "malformed_json",
                                     "deeply_nested_json"])
def test_cli_invalid_plan_has_no_trusted_denominator(synthetic, failure):
    plan, packet, root = synthetic
    plan_path, packet_path = root / "plan.json", root / "packet.json"
    packet_path.write_text(json.dumps(packet), encoding="utf-8")
    if failure == "duplicate_identity":
        plan["expected"].append(deepcopy(plan["expected"][0]))
        plan_path.write_text(json.dumps(plan), encoding="utf-8")
    elif failure == "unreadable_file":
        plan_path.mkdir()
    elif failure == "malformed_json":
        plan_path.write_text('{"expected": [', encoding="utf-8")
    elif failure == "deeply_nested_json":
        plan_path.write_text('{"expected":' + "[" * 1200 + "0" + "]" * 1200 + "}", encoding="utf-8")
    report = run_cli(plan_path, packet_path, root)
    assert "coverage" not in report and "cases" not in report
    if failure == "deeply_nested_json":
        assert report["errors"] == ["JSON nesting is too deep"]
