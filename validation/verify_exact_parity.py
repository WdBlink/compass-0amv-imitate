"""Score finite disclosed labels, not recovery of the official algorithm.

CSV schemas are date/official_close/official_change_pct, date/computed_close,
and date for the full trading calendar. Dates must be unique sorted ISO dates.
Extra columns are rejected. Earlier prediction warmup is allowed, but every
calendar date from its start through the final observation must be present.

Official labels are scoring-only, never allowed solver inputs. Copying or
fitting those labels is forbidden; this gate alone cannot detect such leakage.
Matching disclosed labels does not establish official formula replication.
Comparisons use exact rational arithmetic after Decimal parsing. JSON numeric
diagnostics are strings with up to 50 significant digits, not scoring inputs.
"""
from __future__ import annotations

import argparse
import csv
from datetime import date
from decimal import Decimal, InvalidOperation, localcontext
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re
import sys


LEVEL_HALF = Fraction(1, 20)
RETURN_HALF = Fraction(1, 200)
LABEL_EXPOSURE = {
    "role": "disclosed_scoring_only",
    "solver_use": "forbidden_including_copying_or_fitting_official_labels",
    "leakage_detection": "not_established_by_this_gate",
}


def _read(path: Path, role: str, columns: set[str]) -> tuple[list[dict], str]:
    try:
        data = Path(path).read_bytes()
    except OSError as exc:
        raise ValueError(f"{role}: cannot read input ({exc.strerror})") from exc
    rows = list(csv.reader(io.StringIO(data.decode("utf-8-sig")), strict=True))
    if not rows or len(rows[0]) != len(columns) or set(rows[0]) != columns:
        raise ValueError(f"{role}: expected exactly columns {sorted(columns)}")
    if len(rows) == 1:
        raise ValueError(f"{role}: no observations")
    result, previous = [], ""
    for number, cells in enumerate(rows[1:], 2):
        if len(cells) != len(columns) or any(not cell or cell != cell.strip() for cell in cells):
            raise ValueError(f"{role}: empty, missing, or malformed row {number}")
        row = dict(zip(rows[0], cells))
        day = row["date"]
        if not re.fullmatch(r"[0-9]{4}-[0-9]{2}-[0-9]{2}", day):
            raise ValueError(f"{role}: invalid ISO date at row {number}")
        date.fromisoformat(day)
        if day <= previous:
            raise ValueError(f"{role}: duplicate or unsorted dates at row {number}")
        previous = day
        for field in columns - {"date"}:
            try:
                value = Decimal(row[field])
            except InvalidOperation as exc:
                raise ValueError(f"{role}: invalid {field} at {day}") from exc
            if not value.is_finite() or value <= (-100 if field.endswith("pct") else 0):
                raise ValueError(f"{role}: invalid {field} at {day}")
            row[field] = Fraction(value)
            grid = Fraction(1, 100) if field.endswith("pct") else Fraction(1, 10)
            if role == "official" and (row[field] / grid).denominator != 1:
                raise ValueError(f"official: {field} is not on the display grid at {day}")
        result.append(row)
    return result, hashlib.sha256(data).hexdigest()


def _status(error: Fraction, half: Fraction) -> str:
    return "pass" if abs(error) < half else "ambiguous" if abs(error) == half else "fail"


def _combine(statuses) -> str:
    values = set(statuses)
    return "fail" if "fail" in values else "ambiguous" if "ambiguous" in values else "pass"


def _signal(value: Fraction) -> str:
    return "up_trigger" if value >= 4 else "down_trigger" if value <= Fraction(-23, 10) else "neither"


def _number(value: Fraction) -> str:
    with localcontext() as context:
        context.prec = 50
        return str(Decimal(value.numerator) / Decimal(value.denominator))


def verify_exact_parity(official: Path, predictions: Path, calendar: Path) -> dict:
    """Validate all inputs and return display-only evidence; invalid input raises ValueError."""
    labels, official_hash = _read(official, "official", {"date", "official_close", "official_change_pct"})
    levels, prediction_hash = _read(predictions, "predictions", {"date", "computed_close"})
    dates, calendar_hash = _read(calendar, "calendar", {"date"})
    days = [row["date"] for row in dates]
    positions = {day: index for index, day in enumerate(days)}
    if any(row["date"] not in positions for row in labels + levels):
        raise ValueError("official or prediction dates are missing from the calendar")
    first, last = positions[labels[0]["date"]], positions[labels[-1]["date"]]
    start = positions[levels[0]["date"]]
    if first == 0 or start >= first:
        raise ValueError("predictions and calendar require the first preceding market day")
    if [row["date"] for row in levels] != days[start:last + 1]:
        raise ValueError("predictions must cover the contiguous calendar slice without missing or extra dates")
    by_day = {row["date"]: row["computed_close"] for row in levels}
    details = []
    for label in labels:
        day = label["date"]
        previous = days[positions[day] - 1]
        computed = by_day[day]
        change = (computed / by_day[previous] - 1) * 100
        observed = label["official_change_pct"]
        close_error, return_error = computed - label["official_close"], change - observed
        level_status, return_status = _status(close_error, LEVEL_HALF), _status(return_error, RETURN_HALF)
        possible_signals = {_signal(observed - RETURN_HALF), _signal(observed + RETURN_HALF)}
        ambiguous = len(possible_signals) > 1
        signal_match = _signal(change) == _signal(observed)
        details.append({
            "date": day, "previous_market_date": previous,
            "official_close": _number(label["official_close"]), "computed_close": _number(computed),
            "official_change_pct": _number(observed), "computed_change_pct": _number(change),
            "close_error": _number(close_error), "change_error_pp": _number(return_error),
            "level_display_match": level_status, "return_display_match": return_status,
            "observed_display_match": _combine((level_status, return_status)),
            "official_display_signal": _signal(observed), "computed_raw_signal": _signal(change),
            "official_possible_raw_signals": sorted(possible_signals),
            "signal_display_match": signal_match, "signal_rounding_ambiguous": ambiguous,
            "known_signal_mismatch": _signal(change) not in possible_signals,
        })
    numeric_status = _combine(row["observed_display_match"] for row in details)
    ambiguous_count = sum(row["signal_rounding_ambiguous"] for row in details)
    known_mismatches = sum(row["known_signal_mismatch"] for row in details)
    counts = {"observations": len(details), "calendar_rows": len(days), "prediction_rows": len(levels),
              "signal_display_matches": sum(row["signal_display_match"] for row in details),
              "signal_rounding_ambiguous": ambiguous_count, "signal_known_mismatches": known_mismatches}
    for name, field in (("numeric", "observed_display_match"), ("level", "level_display_match"),
                        ("return", "return_display_match")):
        counts.update({f"{name}_{status}": sum(row[field] == status for row in details)
                       for status in ("pass", "fail", "ambiguous")})
    return {
        "schema_version": 1, "input_status": "valid",
        "observed_display_match": numeric_status,
        "strict_gate": _combine((numeric_status, "fail" if known_mismatches else
                                  "ambiguous" if ambiguous_count else "pass")),
        "signal_display_agreement": "pass" if all(row["signal_display_match"] for row in details) else "fail",
        "exact_trigger_status": "fail" if known_mismatches else "not_established" if ambiguous_count else "pass",
        "official_formula_replication": "not_established",
        "label_exposure": dict(LABEL_EXPOSURE),
        "scope": "Only supplied daily observations; stateful regimes and undisclosed dates are not assessed.",
        "rules": {"level_absolute_error": "<0.05", "return_absolute_error_pp": "<0.005",
                  "half_unit_ties": "ambiguous_official_rounding_policy_unknown",
                  "up_trigger": ">=4.0", "down_trigger": "<=-2.3",
                  "returns": "(computed_close / previous_market_computed_close - 1) * 100",
                  "comparison_arithmetic": "exact_rational", "diagnostic_significant_digits": 50},
        "input_sha256": {"official": official_hash, "predictions": prediction_hash, "calendar": calendar_hash},
        "verifier_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "counts": counts, "details": details,
    }


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "predictions", "calendar", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args(argv)
    sources = (args.official, args.predictions, args.calendar)
    if any(args.output.resolve() == path.resolve() or
           (args.output.exists() and path.exists() and args.output.samefile(path)) for path in sources):
        print("output must not overwrite an input file", file=sys.stderr)
        return 2
    try:
        report = verify_exact_parity(*sources)
        code = 0 if report["strict_gate"] == "pass" else 1
    except (ValueError, csv.Error, UnicodeError) as exc:
        report = {"input_status": "invalid", "error": str(exc),
                  "official_formula_replication": "not_established", "label_exposure": dict(LABEL_EXPOSURE)}
        code = 2
    try:
        args.output.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    except OSError as exc:
        print(f"cannot write report ({exc.strerror})", file=sys.stderr)
        return 2
    return code


if __name__ == "__main__":
    raise SystemExit(main())
