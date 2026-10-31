"""Score a nonnegative-subset lower bound; never treat the subset as a point estimate."""
from __future__ import annotations

import argparse
import csv
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

from validation.verify_exact_parity import LEVEL_HALF, _number, _read


SCOPES = ("SH", "SH_SZ")
PERIODS = ("5", "13", "34")


def score(official: Path, calendar: Path, bounds: Path, frozen_sha256: str) -> dict:
    raw = bounds.read_bytes()
    if hashlib.sha256(raw).hexdigest() != frozen_sha256:
        raise ValueError("bound file differs from the supplied prediction freeze")
    labels, label_hash = _read(official, "official", {"date", "official_close", "official_change_pct"})
    sessions, calendar_hash = _read(calendar, "calendar", {"date"})
    closes = {row["date"]: row["official_close"] for row in labels}
    if list(closes) != [row["date"] for row in sessions[1:]]:
        raise ValueError("require every scored date plus exactly one preceding market session")
    date_list = list(closes)
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), strict=True)
    required = {"date", "market", "window", "eligible", "lower_yi"}
    header = reader.fieldnames or []
    if len(set(header)) != len(header) or not required.issubset(header):
        raise ValueError("missing or duplicate bound column")
    parsed = {}
    for row in reader:
        day, market, window = key = (row["date"], row["market"], row["window"])
        if (None in row or any(value is None for value in row.values()) or key in parsed
                or day not in closes or market not in SCOPES or window not in PERIODS):
            raise ValueError("malformed, duplicate or unexpected bound row")
        if not (re.fullmatch(r"[0-9]{1,9}", row["eligible"])
                and re.fullmatch(r"[0-9]{1,32}(?:\.[0-9]{1,32})?", row["lower_yi"])):
            raise ValueError("invalid coverage count or decimal CNY 100M lower bound")
        eligible, lower = int(row["eligible"]), Fraction(row["lower_yi"])
        if not eligible and lower:
            raise ValueError("nonzero lower bound with zero eligible contributors")
        parsed[key] = eligible, lower
    if len(parsed) != len(closes) * len(SCOPES) * len(PERIODS):
        raise ValueError("incomplete date/market/window grid")
    for d in date_list:
        for window in PERIODS:
            small, big = parsed[d, "SH", window], parsed[d, "SH_SZ", window]
            if small[0] > big[0] or small[1] > big[1]:
                raise ValueError("combined scope must include Shanghai contributors and value")

    summaries, details = [], []
    for market in SCOPES:
        for window in PERIODS:
            gaps = []
            for d in date_list:
                eligible, lower = parsed[d, market, window]
                margin = lower - (closes[d] + LEVEL_HALF)
                gaps.append((margin, d))
                details.append({"date": d, "market": market, "window": int(window), "eligible": eligible,
                                "lower_yi": _number(lower), "official_display": _number(closes[d]),
                                "gap_above_upper": _number(margin), "excluded_on_date": margin > 0})
            count = sum(gap > 0 for gap, _ in gaps)
            peak, peak_date = max(gaps)
            summaries.append({"market": market, "window": int(window), "days": len(date_list),
                              "excluded_days": count, "undecided_days": len(date_list) - count,
                              "largest_gap_above_upper": _number(peak), "largest_gap_date": peak_date,
                              "status": "conditionally_refuted_by_lower_bound" if count
                              else "not_excluded_by_lower_bound"})
    return {"assessment": "disclosed_days_one_sided_only", "scale_assumption": "CNY 100M per display unit",
            "official_formula_replication": "not_established", "change_and_trigger": "not_scored",
            "input_sha256": {"official": label_hash, "calendar": calendar_hash, "bounds": frozen_sha256},
            "summary": summaries, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "calendar", "bounds", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frozen-sha256", required=True)
    args = parser.parse_args()
    result = score(args.official, args.calendar, args.bounds, args.frozen_sha256)
    with args.out.open("x", encoding="utf-8") as out:
        json.dump(result, out, ensure_ascii=False, indent=2)
        out.write("\n")
    print(json.dumps(result["summary"], ensure_ascii=False))


if __name__ == "__main__":
    main()
