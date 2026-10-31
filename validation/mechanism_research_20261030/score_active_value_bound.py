"""Score a nonnegative-subset lower bound; never treat the subset as a point estimate."""
from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from fractions import Fraction
import hashlib
import io
import json
from pathlib import Path
import re

from validation.mechanism_research_20261030.active_value_bound import COLUMNS, EXCLUSIONS
from validation.verify_exact_parity import LEVEL_HALF, _number, _read


SCOPES = ("SH", "SH_SZ")
PERIODS = ("5", "13", "34")
REASONS = tuple("ex_" + reason for reason in EXCLUSIONS)
# Scoring-only pins from the reference manifest and the original pre-score prediction receipt.
OFFICIAL_SHA256 = "176e0335f5c062fa3d93951ccb15afcaa83641f0b55da693d8c9fa90d47c75ea"
CALENDAR_SHA256 = "de73fe87079197668e5aa6e689907752d5ccf271273a34b3b6704b58d0e677ed"
BOUNDS_SHA256 = "4bafb34e9da286d7ba46f5543bdc3748bf3bb591a7fd5667753a97c2e140c752"
RECEIPT_SHA256 = "631faba1aa6891d93071114a00712bc5a6e00835f2b33d44dc2a6b57a2890d19"


@dataclass(frozen=True)
class Bound:
    eligible: int
    lower: Fraction
    reasons: tuple[int, ...]

    @property
    def seen(self) -> int:
        return self.eligible + sum(self.reasons)


def _read_grid(raw: bytes, dates: list[str]) -> dict[tuple[str, str, str], Bound]:
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8")), strict=True)
    header = reader.fieldnames or []
    if len(header) != len(COLUMNS) or set(header) != set(COLUMNS):
        raise ValueError("require the complete native bound columns, including all coverage counts")
    parsed = {}
    date_set = set(dates)
    for row in reader:
        day, market, window = key = (row["date"], row["market"], row["window"])
        if (None in row or any(value is None for value in row.values()) or key in parsed
                or day not in date_set or market not in SCOPES or window not in PERIODS):
            raise ValueError("malformed, duplicate or unexpected bound row")
        if not (all(re.fullmatch(r"[0-9]{1,9}", row[field]) for field in ("eligible", "seen_asof", "excluded", *REASONS))
                and re.fullmatch(r"[0-9]{1,32}(?:\.[0-9]{1,32})?", row["lower_yi"])):
            raise ValueError("invalid coverage count or decimal CNY 100M lower bound")
        cell = Bound(int(row["eligible"]), Fraction(row["lower_yi"]), tuple(int(row[field]) for field in REASONS))
        excluded = int(row["excluded"])
        if cell.eligible + excluded != int(row["seen_asof"]) or sum(cell.reasons) != excluded:
            raise ValueError(f"{key}: invalid population or exclusion reason totals")
        parsed[key] = cell
    if len(parsed) != len(dates) * len(SCOPES) * len(PERIODS):
        raise ValueError("incomplete date/market/window grid")
    prior_seen = {"SH": 0, "SZ": 0}
    for day in dates:
        sh = [parsed[day, "SH", w] for w in PERIODS]
        both = [parsed[day, "SH_SZ", w] for w in PERIODS]
        # The producer adds two disjoint markets. Check the implied Shenzhen slice too.
        sz = [Bound(b.eligible - s.eligible, b.lower - s.lower, tuple(y - x for x, y in zip(s.reasons, b.reasons)))
              for s, b in zip(sh, both)]
        for market, cells in (("SH", sh), ("SZ", sz)):
            if any(min(c.eligible, c.lower, *c.reasons) < 0 or (not c.eligible and c.lower) for c in cells):
                raise ValueError(f"{day} {market}: invalid contributor counts or value for market scope")
            seen = cells[0].seen
            if any(c.seen != seen for c in cells) or seen < prior_seen[market]:
                raise ValueError(f"{day} {market}: seen_asof must agree across windows and not decrease across dates")
            prior_seen[market] = seen
            for short, long in zip(cells, cells[1:]):
                if short.eligible < long.eligible or any(a > b for a, b in zip(short.reasons, long.reasons)):
                    raise ValueError(f"{day} {market}: eligible and exclusion counts are not nested across windows")
                if short.eligible == long.eligible and long.lower < short.lower:
                    raise ValueError(f"{day} {market}: lower bound decreases for the same eligible cohort")
    return parsed


def score(official: Path, calendar: Path, bounds: Path, frozen_sha256: str | None = None, *,
          generic: bool = False, freeze_receipt: Path | None = None) -> dict:
    """The default replays the frozen 64 disclosed days; generic diagnostics have no campaign authority."""
    if generic and (not frozen_sha256 or freeze_receipt is not None):
        raise ValueError("generic diagnostics require a caller-supplied freeze hash and no campaign receipt")
    labels, label_hash = _read(official, "official", {"date", "official_close", "official_change_pct"})
    sessions, calendar_hash = _read(calendar, "calendar", {"date"})
    closes = {row["date"]: row["official_close"] for row in labels}
    date_list = list(closes)
    if not generic:
        if label_hash != OFFICIAL_SHA256:
            raise ValueError("official labels do not match the disclosed 64-day freeze")
        if calendar_hash != CALENDAR_SHA256:
            raise ValueError("calendar does not match the frozen trading sessions and predecessor")
        if (len(labels), sessions[0]["date"], date_list[0], date_list[-1]) != (
            64, "2026-07-24", "2026-07-27", "2026-10-30"
        ):
            raise ValueError("unexpected disclosed campaign window")
        if freeze_receipt is not None and hashlib.sha256(freeze_receipt.read_bytes()).hexdigest() != RECEIPT_SHA256:
            raise ValueError("prediction receipt differs from the original pre-score freeze receipt")
    if date_list != [row["date"] for row in sessions[1:]]:
        raise ValueError("require every scored date plus exactly one preceding market session")
    raw = bounds.read_bytes()
    bound_hash = hashlib.sha256(raw).hexdigest()
    if frozen_sha256 is not None and bound_hash != frozen_sha256:
        raise ValueError("bound file differs from the supplied prediction freeze")
    parsed = _read_grid(raw, date_list)
    if not generic and bound_hash != BOUNDS_SHA256:
        raise ValueError("bound file differs from the original disclosed campaign prediction freeze")

    summaries, details = [], []
    for market in SCOPES:
        for window in PERIODS:
            gaps = []
            for d in date_list:
                cell = parsed[d, market, window]
                eligible, lower = cell.eligible, cell.lower
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
    return {"assessment": "generic_diagnostic_only_not_campaign_evidence" if generic else "disclosed_days_one_sided_only",
            "scale_assumption": "CNY 100M per display unit",
            "official_formula_replication": "not_established", "change_and_trigger": "not_scored",
            "input_sha256": {"official": label_hash, "calendar": calendar_hash, "bounds": bound_hash},
            **({} if generic else {"archived_prediction_receipt_sha256": RECEIPT_SHA256}),
            "summary": summaries, "details": details}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("official", "calendar", "bounds", "out"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--frozen-sha256", help="additional bounds hash assertion; required for generic diagnostics")
    parser.add_argument("--freeze-receipt", type=Path, help="also verify the original archived prediction receipt")
    parser.add_argument("--generic", action="store_true", help="score local fixtures only, never campaign evidence")
    args = parser.parse_args()
    try:
        result = score(args.official, args.calendar, args.bounds, args.frozen_sha256,
                       generic=args.generic, freeze_receipt=args.freeze_receipt)
        with args.out.open("x", encoding="utf-8") as out:
            json.dump(result, out, ensure_ascii=False, indent=2)
            out.write("\n")
    except (OSError, ValueError, csv.Error) as exc:
        parser.exit(2, f"score_active_value_bound: {exc}\n")
    summary = result["summary"]
    if args.generic:
        summary = {"assessment": result["assessment"], "summary": summary}
    print(json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
