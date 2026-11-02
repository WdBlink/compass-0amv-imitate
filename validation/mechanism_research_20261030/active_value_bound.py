"""Offline conditional lower bounds, not market levels or a Compass formula.

For each eligible stock, value F * close * (1 - product(1 - volume/F))
at the current unadjusted close. Use exact rational arithmetic and floor each
stock to 10^-12 CNY yi before summing. Ineligible or unseen stocks are
unidentified nonnegative terms, not observations of zero.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter, deque
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from datetime import timedelta
from decimal import Decimal
from fractions import Fraction
import io
import json
from pathlib import Path
import re
import shlex
import sys

from . import market_snapshot as snapshot


WINDOWS = (5, 13, 34)
REPORT_START, REPORT_END = "20260727", "20261030"
TICKS_PER_YI = 10**12
TICKS_PER_CNY = TICKS_PER_YI // 10**8
ENDPOINTS = ("daily", "daily_basic", "adj_factor")
# 689009 is a depositary receipt, not an ordinary A share (SSE notice, 2020-10-28).
SH_A = re.compile(r"(?:60[0-9]{4}|688[0-9]{3})\.SH\Z")
SZ_A = re.compile(r"(?:00|30)[0-9]{4}\.SZ\Z")
EXCLUSIONS = ("initial_or_new", "missing", "ohlc_vwap", "close_mismatch", "bad_input", "bad_q", "action")
COLUMNS = ("date", "market", "window", "eligible", "lower_yi", "seen_asof", "excluded",
           *("ex_" + reason for reason in EXCLUSIONS))


@dataclass(frozen=True, slots=True)
class Panel:
    """One independently dated session of provider-unit endpoint records."""

    date: str
    daily: Sequence[dict]
    basic: Sequence[dict]
    factors: Sequence[dict]


@dataclass(slots=True)
class History:
    """Latest consecutive, valid, constant-float/factor segment of one stock."""

    market: str
    survival: deque = field(default_factory=lambda: deque(maxlen=max(WINDOWS)))
    regime: tuple[Fraction, Fraction] | None = None
    last_break: str = "initial_or_new"

    def reset(self, reason: str) -> None:
        self.survival.clear()
        self.regime = None
        self.last_break = reason


def _exact(value: object) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (int, str, Decimal, Fraction)):
        raise ValueError("not an exact numeric field")
    return Fraction(value)


def _stock_day(d: dict, b: dict, a: dict) -> str | tuple[Fraction, Fraction, Fraction, Fraction]:
    try:
        o, high, low, close, lots, thousands = (_exact(d[k]) for k in ("open", "high", "low", "close", "vol", "amount"))
        basic_close, shares, factor = _exact(b["close"]), _exact(b["float_share"]) * 10000, _exact(a["adj_factor"])
    except (KeyError, ValueError, OverflowError, ZeroDivisionError):
        return "bad_input"
    volume, amount = lots * 100, thousands * 1000
    if min(o, high, low, close, basic_close, factor) <= 0 or min(volume, amount) < 0:
        return "bad_input"
    if not (low <= o <= high and low <= close <= high and low * volume <= amount <= high * volume):
        return "ohlc_vwap"
    if basic_close != close:
        return "close_mismatch"
    if not 0 <= volume < shares:
        return "bad_q"
    return shares, factor, 1 - volume / shares, close * shares * TICKS_PER_CNY


def _index(rows: Sequence[dict], day: str) -> dict[str, dict]:
    index = {}
    for row in rows:
        symbol = row["ts_code"]
        if row["trade_date"] != day:
            raise ValueError("endpoint row outside requested market date")
        if symbol in index:
            raise ValueError("duplicate symbol within endpoint and date")
        index[symbol] = row
    return index


def _add_stock(state: History, observation: str | tuple, counts: dict[int, Counter]) -> None:
    if isinstance(observation, str):
        state.reset(observation)
    else:
        shares, factor, survival, cap = observation
        if state.regime is not None and state.regime != (shares, factor):
            state.reset("action")
        state.regime = (shares, factor)
        state.survival.append(survival.as_integer_ratio())
        num = den = 1
        for age, (n, d) in enumerate(reversed(state.survival), 1):
            num *= n
            den *= d
            if age in counts:
                counts[age]["eligible"] += 1
                counts[age]["ticks"] += cap.numerator * (den - num) // (cap.denominator * den)
    for n in WINDOWS:
        if len(state.survival) < n:
            counts[n][state.last_break] += 1


def calculate_bounds(calendar: Iterable[str], panels: Iterable[Panel]) -> Iterator[dict]:
    """Emit six rows per declared session using only that session and its prefix.

    seen_asof counts only ordinary-stock codes observed in any of the three
    endpoints through t. It is not historical market membership. Ineligible
    codes are partitioned by the most recent interruption to their clean
    streak; newly seen securities need N sessions just like older stocks.
    """
    histories: dict[str, History] = {}
    prior = ""
    for day, panel in zip(calendar, panels, strict=True):
        snapshot.valid_date(day)
        if day <= prior or panel.date != day:
            raise ValueError("require exact ordered exchange sessions and dated panels")
        prior = day
        daily, basic, factors = (_index(rows, day) for rows in (panel.daily, panel.basic, panel.factors))
        for symbol in (daily.keys() | basic.keys() | factors.keys()) - histories.keys():
            if SH_A.fullmatch(symbol):
                histories[symbol] = History("SH")
            elif SZ_A.fullmatch(symbol):
                histories[symbol] = History("SZ")
        counts = {ex: {n: Counter() for n in WINDOWS} for ex in ("SH", "SZ")}
        for symbol, state in histories.items():
            if symbol not in daily or symbol not in basic or symbol not in factors:
                obs = "missing"
            else:
                obs = _stock_day(daily[symbol], basic[symbol], factors[symbol])
            _add_stock(state, obs, counts[state.market])
        for market in ("SH", "SH_SZ"):
            for n in WINDOWS:
                c = counts["SH"][n] if market == "SH" else counts["SH"][n] + counts["SZ"][n]
                whole, fractional = divmod(c["ticks"], TICKS_PER_YI)
                excluded = sum(c[reason] for reason in EXCLUSIONS)
                yield dict(date=snapshot.parse_date(day).isoformat(), market=market, window=n,
                           eligible=c["eligible"], lower_yi=f"{whole}.{fractional:012d}",
                           seen_asof=c["eligible"] + excluded, excluded=excluded,
                           **{"ex_" + reason: c[reason] for reason in EXCLUSIONS})


def _calendar(root: Path, manifest: dict) -> list[str]:
    if (manifest["requested_start"], manifest["requested_end"]) != (snapshot.START, snapshot.END):
        raise ValueError("unexpected snapshot range")
    if manifest["fields"] != snapshot.FIELDS:
        raise ValueError("unexpected snapshot field contract")
    start, end = map(snapshot.parse_date, (snapshot.START, snapshot.END))
    civil = {(start + timedelta(days=i)).strftime("%Y%m%d") for i in range((end - start).days + 1)}
    calendars = []
    for exchange in ("SSE", "SZSE"):
        rows = snapshot.load_group(root, manifest, f"trade_cal/{exchange}")
        if {r["cal_date"] for r in rows} != civil:
            raise ValueError("incomplete exchange calendar")
        calendars.append(sorted(r["cal_date"] for r in rows if r["is_open"] == 1))
    days = calendars[0]
    report_days = [d for d in days if REPORT_START <= d <= REPORT_END]
    if days != calendars[1] or len(days) != 121 or len(report_days) != 64 or (
        report_days[0], report_days[-1]
    ) != (REPORT_START, REPORT_END):
        raise ValueError("calendar differs from preregistered 121-session/64-day coverage")
    required = {f"{ep}/{day}" for day in days for ep in ENDPOINTS}
    metadata = {f"stock_basic/{status}" for status in "LDP"}
    if set(manifest["groups"]) != required | metadata | {"trade_cal/SSE", "trade_cal/SZSE"}:
        raise ValueError("missing or unexpected snapshot group/date")
    return days


def snapshot_bounds(root: Path, manifest_sha256: str) -> list[dict]:
    """Authenticate market-only bytes against an externally supplied freeze."""
    raw = (root / "manifest.json").read_bytes()
    if snapshot.digest(raw) != manifest_sha256:
        raise ValueError("snapshot manifest hash mismatch")
    manifest = json.loads(raw)
    days = _calendar(root, manifest)

    def panels():
        for day in days:
            groups = [snapshot.load_group(root, manifest, f"{ep}/{day}") for ep in ENDPOINTS]
            if not all(groups):
                raise ValueError(f"empty endpoint response for open date {day}")
            yield Panel(day, *groups)

    result = [r for r in calculate_bounds(days, panels()) if r["date"] >= snapshot.parse_date(REPORT_START).isoformat()]
    # Verify the unused metadata bytes too, without feeding their current classifications to histories.
    for status in "LDP":
        snapshot.load_group(root, manifest, f"stock_basic/{status}")
    return result


def csv_bytes(rows: Iterable[dict]) -> bytes:
    buffer = io.StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buffer.getvalue().encode("utf-8")


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, required=True, help="read-only frozen market snapshot directory")
    p.add_argument("--manifest-sha256", required=True, help="independently frozen SHA-256 of manifest.json")
    p.add_argument("--output", type=Path, required=True, help="new CSV outside the snapshot")
    p.add_argument("--receipt", type=Path, required=True, help="new JSON recording command and hashes after CSV close")
    args = p.parse_args()
    root = args.root.resolve()
    targets = (args.output, args.receipt)
    if len({path.resolve() for path in targets}) != 2:
        p.error("output and receipt must be different files")
    for path in targets:
        if path.exists() or path.is_symlink() or path.resolve().is_relative_to(root) or not path.parent.is_dir():
            p.error("require fresh output paths outside the snapshot in existing directories")
    started = snapshot.utc()
    source_hashes = {Path(module).name: snapshot.digest(Path(module).read_bytes())
                     for module in (__file__, snapshot.__file__)}
    rows = snapshot_bounds(root, args.manifest_sha256)
    data = csv_bytes(rows)
    with args.output.open("xb") as f:
        f.write(data)
    closed = snapshot.utc()
    checksum = snapshot.digest(args.output.read_bytes())
    if checksum != snapshot.digest(data):
        raise ValueError("closed CSV differs from generated bytes")
    receipt = {
        "started_at_utc": started, "csv_closed_at_utc": closed, "receipt_at_utc": snapshot.utc(),
        "command": shlex.join(sys.orig_argv), "cwd": str(Path.cwd()), "source_sha256": source_hashes,
        "snapshot_manifest_sha256": args.manifest_sha256, "bounds_sha256": checksum,
        "rows": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "arithmetic": "Exact rationals; floor each valid stock value in units of 10^-12 yi, then sum integer ticks.",
        "coverage": "Observed A-share code subset only. Missing, recent or invalid stock windows are not known zero.",
    }
    with args.receipt.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "bounds_sha256": checksum, "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
