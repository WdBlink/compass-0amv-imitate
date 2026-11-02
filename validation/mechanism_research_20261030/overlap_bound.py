"""Conditional current-price bounds for recently traded ordinary shares, not a Compass level.

Assume each session's volume counts distinct shares that day. Across sessions their
identities can coincide arbitrarily. Missing stocks are not measured zero; only the
lower endpoint also bounds a larger nonnegative universe containing the covered stocks.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass, field
from fractions import Fraction
import io
import json
from pathlib import Path
import re
import shlex
import sys

from . import active_value_bound as spot
from . import market_snapshot as snapshot


COLUMNS = ("date", "market", "window", "eligible", "lower_yi", "upper_covered_yi", "seen_asof", "excluded",
           *("ex_" + reason for reason in spot.EXCLUSIONS))


@dataclass(slots=True)
class History:
    market: str
    volumes: deque[int | Fraction] = field(default_factory=lambda: deque(maxlen=max(spot.WINDOWS)))
    regime: tuple[Fraction, Fraction] | None = None
    last_break: str = "initial_or_new"

    def reset(self, reason: str) -> None:
        self.volumes.clear()
        self.regime, self.last_break = None, reason


def _accumulate(volumes: deque, shares: Fraction, price_ticks: Fraction, counts: dict[int, Counter]) -> None:
    max_q = total_q = 0
    pn, pd = price_ticks.as_integer_ratio()
    for age, volume in enumerate(reversed(volumes), 1):
        max_q = max(max_q, volume)
        total_q += volume
        if age in counts:
            c = counts[age]
            c["eligible"] += 1
            c["lower_ticks"] += (max_q * pn) // pd
            c["upper_covered_ticks"] += -((-min(shares, total_q) * pn) // pd)


def _yi(ticks: int) -> str:
    whole, decimal = divmod(ticks, spot.TICKS_PER_YI)
    return f"{whole}.{decimal:012d}"


def calculate_bounds(calendar: Iterable[str], panels: Iterable[spot.Panel]) -> Iterator[dict]:
    """Emit six rows each session, using the same as-of cohort and breaks as the spot control."""
    histories: dict[str, History] = {}
    prior = ""
    for day, panel in zip(calendar, panels, strict=True):
        snapshot.valid_date(day)
        if day <= prior or day != panel.date:
            raise ValueError("require exact ordered exchange sessions and dated panels")
        prior = day
        daily, basic, factors = (spot._index(records, day) for records in (panel.daily, panel.basic, panel.factors))
        for code in (daily.keys() | basic.keys() | factors.keys()) - histories.keys():
            if spot.SH_A.fullmatch(code):
                histories[code] = History("SH")
            elif spot.SZ_A.fullmatch(code):
                histories[code] = History("SZ")
        counts = {ex: {n: Counter() for n in spot.WINDOWS} for ex in ("SH", "SZ")}
        for code, state in histories.items():
            obs = (spot._stock_day(daily[code], basic[code], factors[code])
                   if code in daily and code in basic and code in factors else "missing")
            if isinstance(obs, str):
                state.reset(obs)
            else:
                shares, factor, _, _ = obs
                if state.regime is not None and state.regime != (shares, factor):
                    state.reset("action")
                state.regime = (shares, factor)
                volume = spot._exact(daily[code]["vol"]) * 100
                state.volumes.append(volume.numerator if volume.denominator == 1 else volume)
                price_ticks = spot._exact(daily[code]["close"]) * spot.TICKS_PER_CNY
                _accumulate(state.volumes, shares, price_ticks, counts[state.market])
            for n in spot.WINDOWS:
                if len(state.volumes) < n:
                    counts[state.market][n][state.last_break] += 1
        for market in ("SH", "SH_SZ"):
            for n in spot.WINDOWS:
                c = counts["SH"][n] if market == "SH" else counts["SH"][n] + counts["SZ"][n]
                excluded = sum(c[r] for r in spot.EXCLUSIONS)
                yield dict(date=snapshot.parse_date(day).isoformat(), market=market, window=n, eligible=c["eligible"],
                           lower_yi=_yi(c["lower_ticks"]), upper_covered_yi=_yi(c["upper_covered_ticks"]),
                           seen_asof=excluded + c["eligible"], excluded=excluded,
                           **{"ex_" + r: c[r] for r in spot.EXCLUSIONS})


def snapshot_bounds(root: Path, manifest_sha256: str, calendar_sha256: str) -> list[dict]:
    manifest_raw = (root / "manifest.json").read_bytes()
    if snapshot.digest(manifest_raw) != manifest_sha256:
        raise ValueError("snapshot manifest hash mismatch")
    manifest = json.loads(manifest_raw)
    days = spot._calendar(root, manifest)
    first = days.index(spot.REPORT_START)
    # The precommitted dates file contains the previous session, then the 64 development sessions.
    calendar = ("date\n" + "".join(f"{snapshot.parse_date(d)}\n" for d in days[first-1:])).encode()
    if snapshot.digest(calendar) != calendar_sha256:
        raise ValueError("market-derived development calendar hash mismatch")

    def panels():
        for day in days:
            groups = [snapshot.load_group(root, manifest, f"{ep}/{day}") for ep in spot.ENDPOINTS]
            if not all(groups):
                raise ValueError(f"empty endpoint response for open date {day}")
            yield spot.Panel(day, *groups)

    start = snapshot.parse_date(spot.REPORT_START).isoformat()
    result = [r for r in calculate_bounds(days, panels()) if r["date"] >= start]
    for status in "LDP":
        snapshot.load_group(root, manifest, f"stock_basic/{status}")  # Authenticate, never use current listing status.
    return result


def csv_bytes(rows: Iterable[dict]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="read-only frozen market snapshot")
    parser.add_argument("--manifest-sha256", required=True)
    parser.add_argument("--calendar-sha256", required=True, help="precommitted date-only file hash")
    parser.add_argument("--protocol-sha256", required=True, help="digest of the externally frozen specification")
    parser.add_argument("--output", type=Path, required=True, help="new local prediction CSV")
    parser.add_argument("--receipt", type=Path, required=True, help="new local pre-score receipt")
    args = parser.parse_args()
    if not all(re.fullmatch("[0-9a-f]{64}", pin) for pin in (
        args.manifest_sha256, args.calendar_sha256, args.protocol_sha256
    )):
        parser.error("require SHA-256 pins as 64 lowercase hexadecimal digits")
    root = args.root.resolve()
    targets = (args.output, args.receipt)
    if len({p.resolve() for p in targets}) != 2 or any(
        p.exists() or p.is_symlink() or not p.parent.is_dir() or p.resolve().is_relative_to(root) for p in targets
    ):
        parser.error("require two distinct fresh files outside the snapshot in existing directories")
    started = snapshot.utc()
    source_hashes = {Path(p).name: snapshot.digest(Path(p).read_bytes())
                     for p in (__file__, spot.__file__, snapshot.__file__)}
    rows = snapshot_bounds(root, args.manifest_sha256, args.calendar_sha256)
    data = csv_bytes(rows)
    with args.output.open("xb") as f:
        f.write(data)
    closed = snapshot.utc()
    digest = snapshot.digest(args.output.read_bytes())
    if digest != snapshot.digest(data):
        raise ValueError("closed prediction CSV differs from generated bytes")
    receipt = {
        "started_at_utc": started, "csv_closed_at_utc": closed, "receipt_at_utc": snapshot.utc(),
        "command": shlex.join(sys.orig_argv), "predictor_argv": sys.argv, "cwd": str(Path.cwd()),
        "source_sha256": source_hashes, "protocol_sha256": args.protocol_sha256,
        "snapshot_manifest_sha256": args.manifest_sha256, "calendar_sha256": args.calendar_sha256,
        "prediction_sha256": digest, "rows": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "arithmetic": "Per stock floor/ceil exact rational current-price value in 1e-12 yi ticks, then sum integers.",
        "coverage": "Under distinct intraday shares and arbitrary cross-session overlap; upper is covered-only.",
    }
    with args.receipt.open("x", encoding="utf-8") as f:
        f.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"prediction_sha256": digest, "rows": len(rows), "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
