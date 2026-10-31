"""Offline acquisition costs for uniform latest-sale cohorts, not a Compass point formula.

Low and VWAP are rounded down on covered stocks; high is rounded up for
those stocks only. Unobserved or excluded holdings have unknown value.
"""
from __future__ import annotations

import argparse
from collections import Counter, deque
from collections.abc import Iterable, Iterator
import csv
from dataclasses import dataclass
from datetime import date
from fractions import Fraction
import io
import json
from math import lcm
from pathlib import Path
import shlex
import sys
import time
from typing import NamedTuple

from . import active_value_bound as spot
from . import market_snapshot as snapshot


COLUMNS = ("date", "market", "window", "eligible", "low_yi", "vwap_yi", "high_yi", "seen_asof", "excluded",
           *("ex_" + reason for reason in spot.EXCLUSIONS))


class Cohort(NamedTuple):
    survival_num: int
    survival_den: int
    low: int
    vwap: int
    high: int
    cash_den: int


@dataclass(slots=True)
class History:
    market: str
    cohorts: deque[Cohort]
    regime: tuple[Fraction, Fraction] | None = None
    last_break: str = "initial_or_new"


def _cohort(daily: dict, survival: Fraction) -> Cohort:
    # At VWAP the original cost of new shares is the actual amount, including 0 for 0 volume.
    volume_ticks = spot._exact(daily["vol"]) * (100 * spot.TICKS_PER_CNY)
    cash = (spot._exact(daily["low"]) * volume_ticks,
            spot._exact(daily["amount"]) * (1000 * spot.TICKS_PER_CNY),
            spot._exact(daily["high"]) * volume_ticks)
    den = lcm(*(v.denominator for v in cash))
    return Cohort(*survival.as_integer_ratio(), *(v.numerator * (den // v.denominator) for v in cash), den)


def _accumulate(cohorts: deque[Cohort], counts: dict[int, Counter]) -> None:
    if len(cohorts) < min(counts):
        return
    # All cash amounts use one exact tick denominator; no per-day quantization.
    cash_den = lcm(*(c.cash_den for c in cohorts))
    p = q = 1
    low = vwap = high = 0
    for age, (surv_num, surv_den, lo, vw, hi, scale) in enumerate(reversed(cohorts), 1):
        # p/q is the survival product of strictly later sessions. Cash totals
        # use denominator q*cash_den. Extend that denominator only after valuing s.
        weight = p if scale == cash_den else p * (cash_den // scale)
        low += lo * weight
        vwap += vw * weight
        high += hi * weight
        if age in counts:
            c = counts[age]
            den = q * cash_den
            c["eligible"] += 1
            c["low"] += low // den
            c["vwap"] += vwap // den
            c["high"] += -(-high // den)
        p *= surv_num
        q *= surv_den
        low *= surv_den
        vwap *= surv_den
        high *= surv_den


def _value(ticks: int) -> str:
    whole, decimal = divmod(ticks, spot.TICKS_PER_YI)
    return f"{whole}.{decimal:012d}"


def calculate_costs(calendar: Iterable[str], panels: Iterable[spot.Panel], *,
                    windows: tuple[int, ...] = spot.WINDOWS) -> Iterator[dict]:
    """Emit causal subset costs; production uses only the three preregistered windows."""
    if not windows or any(type(n) is not int or n < 1 for n in windows) or tuple(sorted(set(windows))) != windows:
        raise ValueError("windows must be strictly increasing positive integers")
    histories: dict[str, History] = {}
    prior = ""
    for day, panel in zip(calendar, panels, strict=True):
        snapshot.valid_date(day)
        if day <= prior or day != panel.date:
            raise ValueError("require exact ordered exchange sessions and dated panels")
        prior = day
        daily, basic, factors = (spot._index(records, day) for records in (panel.daily, panel.basic, panel.factors))
        for code in (daily.keys() | basic.keys() | factors.keys()) - histories.keys():
            market = "SH" if spot.SH_A.fullmatch(code) else "SZ" if spot.SZ_A.fullmatch(code) else None
            if market:
                histories[code] = History(market, deque(maxlen=windows[-1]))
        counts = {ex: {n: Counter() for n in windows} for ex in ("SH", "SZ")}
        for code, state in histories.items():
            obs = (spot._stock_day(daily[code], basic[code], factors[code])
                   if code in daily and code in basic and code in factors else "missing")
            if isinstance(obs, str):
                state.cohorts.clear()
                state.regime, state.last_break = None, obs
            else:
                shares, factor, survival, _ = obs
                if state.regime is not None and state.regime != (shares, factor):
                    state.cohorts.clear()
                    state.last_break = "action"
                state.regime = (shares, factor)
                state.cohorts.append(_cohort(daily[code], survival))
                _accumulate(state.cohorts, counts[state.market])
            for n in windows:
                if len(state.cohorts) < n:
                    counts[state.market][n][state.last_break] += 1
        for market in ("SH", "SH_SZ"):
            for n in windows:
                c = counts["SH"][n] if market == "SH" else counts["SH"][n] + counts["SZ"][n]
                excluded = sum(c[reason] for reason in spot.EXCLUSIONS)
                yield dict(date=date.fromisoformat(day).isoformat(), market=market, window=n, eligible=c["eligible"],
                           low_yi=_value(c["low"]), vwap_yi=_value(c["vwap"]), high_yi=_value(c["high"]),
                           seen_asof=excluded + c["eligible"], excluded=excluded,
                           **{"ex_" + reason: c[reason] for reason in spot.EXCLUSIONS})


def snapshot_costs(root: Path, manifest_sha256: str) -> list[dict]:
    raw = (root / "manifest.json").read_bytes()
    if snapshot.digest(raw) != manifest_sha256:
        raise ValueError("snapshot manifest hash mismatch")
    manifest = json.loads(raw)
    days = spot._calendar(root, manifest)

    def panels():
        for day in days:
            groups = [snapshot.load_group(root, manifest, f"{ep}/{day}") for ep in spot.ENDPOINTS]
            if not all(groups):
                raise ValueError(f"empty endpoint response for open date {day}")
            yield spot.Panel(day, *groups)

    start = date.fromisoformat(spot.REPORT_START).isoformat()
    rows = [r for r in calculate_costs(days, panels()) if r["date"] >= start]
    for status in "LDP":
        snapshot.load_group(root, manifest, f"stock_basic/{status}")  # Authenticate; never use today's membership.
    return rows


def csv_bytes(rows: Iterable[dict]) -> bytes:
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return buf.getvalue().encode("utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True, help="read-only frozen market-only snapshot")
    parser.add_argument("--manifest-sha256", required=True, help="independent SHA-256 of manifest.json")
    parser.add_argument("--output", type=Path, required=True, help="fresh prediction CSV outside the snapshot")
    parser.add_argument("--receipt", type=Path, required=True, help="fresh pre-score JSON receipt")
    args = parser.parse_args()
    root = args.root.resolve()
    targets = args.output, args.receipt
    if len({p.resolve() for p in targets}) != 2 or any(
        p.exists() or p.is_symlink() or not p.parent.is_dir() or p.resolve().is_relative_to(root) for p in targets
    ):
        parser.error("require two different fresh output paths outside the snapshot in existing directories")
    started, timer = snapshot.utc(), time.monotonic()
    sources = {Path(f).name: snapshot.digest(Path(f).read_bytes()) for f in (__file__, spot.__file__, snapshot.__file__)}
    rows = snapshot_costs(root, args.manifest_sha256)
    data = csv_bytes(rows)
    with args.output.open("xb") as file:
        file.write(data)
    closed = snapshot.utc()
    checksum = snapshot.digest(args.output.read_bytes())
    if checksum != snapshot.digest(data):
        raise ValueError("closed CSV differs from generated bytes")
    receipt = {
        "started_at_utc": started, "csv_closed_at_utc": closed, "receipt_at_utc": snapshot.utc(),
        "elapsed_seconds": round(time.monotonic() - timer, 3), "command": shlex.join(sys.orig_argv),
        "cwd": str(Path.cwd()), "source_sha256": sources, "snapshot_manifest_sha256": args.manifest_sha256,
        "prediction_sha256": checksum, "rows": len(rows), "first_date": rows[0]["date"], "last_date": rows[-1]["date"],
        "arithmetic": "Exact rational costs and survival. Per stock round low/VWAP down and covered high up to 1e-12 yi; sum integer ticks.",
        "coverage": "Same nonnegative covered ordinary-A subset as the spot control. Missing terms are unknown; the high is not a full-market upper bound.",
    }
    with args.receipt.open("x", encoding="utf-8") as file:
        file.write(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"rows": len(rows), "prediction_sha256": checksum, "elapsed_seconds": receipt["elapsed_seconds"],
                      "receipt": str(args.receipt)}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
