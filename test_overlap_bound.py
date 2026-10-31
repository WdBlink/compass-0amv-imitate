import csv
from dataclasses import replace
from datetime import date
from fractions import Fraction
from itertools import islice
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from test_active_value_bound import at, make_snapshot, panel, records, series
from validation.mechanism_research_20261030 import active_value_bound as spot
from validation.mechanism_research_20261030 import market_snapshot as snap
from validation.mechanism_research_20261030 import overlap_bound as overlap


def rows(panels):
    return list(overlap.calculate_bounds([p.date for p in panels], panels))


@pytest.mark.parametrize(("shares", "ids", "min_units", "max_units"), [
    (12, [[1, 2]] * 5, 2, 10),
    (12, [[2*i, 2*i+1] for i in range(5)], 2, 10),
    (8, [[1, 2, 3, 4, 5], [4, 5, 6], [2, 7, 8], [7], [3]], 5, 8),
    (8, [[], [], [], [], []], 0, 0),
])
def test_identity_witnesses_and_float_cap(shares, ids, min_units, max_units):
    ps = series(5, lambda i: [records(price=17 if i == 4 else 3, shares=shares, volume=len(ids[i]))])
    result = at(rows(ps), ps[-1].date)
    to_shares = Fraction(10**8, 17)
    lo, hi = (Fraction(result[k]) * to_shares for k in ("lower_yi", "upper_covered_yi"))
    assert result["eligible"] == 1
    assert (lo, hi) == (min_units, max_units)
    assert len(set().union(*ids)) in (lo, hi)


def test_exact_outward_per_stock_rounding_three_windows_and_rolling_current_price():
    ps = series(35, lambda i: [
        records(s, shares=11, volume=1 if i == 0 else 0,
                price=Fraction(1, 30000) if i == 4 else 3)
        for s in ("600000.SH", "000001.SZ")
    ])
    out = rows(ps)
    assert len(out) == 35 * 6
    for n in (5, 13, 34):
        for market, size in (("SH", 1), ("SH_SZ", 2)):
            assert at(out, ps[n-2].date, n, market)["ex_initial_or_new"] == size
            first, after = (at(out, ps[i].date, n, market) for i in (n-1, n))
            assert (first["eligible"], after["eligible"]) == (size, size)
            assert (Fraction(first["lower_yi"]), Fraction(first["upper_covered_yi"])) == (
                (0, Fraction(size, 10**12)) if n == 5 else (Fraction(3*size, 10**8),)*2
            )
            assert (Fraction(after["lower_yi"]), Fraction(after["upper_covered_yi"])) == (0, 0)


def test_missing_not_backfilled_actions_and_every_existing_exclusion_counter():
    ps = series(42, lambda i: [
        records("600001.SH", shares=20 if i < 35 else 30, volume=1),
        records("000001.SZ", shares=20, volume=1, factor=1 if i < 35 else 2),
        *(records(f"60000{n}.SH", shares=20, volume=1) for n in range(2, 8)),
        *([records("688001.SH", shares=20, volume=1)] if i >= 35 else []),
        records("689009.SH", shares=20, volume=1),
    ])
    day = ps[35]
    day.daily[:] = [r for r in day.daily if r["ts_code"] != "600002.SH"]
    lookup = {r["ts_code"]: r for r in day.daily}
    lookup["600003.SH"]["open"] = 99
    lookup["600004.SH"]["vol"] = -1
    for r in day.basic:
        if r["ts_code"] == "600005.SH":
            r["close"] = 11
        if r["ts_code"] == "600006.SH":
            r["float_share"] = Fraction(1, 10000)
    out = rows(ps)
    ref = list(spot.calculate_bounds([p.date for p in ps], ps))
    coverage = ("date", "market", "window", "eligible", "seen_asof", "excluded",
                *("ex_" + r for r in spot.EXCLUSIONS))
    assert [[r[k] for k in coverage] for r in out] == [[r[k] for k in coverage] for r in ref]
    mixed = at(out, day.date, market="SH_SZ")
    assert mixed["eligible"] == 1
    assert [mixed["ex_" + r] for r in spot.EXCLUSIONS] == [1, 1, 1, 1, 1, 1, 2]
    assert all(at(out, ps[i].date)["ex_missing"] == 1 for i in range(35, 40))
    assert at(out, ps[40].date)["ex_missing"] == 0
    assert at(out, ps[39].date)["ex_action"] == 0


def test_out_of_date_duplicate_and_unaligned_panel_rejected():
    ps = series(5)
    for bad in ([*ps[:-1], replace(ps[-1], date=ps[0].date)], ps[:-1], ps[::-1]):
        with pytest.raises(ValueError):
            list(overlap.calculate_bounds([p.date for p in ps], bad))
    ps[-1].daily.append(ps[-1].daily[0])
    with pytest.raises(ValueError, match="duplicate"):
        rows(ps)


def test_nonempty_prefix_and_future_poison():
    fixtures = series(42)
    prefix = overlap.csv_bytes(rows(fixtures[:35]))
    changed = [*fixtures[:35], *(panel(p.date, [records("000001.SZ", price=71)]) for p in fixtures[35:])]
    base, other = rows(fixtures), rows(changed)
    assert prefix == overlap.csv_bytes(base[:35*6]) == overlap.csv_bytes(other[:35*6])
    assert at(base, fixtures[34].date, 34)["eligible"] > 0
    assert overlap.csv_bytes(base[35*6:]) != overlap.csv_bytes(other[35*6:])


def test_cli_authenticates_market_calendar_hashes_and_never_reads_decoy(tmp_path):
    root = tmp_path / "market"
    pin = make_snapshot(root)
    manifest = json.loads((root / "manifest.json").read_bytes())
    days = spot._calendar(root, manifest)
    calendar = snap.digest(("date\n" + "".join(f"{date.fromisoformat(d)}\n" for d in days[-65:])).encode())
    decoy = root / "unrelated.csv"
    decoy.write_text("unused")
    guarded = """import os, runpy, sys
decoy, module = sys.argv[1:3]
def guard(event, args):
    if event == 'open' and isinstance(args[0], str) and os.path.realpath(args[0]) == decoy:
        raise AssertionError('nonmarket access')
sys.addaudithook(guard)
sys.argv = [module, *sys.argv[3:]]
runpy.run_module(module, run_name='__main__')
"""
    def run(out, receipt, m=pin, c=calendar):
        return subprocess.run([sys.executable, "-B", "-c", guarded, str(decoy), overlap.__name__,
                               "--root", str(root), "--manifest-sha256", m, "--calendar-sha256", c,
                               "--protocol-sha256", snap.digest(b"synthetic overlap protocol"),
                               "--output", str(out), "--receipt", str(receipt)],
                              cwd=Path(__file__).resolve().parent, capture_output=True, text=True)

    out, receipt = tmp_path / "bound.csv", tmp_path / "receipt.json"
    for m, c in (("0"*64, calendar), (pin, "0"*64)):
        assert run(out, receipt, m, c).returncode != 0
        assert not out.exists() and not receipt.exists()
    done = run(out, receipt)
    assert done.returncode == 0, done.stderr
    body, proof = out.read_bytes(), json.loads(receipt.read_bytes())
    assert proof["prediction_sha256"] == snap.digest(body)
    assert proof["csv_closed_at_utc"] <= proof["receipt_at_utc"]
    assert len(list(csv.DictReader(body.decode().splitlines()))) == 384
    assert run(out, receipt).returncode != 0
    assert out.read_bytes() == body
    decoy.write_text("poisoned unrelated numbers")
    another, receipt2 = tmp_path / "bound2.csv", tmp_path / "receipt2.json"
    again = run(another, receipt2)
    assert again.returncode == 0 and another.read_bytes() == body, again.stderr
    (root / "raw" / "daily-20261030.json").write_bytes(b"corrupt")
    failed = run(tmp_path / "corrupt.csv", tmp_path / "corrupt.json")
    assert failed.returncode != 0 and "hash mismatch" in failed.stderr
    assert not (tmp_path / "corrupt.csv").exists()


@pytest.mark.skipif("OVERLAP_FROZEN_PREDICTION" not in os.environ, reason="opt-in private full-market replay")
def test_real_market_prefix_then_poison_matches_frozen_predictions():
    root = Path(os.environ["OVERLAP_MARKET_ROOT"])
    raw = (root / "manifest.json").read_bytes()
    assert snap.digest(raw) == os.environ["OVERLAP_MANIFEST_SHA256"]
    manifest = json.loads(raw)
    days = spot._calendar(root, manifest)
    split = days.index(spot.REPORT_START) + 2
    future = days[split:split+5]
    tail_allowed = False

    def stream():
        for d in days[:split]:
            groups = [snap.load_group(root, manifest, f"{ep}/{d}") for ep in spot.ENDPOINTS]
            yield spot.Panel(d, *groups)
        for d in future:
            assert tail_allowed, "the predictor requested future data to emit an earlier value"
            yield panel(d, [records("688999.SH"), records("301999.SZ", price=71)])

    calculation = overlap.calculate_bounds(days[:split+len(future)], stream())
    prefix = list(islice(calculation, split*6))[-12:]
    tail_allowed = True
    poisoned = list(calculation)
    with Path(os.environ["OVERLAP_FROZEN_PREDICTION"]).open(newline="") as f:
        frozen = list(csv.DictReader(f))
    assert len(frozen) == 384 and all(r["eligible"] > 0 for r in prefix)
    assert overlap.csv_bytes(prefix) == overlap.csv_bytes(frozen[:12])
    assert len(poisoned) == 30 and overlap.csv_bytes(poisoned) != overlap.csv_bytes(frozen[12:42])
    assert all(r["eligible"] == 0 for r in poisoned[:6])
    print(json.dumps({"prefix_sessions": split, "full_market_report_rows_compared": len(prefix),
                      "prefix_through": days[split-1], "prefix_sha256": snap.digest(overlap.csv_bytes(prefix)),
                      "poisoned_future_rows": len(poisoned), "first_poisoned_date": future[0]}))
