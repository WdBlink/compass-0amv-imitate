import csv
from dataclasses import replace
from fractions import Fraction
import json
from pathlib import Path
import subprocess
import sys

import pytest

from test_active_value_bound import make_snapshot, panel, records, series
from validation.mechanism_research_20261030 import acquisition_cost_bound as cost
from validation.mechanism_research_20261030 import active_value_bound as spot
from validation.mechanism_research_20261030 import market_snapshot as snapshot


def basis(volume, low, vwap, high, close, *, shares=100, symbol="600000.SH", factor=1):
    code, (d, b, a) = records(symbol, volume=volume, price=close, shares=shares, factor=factor)
    return code, ({**d, "low": low, "high": high, "amount": Fraction(vwap) * volume / 1000}, b, a)


def run(panels, windows=spot.WINDOWS):
    return list(cost.calculate_costs([p.date for p in panels], panels, windows=windows))


def values(row):
    return tuple(Fraction(row[k]) * 10**8 for k in ("low_yi", "vwap_yi", "high_yi"))


def test_two_day_partial_survival_cost_vs_spot_rolloff_and_zero_trades():
    days = [panel("20000103", [basis(40, 9, 10, 11, 10)]), panel("20000104", [basis(25, 18, 20, 22, 21)]),
            panel("20000105", [basis(0, 100, 105, 110, 109)])]
    rows = {(r["date"], r["window"]): r for r in run(days, (1, 2, 3)) if r["market"] == "SH"}
    # 40 shares bought yesterday leave 30; all 25 bought today survive.
    assert values(rows["2000-01-04", 2]) == (30*9 + 25*18, 30*10 + 25*20, 30*11 + 25*22)
    assert values(rows["2000-01-04", 2])[1] != (30 + 25) * 21
    assert values(rows["2000-01-05", 3]) == values(rows["2000-01-04", 2])
    assert rows["2000-01-05", 1]["eligible"] == 1 and values(rows["2000-01-05", 1]) == (0, 0, 0)
    assert values(rows["2000-01-05", 2]) == (25*18, 25*20, 25*22)


def test_fractional_subtick_prices_round_each_stock_outward_not_each_day():
    prices = (Fraction(1, 7), Fraction(2, 13), Fraction(3, 17))
    days = series(3, lambda i: [basis(1, prices[i], prices[i], prices[i], prices[i], shares=3, symbol=code)
                                for code in ("600000.SH", "300001.SZ")])
    row = run(days, (3,))[-1]
    exact = sum(p * Fraction(2, 3)**(2-i) for i, p in enumerate(prices)) * spot.TICKS_PER_CNY
    lower = (exact.numerator // exact.denominator) * 2
    upper = -(-exact.numerator // exact.denominator) * 2
    assert lower < 2 * exact < upper
    assert tuple(x * spot.TICKS_PER_CNY for x in values(row)) == (lower, lower, upper)
    flat = series(5, lambda _: [basis(20, 8, 8, 8, 8)])
    control = list(spot.calculate_bounds([p.date for p in flat], flat))[-3]
    same = run(flat)[-3]
    assert same["low_yi"] == same["vwap_yi"] == same["high_yi"] == control["lower_yi"]


@pytest.mark.parametrize("event,reason", [("factor", "action"), ("float", "action"), ("missing", "missing"),
                                         ("volume", "bad_q"), ("amount", "ohlc_vwap"), ("price", "bad_input"),
                                         ("different", "close_mismatch")])
def test_reuse_spot_membership_and_interruption_reasons(event, reason):
    def stocks(i):
        s = records(factor=2 if event == "factor" and i >= 5 else 1,
                    shares=2*10**8 if event == "float" and i >= 5 else 10**8)
        if i == 5:
            if event == "volume":
                s[1][0]["vol"] = 10**6
                s[1][0]["amount"] = 10**6
            elif event == "amount":
                s[1][0].update(vol=0, amount=1)
            elif event == "price":
                s[1][0]["low"] = "NaN"
            elif event == "different":
                s[1][1]["close"] = 11
        return [s, records("689009.SH"), records("900001.SH"), records("920001.BJ"),
                *([records("300001.SZ")] if i >= 5 else [])]

    days = series(12, stocks)
    if event == "missing":
        days[5] = replace(days[5], factors=[r for r in days[5].factors if r["ts_code"] != "600000.SH"])
    actual = run(days)
    expected = list(spot.calculate_bounds([p.date for p in days], days))
    for a, b in zip(actual, expected, strict=True):
        assert {k: a[k] for k in b if k != "lower_yi"} == {k: v for k, v in b.items() if k != "lower_yi"}
    assert (actual[30]["eligible"], actual[30]["ex_" + reason]) == (0, 1)
    assert (actual[33]["seen_asof"], actual[33]["ex_initial_or_new"]) == (2, 1)
    assert actual[-3]["eligible"] == 2


def test_prefix_poison_and_nonmarket_fields_cannot_change_prior_bytes():
    days = series(40)
    prefix = cost.csv_bytes(run(days[:34]))
    tail = series(40, lambda _: [records("300099.SZ", price=700)])
    assert cost.csv_bytes(run(days)[:34*6]) == prefix == cost.csv_bytes(run(days[:34] + tail[34:])[:34*6])
    for d in days[:34]:
        for r in d.daily:
            r.update(official_close="poison", indicator_change="unrelated")
    assert cost.csv_bytes(run(days[:34])) == prefix


def test_misaligned_reversed_duplicate_and_invalid_windows_fail_closed():
    days = series(3)
    dates = [d.date for d in days]
    for calendar, panels in ((dates, days[::-1]), (dates[::-1], days[::-1]), (dates, days[:-1]),
                             (dates[:1]*2, days[:1]*2), (dates[:1], [replace(days[0], daily=days[0].daily*2)])):
        with pytest.raises(ValueError):
            list(cost.calculate_costs(calendar, panels))
    for windows in ((), (0,), (True,), (3, 2), (2, 2)):
        with pytest.raises(ValueError):
            run(days, windows)


def test_cli_market_only_byte_freeze_fresh_output_and_authenticated_data(tmp_path):
    root = tmp_path / "snapshot"
    pin = make_snapshot(root)
    decoy = root / "official.csv"  # Synthetic poison, never a historical target.
    decoy.write_text("fabricated 123")
    module = "validation.mechanism_research_20261030.acquisition_cost_bound"
    guard = """import os, pathlib, runpy, sys
root, output = map(pathlib.Path, sys.argv[1:3]); module = sys.argv[3]
def audit(event, args):
    if event in ('socket.connect', 'socket.getaddrinfo'):
        raise RuntimeError('network access')
    if event == 'open' and isinstance(args[0], str) and isinstance(args[1], str) and 'r' in args[1]:
        path = pathlib.Path(args[0]).resolve()
        if path.suffix in ('.csv', '.json') and not (path in (root/'manifest.json', output) or path.is_relative_to(root/'raw')):
            raise RuntimeError('nonmarket data read')
sys.addaudithook(audit); sys.argv = [module, *sys.argv[4:]]; runpy.run_module(module, run_name='__main__')"""

    def call(output, receipt, sha=pin):
        return subprocess.run([sys.executable, "-B", "-c", guard, str(root), str(output), module, "--root", str(root),
                               "--manifest-sha256", sha, "--output", str(output), "--receipt", str(receipt)],
                              cwd=Path(__file__).parent, capture_output=True, text=True)

    out, proof = tmp_path/"pred.csv", tmp_path/"receipt.json"
    result = call(out, proof)
    assert result.returncode == 0, result.stderr
    frozen = out.read_bytes()
    receipt = json.loads(proof.read_bytes())
    assert receipt["prediction_sha256"] == snapshot.digest(frozen) and receipt["snapshot_manifest_sha256"] == pin
    assert receipt["started_at_utc"] <= receipt["csv_closed_at_utc"] <= receipt["receipt_at_utc"]
    with out.open() as file:
        rows = list(csv.DictReader(file))
    assert len(rows) == len({(r["date"], r["market"], r["window"]) for r in rows}) == 384
    assert (rows[0]["date"], rows[-1]["date"]) == ("2026-07-27", "2026-10-30")
    for r in rows:
        n = int(r["window"])
        exact = 10 * (1 - Fraction(9, 10)**n) * spot.TICKS_PER_YI
        stocks = 1 if r["market"] == "SH" else 2
        lo, hi = exact.numerator // exact.denominator, -(-exact.numerator // exact.denominator)
        assert tuple(x*spot.TICKS_PER_CNY for x in values(r)) == (stocks*lo, stocks*lo, stocks*hi)
    assert call(out, proof).returncode != 0 and out.read_bytes() == frozen
    decoy.write_text("every synthetic target replaced")
    other, other_proof = tmp_path/"other.csv", tmp_path/"other.json"
    assert call(other, other_proof).returncode == 0 and other.read_bytes() == frozen
    missing, receipt_path = tmp_path/"unused.csv", tmp_path/"unused.json"
    assert call(missing, receipt_path, "0" * 64).returncode != 0
    assert call(root/"leak.csv", receipt_path).returncode != 0 and not (root/"leak.csv").exists()
    (root/"raw"/"daily-20261030.json").write_bytes(b"corrupt")
    failure = call(missing, receipt_path)
    assert failure.returncode != 0 and "hash mismatch" in failure.stderr
    assert not missing.exists() and not receipt_path.exists()
