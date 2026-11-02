import csv
from dataclasses import replace
from datetime import date, datetime, timedelta
from decimal import Decimal
from fractions import Fraction
import io
import json
from pathlib import Path
import subprocess
import sys

import pytest

from research.mechanism_research_20261030 import active_value_bound as av
from research.mechanism_research_20261030 import market_snapshot as snap


def records(symbol="600000.SH", *, price=10, shares=10**8, volume=10**7, factor=1):
    price, shares, volume = map(Fraction, (price, shares, volume))
    d = dict(open=price, high=price, low=price, close=price, vol=volume / 100, amount=price * volume / 1000)
    b = dict(close=price, float_share=shares / 10000, total_share=shares / 10000, free_share=0,
             turnover_rate=0, total_mv=0, circ_mv=0)
    return symbol, (d, b, {"adj_factor": factor})


def panel(day, stocks):
    groups = [[], [], []]
    for symbol, rows in stocks:
        for group, row in zip(groups, rows, strict=True):
            group.append({**row, "ts_code": symbol, "trade_date": day})
    return av.Panel(day, *groups)


def series(n, make=lambda i: [records()]):
    return [panel((date(2026, 1, 1) + timedelta(days=i)).strftime("%Y%m%d"), make(i)) for i in range(n)]


def rows(panels):
    return list(av.calculate_bounds([p.date for p in panels], panels))


def at(result, day, n=5, market="SH"):
    key = datetime.strptime(day, "%Y%m%d").date().isoformat()
    return next(r for r in result if (r["date"], r["market"], r["window"]) == (key, market, n))


def test_exact_values_window_rolloff_current_valuation_and_zero_activity():
    ps = series(35, lambda i: [records(volume=5*10**7 if i == 0 else 0, price=12 if i == 4 else 10)])
    out = rows(ps)
    assert len(out) == 35 * 6
    for n in (5, 13, 34):
        prior = at(out, ps[n-2].date, n)
        assert (prior["eligible"], prior["ex_initial_or_new"]) == (0, 1)
        first = at(out, ps[n-1].date, n)
        assert (first["eligible"], Fraction(first["lower_yi"])) == (1, 6 if n == 5 else 5)
        after = at(out, ps[n].date, n)
        assert (after["eligible"], Fraction(after["lower_yi"])) == (1, 0)


def test_exact_expanded_cohort_oracle_and_per_stock_floor_even_at_low_decimal_precision():
    volumes = (1, 0, 2, 1, 3)
    ps = series(5, lambda i: [records("600000.SH", price=1, shares=7, volume=volumes[i]),
                             records("000001.SZ", price=3, shares=11, volume=volumes[4-i])])
    from decimal import localcontext
    with localcontext() as ctx:
        ctx.prec = 2
        out = rows(ps)

    def cohorts(shares, price, vols):
        live = []
        for v in vols:
            live = [old * (1 - Fraction(v, shares)) for old in live] + [Fraction(v, shares)]
        return sum(live) * price * shares / 10**8

    exact = [cohorts(7, 1, volumes), cohorts(11, 3, volumes[::-1])]
    ticks = 10**12
    sh = Fraction(at(out, ps[-1].date)["lower_yi"])
    total = Fraction(at(out, ps[-1].date, market="SH_SZ")["lower_yi"])
    assert sh == Fraction(int(exact[0] * ticks), ticks)
    assert total == Fraction(sum(int(x*ticks) for x in exact), ticks)
    assert 0 < sum(exact) - total < Fraction(2, ticks)


def test_monotone_turnover_price_windows_and_market_on_same_valid_population():
    def value(volume, price):
        ps = series(34, lambda _: [records(volume=volume, price=price), records("300001.SZ", volume=volume, price=price)])
        output = rows(ps)
        return [Fraction(at(output, ps[-1].date, n, m)["lower_yi"]) for m in ("SH", "SH_SZ") for n in av.WINDOWS]
    baseline, more, dearer, zero = value(10**6, 6), value(3*10**6, 6), value(10**6, 12), value(0, 6)
    assert all(0 < a < b and a < c for a, b, c in zip(baseline, more, dearer))
    assert baseline[0] < baseline[1] < baseline[2]
    assert baseline[3:] == [v * 2 for v in baseline[:3]]
    assert zero == [0] * 6


@pytest.mark.parametrize(("which", "changes", "reason"), [
    ("daily", {"open": 12}, "ohlc_vwap"), ("daily", {"amount": Decimal("99999.99999999999")}, "ohlc_vwap"),
    ("daily", {"vol": 0}, "ohlc_vwap"), ("basic", {"close": 11}, "close_mismatch"),
    ("basic", {"float_share": 1000}, "bad_q"), ("basic", {"float_share": 0}, "bad_q"),
    ("daily", {"open": float("nan")}, "bad_input"), ("daily", {"vol": -1}, "bad_input"),
    ("factors", {"adj_factor": 0}, "bad_input"), ("factors", {"adj_factor": True}, "bad_input"),
])
def test_single_invalid_day_rejects_whole_window_without_dropping_dates(which, changes, reason):
    ps = series(12)
    getattr(ps[6], which)[0].update(changes)
    out = rows(ps)
    assert at(out, ps[5].date)["eligible"] == 1
    for i in range(6, 11):
        r = at(out, ps[i].date)
        assert (r["eligible"], r["excluded"], r["ex_"+reason], Fraction(r["lower_yi"])) == (0, 1, 1, 0)
    assert at(out, ps[11].date)["eligible"] == 1


@pytest.mark.parametrize("endpoint", ["daily", "basic", "factors", "all"])
def test_absent_observation_not_zero_or_suspension_and_then_recovery(endpoint):
    ps = series(12)
    missing = ("daily", "basic", "factors") if endpoint == "all" else (endpoint,)
    ps[6] = replace(ps[6], **{k: [] for k in missing})
    out = rows(ps)
    assert all(at(out, ps[i].date)["ex_missing"] == 1 for i in range(6, 11))
    assert at(out, ps[11].date)["eligible"] == 1
    zero = series(12, lambda i: [records(volume=0 if i == 6 else 10**7)])
    assert at(rows(zero), zero[6].date)["eligible"] == 1


@pytest.mark.parametrize("what", ["float", "factor"])
def test_corporate_action_only_excludes_windows_which_span_it(what):
    ps = series(12, lambda i: [records(shares=2*10**8 if what == "float" and i >= 6 else 10**8,
                                       factor=2 if what == "factor" and i >= 6 else 1)])
    out = rows(ps)
    assert at(out, ps[5].date)["eligible"] == 1
    assert all(at(out, ps[i].date)["ex_action"] == 1 for i in range(6, 10))
    assert at(out, ps[10].date)["eligible"] == 1


def test_q_boundaries_and_only_confirmed_share_families():
    candidates = ("600000.SH", "688001.SH", "689009.SH", "900901.SH", "000001.SZ", "301001.SZ",
                  "300001.SH", "600000.SZ", "200001.SZ", "830001.BJ")
    ps = series(5, lambda i: [records(s, shares=10**8, volume=10**8-1) for s in candidates])
    out = rows(ps)
    assert (at(out, ps[-1].date)["eligible"], at(out, ps[-1].date, market="SH_SZ")["eligible"]) == (2, 4)
    assert at(out, ps[-1].date, market="SH_SZ")["seen_asof"] == 4
    over = series(5, lambda i: [records(shares=10**8, volume=10**8+1)])
    assert at(rows(over), over[-1].date)["ex_bad_q"] == 1


def test_future_introductions_exit_and_poison_cannot_change_prefix_bytes():
    ps = series(42, lambda i: [records(), *([records("300001.SZ")] if i >= 37 else [])])
    prefix = av.csv_bytes(rows(ps[:35]))
    output = rows(ps)
    assert av.csv_bytes(output[:35*6]) == prefix
    assert at(output, ps[37].date, market="SH_SZ")["ex_initial_or_new"] == 1
    assert at(output, ps[41].date, market="SH_SZ")["eligible"] == 2
    poison = series(42, lambda i: [records()] if i < 35 else [records("000001.SZ", price=99, shares=10**8, volume=9*10**7)])
    other = rows(poison)
    assert av.csv_bytes(other[:35*6]) == prefix
    assert av.csv_bytes(other[35*6:]) != av.csv_bytes(output[35*6:])
    assert at(other, poison[-1].date, market="SH_SZ")["ex_missing"] == 1


@pytest.mark.parametrize("endpoint", ["daily", "basic", "factors"])
def test_leaks_and_duplicate_keys_rejected_even_outside_eligible_universe(endpoint):
    ps = series(2)
    getattr(ps[0], endpoint).append({"ts_code": "830001.BJ", "trade_date": ps[1].date})
    with pytest.raises(ValueError, match="outside"):
        rows(ps)
    ps = series(2)
    getattr(ps[0], endpoint).append(getattr(ps[0], endpoint)[0])
    with pytest.raises(ValueError, match="duplicate"):
        rows(ps)


def test_calendar_cannot_skip_reorder_duplicate_or_misdate_a_panel():
    ps = series(5)
    dates = [p.date for p in ps]
    for calendar, data in ((dates, ps[:-1]), (dates[:-1], ps), (dates, ps[::-1]), (dates[:1]*5, ps[:1]*5),
                           (dates, [*ps[:2], *ps[3:]])):
        with pytest.raises(ValueError):
            list(av.calculate_bounds(calendar, data))


def make_snapshot(root):
    root.mkdir()
    (root / "raw").mkdir()
    first, last = (datetime.strptime(value, "%Y%m%d").date() for value in (snap.START, snap.END))
    civil = [(first + timedelta(days=i)).strftime("%Y%m%d") for i in range((last-first).days+1)]
    weekdays = [v for v in civil if datetime.strptime(v, "%Y%m%d").date().weekday() < 5]
    sessions = [v for v in weekdays if v < av.REPORT_START][-57:]
    sessions += [v for v in weekdays if v >= av.REPORT_START][:63] + [av.REPORT_END]
    manifest = dict(requested_start=snap.START, requested_end=snap.END, fields=snap.FIELDS, groups={})

    def write_group(ep, key, params, items):
        fields = snap.FIELDS[ep].split(",")
        raw = json.dumps({"code": 0, "data": {"fields": fields, "items": [[str(x[k].numerator)
                         if isinstance(x[k], Fraction) and x[k].denominator == 1 else x[k] for k in fields]
                         for x in items]}}).encode()
        name = f"raw/{ep}-{key}.json"
        (root / name).write_bytes(raw)
        page = dict(file=name, rows=len(items), offset=0, sha256=snap.digest(raw))
        manifest["groups"][f"{ep}/{key}"] = dict(endpoint=ep, params=params, pages=[page], rows=len(items), status="complete")

    for ex in ("SSE", "SZSE"):
        calendar = [dict(exchange=ex, cal_date=d, is_open=int(d in sessions), pretrade_date=None) for d in civil]
        write_group("trade_cal", ex, dict(exchange=ex, start_date=snap.START, end_date=snap.END), calendar)
    for day in sessions:
        p = panel(day, [records(), records("300001.SZ")])
        for ep, items in zip(av.ENDPOINTS, [p.daily, p.basic, p.factors]):
            write_group(ep, day, dict(trade_date=day), items)
    for status in "LDP":
        write_group("stock_basic", status, dict(list_status=status), [])
    data = json.dumps(manifest).encode()
    (root / "manifest.json").write_bytes(data)
    return snap.digest(data)


def test_real_cli_freezes_then_refuses_overwrite_corruption_and_nonmarket_file_reads(tmp_path):
    root = tmp_path / "snapshot"
    digest = make_snapshot(root)
    decoy = root / "official.csv"
    decoy.write_text("unrelated numbers")
    module = "research.mechanism_research_20261030.active_value_bound"
    # Abort on any attempt to read or write the decoy. Both runs must also be byte-identical.
    guarded = """import os, runpy, sys
decoy, module = sys.argv[1:3]
def guard(event, args):
    if event == 'open' and isinstance(args[0], str) and os.path.realpath(args[0]) == decoy:
        raise RuntimeError('attempt to access the non-market file')
sys.addaudithook(guard)
sys.argv = [module, *sys.argv[3:]]
runpy.run_module(module, run_name='__main__')
"""
    def call(out, receipt, pin=digest, guard=False):
        command = [sys.executable, "-B", "-c", guarded, str(decoy), module] if guard else [sys.executable, "-B", "-m", module]
        return subprocess.run([*command, "--root", str(root), "--manifest-sha256", pin,
                               "--output", str(out), "--receipt", str(receipt)],
                              cwd=Path(__file__).resolve().parents[1], capture_output=True, text=True)

    out, receipt = tmp_path/"bound.csv", tmp_path/"receipt.json"
    done = call(out, receipt, guard=True)
    assert done.returncode == 0, done.stderr
    frozen = out.read_bytes()
    proof = json.loads(receipt.read_bytes())
    assert proof["bounds_sha256"] == snap.digest(frozen)
    assert proof["snapshot_manifest_sha256"] == digest and proof["csv_closed_at_utc"] <= proof["receipt_at_utc"]
    assert "--manifest-sha256" in proof["command"] and module in proof["command"]
    values = list(csv.DictReader(io.StringIO(frozen.decode())))
    assert len(values) == 384 and (values[0]["date"], values[-1]["date"]) == ("2026-07-27", "2026-10-30")
    for v in values:
        n = int(v["window"])
        exact = 10 * (1 - Fraction(9, 10)**n)
        nsh = 1 if v["market"] == "SH" else 2
        assert (int(v["eligible"]), int(v["excluded"])) == (nsh, 0)
        assert Fraction(v["lower_yi"]) == nsh * Fraction(int(exact*10**12), 10**12)
    before = (out.read_bytes(), receipt.read_bytes())
    assert call(out, receipt).returncode != 0
    assert (out.read_bytes(), receipt.read_bytes()) == before
    decoy.write_text("poison: all labels replaced")
    out2, receipt2 = tmp_path/"fresh.csv", tmp_path/"fresh.json"
    again = call(out2, receipt2, guard=True)
    assert again.returncode == 0 and out2.read_bytes() == frozen, again.stderr
    unused, unused_receipt = tmp_path/"unused", tmp_path/"unused.json"
    assert call(unused, unused_receipt, "0"*64).returncode != 0
    assert call(root/"must_not_write", unused_receipt).returncode != 0 and not (root/"must_not_write").exists()
    (root/"raw"/"daily-20261030.json").write_bytes(b"damaged input")
    bad = call(unused, unused_receipt)
    assert bad.returncode != 0 and "hash mismatch" in bad.stderr
    assert not unused.exists() and not unused_receipt.exists()
