import json
import datetime as dt
import sys
from copy import deepcopy
from decimal import Decimal

import pytest

from research.mechanism_research_20261030 import market_snapshot as m


def daily(**updates):
    row = dict(ts_code="000001.SZ", trade_date=m.END, open=10, high=12, low=9, close=11, vol=100, amount=105)
    return {**row, **updates}


def payload(endpoint, rows):
    fields = m.FIELDS[endpoint].split(",")
    return json.dumps({"code": 0, "data": {"fields": fields, "items": [[r[k] for k in fields] for r in rows]}}).encode()


def test_exact_unit_conversions_and_unadjusted_close():
    row = m.parse_response(payload("daily", [daily()]), "daily", {"trade_date": m.END})[0]
    assert m.normalize_daily(row) == {"symbol": "000001.SZ", "date": m.END, "raw_close": Decimal(11),
                                      "volume_shares": Decimal(10000), "amount_cny": Decimal(105000)}
    assert "adj_factor" not in m.normalize_daily(row)
    basic = {key: "1.2345" for key in ("total_share", "float_share", "free_share", "total_mv", "circ_mv")}
    assert set(m.normalize_basic(basic).values()) == {Decimal(12345)}


@pytest.mark.parametrize("updates", [{"vol": float("nan")}, {"amount": float("inf")}, {"close": None},
                                    {"open": True}, {"amount": -1}, {"close": 0}, {"trade_date": "20261029"},
                                    {"ts_code": "not-a-stock"}, {"trade_date": "20260230"}])
def test_bad_numeric_dates_or_symbols_fail(updates):
    with pytest.raises(ValueError):
        m.parse_response(payload("daily", [daily(**updates)]), "daily", {"trade_date": m.END})


def test_duplicate_and_malformed_rows_fail():
    with pytest.raises(ValueError, match="duplicate"):
        m.parse_response(payload("daily", [daily(), daily()]), "daily", {"trade_date": m.END})
    bad = json.loads(payload("daily", [daily()]))
    bad["data"]["items"][0].pop()
    with pytest.raises(ValueError, match="malformed row"):
        m.parse_response(json.dumps(bad).encode(), "daily", {"trade_date": m.END})
    bad["data"]["fields"][0] = "unexpected"
    with pytest.raises(ValueError, match="schema"):
        m.parse_response(json.dumps(bad).encode(), "daily", {"trade_date": m.END})


def test_pagination_needs_terminal_short_page_and_unique_keys():
    a, b = daily(), daily(ts_code="600000.SH")
    with pytest.raises(ValueError, match="truncated"):
        m.combine_pages([[a, b]], "daily", page_size=2)
    assert m.combine_pages([[a, b], []], "daily", page_size=2) == [a, b]
    with pytest.raises(ValueError, match="duplicate"):
        m.combine_pages([[a, b], [a]], "daily", page_size=2)
    with pytest.raises(ValueError, match="sequence"):
        m.combine_pages([[a], []], "daily", page_size=2)


def test_bounds_fail_without_noise_tuned_tolerance():
    assert m.bar_issues(daily()) == []
    assert m.bar_issues(daily(amount=90)) == []
    assert m.bar_issues(daily(amount=120)) == []
    assert m.bar_issues(daily(close=13)) == ["ohlc_bounds"]
    assert m.bar_issues(daily(amount=120.001)) == ["vwap_outside_ohlc"]
    assert m.bar_issues(daily(vol=0, amount=0)) == []
    assert m.bar_issues(daily(vol=0)) == ["amount_without_volume"]


def test_cross_endpoint_gaps_are_not_suspensions_or_full_market():
    metadata = [dict(ts_code="000001.SZ", list_date="19910403", delist_date=None, market="主板"),
                dict(ts_code="600000.SH", list_date="19991110", delist_date=None, market="主板")]
    report = m.audit_day([daily()], [], [], metadata, m.END)
    assert report["missing_endpoint_keys"] == {"daily_basic": ["000001.SZ"], "adj_factor": ["000001.SZ"]}
    assert report["missing_daily_vs_metadata"] == ["600000.SH"]
    assert report["amount_cny_sum"] == "105000"
    assert report["numeric_panel_valid"] is False
    assert report["all_market_completeness_proven"] is False
    assert "suspensions" not in report


def test_listing_dates_bound_snapshot_membership():
    metadata = [dict(ts_code="000001.SZ", list_date="20261031", delist_date=None, market="主板"),
                dict(ts_code="600000.SH", list_date="19991110", delist_date="20261029", market="主板")]
    report = m.audit_day([daily()], [daily()], [daily()], metadata, m.END)
    assert report["metadata_asof_eligible"] == 0
    assert report["daily_not_in_metadata_asof"] == ["000001.SZ"]


def test_private_raw_hash_replay_and_partial_refusal(tmp_path):
    raw = payload("daily", [daily()])
    (tmp_path / "raw").mkdir()
    (tmp_path / "raw" / "test.json").write_bytes(raw)
    page = dict(file="raw/test.json", sha256=m.digest(raw), rows=1, offset=0)
    group = dict(endpoint="daily", params={"trade_date": m.END}, status="complete", pages=[page], rows=1)
    manifest = {"groups": {"daily/20261030": group}}
    assert len(m.load_group(tmp_path, manifest, "daily/20261030")) == 1
    group["status"] = "partial"
    with pytest.raises(ValueError, match="partial"):
        m.load_group(tmp_path, manifest, "daily/20261030")
    group["status"] = "complete"
    (tmp_path / "raw" / "test.json").write_bytes(raw + b" ")
    with pytest.raises(ValueError, match="hash"):
        m.load_group(tmp_path, manifest, "daily/20261030")


def test_raw_path_escape_is_refused(tmp_path):
    with pytest.raises(ValueError, match="outside"):
        m.read_page(tmp_path, "daily", {}, {"file": "../outside.json"})


def test_retired_metadata_symbol_is_not_a_trading_symbol():
    row = dict(ts_code="T600018.SH", symbol="T600018", name="historical", market="主板", exchange="SSE",
               curr_type="CNY", list_status="D", list_date="20000719", delist_date="20061020")
    assert m.parse_response(payload("stock_basic", [row]), "stock_basic", {"list_status": "D"})[0] == row
    with pytest.raises(ValueError, match="symbol"):
        m.parse_response(payload("daily", [daily(ts_code="T600018.SH")]), "daily", {"trade_date": m.END})


def test_loader_refuses_missing_endpoint_group(tmp_path):
    (tmp_path / "manifest.json").write_text(json.dumps({"groups": {}}))
    with pytest.raises(ValueError, match="missing endpoint"):
        m.load_daily_panel(tmp_path, m.END)


def test_network_budget_prevents_even_a_request(tmp_path, monkeypatch):
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-value")
    collector = m.Collector(tmp_path)
    collector.manifest["successful_calls"] = 450
    assert collector.get("daily", m.END, {"trade_date": m.END}) is None
    assert collector.manifest["requests"] == []
    assert collector.manifest["groups"][f"daily/{m.END}"]["stop_reason"] == "bounded_budget"


def test_transport_failure_is_sanitized_and_not_retried(tmp_path, monkeypatch):
    token = "test-secret-value"
    monkeypatch.setenv("TUSHARE_TOKEN", token)
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    collector = m.Collector(tmp_path)
    class FailingOpener:
        count = 0

        def open(self, *args, **kwargs):
            self.count += 1
            raise ValueError("unsafe provider diagnostic " + token)
    opener = FailingOpener()
    collector.opener = opener
    assert collector.get("daily", m.END, {"trade_date": m.END}) is None
    assert collector.get("daily", m.END, {"trade_date": m.END}) is None
    assert opener.count == 1
    assert token not in (tmp_path / "manifest.json").read_text()


def test_redirect_is_refused_before_credentials_can_leave_provider():
    with pytest.raises(ValueError, match="redirect"):
        m.NoRedirect().redirect_request(None, None, 302, "", {}, "https://example.org")


@pytest.mark.parametrize("code", [False, None, "0", 1001])
def test_provider_code_must_be_numeric_success(code):
    raw = json.loads(payload("daily", [daily()]))
    raw["code"] = code
    with pytest.raises(ValueError, match="unsuccessful"):
        m.parse_response(json.dumps(raw).encode(), "daily", {"trade_date": m.END})


def test_malformed_calendar_open_flag():
    row = dict(exchange="SSE", cal_date=m.END, is_open=True, pretrade_date="20261029")
    with pytest.raises(ValueError, match="open flag"):
        m.parse_response(payload("trade_cal", [row]), "trade_cal", {"exchange": "SSE"})


def test_loader_checks_pagination_offsets(tmp_path):
    group = dict(endpoint="daily", params={"trade_date": m.END}, status="complete",
                 pages=[{"offset": 6000, "rows": 1}], rows=1)
    with pytest.raises(ValueError, match="offset"):
        m.load_group(tmp_path, {"groups": {"daily/20261030": group}}, "daily/20261030")


def panel_rows(day):
    return {
        "daily": [daily(trade_date=day)],
        "daily_basic": [dict(ts_code="000001.SZ", trade_date=day, close=11, turnover_rate=1,
                             total_share=100, float_share=50, free_share=40, total_mv=1100, circ_mv=550)],
        "adj_factor": [dict(ts_code="000001.SZ", trade_date=day, adj_factor=2)],
    }


@pytest.fixture
def offline_collector(tmp_path, monkeypatch):
    class Response:
        status = 200

        def __init__(self, raw):
            self.raw = raw

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self, size):
            return self.raw[:size]

    class Opener:
        def __init__(self):
            self.calls = []
            self.responses = {}

        def open(self, request, timeout):
            body = json.loads(request.data)
            endpoint, params = body["api_name"], body["params"]
            self.calls.append((endpoint, params))
            selector = params.get("trade_date", params.get("exchange", params.get("list_status")))
            raw = self.responses.get((endpoint, selector))
            if raw is None:
                raw = payload(endpoint, panel_rows(selector)[endpoint])
            return Response(raw)

    opener = Opener()
    monkeypatch.setenv("TUSHARE_TOKEN", "test-only-value")
    monkeypatch.setattr(m.time, "sleep", lambda _: None)
    monkeypatch.setattr(m.urllib.request, "build_opener", lambda *_: opener)
    return m.Collector(tmp_path), opener


def collect_panel(collector, day):
    for endpoint in ("daily", "daily_basic", "adj_factor"):
        assert collector.get(endpoint, day, {"trade_date": day}) is not None


def test_collector_to_strict_panel_keeps_units_and_raw_date(offline_collector, tmp_path):
    collector, opener = offline_collector
    collect_panel(collector, m.END)
    assert len(opener.calls) == 3
    assert len(list((tmp_path / "raw").iterdir())) == 3
    assert m.load_daily_panel(tmp_path, m.END) == [dict(
        symbol="000001.SZ", date=m.END, raw_close=Decimal(11), volume_shares=Decimal(10000),
        amount_cny=Decimal(105000), total_share=Decimal(1000000), float_share=Decimal(500000),
        free_share=Decimal(400000), total_mv=Decimal(11000000), circ_mv=Decimal(5500000), adj_factor=Decimal(2))]


@pytest.mark.parametrize("endpoint,selector,params", [
    ("daily", "20261029", {"trade_date": m.END}),
    ("daily_basic", m.END, {"trade_date": m.END, "ts_code": "000001.SZ"}),
    ("adj_factor", m.END, {}),
    ("daily", "20260430", {"trade_date": "20260430"}),
    ("stock_basic", "L", {"list_status": "D"}),
    ("stock_basic", "L", {"list_status": "L", "exchange": "SSE"}),
    ("trade_cal", "SSE", {"exchange": "SZSE", "start_date": m.START, "end_date": m.END}),
    ("trade_cal", "SSE", {"exchange": "SSE", "start_date": m.END, "end_date": m.END}),
    ("unknown", m.END, {"trade_date": m.END}),
])
def test_collector_rejects_selector_and_restricted_scope_before_request(offline_collector, endpoint, selector, params):
    collector, opener = offline_collector
    with pytest.raises(ValueError):
        collector.get(endpoint, selector, params)
    assert opener.calls == []
    assert collector.manifest["groups"] == {}


@pytest.mark.parametrize("change", ["key", "endpoint", "params", "scope"])
def test_manifest_request_binding_rejects_unchanged_raw(offline_collector, tmp_path, change):
    collector, opener = offline_collector
    collect_panel(collector, m.END)
    original_raw = {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()}
    original_pages = deepcopy([g["pages"] for g in collector.manifest["groups"].values()])
    groups = collector.manifest["groups"]
    day = m.END
    if change == "key":
        day = "20261029"
        collector.manifest["groups"] = {key.replace(m.END, day): value for key, value in groups.items()}
    elif change == "endpoint":
        groups[f"daily/{day}"]["endpoint"] = "daily_basic"
    elif change == "params":
        groups[f"daily/{day}"]["params"]["trade_date"] = "20261029"
    else:
        groups[f"daily/{day}"]["params"]["ts_code"] = "000001.SZ"
    m.save_manifest(tmp_path, collector.manifest)
    with pytest.raises(ValueError, match="mismatch|scope"):
        m.load_daily_panel(tmp_path, day)
    with pytest.raises(ValueError, match="mismatch|scope"):
        m.audit_snapshot(tmp_path)
    with pytest.raises(ValueError, match="mismatch|scope"):
        collector.get("daily", day, {"trade_date": day})
    assert len(opener.calls) == 3
    assert original_raw == {p.name: p.read_bytes() for p in (tmp_path / "raw").iterdir()}
    assert original_pages == [g["pages"] for g in collector.manifest["groups"].values()]


@pytest.mark.parametrize("endpoint", ["daily", "daily_basic", "adj_factor"])
def test_wrong_date_in_any_endpoint_is_rejected_before_join(offline_collector, tmp_path, endpoint):
    collector, _ = offline_collector
    collect_panel(collector, m.END)
    group = collector.manifest["groups"][f"{endpoint}/{m.END}"]
    page = group["pages"][0]
    rows = panel_rows(m.END)[endpoint] + [dict(panel_rows("20261029")[endpoint][0], ts_code="600000.SH")]
    raw = payload(endpoint, rows)
    (tmp_path / page["file"]).write_bytes(raw)
    page.update(sha256=m.digest(raw), rows=2)
    group["rows"] = 2
    m.save_manifest(tmp_path, collector.manifest)
    with pytest.raises(ValueError, match="trade date request mismatch"):
        m.load_daily_panel(tmp_path, m.END)


@pytest.mark.parametrize("failure", ["missing_file", "missing_endpoint", "partial", "reordered", "duplicate", "corrupt"])
def test_collected_panel_retains_fail_closed_replay(offline_collector, tmp_path, failure):
    collector, _ = offline_collector
    collect_panel(collector, m.END)
    group = collector.manifest["groups"][f"daily/{m.END}"]
    page = group["pages"][0]
    path = tmp_path / page["file"]
    if failure == "missing_file":
        path.unlink()
    elif failure == "missing_endpoint":
        del collector.manifest["groups"][f"daily_basic/{m.END}"]
    elif failure == "partial":
        group["status"] = "partial"
    elif failure == "reordered":
        page["offset"] = 6000
    elif failure == "duplicate":
        raw = payload("daily", [daily(), daily()])
        path.write_bytes(raw)
        page.update(sha256=m.digest(raw), rows=2)
        group["rows"] = 2
    else:
        path.write_bytes(path.read_bytes() + b" ")
    m.save_manifest(tmp_path, collector.manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        m.load_daily_panel(tmp_path, m.END)


@pytest.mark.parametrize("flag,expected", [(1, 1), (1.0, 1), ("1", 1), (0, 0), (0.0, 0), ("0", 0)])
def test_calendar_flags_audit_and_continue_use_same_open_days(offline_collector, tmp_path, monkeypatch, flag, expected):
    collector, opener = offline_collector
    first = dt.datetime.strptime(m.START, "%Y%m%d").date()
    last = dt.datetime.strptime(m.END, "%Y%m%d").date()
    days = [(first + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((last - first).days + 1)]
    flags = []
    for exchange in ("SSE", "SZSE"):
        rows = [dict(exchange=exchange, cal_date=day, is_open=flag if day == m.END else 0,
                     pretrade_date="20260430") for day in days]
        opener.responses[("trade_cal", exchange)] = payload("trade_cal", rows)
        result = collector.get("trade_cal", exchange,
                               {"exchange": exchange, "start_date": m.START, "end_date": m.END})
        flags.extend([result[-1]["is_open"],
                      m.load_group(tmp_path, collector.manifest, f"trade_cal/{exchange}")[-1]["is_open"]])
    report = m.audit_snapshot(tmp_path)
    requested = [m.END] if expected else []
    assert report["calendar_missing_dates"] == {"SSE": [], "SZSE": []}
    assert report["expected_open_days"] == requested
    assert report["missing_requested_dates"] == {e: requested for e in ("daily", "daily_basic", "adj_factor")}
    assert report["actual_endpoint_dates"] == {e: [] for e in ("daily", "daily_basic", "adj_factor")}
    assert all(type(value) is int and value == expected for value in flags)
    (tmp_path / "pilot-audit.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["market_snapshot", "continue", "--root", str(tmp_path)])
    assert m.main() == 0
    assert [(endpoint, params["trade_date"]) for endpoint, params in opener.calls[2:]] == [
        (endpoint, m.END) for endpoint in ("daily", "daily_basic", "adj_factor") if expected]
    after = json.loads((tmp_path / "coverage-audit.json").read_text())
    assert after["expected_open_days"] == requested
    assert all(not missing for missing in after["missing_requested_dates"].values())


@pytest.mark.parametrize("flag", [True, False, 2, -1, 0.5, "1.0", "yes", None])
def test_invalid_calendar_flag_blocks_replay_and_continuation(offline_collector, tmp_path, monkeypatch, flag):
    collector, opener = offline_collector
    row = dict(exchange="SSE", cal_date=m.END, is_open=flag, pretrade_date="20261029")
    opener.responses[("trade_cal", "SSE")] = payload("trade_cal", [row])
    assert collector.get("trade_cal", "SSE", {"exchange": "SSE", "start_date": m.START, "end_date": m.END}) is None
    assert collector.manifest["groups"]["trade_cal/SSE"]["status"] == "partial"
    assert len(list((tmp_path / "raw").iterdir())) == 1
    with pytest.raises(ValueError, match="partial"):
        m.load_group(tmp_path, collector.manifest, "trade_cal/SSE")
    report = m.audit_snapshot(tmp_path)
    assert report["blocked_endpoints"] == {"trade_cal": "ValueError"}
    assert report["calendar_exchange_agreement"] is False
    (tmp_path / "pilot-audit.json").write_text(json.dumps(report))
    monkeypatch.setattr(sys, "argv", ["market_snapshot", "continue", "--root", str(tmp_path)])
    with pytest.raises(ValueError, match="partial"):
        m.main()
    assert len(opener.calls) == 1


def test_consumer_prefix_is_unchanged_by_future_tail_poison(offline_collector, tmp_path):
    collector, _ = offline_collector
    prefix_days = ["20261028", "20261029"]
    for day in prefix_days:
        collect_panel(collector, day)
    prefix = [m.load_daily_panel(tmp_path, day) for day in prefix_days]
    collect_panel(collector, m.END)
    assert prefix == [m.load_daily_panel(tmp_path, day) for day in prefix_days]
    for endpoint in ("daily", "daily_basic", "adj_factor"):
        group = collector.manifest["groups"][f"{endpoint}/{m.END}"]
        for page in group["pages"]:
            (tmp_path / page["file"]).write_bytes(b"poisoned future input")
    assert prefix == [m.load_daily_panel(tmp_path, day) for day in prefix_days]
    with pytest.raises(ValueError, match="hash"):
        m.load_daily_panel(tmp_path, m.END)
