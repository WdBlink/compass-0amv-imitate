import json
from decimal import Decimal

import pytest

from validation.mechanism_research_20261030 import market_snapshot as m


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
