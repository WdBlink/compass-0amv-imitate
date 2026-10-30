"""Private, resumable Tushare snapshot collection and network-free input audit.

No official indicator labels are read. Raw close stays unadjusted; factors are separate.
"""
from __future__ import annotations

import argparse
from collections import Counter
import datetime as dt
from decimal import Decimal, InvalidOperation
import hashlib
import json
import os
from pathlib import Path
import re
import time
import urllib.request


START, END = "20260501", "20261030"
BASE = "https://api.waditu.com/dataapi"
SDK_BASE = "http://api.waditu.com/dataapi"
CAP = 6000
FIELDS = {
    "daily": "ts_code,trade_date,open,high,low,close,vol,amount",
    "daily_basic": "ts_code,trade_date,close,turnover_rate,total_share,float_share,free_share,total_mv,circ_mv",
    "adj_factor": "ts_code,trade_date,adj_factor",
    "trade_cal": "exchange,cal_date,is_open,pretrade_date",
    "stock_basic": "ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date",
}
DOCS = {name: f"https://tushare.pro/document/2?doc_id={number}" for name, number in
        zip(FIELDS, (27, 32, 28, 26, 25))}
UNITS = {"daily.vol": "hands; normalized shares = vol * 100",
         "daily.amount": "thousand CNY; normalized CNY = amount * 1000",
         "daily.close": "unadjusted CNY/share; never replaced with adjusted close",
         "daily_basic.share_fields": "10,000 shares; normalized shares = value * 10000",
         "daily_basic.mv_fields": "10,000 CNY; normalized CNY = value * 10000",
         "daily_basic.turnover_rate": "percent", "adj_factor.adj_factor": "dimensionless; separate"}


def digest(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).isoformat()


def valid_date(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"\d{8}", value):
        raise ValueError("invalid date")
    dt.datetime.strptime(value, "%Y%m%d")
    return value


def number(value: object) -> Decimal:
    if isinstance(value, bool):
        raise ValueError("boolean numeric value")
    try:
        result = Decimal(str(value))
    except (InvalidOperation, TypeError):
        raise ValueError("malformed numeric value") from None
    if not result.is_finite():
        raise ValueError("nonfinite numeric value")
    return result


def row_key(endpoint: str, row: dict) -> tuple:
    if endpoint == "trade_cal":
        return row["exchange"], row["cal_date"]
    return (row["ts_code"],) if endpoint == "stock_basic" else (row["ts_code"], row["trade_date"])


def parse_response(raw: bytes, endpoint: str, params: dict) -> list[dict]:
    """Validate provider structure, numeric values, requested scope and uniqueness."""
    payload = json.loads(raw, parse_float=Decimal)
    if not isinstance(payload, dict) or type(payload.get("code")) is not int or payload.get("code") != 0:
        raise ValueError("unsuccessful provider response")
    data = payload.get("data")
    if not isinstance(data, dict):
        raise ValueError("malformed data")
    fields, items = data.get("fields"), data.get("items")
    expected = FIELDS[endpoint].split(",")
    if not isinstance(fields, list) or len(fields) != len(expected) or set(fields) != set(expected):
        raise ValueError("unexpected schema")
    if not isinstance(items, list):
        raise ValueError("malformed items")
    result, keys = [], set()
    for item in items:
        if not isinstance(item, list) or len(item) != len(fields):
            raise ValueError("malformed row")
        row = dict(zip(fields, item))
        if endpoint == "trade_cal":
            day = valid_date(row["cal_date"])
            if row["exchange"] != params["exchange"] or not START <= day <= END:
                raise ValueError("calendar request mismatch")
            if isinstance(row["is_open"], bool) or row["is_open"] not in (0, 1, "0", "1"):
                raise ValueError("invalid calendar open flag")
            if row["pretrade_date"] is not None:
                valid_date(row["pretrade_date"])
        else:
            pattern = r"T?\d{6}\.(SH|SZ|BJ)" if endpoint == "stock_basic" else r"\d{6}\.(SH|SZ|BJ)"
            if not isinstance(row["ts_code"], str) or not re.fullmatch(pattern, row["ts_code"]):
                raise ValueError("invalid symbol")
            if endpoint == "stock_basic":
                if row["list_status"] != params["list_status"]:
                    raise ValueError("status request mismatch")
                valid_date(row["list_date"])
                if row["delist_date"]:
                    valid_date(row["delist_date"])
                if row["symbol"] != row["ts_code"].split(".")[0]:
                    raise ValueError("symbol metadata mismatch")
            else:
                if valid_date(row["trade_date"]) != params["trade_date"]:
                    raise ValueError("trade date request mismatch")
                for field in set(fields) - {"ts_code", "trade_date"}:
                    row[field] = number(row[field])
                    if row[field] < 0 or (field in {"open", "high", "low", "close", "adj_factor"} and not row[field]):
                        raise ValueError("negative or zero numeric value")
        key = row_key(endpoint, row)
        if key in keys:
            raise ValueError("duplicate endpoint key")
        keys.add(key)
        result.append(row)
    return result


def combine_pages(pages: list[list[dict]], endpoint: str, page_size: int = CAP) -> list[dict]:
    if not pages or len(pages[-1]) >= page_size:
        raise ValueError("truncated or unterminated pagination")
    if any(len(page) != page_size for page in pages[:-1]):
        raise ValueError("invalid pagination sequence")
    rows, keys = [], set()
    for page in pages:
        for row in page:
            key = row_key(endpoint, row)
            if key in keys:
                raise ValueError("duplicate paginated key")
            keys.add(key)
            rows.append(row)
    return rows


def normalize_daily(row: dict) -> dict:
    return {"symbol": row["ts_code"], "date": row["trade_date"],
            "raw_close": number(row["close"]), "volume_shares": number(row["vol"]) * 100,
            "amount_cny": number(row["amount"]) * 1000}


def normalize_basic(row: dict) -> dict:
    return {key: number(row[key]) * 10000 for key in
            ("total_share", "float_share", "free_share", "total_mv", "circ_mv")}


def bar_issues(row: dict) -> list[str]:
    """Strict OHLC/VWAP bounds; no undocumented tolerance calibrated to data."""
    o, h, low, close, vol, amount = (number(row[k]) for k in ("open", "high", "low", "close", "vol", "amount"))
    issues = []
    if not (low <= o <= h and low <= close <= h):
        issues.append("ohlc_bounds")
    if vol == 0:
        if amount != 0:
            issues.append("amount_without_volume")
    elif not low * vol * 100 <= amount * 1000 <= h * vol * 100:
        issues.append("vwap_outside_ohlc")
    return issues


def audit_day(daily: list[dict], basic: list[dict], factors: list[dict], metadata: list[dict], day: str) -> dict:
    sets = {name: {r["ts_code"] for r in rows} for name, rows in
            (("daily", daily), ("daily_basic", basic), ("adj_factor", factors))}
    expected = {r["ts_code"] for r in metadata if r["list_date"] <= day and
                (not r["delist_date"] or day <= r["delist_date"])}
    lookup = {r["ts_code"]: r for r in metadata}
    violations = [{"symbol": r["ts_code"], "issues": bar_issues(r)} for r in daily if bar_issues(r)]
    close_by_symbol = {r["ts_code"]: r["close"] for r in basic}
    cross_close = [r["ts_code"] for r in daily if r["ts_code"] in close_by_symbol and
                   r["close"] != close_by_symbol[r["ts_code"]]]
    missing = {name: sorted(sets["daily"] - sets[name]) for name in ("daily_basic", "adj_factor")}
    return {"date": day, "row_counts": {k: len(v) for k, v in sets.items()},
            "market_counts": dict(Counter(lookup.get(r["ts_code"], {}).get("market", "UNKNOWN") for r in daily)),
            "exchange_suffix_counts": dict(Counter(r["ts_code"].split(".")[1] for r in daily)),
            "amount_cny_sum": str(sum((normalize_daily(r)["amount_cny"] for r in daily), Decimal(0))),
            "metadata_asof_eligible": len(expected), "missing_daily_vs_metadata": sorted(expected - sets["daily"]),
            "daily_not_in_metadata_asof": sorted(sets["daily"] - expected), "missing_endpoint_keys": missing,
            "extra_endpoint_keys": {k: sorted(sets[k] - sets["daily"]) for k in ("daily_basic", "adj_factor")},
            "bar_violations": violations, "cross_endpoint_close_mismatches": cross_close,
            "numeric_panel_valid": bool(daily) and not violations and not cross_close and not any(missing.values()),
            "all_market_completeness_proven": False}


def save_manifest(root: Path, manifest: dict) -> None:
    payload = (json.dumps(manifest, ensure_ascii=False, indent=2) + "\n").encode()
    temporary = root / "manifest.pending"
    temporary.write_bytes(payload)
    temporary.replace(root / "manifest.json")


class NoRedirect(urllib.request.HTTPRedirectHandler):
    def redirect_request(self, req, fp, code, msg, headers, newurl):
        raise ValueError("provider redirect refused")


class Collector:
    def __init__(self, root: Path):
        self.root = root
        root.mkdir(parents=True, exist_ok=True)
        (root / "raw").mkdir(exist_ok=True)
        self.manifest = json.loads((root / "manifest.json").read_text()) if (root / "manifest.json").exists() else {
            "version": 1, "created_at_utc": utc(), "requested_start": START, "requested_end": END,
            "fields": FIELDS, "units": UNITS, "docs": DOCS, "groups": {}, "requests": [], "blocked_endpoints": {},
            "successful_calls": 0, "fetch_seconds": 0.0, "all_market_completeness_proven": False,
            "limit": {"successful_calls": 450, "fetch_seconds": 720, "spacing_seconds": 1.3, "timeout_seconds": 25}}
        self.token = os.environ.get("TUSHARE_TOKEN")
        if not self.token:
            raise ValueError("TUSHARE_TOKEN unavailable")
        self.opener = urllib.request.build_opener(NoRedirect())
        self.revalidate_saved_schema_failures()

    def revalidate_saved_schema_failures(self) -> None:
        """Repair local parsing only. Never retry a provider denial or transport failure."""
        for receipt in self.manifest["requests"]:
            endpoint = receipt["api_name"]
            if (endpoint not in self.manifest["blocked_endpoints"] or receipt["status"] != "failed"
                    or receipt.get("provider_code") != 0 or "response_file" not in receipt
                    or "local_schema_revalidated_at_utc" in receipt):
                continue
            params = {k: v for k, v in receipt["params"].items() if k not in {"limit", "offset", "ts_type_name"}}
            selector = params.get("trade_date", params.get("list_status", params.get("exchange")))
            raw = (self.root / receipt["response_file"]).read_bytes()
            if digest(raw) != receipt["sha256"]:
                raise ValueError("saved failure hash mismatch")
            try:
                rows = parse_response(raw, endpoint, params)
            except ValueError:
                continue
            group = self.manifest["groups"][f"{endpoint}/{selector}"]
            if group["pages"] or receipt["params"]["offset"] != 0 or len(rows) >= CAP:
                continue
            group.update(status="complete", rows=len(rows), pages=[{
                "file": receipt["response_file"], "sha256": receipt["sha256"], "rows": len(rows), "offset": 0}])
            receipt["local_schema_revalidated_at_utc"] = utc()
            del self.manifest["blocked_endpoints"][endpoint]
            save_manifest(self.root, self.manifest)

    def get(self, endpoint: str, selector: str, params: dict) -> list[dict] | None:
        group_key = f"{endpoint}/{selector}"
        groups = self.manifest["groups"]
        group = groups.setdefault(group_key, {"endpoint": endpoint, "params": params, "pages": [], "status": "partial"})
        if group["status"] == "complete":
            return load_group(self.root, self.manifest, group_key)
        if endpoint in self.manifest["blocked_endpoints"]:
            return None
        pages = [read_page(self.root, endpoint, params, page) for page in group["pages"]]
        while not pages or len(pages[-1]) == CAP:
            if self.manifest["successful_calls"] >= 450 or self.manifest["fetch_seconds"] >= 693:
                group["stop_reason"] = "bounded_budget"
                save_manifest(self.root, self.manifest)
                return None
            offset = sum(len(page) for page in pages)
            request_params = {**params, "limit": CAP, "offset": offset, "ts_type_name": SDK_BASE}
            receipt = {"endpoint": f"{BASE}/{endpoint}", "api_name": endpoint, "params": request_params,
                       "fields": FIELDS[endpoint], "started_at_utc": utc(), "status": "failed"}
            started = time.monotonic()
            try:
                time.sleep(1.3)
                body = json.dumps({"api_name": endpoint, "params": request_params,
                                   "fields": FIELDS[endpoint], "token": self.token}).encode()
                req = urllib.request.Request(f"{BASE}/{endpoint}", data=body, method="POST",
                                             headers={"Content-Type": "application/json"})
                with self.opener.open(req, timeout=25) as response:
                    raw = response.read(20_000_001)
                    receipt["http_status"] = response.status
                if len(raw) > 20_000_000 or self.token.encode() in raw:
                    raise ValueError("unsafe response")
                payload = json.loads(raw)
                code = payload.get("code") if isinstance(payload, dict) else None
                receipt["provider_code"] = code if type(code) is int else None
                if receipt["provider_code"] != 0:
                    raise ValueError("provider rejected request; message suppressed")
                self.manifest["successful_calls"] += 1
                name = f"raw/{endpoint}_{selector}_{offset:06d}_{digest(raw)}.json"
                path = self.root / name
                if not path.exists():
                    with path.open("xb") as stream:
                        stream.write(raw)
                receipt.update(response_file=name, sha256=digest(raw), bytes=len(raw))
                rows = parse_response(raw, endpoint, params)
                if len(rows) > CAP:
                    raise ValueError("provider exceeded page size")
                receipt.update(status="success", rows=len(rows), observed_dates=sorted({
                    r.get("trade_date", r.get("cal_date")) for r in rows} - {None}))
                group["pages"].append({"file": name, "sha256": digest(raw), "rows": len(rows), "offset": offset})
                pages.append(rows)
                if sum(map(len, pages)) != len({row_key(endpoint, r) for page in pages for r in page}):
                    raise ValueError("pagination ignored or duplicate keys")
            except Exception as exc:
                receipt.update(status="failed", error_type=type(exc).__name__,
                               error="Request/schema failure; provider text suppressed")
                self.manifest["blocked_endpoints"][endpoint] = receipt["error_type"]
                group["status"] = "partial"
                return None
            finally:
                self.manifest["fetch_seconds"] += time.monotonic() - started
                receipt["finished_at_utc"] = utc()
                self.manifest["requests"].append(receipt)
                save_manifest(self.root, self.manifest)
        rows = combine_pages(pages, endpoint)
        group.update(status="complete", rows=len(rows))
        save_manifest(self.root, self.manifest)
        return rows


def read_page(root: Path, endpoint: str, params: dict, page: dict) -> list[dict]:
    path = (root / page["file"]).resolve()
    if not path.is_relative_to(root.resolve() / "raw"):
        raise ValueError("raw path outside snapshot")
    raw = path.read_bytes()
    if digest(raw) != page["sha256"]:
        raise ValueError("raw response hash mismatch")
    rows = parse_response(raw, endpoint, params)
    if len(rows) != page["rows"]:
        raise ValueError("page count mismatch")
    return rows


def load_group(root: Path, manifest: dict, key: str) -> list[dict]:
    group = manifest["groups"][key]
    if group["status"] != "complete":
        raise ValueError("partial endpoint group")
    offset = 0
    for page in group["pages"]:
        if page["offset"] != offset:
            raise ValueError("missing or reordered pagination offset")
        offset += page["rows"]
    rows = combine_pages([read_page(root, group["endpoint"], group["params"], p) for p in group["pages"]], group["endpoint"])
    if len(rows) != group["rows"]:
        raise ValueError("group count mismatch")
    return rows


def load_daily_panel(root: Path, day: str) -> list[dict]:
    """Strict numerical consumer; raw forensic replay is intentionally separate."""
    valid_date(day)
    manifest = json.loads((root / "manifest.json").read_text())
    try:
        daily, basic, factors = [load_group(root, manifest, f"{endpoint}/{day}")
                                 for endpoint in ("daily", "daily_basic", "adj_factor")]
    except KeyError:
        raise ValueError("missing endpoint group") from None
    report = audit_day(daily, basic, factors, [], day)
    if not report["numeric_panel_valid"]:
        raise ValueError("invalid numerical panel: bounds or cross-endpoint keys/close")
    basic_by_symbol = {r["ts_code"]: r for r in basic}
    factors_by_symbol = {r["ts_code"]: r for r in factors}
    return [{**normalize_daily(row), **normalize_basic(basic_by_symbol[row["ts_code"]]),
             "adj_factor": factors_by_symbol[row["ts_code"]]["adj_factor"]} for row in daily]


def audit_snapshot(root: Path) -> dict:
    manifest_raw = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_raw)
    groups = manifest["groups"]
    metadata = [r for state in "LDP" if groups.get(f"stock_basic/{state}", {}).get("status") == "complete"
                for r in load_group(root, manifest, f"stock_basic/{state}")]
    if len(metadata) != len({r["ts_code"] for r in metadata}):
        raise ValueError("duplicate stock across statuses")
    expected_days, calendar_coverage, calendar_missing, calendar_open_dates = set(), {}, {}, {}
    first, last = (dt.datetime.strptime(d, "%Y%m%d").date() for d in (START, END))
    expected_calendar = {(first + dt.timedelta(days=i)).strftime("%Y%m%d") for i in range((last - first).days + 1)}
    for exchange in ("SSE", "SZSE"):
        key = f"trade_cal/{exchange}"
        if groups.get(key, {}).get("status") == "complete":
            rows = load_group(root, manifest, key)
            calendar_coverage[exchange] = len(rows)
            calendar_missing[exchange] = sorted(expected_calendar - {r["cal_date"] for r in rows})
            calendar_open_dates[exchange] = {r["cal_date"] for r in rows if str(r["is_open"]) == "1"}
            expected_days.update(calendar_open_dates[exchange])
    available = {endpoint: sorted(key.split("/")[1] for key, group in groups.items()
                                if group["endpoint"] == endpoint and group["status"] == "complete")
                 for endpoint in ("daily", "daily_basic", "adj_factor")}
    all_dates = sorted(set().union(*(set(dates) for dates in available.values())))
    reports = []
    for day in all_dates:
        rows = [load_group(root, manifest, f"{endpoint}/{day}") if day in available[endpoint] else []
                for endpoint in ("daily", "daily_basic", "adj_factor")]
        reports.append(audit_day(*rows, metadata, day))
    return {"created_at_utc": utc(), "manifest_sha256": digest(manifest_raw),
            "audit_code_sha256": digest(Path(__file__).read_bytes()),
            "requested_range": [START, END], "calendar_rows": calendar_coverage,
            "calendar_missing_dates": calendar_missing,
            "calendar_exchange_agreement": (len(calendar_open_dates) == 2 and
                                              calendar_open_dates["SSE"] == calendar_open_dates["SZSE"]),
            "expected_open_days": sorted(expected_days), "actual_endpoint_dates": available,
            "missing_requested_dates": {e: sorted(expected_days - set(d)) for e, d in available.items()},
            "metadata_status_rows": {s: groups.get(f"stock_basic/{s}", {}).get("rows") for s in "LDP"},
            "metadata_is_current_snapshot_not_historical": True, "days": reports,
            "all_market_completeness_proven": False, "successful_calls": manifest["successful_calls"],
            "fetch_seconds": manifest["fetch_seconds"], "blocked_endpoints": manifest["blocked_endpoints"]}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("pilot", "continue", "audit"))
    parser.add_argument("--root", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    if args.command != "audit":
        collector = Collector(root)
        if args.command == "pilot":
            for endpoint in ("daily", "daily_basic", "adj_factor"):
                collector.get(endpoint, END, {"trade_date": END})
            for status in "LDP":
                collector.get("stock_basic", status, {"list_status": status})
            for exchange in ("SSE", "SZSE"):
                collector.get("trade_cal", exchange, {"exchange": exchange, "start_date": START, "end_date": END})
        else:
            if not (root / "pilot-audit.json").exists():
                raise ValueError("run and inspect pilot before continuation")
            calendars = [load_group(root, collector.manifest, f"trade_cal/{ex}") for ex in ("SSE", "SZSE")]
            dates = sorted({r["cal_date"] for rows in calendars for r in rows if str(r["is_open"]) == "1"})
            for day in dates:
                for endpoint in ("daily", "daily_basic", "adj_factor"):
                    collector.get(endpoint, day, {"trade_date": day})
                if collector.manifest["fetch_seconds"] >= 693 or collector.manifest["successful_calls"] >= 450:
                    break
                print(json.dumps({"date": day, "successful_calls": collector.manifest["successful_calls"]}), flush=True)
    report = audit_snapshot(root)
    name = "pilot-audit.json" if args.command == "pilot" else "coverage-audit.json"
    (root / name).write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({"report": name, "endpoint_days": {k: len(v) for k, v in report["actual_endpoint_dates"].items()},
                      "successful_calls": report["successful_calls"], "blocked_endpoints": report["blocked_endpoints"],
                      "day_counts": [{"date": r["date"], "rows": r["row_counts"],
                                      "bar_violations": len(r["bar_violations"])} for r in report["days"]]}))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
