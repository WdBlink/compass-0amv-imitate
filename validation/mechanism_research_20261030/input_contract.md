# R2 market input contract

This input acquisition is independent of official 0AMV labels. The requested interval is 2026-05-01 through 2026-10-30. It supplies finite-age warmup and exposed-day research inputs. It does not establish infinite-state initialization or exact replication.

## Provider fields and units

| Endpoint | Requested fields | Normalized interpretation |
| --- | --- | --- |
| `daily` | `ts_code,trade_date,open,high,low,close,vol,amount` | OHLC remains unadjusted. `vol * 100` gives shares. `amount * 1000` gives CNY. |
| `daily_basic` | `ts_code,trade_date,close,turnover_rate,total_share,float_share,free_share,total_mv,circ_mv` | Shares multiply by 10,000. Market values multiply by 10,000 CNY. Turnover stays percent. |
| `adj_factor` | `ts_code,trade_date,adj_factor` | A separate dimensionless factor. No automatic price adjustment. |
| `trade_cal` | `exchange,cal_date,is_open,pretrade_date` | Separate SSE and SZSE calendars. BSE calendar equivalence is a documented provider statement, not a separate BSE response. |
| `stock_basic` | `ts_code,symbol,name,market,exchange,curr_type,list_status,list_date,delist_date` | Separate L, D and P requests. Current metadata snapshot with listing-date bounds, not historical metadata snapshots. |

Primary documentation checked 2026-10-30: [daily](https://tushare.pro/document/2?doc_id=27), [daily_basic](https://tushare.pro/document/2?doc_id=32), [adj_factor](https://tushare.pro/document/2?doc_id=28), [trade_cal](https://tushare.pro/document/2?doc_id=26), [stock_basic](https://tushare.pro/document/2?doc_id=25).

The daily documentation describes unadjusted bars and no observations during suspensions. A missing bar alone still does not prove suspension. No suspension endpoint is included. `amount` is the provider field, never `close * volume` or typical-price turnover.

The stock metadata permits an optional historical `T` prefix, observed in a retired security. Trading bars still require six digits and `.SH`, `.SZ` or `.BJ`. No metadata row is silently discarded to force the current trading-symbol rule.

## Raw response and request boundaries

The CLI requires an explicit private `--root`. Raw data, per-request receipts, pilot audit and coverage audit remain outside Git. Successful raw responses are saved once under content-hashed filenames. The manifest stores token-free request parameters, endpoint, returned dates, row counts, SHA-256, and request times. Existing response bytes are never replaced. Offline replay checks each hash again.

Collection and replay bind each group key to its endpoint, selector and exact request parameters. Daily groups request one day inside the fixed interval, stock metadata requests one L/D/P status, and calendars request one SSE/SZSE exchange over the full interval. Extra filters or mismatched selectors are rejected. The strict panel checks every row from all three daily endpoints against its requested day before joining; it never rewrites dates. Calendar `is_open` accepts JSON 0/1, numeric 0.0/1.0 and strings `"0"`/`"1"`, normalizing them to integer 0/1 at parsing. Booleans and other values are rejected. Audit and continuation use that same normalized flag.

Requests use HTTPS on `api.waditu.com/dataapi/<endpoint>`, matching the installed SDK route with an HTTPS upgrade. Redirects are refused. The token comes only from the environment. Failed provider messages, request bodies and exception text are not logged.

The cumulative budget is 450 successful API calls and 720 seconds of actual fetch work, including pacing. Each request has a 25-second timeout and 1.3-second sequential spacing. New requests stop before 693 cumulative seconds to reserve timeout overhead. The manifest preserves counters across continuation. An endpoint failure blocks further network calls to that endpoint. A saved successful response rejected by a local schema rule can be revalidated offline once. This is not a retry of an access denial.

Daily, daily-basic and stock-basic docs specify a 6,000-row maximum. All endpoints use a conservative 6,000-row page guard. A full page requires a subsequent page with fewer rows, including an empty terminal page. Offset continuity and key uniqueness are checked. A short page means the requested provider query terminated, not that historical all-market completeness is proven. The `adj_factor` and calendar pages do not themselves document the same cap.

## Numerical and coverage checks

`parse_response` rejects malformed schemas, row widths, dates, symbols, duplicate endpoint keys, missing numeric values, booleans, nonfinite numbers and invalid signs. `load_group` replays raw data for forensic inspection. `load_daily_panel` is the strict numerical consumer and rejects missing endpoint groups, daily keys without basic data or factors, close disagreements and failed bar constraints.

The bar constraints are `low <= open,close <= high` and `low <= amount * 1000 / (vol * 100) <= high` when volume is positive. Zero volume requires zero amount. The implementation adds no tolerance. The checked provider docs establish units but do not specify a rounding-error guarantee sufficient to relax VWAP bounds. Any small violation remains an explicit anomaly until a provider rounding contract or independent source resolves it. It is not removed, repaired, or used to tune a tolerance.

The coverage audit records expected open dates, actual endpoint dates, missing dates, exchange-calendar agreement, counts by market and exchange suffix, endpoint key differences, metadata date-bound membership differences, numerical anomalies and exact decimal turnover sums. Metadata listing and delisting dates only bound possible membership. Current names, market classifications and current status are not historical classifications. Missing symbols are not automatically suspensions. `all_market_completeness_proven` remains false.

The test suite uses synthetic data only. Network-free audit and strict loader are separate so that anomalies remain inspectable without being accepted as valid numerical research input.
