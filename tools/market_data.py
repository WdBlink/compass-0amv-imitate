"""下载可复现的沪深市场成交额；不再用指数价格和随机数伪造 amount。"""
from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

import pandas as pd


_URL = "https://push2his.eastmoney.com/api/qt/stock/kline/get"


def fetch_index_daily(secid: str, start: str, end: str) -> pd.DataFrame:
    query = urlencode(
        {
            "secid": secid,
            "fields1": "f1,f2,f3,f4,f5,f6",
            "fields2": "f51,f52,f53,f54,f55,f56,f57",
            "klt": "101",
            "fqt": "0",
            "beg": start.replace("-", ""),
            "end": end.replace("-", ""),
        }
    )
    request = Request(f"{_URL}?{query}", headers={"User-Agent": "Mozilla/5.0"})
    with urlopen(request, timeout=30) as response:
        payload = json.load(response)
    if payload.get("data") is None:
        raise RuntimeError(f"东方财富未返回 {secid} 的数据: {payload}")
    rows = [row.split(",") for row in payload["data"]["klines"]]
    return pd.DataFrame(
        {
            "date": pd.to_datetime([row[0] for row in rows]),
            "amount": [float(row[6]) for row in rows],
        }
    ).set_index("date")


def load_mainland_market_amount(start: str, end: str) -> pd.DataFrame:
    """以上证指数和深证综指成交额之和近似沪深全市场成交额（元）。"""
    sh = fetch_index_daily("1.000001", start, end).rename(columns={"amount": "sh_amount"})
    sz = fetch_index_daily("0.399106", start, end).rename(columns={"amount": "sz_amount"})
    market = sh.join(sz, how="inner")
    if market.empty:
        raise RuntimeError("沪深指数数据没有重叠日期")
    market["amount"] = market["sh_amount"] + market["sz_amount"]
    return market
