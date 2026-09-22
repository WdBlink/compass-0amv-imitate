"""TradingAnalyze / qlib 集成片段。

重要：qlib 的单标的 ``$amount`` 不能直接代表沪深全市场成交额。只有在输入
instrument 本身就是预先聚合的全市场序列时，下面的表达式才有 0AMV 代理含义。
"""


FACTORS = {
    # EMA(X, 19) 在收敛后等价于通达信 SMA(X, 10, 1)。
    "0amv_amount_smooth": "EMA($amount, 19)",
    "0amv_close": "EMA($amount, 19) / 1e7",
    "0amv_open": "Ref(EMA($amount, 19) / 1e7, 1)",
    "0amv_life_line": "EMA(EMA($amount, 19) / 1e7, 12)",
    "0amv_change_pct": "EMA($amount, 19) / Ref(EMA($amount, 19), 1) - 1",
    "0amv_color": "If(EMA($amount, 19) > Ref(EMA($amount, 19), 1), 1, 0)",
}
