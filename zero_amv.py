"""
指南针软件 0AMV（活跃市值 / 活筹指数）公式的 Python 近似实现。

================================================================
重要前提
================================================================
0AMV 是「指南针」股票软件创始人（指南针软件 / 沈阳指南针）的私有指标，
官方只对客户端开放 K 线，不公开算法源代码。网上能查到的所有源码都
是逆向 / 民间复刻版，本文件同样如此。

本模块实现的是**通达信 / 同花顺 / 飞狐 仿制版**（公开流通最广），
公开资料存在两种不同公式，本模块默认采用明确标为 0AMV 的成交额版本：

    0AMV_close = SMA(全市场 AMOUNT, 10, 1) / 1e7

另一个常见的 ``× CLOSE / MA(REF(CLOSE, 1), 5)`` 版本更常以“资金起爆”
等名称传播，本模块仅作为 ``price_adjusted`` 兼容口径保留。两者都不是
指南针公开源码，不能把民间公式一致性当作原版拟合精度。

================================================================
三个输出层级
================================================================
- 简化版（fit_lite）：仅 0AMV 收盘价 + 生命线。
  用途：快速看大盘资金流入/流出方向。

- 标准版（fit_standard）：0AMV 完整 K 线（开/高/低/收）+ 生命线。
  用途：画民间公式 K 线和 EMA12 平滑线。
  来源：通达信仿指南针源码 / 公式网 / 东方财富博客。

- 完整版（fit_full）：标准版 + C5 / C13 / C34 三条活筹成本均线。
  用途：复刻指南针软件最完整的资金面分析模块。

================================================================
输入要求
================================================================
本指标是**大盘级别**的「活跃市值」——输入是全市场（沪深 A 股）
聚合的 daily bar：
    amount: 当日全市场成交额（元）
    open/close/high/low: 仅 ``price_adjusted`` 兼容口径需要
    vol: 当日全市场成交量
    capital: 当日全市场流通股本（用于换手率归一化）

单只股票成交额代入后只是个股成交额代理，不再具有大盘 0AMV 的定义。

================================================================
0AMV vs 0号指数（流通市值）的关系
================================================================
    0号指数 (流通市值) = 0AMV (活筹) + 0DMV (死筹)

死筹 = 长期锁定的筹码（"近期不参与交易"的部分），活筹 = 短
期交易活跃的部分。0AMV 上升 + 0号指数横盘 = 资金被激活（好
信号）；0AMV 横盘 + 0号指数上升 = 新股发行 / 解禁（中性）。

================================================================
依赖
================================================================
- pandas
- numpy
(可选) qlib —— 仅当使用 QlibExpressionAdapter 时需要。
"""
from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Iterable

import numpy as np
import pandas as pd


# ---------------------------------------------------------------------------
# 核心数学函数（与 qlib / 通达信 / 同花顺 保持一致）
# ---------------------------------------------------------------------------

def sma(series: pd.Series, n: int, m: int = 1) -> pd.Series:
    """通达信 SMA(X, N, M) 平滑移动平均。

    等价于递归：
        Y[0] = X[0]
        Y[t] = (M * X[t] + (N - M) * Y[t-1]) / N

    在 m == 1 时，Y[t] = (X[t] + (N-1) * Y[t-1]) / N，
    等价于 qlib 的 EMA(X, 2N-1)（adjust=True 下需 ~30 bar 收敛）。

    重要：与 qstock / qlib 不同的是，ta-lib 的 SMA / EMA 是
    简单 / 指数移动平均，**不等于** 通达信 SMA(N, 1)。请勿混用。
    """
    if m < 1 or m > n:
        raise ValueError(f"SMA 要求 1 <= m <= n，得到 m={m}, n={n}")
    out = series.copy().astype(float)
    out.iloc[0] = series.iloc[0]
    alpha = m / n
    for i in range(1, len(series)):
        if pd.isna(series.iloc[i]):
            out.iloc[i] = out.iloc[i - 1]
        else:
            out.iloc[i] = alpha * series.iloc[i] + (1 - alpha) * out.iloc[i - 1]
    return out


def ema(series: pd.Series, n: int) -> pd.Series:
    """标准 EMA —— qlib 与绝大多数库都用这个，用于 0AMV 的「生命线」。"""
    return series.ewm(span=n, adjust=False).mean()


def ma_ref(series: pd.Series, n: int) -> pd.Series:
    """MA(REF(CLOSE, 1), 5) —— 0AMV 公式里的分母。

    注意：用的是「昨日收盘价」过去 N 天的简单移动平均，
    不是「当日收盘价」过去 N 天的简单移动平均。这是仿版与
    网上一些简化版的主要差异之一。
    """
    return series.shift(1).rolling(n).mean()


# ---------------------------------------------------------------------------
# 三个层级的拟合
# ---------------------------------------------------------------------------

class FitLevel(str, Enum):
    LITE = "lite"            # 简化版
    STANDARD = "standard"    # 标准版（推荐）
    FULL = "full"            # 完整版


class FormulaVariant(str, Enum):
    """公开流传的两种公式口径。"""

    AMOUNT_ONLY = "amount_only"
    PRICE_ADJUSTED = "price_adjusted"


@dataclass
class MarketBar:
    """0AMV 输入的全市场聚合 bar。"""
    date: pd.Timestamp
    open: float
    high: float
    low: float
    close: float
    amount: float   # 元
    volume: float   # 股
    capital: float  # 流通股本

    @classmethod
    def from_dataframe(cls, df: pd.DataFrame) -> list["MarketBar"]:
        required = {"open", "high", "low", "close", "amount", "volume", "capital"}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"输入 df 缺少列: {missing}")
        bars = []
        for idx, row in df.iterrows():
            bars.append(
                cls(
                    date=idx if isinstance(idx, pd.Timestamp) else pd.Timestamp(idx),
                    open=float(row["open"]),
                    high=float(row["high"]),
                    low=float(row["low"]),
                    close=float(row["close"]),
                    amount=float(row["amount"]),
                    volume=float(row["volume"]),
                    capital=float(row["capital"]),
                )
            )
        return bars


def _to_series(bars: Iterable[MarketBar], field: str) -> pd.Series:
    """把 MarketBar 列表转成 pd.Series，索引为 date。"""
    data = {b.date: getattr(b, field) for b in bars}
    s = pd.Series(data).sort_index()
    s.index = pd.DatetimeIndex(s.index)
    return s


def compute_0amv(
    df: pd.DataFrame,
    fit_level: FitLevel | str = FitLevel.STANDARD,
    formula_variant: FormulaVariant | str = FormulaVariant.AMOUNT_ONLY,
    smooth_n: int = 10,
    ref_ma_n: int = 5,
    life_line_ema: int = 12,
    scale_factor: float = 1.0,
) -> pd.DataFrame:
    """计算 0AMV 指标。

    Args:
        df: 默认口径只要求 amount；full 还要求 volume / capital；
            price_adjusted 还要求 OHLC。索引为可解析日期。
        fit_level: "lite" / "standard" / "full"。
        smooth_n: SMA(AMOUNT, N, 1) 的 N，默认 10（指南针默认）。
        ref_ma_n: 0AMV 公式分母 MA(REF(CLOSE, 1), N) 的 N，默认 5。
        life_line_ema: 生命线 EMA 的 N，默认 12。
        formula_variant: ``amount_only`` 是公开 0AMV 仿制公式；
            ``price_adjusted`` 保留另一种常见的 OHLC 资金公式。
        scale_factor: 用同日期指南针导出真值校准的乘数，默认不缩放。不要
            把截图十字光标的坐标标签误当成 K 线收盘值。

    Returns:
        DataFrame，索引为 date，包含的列取决于 fit_level：

        lite:
            - 0amv_close
            - 0amv_life_line
            - 0amv_change_pct

        standard (lite +):
            - 0amv_open
            - 0amv_high
            - 0amv_low
            - 0amv_color (1=红/上涨, 0=绿/下跌, NaN=平)

        full (standard +):
            - 0amv_c5  (短线活筹均线, 换手归一化)
            - 0amv_c13 (中线活筹均线)
            - 0amv_c34 (长线活筹均线)
            - 0amv_infinite (∞ 活筹均线的民间公式)

    Note on 0DMV (死筹):
        民间仿制版 0AMV 是一个带「千万 RMB」单位的相对值（资金流量折算），
        严格意义上**不等于**真实活跃市值。因此 0AMV + 0DMV = 流通市值
        这个不变量在仿制版中不成立；本模块不提供 0DMV 列。
    """
    fit_level = FitLevel(fit_level)
    formula_variant = FormulaVariant(formula_variant)

    required = {"amount"}
    if formula_variant == FormulaVariant.PRICE_ADJUSTED:
        required |= {"open", "high", "low", "close"}
    if fit_level == FitLevel.FULL:
        required |= {"volume", "capital"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"输入 df 缺少必需列: {sorted(missing)}")

    if fit_level not in FitLevel:
        raise ValueError(f"fit_level 必须是 {list(FitLevel)}, 得到 {fit_level!r}")

    df = df.copy()
    if len(df) == 0:
        return pd.DataFrame(index=df.index)

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(df.index)
    df = df.sort_index()

    out = pd.DataFrame(index=df.index)

    # -----------------------------------------------------------------------
    # 1) 核心: SMA(AMOUNT, 10, 1) —— 10 日平滑全市场成交额
    # -----------------------------------------------------------------------
    smooth_amount = sma(df["amount"], n=smooth_n, m=1)

    base = smooth_amount * scale_factor / 1e7
    if formula_variant == FormulaVariant.AMOUNT_ONLY:
        out["0amv_close"] = base
    else:
        ref_ma = ma_ref(df["close"], n=ref_ma_n)
        out["0amv_close"] = base * df["close"] / ref_ma

    if fit_level == FitLevel.LITE:
        out["0amv_life_line"] = ema(out["0amv_close"], life_line_ema)
        out["0amv_change_pct"] = out["0amv_close"].pct_change() * 100
        return out

    if formula_variant == FormulaVariant.AMOUNT_ONLY:
        # 公开 0AMV 公式用当日值与昨日值画柱；并没有可验证的日内 OHLC。
        out["0amv_open"] = out["0amv_close"].shift(1)
        out["0amv_high"] = out[["0amv_open", "0amv_close"]].max(axis=1)
        out["0amv_low"] = out[["0amv_open", "0amv_close"]].min(axis=1)
    else:
        for col, src in [("0amv_open", df["open"]), ("0amv_high", df["high"]), ("0amv_low", df["low"])]:
            out[col] = base * src / ref_ma

    out["0amv_color"] = (out["0amv_close"] > out["0amv_open"]).astype(float)
    out.loc[out["0amv_open"].isna(), "0amv_color"] = np.nan
    out.loc[out["0amv_close"] == out["0amv_open"], "0amv_color"] = np.nan

    # 生命线
    out["0amv_life_line"] = ema(out["0amv_close"], life_line_ema)
    out["0amv_change_pct"] = out["0amv_close"].pct_change() * 100

    if fit_level == FitLevel.STANDARD:
        return out

    # -----------------------------------------------------------------------
    # 5) C5 / C13 / C34 / ∞ 活筹成本均线 (用换手归一化)
    #    公开民间公式:
    #       C5  = DMA(SMA(AMOUNT, 3, 1), VOL / 0.02 / CAPITAL)
    #       C13 = DMA(SMA(AMOUNT, 3, 1), VOL / 0.10 / CAPITAL)
    #       C34 = DMA(SMA(AMOUNT, 3, 1), VOL / 0.18 / CAPITAL)
    #       ∞   = DMA(SMA(AMOUNT, 10, 1), VOL / 1.1 / CAPITAL)
    #
    #    DMA(X, A) 是动态移动平均：
    #       Y[t] = (1 - A) * Y[t-1] + A * X[t]
    #    系数 A = VOL[t] / k / CAPITAL[t] 反映"换手率归一化"——
    #    成交量越大的日子权重越高。
    # -----------------------------------------------------------------------
    turnover = df["volume"] / df["capital"]

    def dma(x: pd.Series, alpha_series: pd.Series) -> pd.Series:
        """DMA 动态移动平均。alpha 是逐 bar 变化的系数。

        Y[t] = (1 - alpha[t]) * Y[t-1] + alpha[t] * X[t]
        """
        x_vals = x.astype(float).values
        a_vals = alpha_series.astype(float).values
        n = len(x_vals)
        y = np.full(n, np.nan, dtype=float)
        first_valid = -1
        for i in range(n):
            if not np.isnan(x_vals[i]) and not np.isnan(a_vals[i]):
                y[i] = x_vals[i]
                first_valid = i
                break
        if first_valid < 0:
            return pd.Series(y, index=x.index)
        for i in range(first_valid + 1, n):
            if np.isnan(x_vals[i]) or np.isnan(a_vals[i]):
                y[i] = y[i - 1]
            else:
                a = float(a_vals[i])
                # 截断 alpha 到 [0, 1] 防止数值发散
                a = max(0.0, min(1.0, a))
                y[i] = (1 - a) * y[i - 1] + a * x_vals[i]
        return pd.Series(y, index=x.index)

    nested_sma = sma(base, n=3, m=1)
    out["0amv_c5"] = dma(nested_sma, turnover / 0.02)
    out["0amv_c13"] = dma(nested_sma, turnover / 0.10)
    out["0amv_c34"] = dma(nested_sma, turnover / 0.18)
    out["0amv_infinite"] = dma(base, turnover / 1.10)

    return out


def calibrate_scale(proxy: pd.Series, ground_truth: pd.Series) -> float:
    """用重叠日期真值的中位数比例标定量纲，降低单个截图读数的影响。"""
    aligned = pd.concat({"proxy": proxy, "truth": ground_truth}, axis=1).dropna()
    aligned = aligned[(aligned["proxy"] > 0) & (aligned["truth"] > 0)]
    if aligned.empty:
        raise ValueError("proxy 与 ground_truth 没有可用的重叠正值")
    return float((aligned["truth"] / aligned["proxy"]).median())


# ---------------------------------------------------------------------------
# qlib 表达式版 —— 直接贴进 TradingAnalyze 的 technical_factors.py 用
# ---------------------------------------------------------------------------

QLIB_EXPRESSIONS: dict[str, str] = {
    # 公开 0AMV 成交额公式。qlib 没有通达信 SMA(N,1)，EMA(2N-1) 收敛后等价。
    "0amv_amount_smooth": "EMA($amount, 19)",
    "0amv_close": "EMA($amount, 19) / 1e7",
    "0amv_open": "Ref(EMA($amount, 19) / 1e7, 1)",
    "0amv_high": "If(EMA($amount, 19) > Ref(EMA($amount, 19), 1), EMA($amount, 19), Ref(EMA($amount, 19), 1)) / 1e7",
    "0amv_low": "If(EMA($amount, 19) < Ref(EMA($amount, 19), 1), EMA($amount, 19), Ref(EMA($amount, 19), 1)) / 1e7",

    # 生命线（默认 12 期 EMA）
    "0amv_life_line": "EMA(EMA($amount, 19) / 1e7, 12)",

    # 涨跌幅
    "0amv_change_pct": "(EMA($amount, 19) / Ref(EMA($amount, 19), 1)) - 1",

    # K 线颜色信号（红/绿）
    "0amv_color": "If(EMA($amount, 19) > Ref(EMA($amount, 19), 1), 1, 0)",
}


def explain() -> None:
    """打印 0AMV 公式核心 + 三个版本差异。"""
    print("=" * 70)
    print("0AMV (活跃市值 / 活筹指数) —— 指南针软件指标近似拟合")
    print("=" * 70)
    print("""
核心公式（公开民间版，未经指南针官方确认）:

    0AMV_close = SMA(全市场 AMOUNT, 10, 1) / 1e7

    0AMV_life_line = EMA(0AMV_close, 12)

    0AMV + 0DMV ≈ 0号指数 (全市场流通市值)

本模块提供三个层级:
    1) compute_0amv(df, fit_level='lite')       → 仅 0AMV_close + 生命线
    2) compute_0amv(df, fit_level='standard')   → 标准版，仿指南针 K 线
    3) compute_0amv(df, fit_level='full')       → 完整版，含 C5/C13/C34 均线

输入要求: 默认只需全市场 amount（元）；full 另需同口径的 volume / capital。
""")


if __name__ == "__main__":
    explain()
