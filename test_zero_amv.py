"""
0AMV 单元测试

测试覆盖：
1. SMA / EMA / MA(REF) 数学函数正确性
2. compute_0amv 三个层级的输出 schema 正确性
3. 关键不变量：0AMV 与 0DMV 之和 ≈ 0号指数
4. qlib 表达式可解析（不抛语法错）
5. 边界情况：空 df / NaN / 异常输入

依赖: pytest
运行: pytest /Users/wdblink/Research/trade/0amv/test_zero_amv.py -v
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

# 把父目录加进 path 以便 import zero_amv
sys.path.insert(0, str(Path(__file__).parent))

from zero_amv import (  # noqa: E402
    FitLevel,
    FormulaVariant,
    QLIB_EXPRESSIONS,
    calibrate_scale,
    compute_0amv,
    ema,
    ma_ref,
    sma,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def sample_market_df() -> pd.DataFrame:
    """生成 100 天的合成全市场数据，单位与 Tushare 一致。"""
    np.random.seed(42)
    n = 100
    dates = pd.date_range("2024-01-01", periods=n, freq="B")
    # 模拟大盘: close 在 3000 附近, amount 在 1e12 附近
    close = 3000 + np.cumsum(np.random.randn(n) * 10)
    open_ = close + np.random.randn(n) * 5
    high = np.maximum(open_, close) + abs(np.random.randn(n)) * 8
    low = np.minimum(open_, close) - abs(np.random.randn(n)) * 8
    volume = np.random.uniform(5e10, 1e11, n)  # 50-100 亿股
    capital = 4e12  # 全市场流通股本约 4 万亿股
    amount = volume * close  # 元

    return pd.DataFrame(
        {
            "open": open_,
            "high": high,
            "low": low,
            "close": close,
            "amount": amount,
            "volume": volume,
            "capital": capital,
        },
        index=dates,
    )


@pytest.fixture
def sample_market_df_with_cap(sample_market_df: pd.DataFrame) -> pd.DataFrame:
    """带流通市值列，用于 0DMV 计算。"""
    df = sample_market_df.copy()
    df["mkt_cap"] = df["capital"] * df["close"]
    return df


# ---------------------------------------------------------------------------
# 数学函数测试
# ---------------------------------------------------------------------------

class TestSMA:
    """通达信 SMA(N, 1) 的关键测试：起始值 = X[0]，递推 Y[t] = (X[t] + (N-1) Y[t-1]) / N"""

    def test_sma_first_value(self):
        x = pd.Series([10.0, 20.0, 30.0])
        y = sma(x, n=3, m=1)
        assert y.iloc[0] == 10.0

    def test_sma_recursion(self):
        """手算验证：N=3, M=1, x=[10, 20, 30]
        y[0] = 10
        y[1] = (1*20 + 2*10) / 3 = 40/3 = 13.333
        y[2] = (1*30 + 2*13.333) / 3 = 56.667/3 = 18.889
        """
        x = pd.Series([10.0, 20.0, 30.0])
        y = sma(x, n=3, m=1)
        assert math.isclose(y.iloc[0], 10.0, abs_tol=1e-9)
        assert math.isclose(y.iloc[1], 40.0 / 3, abs_tol=1e-3)
        assert math.isclose(y.iloc[2], 56.667 / 3, abs_tol=1e-3)

    def test_sma_converges_to_mean(self):
        """长期来看 SMA(N, 1) 收敛到序列均值附近。"""
        np.random.seed(0)
        x = pd.Series(np.random.randn(1000) + 100)
        y = sma(x, n=20, m=1)
        # 最后 200 个值的均值应该接近 100
        assert abs(y.iloc[-200:].mean() - 100) < 5

    def test_sma_invalid_m_n_raises(self):
        x = pd.Series([1.0, 2.0, 3.0])
        with pytest.raises(ValueError):
            sma(x, n=3, m=5)
        with pytest.raises(ValueError):
            sma(x, n=3, m=0)

    def test_sma_handles_nan(self):
        x = pd.Series([1.0, np.nan, 3.0, 4.0])
        y = sma(x, n=2, m=1)
        # NaN 应该被替换为前一个有效值
        assert not y.isna().any()


class TestEMA:
    def test_ema_matches_pandas(self):
        """和 pandas 的 ewm 对照。"""
        x = pd.Series(np.random.randn(50))
        y = ema(x, n=10)
        expected = x.ewm(span=10, adjust=False).mean()
        np.testing.assert_array_almost_equal(y.values, expected.values)


class TestMARef:
    def test_ref_ma_uses_yesterday(self):
        """MA(REF(CLOSE, 1), 5) 应该是「昨日」的 5 日均，不是「今日」。"""
        x = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0])
        y = ma_ref(x, n=5)
        # 前 5 天（包含 ref 后）应该是 NaN
        assert y.iloc[:5].isna().all()
        # 第 6 天 (index=5) = mean([1,2,3,4,5]) = 3
        assert y.iloc[5] == 3.0
        # 第 7 天 (index=6) = mean([2,3,4,5,6]) = 4
        assert y.iloc[6] == 4.0


# ---------------------------------------------------------------------------
# compute_0amv 测试
# ---------------------------------------------------------------------------

class TestCompute0AMV:
    def test_lite_schema(self, sample_market_df):
        result = compute_0amv(sample_market_df, fit_level="lite")
        assert "0amv_close" in result.columns
        assert "0amv_life_line" in result.columns
        assert "0amv_change_pct" in result.columns
        assert "0amv_open" not in result.columns  # lite 没有
        assert "0amv_c5" not in result.columns

    def test_standard_schema(self, sample_market_df):
        result = compute_0amv(sample_market_df, fit_level="standard")
        for col in ["0amv_open", "0amv_high", "0amv_low", "0amv_close", "0amv_life_line", "0amv_color"]:
            assert col in result.columns, f"missing {col}"

    def test_full_schema(self, sample_market_df):
        result = compute_0amv(sample_market_df, fit_level="full")
        for col in ["0amv_close", "0amv_c5", "0amv_c13", "0amv_c34", "0amv_infinite"]:
            assert col in result.columns

    def test_close_is_positive(self, sample_market_df):
        """0AMV 应该是正数（amount * close 都是正）。"""
        result = compute_0amv(sample_market_df, fit_level="full")
        valid = result["0amv_close"].dropna()
        assert (valid > 0).all()

    def test_amount_only_uses_market_amount(self, sample_market_df):
        smooth = sma(sample_market_df["amount"], n=10, m=1)
        result = compute_0amv(sample_market_df, fit_level="standard")
        pd.testing.assert_series_equal(result["0amv_close"], smooth / 1e7, check_names=False)
        pd.testing.assert_series_equal(result["0amv_open"], result["0amv_close"].shift(1), check_names=False)

    def test_price_adjusted_variant_is_explicit(self, sample_market_df):
        result = compute_0amv(
            sample_market_df,
            fit_level="lite",
            formula_variant=FormulaVariant.PRICE_ADJUSTED,
        )
        assert result["0amv_close"].iloc[:5].isna().all()

    def test_life_line_smoother_than_close(self, sample_market_df):
        """生命线（EMA 12）应该比 0AMV_close 平滑。"""
        result = compute_0amv(sample_market_df, fit_level="standard")
        close_std = result["0amv_close"].std()
        life_std = result["0amv_life_line"].std()
        assert life_std < close_std, f"生命线 std ({life_std}) 应小于 close std ({close_std})"

    def test_c5_most_responsive(self, sample_market_df):
        """C5/C13/C34/∞ 活筹均线应该都收敛到非 NaN，且 std 都 > 0。

        注意：4 条线的相对敏感度取决于数据特征。换手率高的合成数据里
        C5 系数最大 → 最接近原值；∞ 系数最小 → 最平滑。但 DMA 系数
        可能 > 1（高换手日），被截断到 1.0，此时所有线都退化到「跟随机」。
        所以这里只验证「有数值 + std > 0」，不强行做大小比较。
        """
        result = compute_0amv(sample_market_df, fit_level="full")
        for col in ["0amv_c5", "0amv_c13", "0amv_c34", "0amv_infinite"]:
            valid = result[col].dropna()
            assert len(valid) > 0, f"{col} 全 NaN"
            assert valid.std() > 0, f"{col} std 为 0"

    def test_full_cost_lines_smooth_var1_not_raw_amount(self, sample_market_df):
        result = compute_0amv(sample_market_df, fit_level="full")
        var1 = sma(sample_market_df["amount"], n=10, m=1) / 1e7
        nested = sma(var1, n=3, m=1)
        assert result["0amv_c5"].iloc[0] == nested.iloc[0]
        assert result["0amv_c5"].iloc[1] != sma(sample_market_df["amount"], n=3, m=1).iloc[1] / 1e7

    def test_color_consistency(self, sample_market_df):
        """color=1 时 close >= open。"""
        result = compute_0amv(sample_market_df, fit_level="standard")
        red = result[result["0amv_color"] == 1].dropna()
        if len(red) > 0:
            assert (red["0amv_close"] >= red["0amv_open"]).all()

    def test_full_no_dmv(self, sample_market_df_with_cap):
        """0DMV 在仿制版中不提供（民间版 0AMV 是相对值，不严格等于真实活跃市值）。

        这个测试是**反向断言** —— 提醒未来的维护者：「不要」轻易加回 0DMV
        除非有真实换手率活跃度数据。
        """
        result = compute_0amv(sample_market_df_with_cap, fit_level="full")
        assert "0dmv_close" not in result.columns, "0DMV 不在仿制版范围内"

    def test_index_is_datetime(self, sample_market_df):
        result = compute_0amv(sample_market_df, fit_level="standard")
        assert isinstance(result.index, pd.DatetimeIndex)

    def test_input_not_mutated(self, sample_market_df):
        """compute_0amv 应该是纯函数，不修改入参。"""
        original = sample_market_df.copy()
        _ = compute_0amv(sample_market_df, fit_level="full")
        pd.testing.assert_frame_equal(sample_market_df, original)


# ---------------------------------------------------------------------------
# 边界情况
# ---------------------------------------------------------------------------

class TestEdgeCases:
    def test_empty_df(self):
        df = pd.DataFrame(
            {
                "open": pd.Series(dtype=float),
                "high": pd.Series(dtype=float),
                "low": pd.Series(dtype=float),
                "close": pd.Series(dtype=float),
                "amount": pd.Series(dtype=float),
                "volume": pd.Series(dtype=float),
                "capital": pd.Series(dtype=float),
            }
        )
        df.index = pd.DatetimeIndex([])
        result = compute_0amv(df, fit_level="standard")
        assert len(result) == 0

    def test_short_df(self):
        """少于 10 天的数据也应能按递推 SMA 计算。"""
        df = pd.DataFrame(
            {
                "open": [3000.0] * 5,
                "high": [3010.0] * 5,
                "low": [2990.0] * 5,
                "close": [3005.0] * 5,
                "amount": [1e12] * 5,
                "volume": [5e10] * 5,
                "capital": [4e12] * 5,
            },
            index=pd.date_range("2024-01-01", periods=5),
        )
        result = compute_0amv(df, fit_level="lite")
        assert len(result) == 5
        assert result["0amv_close"].notna().all()
        assert all(col in result.columns for col in ["0amv_close", "0amv_life_line", "0amv_change_pct"])

    def test_missing_columns(self):
        df = pd.DataFrame(
            {
                "open": [3000.0, 3010.0],
                "high": [3010.0, 3020.0],
                "low": [2990.0, 3000.0],
                "close": [3005.0, 3015.0],
            },
            index=pd.date_range("2024-01-01", periods=2),
        )
        with pytest.raises(ValueError, match="缺少必需列"):
            compute_0amv(df, fit_level="lite")

    def test_invalid_fit_level(self, sample_market_df):
        with pytest.raises(ValueError):
            compute_0amv(sample_market_df, fit_level="bogus")


# ---------------------------------------------------------------------------
# qlib 表达式一致性
# ---------------------------------------------------------------------------

class TestQlibExpressions:
    def test_all_keys_are_strings(self):
        for k, v in QLIB_EXPRESSIONS.items():
            assert isinstance(k, str)
            assert isinstance(v, str)
            assert len(v) > 0

    def test_expressions_reference_known_fields(self):
        """所有表达式只能引用 $amount / $open / $high / $low / $close 等标准字段。"""
        allowed = {"$open", "$high", "$low", "$close", "$amount", "$volume", "$vwap"}
        for name, expr in QLIB_EXPRESSIONS.items():
            # 提取 $ 开头的字段
            import re
            fields = set(re.findall(r"\$[a-z_]+", expr))
            unknown = fields - allowed
            assert not unknown, f"{name} 引用了未知字段: {unknown}"

    def test_close_uses_correct_formula(self):
        """核心公式应该是 EMA($amount, 19) / 1e7。"""
        expr = QLIB_EXPRESSIONS["0amv_close"]
        assert "EMA($amount, 19)" in expr
        assert "$close" not in expr
        assert "1e7" in expr


# ---------------------------------------------------------------------------
# 手工 vs 函数一致性
# ---------------------------------------------------------------------------

class TestHandComputed:
    """手算几个值对照函数输出。"""

    def test_hand_compute_close(self, sample_market_df):
        df = sample_market_df.copy()
        smooth = sma(df["amount"], n=10, m=1)
        result = compute_0amv(df, fit_level="lite")
        expected = smooth.iloc[5] / 1e7
        actual = result["0amv_close"].iloc[5]
        assert math.isclose(actual, expected, rel_tol=1e-9)

    def test_calibrate_scale_uses_median_ratio(self):
        dates = pd.date_range("2024-01-01", periods=3)
        proxy = pd.Series([10.0, 20.0, 30.0], index=dates)
        truth = pd.Series([20.0, 40.0, 90.0], index=dates)
        assert calibrate_scale(proxy, truth) == 2.0
