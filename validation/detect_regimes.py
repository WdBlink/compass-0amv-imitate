"""
0AMV 多空区间识别器

规则（用户定义）:
- 多头区间进入: 单日 0AMV 涨跌幅 >= +3.0%  OR  过去两日累计 >= +4.0%
- 空头区间进入: 单日 0AMV 涨跌幅 <= -2.3%

策略:
- 区间开始: 触发日
- 区间结束: 下一日不满足"保持条件"（保持条件可调：触发后只要不出现反向触发就算继续）
- 实际采用更宽松的规则: 触发后连续 5 个交易日内不再触发反向则区间继续
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from zero_amv import compute_0amv, FitLevel
from render_imitate_kline import COLORS, draw_compass_kline

import akshare as ak


# ---------------------------------------------------------------------------
# 规则引擎
# ---------------------------------------------------------------------------

def detect_regimes(
    amv_change_pct: pd.Series,
    long_threshold: float = 3.0,
    long_2day_threshold: float = 4.0,
    short_threshold: float = -2.3,
    exit_1d_long: float = -3.5,
    exit_2d_long: float = -5.0,
    exit_1d_short: float = 4.5,
    exit_2d_short: float = 6.0,
    long_split_days: int = 12,
) -> pd.DataFrame:
    """检测多空区间。

    启动规则（用户给）:
    - 多头进入: 单日 0AMV 涨跌幅 >= +3.0%  OR  过去两日累计 >= +4.0%
    - 空头进入: 单日 0AMV 涨跌幅 <= -2.3%

    退出规则（更强的反向触发 + 长区间放宽）:
    - 短区间 (< long_split_days): 单日 exit_1d_long 触发即退
    - 长区间 (>= long_split_days): 单日 exit_1d_long 触发 OR 累计反转 >= long_split_days 天内的累计涨幅 < 峰值一半

    Args:
        amv_change_pct: 0AMV 日涨跌幅（百分比），索引为 date
        long_threshold: 多头启动单日阈值
        long_2day_threshold: 多头启动两日累计阈值
        short_threshold: 空头启动单日阈值
        exit_1d_long: 多头退出单日阈值（默认 -3.5%）
        exit_2d_long: 多头退出两日累计阈值（默认 -5.0%）
        exit_1d_short: 空头退出单日阈值（默认 +4.5%）
        exit_2d_short: 空头退出两日累计阈值（默认 +6.0%）
        long_split_days: 长短区间分界天数（默认 25 天）

    Returns:
        DataFrame，列: start, end, type, max_runup, max_drawdown, days
    """
    s = amv_change_pct.dropna()
    n = len(s)
    dates = s.index

    # 启动阈值
    long_1d = s.values >= long_threshold
    long_2d = np.zeros(n, dtype=bool)
    for i in range(1, n):
        long_2d[i] = (s.values[i] + s.values[i - 1]) >= long_2day_threshold
    short_1d = s.values <= short_threshold
    long_start = long_1d | long_2d
    short_start = short_1d

    # 退出阈值（单日 + 两日累计）
    long_exit_1d = s.values <= exit_1d_long
    long_exit_2d = np.zeros(n, dtype=bool)
    for i in range(1, n):
        long_exit_2d[i] = (s.values[i] + s.values[i - 1]) <= exit_2d_long
    long_exit = long_exit_1d | long_exit_2d

    short_exit_1d = s.values >= exit_1d_short
    short_exit_2d = np.zeros(n, dtype=bool)
    for i in range(1, n):
        short_exit_2d[i] = (s.values[i] + s.values[i - 1]) >= exit_2d_short
    short_exit = short_exit_1d | short_exit_2d

    # 状态机
    state = "neutral"
    state_start = None
    regimes = []
    cum = 0.0
    peak = 0.0
    trough = 0.0

    def append_regime(state_start_idx, end_idx, regime_type):
        regimes.append({
            "start": dates[state_start_idx],
            "end": dates[end_idx],
            "type": regime_type,
            "max_runup": s.values[state_start_idx:end_idx + 1].max(),
            "max_drawdown": s.values[state_start_idx:end_idx + 1].min(),
            "days": end_idx - state_start_idx + 1,
        })

    for i in range(n):
        if state == "neutral":
            if long_start[i]:
                state = "long"
                state_start = i
                cum = s.values[i]
                peak = cum
            elif short_start[i]:
                state = "short"
                state_start = i
                cum = s.values[i]
                trough = cum
        elif state == "long":
            cum += s.values[i]
            peak = max(peak, cum)
            days_in = i - state_start

            # 退出条件 1: 强单日/两日反向
            exit_now = long_exit[i]

            # 退出条件 2: 长区间（>= long_split_days）的累计涨幅回撤 40%
            if not exit_now and days_in >= long_split_days:
                if peak > 0 and (peak - cum) >= peak * 0.4:
                    exit_now = True
                elif cum <= 0:
                    exit_now = True

            if exit_now:
                append_regime(state_start, i - 1, "long")
                state = "neutral"
                if short_start[i]:
                    state = "short"
                    state_start = i
                    cum = s.values[i]
                    trough = cum
        elif state == "short":
            cum += s.values[i]
            trough = min(trough, cum)
            days_in = i - state_start

            exit_now = short_exit[i]

            if not exit_now and days_in >= long_split_days:
                if trough < 0 and (cum - trough) >= abs(trough) * 0.5:
                    exit_now = True
                elif cum >= 0:
                    exit_now = True

            if exit_now:
                append_regime(state_start, i - 1, "short")
                state = "neutral"
                if long_start[i]:
                    state = "long"
                    state_start = i
                    cum = s.values[i]
                    peak = cum

    # 收尾
    if state != "neutral" and state_start is not None:
        append_regime(state_start, len(s) - 1, state)

    return pd.DataFrame(regimes)


# ---------------------------------------------------------------------------
# 对比评估
# ---------------------------------------------------------------------------

def compare_with_truth(
    detected: pd.DataFrame,
    truth_long_ranges: list[tuple[str, str]],
) -> dict:
    """对比检测到的区间 vs 真实多头区间。

    Args:
        detected: detect_regimes 输出
        truth_long_ranges: 真实多头区间 [(start, end), ...]

    Returns:
        评估报告 dict
    """
    def to_date(s: str) -> pd.Timestamp:
        return pd.Timestamp(s)

    truth = [
        (to_date(s), to_date(e)) for s, e in truth_long_ranges
    ]
    detected_longs = detected[detected["type"] == "long"].copy() if len(detected) > 0 else pd.DataFrame()

    # 命中: 检测到的多头区间跟真实区间有日期重叠
    hits = []
    misses = []  # 没检测到的真实区间
    for ts, te in truth:
        matched = False
        for _, row in detected_longs.iterrows():
            ds, de = row["start"], row["end"]
            # 重叠判定
            if not (de < ts or ds > te):
                hits.append((ts, te, ds, de))
                matched = True
                break
        if not matched:
            misses.append((ts, te))

    # 误报: 检测到的多头区间但没匹配任何真实区间
    false_positives = []
    for _, row in detected_longs.iterrows():
        ds, de = row["start"], row["end"]
        matched = False
        for ts, te in truth:
            if not (de < ts or ds > te):
                matched = True
                break
        if not matched:
            false_positives.append((ds, de))

    precision = len(hits) / max(len(detected_longs), 1) if len(detected_longs) > 0 else 0
    recall = len(hits) / len(truth) if len(truth) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return {
        "truth_count": len(truth),
        "detected_count": len(detected_longs),
        "hits": hits,
        "misses": misses,
        "false_positives": false_positives,
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


# ---------------------------------------------------------------------------
# 主流程
# ---------------------------------------------------------------------------

# 用户提供的指南针真实多头区间 (12 个)
TRUTH_LONG_RANGES = [
    ("2026-04-08", "2026-05-27"),
    ("2026-01-05", "2026-02-02"),
    ("2025-06-25", "2025-09-04"),
    ("2025-04-08", "2025-04-16"),
    ("2025-02-06", "2025-02-28"),
    ("2025-01-14", "2025-01-27"),
    ("2024-08-30", "2024-11-14"),
    ("2024-04-26", "2024-05-15"),
    ("2024-07-09", "2024-07-23"),
    ("2024-07-31", "2024-08-12"),
    ("2024-04-17", "2024-05-15"),
    ("2024-02-06", "2024-03-25"),
    ("2023-12-28", "2024-01-17"),
]


def main() -> None:
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 拉覆盖 2023-12 到 2026-06 的沪深 300 数据
    print("拉真实 A 股数据 (2023-12 ~ 2026-06)...")
    df = ak.stock_zh_index_daily(symbol="sh000300")
    df["date"] = pd.to_datetime(df["date"])
    df = df[df["date"] >= "2023-12-01"].sort_values("date").reset_index(drop=True)
    print(f"  数据范围: {df['date'].iloc[0].date()} — {df['date'].iloc[-1].date()}, {len(df)} 个交易日")

    # 准备 amount (跟之前验证一样：独立噪声)
    # 用更小的 noise 让 amount 更平滑（指南针原版 amount 是全市场聚合，相对稳定）
    np.random.seed(42)
    base_amount = df["close"] * df["volume"]
    activity = np.random.lognormal(mean=0.0, sigma=0.05, size=len(df))  # 5% 日波动
    close_trend = df["close"].pct_change().fillna(0)
    activity = activity * (1 + close_trend * 3)  # 弱相关
    df["amount"] = base_amount * activity
    df["capital"] = 4e12
    df = df.set_index("date")

    # 跑 0AMV
    print("跑 0AMV 计算...")
    result = compute_0amv(df, fit_level=FitLevel.FULL)

    # 检测区间
    print("应用规则检测多空区间...")
    regimes = detect_regimes(
        result["0amv_change_pct"],
        long_threshold=3.0,
        long_2day_threshold=4.0,
        short_threshold=-2.3,
        exit_1d_long=-4.0,
        exit_2d_long=-6.0,
        exit_1d_short=4.5,
        exit_2d_short=6.0,
        long_split_days=20,
    )

    print(f"\n检测到的总区间数: {len(regimes)}")
    print(f"  多头: {(regimes['type'] == 'long').sum() if len(regimes) > 0 else 0}")
    print(f"  空头: {(regimes['type'] == 'short').sum() if len(regimes) > 0 else 0}")

    # 评估
    eval_result = compare_with_truth(regimes, TRUTH_LONG_RANGES)

    print(f"\n=== 拟合度评估 (vs 指南针真实多头区间) ===")
    print(f"  真实多头区间数: {eval_result['truth_count']}")
    print(f"  检测到多头区间数: {eval_result['detected_count']}")
    print(f"  命中: {len(eval_result['hits'])}")
    print(f"  漏报 (真实有但没检出): {len(eval_result['misses'])}")
    print(f"  误报 (检出但不在真实列表): {len(eval_result['false_positives'])}")
    print(f"  Precision: {eval_result['precision']:.1%}")
    print(f"  Recall:    {eval_result['recall']:.1%}")
    print(f"  F1:        {eval_result['f1']:.1%}")

    print(f"\n=== 命中详情 ===")
    for ts, te, ds, de in eval_result["hits"]:
        print(f"  真实 {ts.date()} ~ {te.date()}  -> 检测 {ds.date()} ~ {de.date()}")

    print(f"\n=== 漏报 (真实有但没检出) ===")
    for ts, te in eval_result["misses"]:
        # 看这段时间的 0AMV 实际变化
        sub = result.loc[ts:te, "0amv_change_pct"]
        if len(sub) > 0:
            print(f"  {ts.date()} ~ {te.date()}: "
                  f"变化率 max={sub.max():.2f}%, min={sub.min():.2f}%, "
                  f"两日累计 max={sub.rolling(2).sum().max():.2f}%")
        else:
            print(f"  {ts.date()} ~ {te.date()}: (数据缺失)")

    print(f"\n=== 误报 (检测到但不在真实列表) ===")
    for ds, de in eval_result["false_positives"]:
        sub = result.loc[ds:de, "0amv_change_pct"]
        if len(sub) > 0:
            print(f"  {ds.date()} ~ {de.date()}: "
                  f"变化率 max={sub.max():.2f}%, min={sub.min():.2f}%, "
                  f"days={len(sub)}")

    # 保存结果
    regimes.to_csv(out_dir / "detected_regimes.csv", index=False)
    print(f"\n✅ 区间列表保存到: {out_dir / 'detected_regimes.csv'}")

    # ====== 画对比图 ======
    fig, axes = plt.subplots(2, 1, figsize=(18, 10), facecolor=COLORS["bg"])

    # 上：K 线 + 0AMV close + 标出真实/检测区间
    draw_compass_kline(axes[0], result, title="A) 0AMV K 线 + 真实 vs 检测区间",
                       n_recent=len(result), show_ma13=True)
    # 画真实区间（绿色半透明背景）
    for ts, te in TRUTH_LONG_RANGES:
        ts_dt = pd.Timestamp(ts)
        te_dt = pd.Timestamp(te)
        if ts_dt < result.index[0] or ts_dt > result.index[-1]:
            continue
        ts_idx = result.index.get_indexer([ts_dt], method="nearest")[0]
        te_idx = result.index.get_indexer([te_dt], method="nearest")[0]
        axes[0].axvspan(ts_idx - 0.5, te_idx + 0.5,
                        color="#00FF00", alpha=0.12, zorder=1)
    # 画检测到的多头区间（黄色边框）
    if len(regimes) > 0:
        for _, row in regimes.iterrows():
            if row["type"] != "long":
                continue
            ds_idx = result.index.get_indexer([row["start"]], method="nearest")[0]
            de_idx = result.index.get_indexer([row["end"]], method="nearest")[0]
            axes[0].axvspan(ds_idx - 0.5, de_idx + 0.5,
                            edgecolor="#FFFF00", facecolor="none",
                            linewidth=1.5, linestyle="--", zorder=2)

    # 下：0AMV 日变化率 + 触发阈值线
    axes[1].set_facecolor(COLORS["bg"])
    x = np.arange(len(result))
    change = result["0amv_change_pct"].fillna(0).values
    colors_bar = [COLORS["k_red"] if c >= 0 else COLORS["k_cyan"] for c in change]
    axes[1].bar(x, change, color=colors_bar, alpha=0.7, width=0.8)
    axes[1].axhline(3.0, color=COLORS["k_red"], linewidth=1, linestyle="--",
                    label="多头 +3.0%", zorder=5)
    axes[1].axhline(-2.3, color=COLORS["k_cyan"], linewidth=1, linestyle="--",
                    label="空头 -2.3%", zorder=5)

    # 两日累计线
    two_day_sum = pd.Series(change).rolling(2).sum().values
    axes[1].plot(x, two_day_sum, color="#FFFF00", linewidth=0.6, alpha=0.6,
                 label="两日累计", zorder=4)
    axes[1].axhline(4.0, color="#FFFF00", linewidth=0.5, linestyle=":",
                    alpha=0.4, zorder=3)

    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="y", colors=COLORS["fg"], labelsize=8)
    axes[1].tick_params(axis="x", colors=COLORS["fg"], labelsize=8)
    axes[1].grid(True, axis="y", color=COLORS["grid"], linestyle=":", alpha=0.5)
    axes[1].set_title("B) 0AMV 日变化率 + 触发阈值", color=COLORS["fg"],
                      family="monospace", fontsize=11)
    axes[1].legend(loc="upper left", facecolor=COLORS["bg"],
                   edgecolor=COLORS["fg"], labelcolor=COLORS["fg"], fontsize=8)
    for spine in axes[1].spines.values():
        spine.set_color(COLORS["fg"])

    # X 轴日期格式化 (下子图)
    n = len(result)
    step = max(1, n // 12)
    axes[1].set_xticks(x[::step])
    axes[1].set_xticklabels([d.strftime("%Y-%m") for d in result.index[::step]],
                             color=COLORS["fg"], fontsize=7)

    # 标出真实/检测区间（下子图也画）
    for ts, te in TRUTH_LONG_RANGES:
        ts_dt = pd.Timestamp(ts)
        te_dt = pd.Timestamp(te)
        if ts_dt < result.index[0] or ts_dt > result.index[-1]:
            continue
        ts_idx = result.index.get_indexer([ts_dt], method="nearest")[0]
        te_idx = result.index.get_indexer([te_dt], method="nearest")[0]
        axes[1].axvspan(ts_idx - 0.5, te_idx + 0.5,
                        color="#00FF00", alpha=0.12, zorder=1)
    if len(regimes) > 0:
        for _, row in regimes.iterrows():
            if row["type"] != "long":
                continue
            ds_idx = result.index.get_indexer([row["start"]], method="nearest")[0]
            de_idx = result.index.get_indexer([row["end"]], method="nearest")[0]
            axes[1].axvspan(ds_idx - 0.5, de_idx + 0.5,
                            edgecolor="#FFFF00", facecolor="none",
                            linewidth=1.5, linestyle="--", zorder=2)

    # 图例
    green_patch = mpatches.Patch(color="#00FF00", alpha=0.3,
                                 label="指南针真实多头区间 (绿底)")
    yellow_patch = mpatches.Patch(facecolor="none", edgecolor="#FFFF00",
                                  linewidth=1.5, linestyle="--",
                                  label="仿版检测多头区间 (黄框)")
    # 找到现有 legend 并添加
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([green_patch, yellow_patch])
    axes[0].legend(handles=handles, loc="upper left", facecolor=COLORS["bg"],
                   edgecolor=COLORS["fg"], labelcolor=COLORS["fg"], fontsize=8)

    plt.tight_layout()
    out = out_dir / "regime_detection.png"
    plt.savefig(out, dpi=110, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"✅ 对比图保存到: {out}")

    return eval_result


import matplotlib.patches as mpatches  # 放在最后避免循环引用


if __name__ == "__main__":
    main()
