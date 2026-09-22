"""
0AMV 仿版 vs 指南针原版 K 线对比图渲染脚本。

输入: 用东方财富沪深指数成交额聚合真实市场 amount，跑 zero_amv.py。
输出:
  - output/imitate_0amv_kline.png  —— 仿版 0AMV K 线图（指南针配色）
  - output/imitate_0amv_full.png   —— 仿版 0AMV K 线 + 生命线 + 5/13 均线
  - output/imitate_0amv_compare.png —— 并排对比图
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
from matplotlib.patches import Rectangle
from matplotlib.lines import Line2D

# 把 0amv 根目录加到 path
sys.path.insert(0, str(Path(__file__).parent.parent))

from zero_amv import compute_0amv, FitLevel
from market_data import load_mainland_market_amount

plt.rcParams["font.sans-serif"] = ["PingFang SC", "Arial Unicode MS", "DejaVu Sans"]
plt.rcParams["font.monospace"] = ["Arial Unicode MS", "DejaVu Sans Mono"]
plt.rcParams["axes.unicode_minus"] = False

# ============================================================================
# 指南针官方配色 (从 M3 视觉分析得出)
# ============================================================================
COLORS = {
    "bg": "#000000",          # 纯黑底
    "fg": "#FFFFFF",          # 白色文字
    "k_red": "#FF0033",       # 红涨（指南针特色：饱和红）
    "k_cyan": "#00AACC",      # 青蓝跌（指南针特色：不是绿色！）
    "ma5": "#FFFF00",         # EMA12 代理线（黄）
    "ma13": "#CC00CC",        # 民间公式 C13（紫）
    "life_line": "#00CCFF",   # 生命线（青色）
    "grid": "#333333",        # 暗灰网格
    "annotation": "#FF6600",  # 数值标注（橙）
    "highlight": "#FF0033",   # 当前值高亮（红底白字）
}


# ============================================================================
# 指南针风格 K 线绘制函数
# ============================================================================

def draw_compass_kline(
    ax: plt.Axes,
    df_amv: pd.DataFrame,
    title: str = "",
    n_recent: int = 100,
    show_ma13: bool = True,
    show_ma34: bool = False,
) -> None:
    """画一张仿指南针风格的 0AMV K 线图（V2：带 3D 立体感 + 多均线）。"""
    if len(df_amv) > n_recent:
        df_amv = df_amv.tail(n_recent).copy()

    # 背景
    ax.set_facecolor(COLORS["bg"])

    # 取 K 线 OHLC
    opens = df_amv["0amv_open"].values
    highs = df_amv["0amv_high"].values
    lows = df_amv["0amv_low"].values
    closes = df_amv["0amv_close"].values
    dates = df_amv.index

    # 计算每根 K 线的 x 位置
    x = np.arange(len(df_amv))
    width = 0.55  # 实体宽度

    # ----------------------------------------------------------------------
    # K 线绘制（3D 立体效果：实色主体 + 深色阴影 + 亮色高光）
    # ----------------------------------------------------------------------
    for i in range(len(df_amv)):
        if np.isnan(opens[i]) or np.isnan(closes[i]):
            continue
        is_red = closes[i] >= opens[i]
        main_color = COLORS["k_red"] if is_red else COLORS["k_cyan"]
        # 指南针 K 线：阳线空心/阴线实心（也常见反过来的）
        # 我们用 阳线=实心+亮边 / 阴线=实心+暗边
        body_low = min(opens[i], closes[i])
        body_high = max(opens[i], closes[i])
        body_height = body_high - body_low

        if body_height < 1e-6:
            # 十字星
            ax.plot([x[i] - width / 2, x[i] + width / 2], [body_low, body_low],
                    color=main_color, linewidth=1.5, zorder=3)
            ax.plot([x[i], x[i]], [lows[i], highs[i]],
                    color=main_color, linewidth=1, zorder=2)
        else:
            # 1) 主体填充
            rect = Rectangle(
                (x[i] - width / 2, body_low), width, body_height,
                facecolor=main_color, edgecolor=main_color,
                linewidth=0.6, zorder=3,
            )
            ax.add_patch(rect)
            # 2) 立体阴影（指南针特色）：在实体右侧加一条更暗的窄条
            shadow_color = "#660000" if is_red else "#003344"
            shadow_width = 0.06
            shadow = Rectangle(
                (x[i] + width / 2 - shadow_width, body_low),
                shadow_width, body_height,
                facecolor=shadow_color, edgecolor="none", zorder=3.1,
            )
            ax.add_patch(shadow)
            # 3) 立体高光（左侧亮条）
            highlight_color = "#FF8888" if is_red else "#88DDFF"
            highlight_width = 0.04
            highlight = Rectangle(
                (x[i] - width / 2, body_low),
                highlight_width, body_height,
                facecolor=highlight_color, edgecolor="none", zorder=3.2,
            )
            ax.add_patch(highlight)
            # 4) 影线
            ax.plot([x[i], x[i]], [lows[i], body_low],
                    color=main_color, linewidth=0.8, zorder=2)
            ax.plot([x[i], x[i]], [body_high, highs[i]],
                    color=main_color, linewidth=0.8, zorder=2)

    # ----------------------------------------------------------------------
    # 平滑线：EMA12（黄色）+ 民间公式 C13/C34（可选）
    # ----------------------------------------------------------------------
    if "0amv_life_line" in df_amv.columns:
        ax.plot(x, df_amv["0amv_life_line"].values,
                color=COLORS["ma5"], linewidth=1.4, zorder=4.5,
                label="EMA12 代理线", antialiased=True)
    if show_ma13 and "0amv_c13" in df_amv.columns:
        c13 = df_amv["0amv_c13"].values
        valid = ~np.isnan(c13)
        if valid.sum() > 0:
            ax.plot(x[valid], c13[valid],
                    color=COLORS["ma13"], linewidth=1.2, zorder=4.3,
                    label="民间公式 C13", antialiased=True)
    if show_ma34 and "0amv_c34" in df_amv.columns:
        c34 = df_amv["0amv_c34"].values
        valid = ~np.isnan(c34)
        if valid.sum() > 0:
            ax.plot(x[valid], c34[valid],
                    color="#FF8800", linewidth=1.0, zorder=4.2,
                    label="活筹 C34", antialiased=True, linestyle="--")

    # X 轴日期
    if len(df_amv) > 0:
        step = max(1, len(df_amv) // 10)
        tick_positions = x[::step]
        tick_labels = [d.strftime("%m.%d") for d in dates[::step]]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, color=COLORS["fg"], fontsize=8)
        last_x = x[-1]
        ax.axvline(last_x, color=COLORS["highlight"], linestyle=":", alpha=0.5, linewidth=0.8)

    # Y 轴右侧
    ax.yaxis.tick_right()
    ax.yaxis.set_label_position("right")
    ax.tick_params(axis="y", colors=COLORS["fg"], labelsize=8)
    ax.tick_params(axis="x", colors=COLORS["fg"], labelsize=8)

    # 横向虚线网格
    ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.5, alpha=0.6)
    ax.set_axisbelow(True)

    # 边框
    for spine in ax.spines.values():
        spine.set_color(COLORS["fg"])
        spine.set_linewidth(0.8)

    # ----------------------------------------------------------------------
    # 左上角：仿指南针"参数面板"UI（多个文本框 + 参数值）
    # ----------------------------------------------------------------------
    last_life = df_amv["0amv_life_line"].iloc[-1] if "0amv_life_line" in df_amv.columns else np.nan
    last_c13 = df_amv["0amv_c13"].iloc[-1] if "0amv_c13" in df_amv.columns else np.nan
    last_close = df_amv["0amv_close"].iloc[-1] if "0amv_close" in df_amv.columns else np.nan
    last_high = df_amv["0amv_high"].iloc[-1] if "0amv_high" in df_amv.columns else np.nan
    last_low = df_amv["0amv_low"].iloc[-1] if "0amv_low" in df_amv.columns else np.nan
    last_open = df_amv["0amv_open"].iloc[-1] if "0amv_open" in df_amv.columns else np.nan

    # 红色边框信息框（指南针"活跃市值"标题）
    info_box_y = 0.97
    ax.text(0.02, info_box_y, "活跃市值 0AMV",
            transform=ax.transAxes, color=COLORS["ma5"],
            fontsize=10, weight="bold", verticalalignment="top",
            family="monospace")
    # 平滑线参数
    if not np.isnan(last_life):
        ax.text(0.02, info_box_y - 0.06,
                "平滑线",
                transform=ax.transAxes, color=COLORS["fg"],
                fontsize=8, family="monospace", verticalalignment="top")
    if not np.isnan(last_life):
        ax.text(0.02, info_box_y - 0.10,
                f"EMA12: {last_life:,.2f}",
                transform=ax.transAxes, color=COLORS["ma5"],
                fontsize=9, family="monospace", weight="bold", verticalalignment="top")
    if not np.isnan(last_c13) and show_ma13:
        ax.text(0.02, info_box_y - 0.14,
                f"C13: {last_c13:,.2f}",
                transform=ax.transAxes, color=COLORS["ma13"],
                fontsize=9, family="monospace", weight="bold", verticalalignment="top")

    # 右上角：当前 OHLC 数值（指南针风格）
    if not np.isnan(last_close):
        # 高亮当前值
        ax.text(0.98, 0.96,
                f"收:{last_close:,.2f}",
                transform=ax.transAxes, color=COLORS["highlight"],
                fontsize=10, family="monospace", weight="bold",
                verticalalignment="top", horizontalalignment="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#330000",
                          edgecolor=COLORS["highlight"], linewidth=0.8))
    # 右上角日期高亮（指南针红底白字）
    last_date = dates[-1]
    ax.text(0.98, 0.02, f"{last_date.strftime('%Y-%m-%d')}",
            transform=ax.transAxes, color=COLORS["fg"],
            fontsize=8, family="monospace",
            verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.3", facecolor=COLORS["highlight"],
                      edgecolor="none"))

    # 右侧 Y 轴数值标签
    valid_closes = closes[~np.isnan(closes)]
    if len(valid_closes) > 0:
        y_min, y_max = valid_closes.min(), valid_closes.max()
        yticks = np.linspace(y_min, y_max, 6)
        ax.set_yticks(yticks)
        ax.set_yticklabels([f"{y:,.0f}" for y in yticks])

    # 标题
    ax.set_title(title, color=COLORS["fg"], fontsize=11, pad=10,
                 family="monospace")


def draw_volume_panel(ax: plt.Axes, df_amv: pd.DataFrame) -> None:
    """画成交量子图（指南针风格）。"""
    ax.set_facecolor(COLORS["bg"])
    if "0amv_volume" in df_amv.columns:
        volumes = df_amv["0amv_volume"].values
    else:
        # 用 amount 当 volume 代理
        volumes = df_amv["0amv_amount"].values if "0amv_amount" in df_amv.columns else None

    if volumes is None or len(df_amv) == 0:
        ax.text(0.5, 0.5, "(no volume data)", transform=ax.transAxes,
                color=COLORS["fg"], ha="center", va="center")
        return

    x = np.arange(len(df_amv))
    closes = df_amv["0amv_close"].values
    opens = df_amv["0amv_open"].values
    width = 0.6

    for i in range(len(df_amv)):
        if np.isnan(volumes[i]):
            continue
        is_red = closes[i] >= opens[i]
        color = COLORS["k_red"] if is_red else COLORS["k_cyan"]
        ax.bar(x[i], volumes[i], width=width, color=color, alpha=0.6, zorder=2)

    ax.yaxis.tick_right()
    ax.tick_params(axis="y", colors=COLORS["fg"], labelsize=7)
    ax.tick_params(axis="x", colors=COLORS["fg"], labelsize=7)
    ax.grid(True, axis="y", color=COLORS["grid"], linestyle=":", linewidth=0.5, alpha=0.6)
    for spine in ax.spines.values():
        spine.set_color(COLORS["fg"])
        spine.set_linewidth(0.5)


# ============================================================================
# 主流程
# ============================================================================

def run_validation(n_days: int = 100) -> dict[str, Path]:
    """用沪深市场真实成交额画仿版 0AMV 图。"""

    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    print("正在拉取沪深市场真实成交额...")
    df = load_mainland_market_amount("2014-01-01", pd.Timestamp.today().strftime("%Y-%m-%d"))

    print(f"数据范围: {df.index[0].date()} — {df.index[-1].date()}, {len(df)} 个交易日")
    print(f"amount 范围 (亿元): {df['amount'].min()/1e8:,.0f} — {df['amount'].max()/1e8:,.0f}")

    # 跑 0AMV
    print("正在计算 0AMV (standard 层级)...")
    result = compute_0amv(df, fit_level=FitLevel.STANDARD)
    print(f"0AMV_close 范围: {result['0amv_close'].min():,.0f} — {result['0amv_close'].max():,.0f}")

    # 给 figure 准备 amount/volume 透传（绘图用）
    result["0amv_amount"] = df["amount"]

    # ====== 1) 主 K 线图（指南针风格） ======
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS["bg"])
    draw_compass_kline(
        ax, result,
        title=f"0AMV 民间公式代理 — 沪深市场真实成交额 / {df.index[0].date()} ~ {df.index[-1].date()}",
        n_recent=n_days,
    )
    out1 = out_dir / "imitate_0amv_kline.png"
    plt.tight_layout()
    plt.savefig(out1, dpi=120, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"✅ 主 K 线图: {out1}")

    # ====== 2) K 线 + 生命线 + 活筹均线 ======
    fig, ax = plt.subplots(figsize=(14, 7), facecolor=COLORS["bg"])
    draw_compass_kline(ax, result, n_recent=n_days)
    # 叠加 C13 / C34
    if "0amv_c13" in result.columns:
        x = np.arange(len(result))
        c13 = result["0amv_c13"].values
        c34 = result["0amv_c34"].values
        valid_c13 = c13[~np.isnan(c13)]
        if len(valid_c13) > 0:
            ax.plot(x, c13, color=COLORS["ma13"], linewidth=1.0, alpha=0.85,
                    label="活筹 C13", zorder=4)
        valid_c34 = c34[~np.isnan(c34)]
        if len(valid_c34) > 0:
            ax.plot(x, c34, color="#FF8800", linewidth=0.8, alpha=0.7,
                    label="活筹 C34", zorder=4)
    ax.legend(loc="upper left", facecolor=COLORS["bg"], edgecolor=COLORS["fg"],
              labelcolor=COLORS["fg"], fontsize=8)
    out2 = out_dir / "imitate_0amv_full.png"
    plt.savefig(out2, dpi=120, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"✅ K 线 + 均线: {out2}")

    # ====== 3) 统计摘要图 ======
    fig, axes = plt.subplots(3, 1, figsize=(14, 10), facecolor=COLORS["bg"])

    # 上：K 线
    draw_compass_kline(axes[0], result, title="0AMV K 线", n_recent=n_days)

    # 中：0AMV_close 与生命线
    axes[1].set_facecolor(COLORS["bg"])
    x = np.arange(len(result))
    axes[1].plot(x, result["0amv_close"], color=COLORS["k_red"],
                 linewidth=1.2, label="0AMV_close")
    axes[1].plot(x, result["0amv_life_line"], color=COLORS["life_line"],
                 linewidth=1.2, label="生命线 (EMA12)")
    axes[1].yaxis.tick_right()
    axes[1].tick_params(axis="y", colors=COLORS["fg"], labelsize=8)
    axes[1].tick_params(axis="x", colors=COLORS["fg"], labelsize=8)
    axes[1].grid(True, color=COLORS["grid"], linestyle=":", alpha=0.5)
    axes[1].set_title("0AMV 收盘价 + 生命线", color=COLORS["fg"],
                      family="sans-serif", fontsize=10)
    axes[1].legend(loc="upper left", facecolor=COLORS["bg"],
                   edgecolor=COLORS["fg"], labelcolor=COLORS["fg"], fontsize=8)
    for spine in axes[1].spines.values():
        spine.set_color(COLORS["fg"])

    # 下：涨跌幅直方图
    axes[2].set_facecolor(COLORS["bg"])
    change_pct = result["0amv_change_pct"].dropna()
    if len(change_pct) > 0:
        colors_hist = [COLORS["k_red"] if x >= 0 else COLORS["k_cyan"] for x in change_pct]
        axes[2].bar(range(len(change_pct)), change_pct.values, color=colors_hist, alpha=0.8)
    axes[2].axhline(0, color=COLORS["fg"], linewidth=0.5)
    axes[2].yaxis.tick_right()
    axes[2].tick_params(axis="y", colors=COLORS["fg"], labelsize=8)
    axes[2].tick_params(axis="x", colors=COLORS["fg"], labelsize=8)
    axes[2].grid(True, axis="y", color=COLORS["grid"], linestyle=":", alpha=0.5)
    axes[2].set_title("0AMV 日涨跌幅 (%)", color=COLORS["fg"],
                      family="sans-serif", fontsize=10)
    for spine in axes[2].spines.values():
        spine.set_color(COLORS["fg"])

    plt.tight_layout()
    out3 = out_dir / "imitate_0amv_full_panel.png"
    plt.savefig(out3, dpi=120, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"✅ 完整统计: {out3}")

    return {
        "kline": out1,
        "kline_with_ma": out2,
        "full_panel": out3,
    }


if __name__ == "__main__":
    paths = run_validation(n_days=100)
    print("\n=== 输出文件 ===")
    for k, v in paths.items():
        print(f"  {k}: {v}")
