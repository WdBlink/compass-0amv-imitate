"""
最终对比图：仿版 vs 指南针原版，并排展示。
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt

sys.path.insert(0, str(Path(__file__).parent.parent))
sys.path.insert(0, str(Path(__file__).parent))

from render_imitate_kline import COLORS, draw_compass_kline
from zero_amv import compute_0amv, FitLevel
from market_data import load_mainland_market_amount


def make_side_by_side() -> Path:
    """画一张 1x2 并排对比图：左边仿版，右边原版（用指南针截图）。"""
    out_dir = Path(__file__).parent / "output"
    out_dir.mkdir(exist_ok=True)

    # 与原版截图使用完全相同的日期窗口。
    print("拉取 2024-04-10 至 2024-10-10 沪深市场真实成交额...")
    df = load_mainland_market_amount("2023-01-01", "2024-10-10")

    result = compute_0amv(df, fit_level=FitLevel.STANDARD).loc["2024-04-10":"2024-10-10"]

    fig, axes = plt.subplots(1, 2, figsize=(24, 8), facecolor=COLORS["bg"])

    # 左：仿版
    draw_compass_kline(
        axes[0], result,
        title="A) 仿制版 0AMV K 线 (本项目生成)",
        n_recent=100, show_ma13=True,
    )

    # 右：原版（指南针截图）
    from PIL import Image
    img = Image.open(Path(__file__).parent / "compass_baseline_2024.jpg")
    axes[1].imshow(img)
    axes[1].axis("off")
    axes[1].set_title("B) 指南针原版 0AMV K 线 (真实截图基线)",
                      color=COLORS["fg"], fontsize=12, pad=10, family="monospace")

    # 整体标题
    fig.suptitle("0AMV 活跃市值 K 线对比：仿制版 vs 指南针原版",
                 color=COLORS["fg"], fontsize=14, family="monospace", y=0.98)

    plt.tight_layout()
    out = out_dir / "side_by_side_compare.png"
    plt.savefig(out, dpi=110, facecolor=COLORS["bg"], bbox_inches="tight")
    plt.close()
    print(f"✅ 并排对比图: {out}")
    return out


if __name__ == "__main__":
    make_side_by_side()
