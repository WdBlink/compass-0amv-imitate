# 0AMV（活跃市值 / 活筹指数）民间公式研究

指南针官方公开了 0AMV 的含义，但没有公开算法。本仓库实现并验证公开流传的公式；输出是成交额代理，**不是已证实的指南针原版，也不是真实活跃筹码市值**。

## 默认公式

```text
VAR1 = SMA(沪深全市场 AMOUNT, 10, 1) / 1e7
close = VAR1
open  = REF(VAR1, 1)
C5    = DMA(SMA(VAR1, 3, 1), VOL / 0.02 / CAPITAL)
C13   = DMA(SMA(VAR1, 3, 1), VOL / 0.10 / CAPITAL)
C34   = DMA(SMA(VAR1, 3, 1), VOL / 0.18 / CAPITAL)
∞     = DMA(VAR1, VOL / 1.10 / CAPITAL)
```

这里的 `AMOUNT` 必须是同一口径的沪深全市场成交额。不要使用沪深300成交量乘指数点位，也不要把随机生成的数据称作真实市场数据。

公开资料中还存在 `VAR1 × CLOSE / MA(REF(CLOSE, 1), 5)` 版本，它更常以“资金起爆”等名称传播。本仓库通过 `formula_variant="price_adjusted"` 显式保留，但不再将它冒充唯一的 0AMV 公式。

## 使用

```python
from zero_amv import compute_0amv

# 默认 amount_only 只要求 amount（元）。
result = compute_0amv(df[["amount"]], fit_level="standard")

# Full 额外要求同口径的 volume（股）与 capital（流通股本，股）。
full = compute_0amv(df[["amount", "volume", "capital"]], fit_level="full")
```

`scale_factor` 默认是 `1.0`。如果有同日期指南针导出值，可用 `calibrate_scale(proxy, truth)` 标定；单点缩放不得跨时期宣称有效。

## 当前验证边界

仓库截图中的红色横轴/纵轴标签来自十字光标，不能当作最后一根 K 线的日期和收盘值。现有截图只能做同窗口形态检查，不能计算绝对误差，也不能支持“95%+”或“绝对值 80%”之类结论。

复现检查：

```bash
python validation/make_side_by_side.py  # 使用与 2024 截图相同的日期窗口
pytest -q
```

网络验证数据来自东方财富历史日线接口；`validation/market_data.py` 同时保存沪、深成交额列，便于检查聚合口径。缺少指南针原版数值序列是当前结论的主要限制。

## 不能从本项目推出的结论

- `0AMV + 0DMV = 流通市值`：只适用于指南针原版定义，民间成交额代理不满足。
- “个股版 0AMV”：把单只股票成交额代入只能得到成交额代理，不是大盘活跃市值。
- 多空区间 F1 等于公式拟合度：区间规则还包含人工阈值和状态机，必须单独评估。

进一步提高精度需要同日期连续的指南针 0AMV 导出序列。拿到后应固定训练/样本外窗口，报告相关系数、方向一致率、MAE/MAPE，并公开股票池、单位和复权口径；在此之前不做时间插值或按截图逐点过拟合。

## License

MIT。公开民间公式仅供研究。
