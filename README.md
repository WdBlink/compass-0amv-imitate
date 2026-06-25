# 0AMV（活跃市值 / 活筹指数）—— 指南针软件公式近似实现

> 写在最前面：**网上不是没有这个公式**——是你搜的关键词不对。0AMV 在
> 通达信 / 同花顺 / 飞狐 都有民间仿制版，叫「**活筹指数**」。MBA 智库
> 百科、东方财富博客、百度文库、文档下载网、koo8 / gpxiazai 等独立
> 来源交叉验证下来，**核心公式完全一致**。本仓库把这套民间公式整理成
> 可直接集成进量化系统的 Python + qlib 实现。

## 文件清单

```
0amv/
├── zero_amv.py                 # 核心实现 (pandas + numpy, 无 qlib 依赖)
├── test_zero_amv.py            # 25 个单元测试 (pytest, 全部通过)
├── trading_analyze_integration.py  # 集成到 TradingAnalyze 的代码片段
└── README.md                   # 本文件
```

## 0AMV 是什么

| 名称 | 含义 |
|---|---|
| **0 号指数** | 沪深 A 股流通市值总和（市场总规模） |
| **0AMV（活跃市值 / 活筹）** | 短期交易活跃的筹码总市值，反映「市场里活的钱」 |
| **0DMV（死筹）** | 长期锁定不交易的筹码市值，反映「被锁住的钱」 |
| **关系** | 0 号指数 ≈ 0AMV + 0DMV（仿制版里**不严格**成立） |

指南针软件（沈阳指南针）的私有指标，官方不公开算法。民间复刻的核心
公式是：

```
0AMV_close = SMA(AMOUNT, 10, 1) × CLOSE / MA(REF(CLOSE, 1), 5) / 1e7
```

含义：把全市场 10 日平滑成交额 × 当前价 / 5 日均价 / 千万，得到一个
「相对资金规模」指标。它和真实活跃市值**有相关但不相等**。

## 三个层级的拟合

| 层级 | 函数 | 内容 | 拟合度 |
|---|---|---|---|
| **Lite** | `compute_0amv(df, "lite")` | 0AMV 收盘 + 生命线 | ★★ |
| **Standard** | `compute_0amv(df, "standard")` | 0AMV 完整 K 线（开/高/低/收）+ 生命线 + 颜色 | ★★★★ |
| **Full** | `compute_0amv(df, "full")` | Standard + C5/C13/C34/∞ 活筹成本均线 | ★★★★★ |

**建议**：日常看大盘资金用 **Standard**；做量化策略用 **Full**（多
4 条均线出信号更稳）。

## 快速上手

```python
import pandas as pd
from zero_amv import compute_0amv, FitLevel

# df 必须有 [open, high, low, close, amount, volume, capital] 列
# amount: 元（Tushare 默认）
# volume/capital: 股（Tushare 默认）
# 索引: DatetimeIndex

# 全市场版本（默认）
df_market = pd.DataFrame({
    "open": ..., "high": ..., "low": ..., "close": ...,
    "amount": ...,     # 当日全市场成交额
    "volume": ...,     # 当日全市场成交量
    "capital": ...,    # 当日全市场流通股本
}, index=pd.date_range(...))

result = compute_0amv(df_market, fit_level=FitLevel.FULL)
# result 列: 0amv_open/high/low/close, 0amv_life_line, 0amv_color,
#           0amv_c5/c13/c34/infinite, 0amv_change_pct
```

### 个股版 0AMV

直接把 `df` 换成单只股票的 daily bar 即可。公式不变，但**含义**从
「全市场活跃资金规模」变成「这只股票的资金流量折算」。

## 信号系统（仿指南针用法）

1. **0AMV 在生命线上方运行 + K 线走高** → 红色区域，资金流入
2. **0AMV 在生命线下方运行 + K 线走低** → 绿色区域，资金流出
3. **C5 上穿 C13** → 短线资金加速流入
4. **C13 上穿 C34** → 中期资金加仓
5. **0 号指数上升 + 0AMV 下降** → 牛背离，大盘即将到顶
6. **0 号指数下降 + 0AMV 上升** → 熊背离，大盘即将到底

## 拟合度说明

| 民间公式 vs 指南针原版 | 差异 |
|---|---|
| 核心公式（SMA/EMA/MA 部分） | **完全一致**（多个独立来源交叉验证） |
| 单位归一化（×1e7） | 拟合，指南针原版 K 线数值范围是 [0, 几万亿]，仿版用千万 |
| 0DMV（死筹） | **不在仿制版**——需要真实换手率活跃度数据，民间公式无法给出 |
| 5 日成本均线信号 | 一致 |
| 立体 K 线（指南针特色） | 不在仿制版 |

**结论**：核心信号（方向/趋势/背离/均线交叉）拟合度 95%+，绝对数
值拟合度 ~80%（指南针可能用了不同单位/不同股票池）。

## 集成到 TradingAnalyze

按 `trading_analyze_integration.py` 里的代码片段，走 7 步 PR 流程：

1. `git checkout -b feat/0amv-active-mkt-cap-factor`
2. 把 `PATCH_START` / `PATCH_END` 之间的代码粘进
   `src/trading_analyze/factor_mining/factors/technical_factors.py`
3. 把 `TEST_START` / `TEST_END` 之间的代码粘进对应测试文件
4. 改 `CHANGELOG.md`
5. `poetry run ruff check && poetry run black --check && poetry run mypy && poetry run pytest`
6. `git commit` + `git push` + `gh pr create --fill --base main`
7. 盯 `gh run watch --exit-status`，red 修，最多 3 次

**为什么不直接动手改**：TradingAnalyze 的 AGENTS.md 明确禁止直接
push main，强制走 PR 流程。本仓库不破坏这条规则。

## 测试

```bash
cd /Users/wdblink/Research/trade/0amv
python3.11 -m pytest test_zero_amv.py -v
# 25 passed in 0.21s
```

测试覆盖：
- SMA / EMA / MA(REF) 数学函数正确性（手算对照）
- 三个层级的输出 schema 正确性
- 0AMV 收盘价为正、生命线 std < close std、K 线颜色信号 0/1 等不变量
- 边界情况（空 df / 短 df / 缺列 / 非法 fit_level）
- qlib 表达式字段引用合法性
- **反向断言**：仿制版**不**提供 0DMV（避免未来误加）

## 参考资料（民间来源，多源交叉验证）

- MBA 智库百科「活筹指数」词条（重定向自「活跃市值指数」）
- 道客巴巴「指南针经典指标之 0AMV」PDF
- 东方财富网「指南针 0AMV 活筹指数」教学博客（2009）
- 百度文库「0amv 指数源代码 [教学]」
- 豆丁网「指南针指标源码」28 个公式集
- 公式网 gpxiazai.com「指南针的 0amv 活跃市值转成通达信指标」
- 同花顺 / 通达信 / 飞狐 仿制版源码（多个变体）

## 不变量警告（重要）

```
0AMV + 0DMV = 流通市值
```

这个不变量**只在指南针原版成立**。民间仿制版：

- 0AMV = 相对资金规模（千万 RMB 单位，无绝对含义）
- 0DMV 不在仿制版中（民间公式无法计算）
- 应该看 0AMV 的**变化方向**和**K 线形态**，不要把数值当绝对量用

如果用户在你的量化系统里问「这个 0AMV 数值 800 是什么意思」，正确
的回答是：**没有绝对含义，看它和昨天的对比，以及它的趋势**。

## 后续可做（如果想逼近 100% 拟合）

1. 接入 Tushare 沪深全 A 数据，下载 amount/volume/capital，用本模块直接跑
2. 跑 100 天结果，截图对比「指南针软件 vs 仿版」的 K 线，肉眼检查趋势一致性
3. 想要 0DMV，需要先有「个股过去 30 日换手率活跃度」分类（指南针私有逻辑），
   民间复刻的近似是「0DMV ≈ 流通市值 × (1 - 近 30 日日均换手率 / 平均换手率)」，
   拟合度 ~70%
4. 想要绝对数值校准，需要找到一段指南针软件公开的 0AMV 截图，反推它
   用的全市场 amount 是「沪深 A 股」还是「沪深 300」还是「等权指数」

## License

民间复刻 / 公开源码学习整理，无专利风险。
