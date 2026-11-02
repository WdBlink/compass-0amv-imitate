# 0AMV 活跃市值指标民间公式研究

本项目整理并实现公开流传的 0AMV（活跃市值／活筹指数）公式，提供 Python 计算、qlib 表达式以及基于真实沪深市场成交额的复现工具。

> [!IMPORTANT]
> 根目录 `zero_amv.py` 仍输出民间公式构造的**成交额代理指标**，不是指南针原版。指南针原版的计算结构已在本仓库复原：逐股活筹筹码池，64 天日涨跌幅 MAE 0.0217 个百分点，见下文「官方原版复原」章节与 [validation/autoresearch/](validation/autoresearch/)。

2026-10-30 更新：[64 日官方显示精度核验](validation/exact_parity_20261030/README.md)已建立。该目录包含可公开复算的读数、预测、严格核验器和固定比例成交额平滑的反证；它是成交额代理的**反证**，不是复原结论——复原结论在 `validation/autoresearch/`。

## 研究状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 民间公式实现 | 已完成 | 支持成交额公式与价格调整公式两种口径 |
| 全市场数据复现 | 已完成 | 使用上证指数与深证综指的真实成交额聚合 |
| 同日期截图对照 | 已完成 | 复现窗口与 2024 年指南针截图一致 |
| 原版数值误差评估 | 官方精度未通过 | 已建立 64 日显示精度核验；读数由 AI 逐图核对，待人工验收；不能视为官方算法复原 |
| 官方活筹池算法复原 | 已完成，64 日日幅 MAE 0.0217 个百分点 | 见 [validation/autoresearch/](validation/autoresearch/)：逐股活筹筹码池公式，非成交额代理；64 日全部已曝光，属开发证据 |
| 0DMV 推导 | 不支持 | 民间成交额代理不满足 `0AMV + 0DMV = 流通市值` |

## 官方原版复原：活筹筹码池（2026-11-02）

[validation/autoresearch/](validation/autoresearch/) 用
[autoresearch](https://github.com/uditgoenka/autoresearch) 方法
（改 → 验 → 留/弃 → 循环）复原出指南针 0AMV 的计算结构。它**不是**成交额代理，
而是逐股的活筹筹码池：

```text
q_i,t    = traded_shares_i,t / float_shares_i,t            # 当日换手率
f_i,t    = q_i,t + λ · (1 − q_i,t) · f_i,t−1                 # 活筹占流通股本比例, λ = 0.9235
0AMV_t   = Σ_i close_i,t · float_shares_i,t · f_i,t          # 全部 A 股（不含北交所）
```

含义：一只股票的筹码是一个池子，每天换手生成新的一批 `q_t`，其余部分（`1 − q_t`）
存活，同时整个池子按 `1 − λ` 逐日老化。官方指数有无限历史，有限面板需要把池子
初始化在准稳态而非从零开始。

64 天官方测试集（2026-07-27 … 2026-10-30）上的结果：

| 指标 | 结果 |
|---|---:|
| 日涨跌幅 MAE | **0.0217 个百分点** |
| 方向一致率 | 64 / 64 |
| `+4% / −2.3%` 触发日分类一致 | 64 / 64 |
| 数值水平比值（官方 / 模型） | 0.997364，变异系数 0.0006 |
| 开发段 MAE（07-27 … 09-08） | 0.0168 个百分点 |
| 留出段 MAE（09-09 … 10-30） | 0.0266 个百分点 |

水平比值在 64 天里几乎恒定（变异系数 0.06%），说明这是结构复原而不是曲线拟合。
λ 在开发段选定，并在留出段独立复核，两段都选中 0.9235。

水平口径：模型水平比官方高约 0.27%，所以对齐要**乘以** 0.997364
（`amv.py --scale`），该因子是常数、不影响日涨跌幅；水平 MAPE 从 0.2643% 降到 0.0516%。

误差最大的 5 天（完整 64 行见 [compare_64d.md](validation/autoresearch/compare_64d.md)
与 [compare_64d.csv](validation/autoresearch/compare_64d.csv)）：

| 日期 | 官方日幅% | 复刻日幅% | 误差pp |
|---|--:|--:|--:|
| 2026-10-26 | -1.18 | -1.31 | -0.127 |
| 2026-10-23 | +4.95 | +5.05 | +0.099 |
| 2026-10-08 | -2.17 | -2.26 | -0.087 |
| 2026-09-15 | -1.83 | -1.91 | -0.081 |
| 2026-08-06 | -2.04 | -1.98 | +0.060 |

可部署实现与测试：

```bash
# panel.csv 列：trade_date,ts_code,close,vol,float_share（Tushare 单位）
python3 validation/autoresearch/amv.py --input panel.csv --out amv.csv
python3 validation/autoresearch/amv.py --input panel.csv --out amv.csv --scale 0.997364
python3 validation/autoresearch/test_amv.py
python3 validation/autoresearch/compare_table.py --csv /tmp/cmp64.csv
```

**限制**：64 天在搜索中全部曝光，0.0217 个百分点是开发证据而非盲测成绩，跨年份
准确率未验证。根目录 `zero_amv.py` 的成交额代理公式未改动，线上监控的保护性降级
状态也未改动。

## 核心公式

默认使用公开资料中明确标为 0AMV／活筹指数的成交额公式：

```text
VAR1 = SMA(全市场 AMOUNT, 10, 1) / 1e7

close = VAR1
open  = REF(VAR1, 1)

C5  = DMA(SMA(VAR1, 3, 1), VOL / 0.02 / CAPITAL)
C13 = DMA(SMA(VAR1, 3, 1), VOL / 0.10 / CAPITAL)
C34 = DMA(SMA(VAR1, 3, 1), VOL / 0.18 / CAPITAL)
C∞  = DMA(VAR1, VOL / 1.10 / CAPITAL)
```

其中：

- `AMOUNT` 为沪深全市场每日成交额，单位为元；
- `VOL` 与 `CAPITAL` 必须采用一致的股数单位；
- `SMA(X, N, 1)` 为通达信递推平滑，不是简单移动平均；
- `DMA(X, A)` 为动态移动平均，权重由换手率决定。

公开资料中还存在以下价格调整版本：

```text
SMA(AMOUNT, 10, 1) × CLOSE / MA(REF(CLOSE, 1), 5) / 1e7
```

该版本在本项目中以 `formula_variant="price_adjusted"` 显式保留，不作为默认 0AMV 口径。

## 环境准备

项目要求 Python 3.10 或更高版本。核心计算依赖 `numpy` 和 `pandas`；图表复现与测试还需要 `matplotlib`、`Pillow` 和 `pytest`。

```bash
git clone https://github.com/WdBlink/compass-0amv-imitate.git
cd compass-0amv-imitate

python3 -m venv .venv
source .venv/bin/activate
python -m pip install numpy pandas matplotlib pillow pytest
```

## 快速开始

以下示例下载指定窗口的沪深市场成交额，并计算标准版 0AMV 代理指标：

```python
from validation.market_data import load_mainland_market_amount
from zero_amv import compute_0amv

market = load_mainland_market_amount("2024-01-01", "2024-10-10")
result = compute_0amv(market[["amount"]], fit_level="standard")

print(result.tail())
```

如果已经准备好自己的全市场数据：

```python
from zero_amv import compute_0amv

# Lite / Standard：仅要求 amount（元）
standard = compute_0amv(df[["amount"]], fit_level="standard")

# Full：额外要求同口径的 volume（股）与 capital（股）
full = compute_0amv(
    df[["amount", "volume", "capital"]],
    fit_level="full",
)
```

### 输出字段

| 层级 | 字段 | 含义 |
|---|---|---|
| Lite | `0amv_close` | 十日递推平滑后的全市场成交额代理 |
| Lite | `0amv_life_line` | `0amv_close` 的 EMA12 平滑线 |
| Lite | `0amv_change_pct` | 代理指标日变化率（百分比） |
| Standard | `0amv_open/high/low` | 由当日值与前一日值构造的绘图区间 |
| Standard | `0amv_color` | 上升为 1，下降为 0，无法判断为缺失值 |
| Full | `0amv_c5/c13/c34/infinite` | 公开民间公式中的换手率动态均线 |

### 公式口径参数

```python
from zero_amv import FormulaVariant, compute_0amv

legacy = compute_0amv(
    df,
    formula_variant=FormulaVariant.PRICE_ADJUSTED,
    fit_level="standard",
)
```

`scale_factor` 默认值为 `1.0`。只有获得同日期、连续的指南针导出值后，才应使用 `calibrate_scale(proxy, ground_truth)` 标定；截图十字光标标签不能作为收盘真值。

## 数据与验证

验证脚本通过东方财富历史日线接口分别获取上证指数和深证综指成交额，再按交易日求和。原始分市场列 `sh_amount`、`sz_amount` 会保留在 DataFrame 中，便于审计聚合口径。

```bash
# 运行公开测试（不需要私有行情；私有集成测试明确跳过）
python -m pytest -q

# 生成与 2024 年原版截图相同时间窗口的并排对照图
python validation/make_side_by_side.py

# 生成最近窗口的指标图
python validation/render_imitate_kline.py

# 重新计算多空区间及其独立评估指标
python validation/detect_regimes.py
```

### 私有面板集成测试

公开测试包含非空合成面板的模型一致性和因果性检查，不把空数组比较视为模型复现成功。只有持有离线快照时才显式启用以下三项真实面板测试：

```bash
COMPASS_RUN_PRIVATE_TESTS=1 \
COMPASS_SNAPSHOT_ROOT=/path/to/offline/snapshot \
python -m pytest -q validation/autoresearch/test_amv.py
```

快照目录须包含 `raw/`。未启用时三项测试明确跳过；启用后缺目录、空数据或缺必要日期会失败，不会静默跳过。显式配置的数据根目录不复用或改写默认共享缓存。公开测试通过不代表私有样本精度、官方算法或盲测得到验证。

### 同窗口形态对照

![0AMV 民间公式与指南针截图的同窗口对照](validation/output/side_by_side_compare.png)

该图可以检查大体方向和拐点是否相似，但不能据此计算相关系数或绝对误差：截图中的红色日期与数值标签来自十字光标，且没有可导出的逐日原始值。

### 多空区间评估

多空区间检测是建立在 0AMV 代理之上的独立规则系统，不等同于公式拟合度。当前真实成交额复现结果为：

| 指标 | 结果 |
|---|---:|
| 事件 Precision | 58.3% |
| 事件 Recall | 53.8% |
| 事件 F1 | 56.0% |
| 逐日 F1 | 51.9% |
| 逐日 IoU | 35.0% |

事件评分采用一对一匹配，避免一段超长预测区间同时命中多个真实区间。结果用于评估当前阈值和状态机，不应表述为“0AMV 拟合准确率”。

## qlib 集成

`QLIB_EXPRESSIONS` 与 `trading_analyze_integration.py` 提供了成交额公式的 qlib 表达式。需要特别注意：普通单股票 instrument 的 `$amount` 只是个股成交额；只有输入本身是预先聚合的全市场序列时，这些表达式才具有 0AMV 代理含义。

## 项目结构

```text
.
├── zero_amv.py                       # 核心计算与 qlib 表达式
├── trading_analyze_integration.py    # TradingAnalyze / qlib 集成片段
├── test_zero_amv.py                  # 核心公式测试
├── test_regime_evaluation.py         # 区间评分测试
└── validation/
    ├── market_data.py                # 沪深市场成交额下载与聚合
    ├── make_side_by_side.py          # 同日期窗口截图对照
    ├── render_imitate_kline.py       # 指标图渲染
    ├── detect_regimes.py             # 多空区间状态机及评估
    ├── autoresearch/                 # 官方活筹池公式复原（autoresearch 循环）
    │   ├── program.md                # 方法、指标、规则与最终结果
    │   ├── amv.py                    # 可部署实现
    │   ├── test_amv.py               # 实现测试
    │   ├── compare_table.py          # 64 天模型 vs 官方对比表
    │   ├── compare_64d.md            # 对比表（Markdown）
    │   ├── compare_64d.csv           # 对比表（原始数据）
    │   └── results.tsv               # 迭代日志
    └── output/                        # 可复现输出
```

## 已知限制

1. 活筹筹码池公式（见上文「官方原版复原」）在 64 天上达到 0.0217 个百分点日幅 MAE，水平比值恒定，指向结构复原而非拟合；但该窗口在搜索中全部曝光、只覆盖 2026 年 7—10 月，不支持跨年份准确率或所谓“拟合度 95%”。
2. 旧基线截图不足以提供逐日真值。新采集的 64 天明细支持该窗口内的误差评价，但不支持跨年份准确率或所谓“拟合度 95%”。
3. 上证指数与深证综指成交额之和是可复现的全市场代理，但仍可能与指南针历史股票池和数据清洗口径不同。
4. 将单只股票数据代入只能得到个股成交额代理，不能称为大盘活跃市值。
5. 本项目不构成投资建议，也不应用于直接生成实盘交易指令。

进一步提高精度需要更长、覆盖不同阶段的指南针 0AMV 原始序列。后续若利用本批样本修改公式，应另采未用于开发的窗口，并报告方向一致率、相关系数、MAE/MAPE 和触发日命中情况。

## 参考资料与数据来源

- [北京指南针：经典指标之 0AMV](https://www.compass.cn/shownews.php?nid=1976015)——官方指标含义与使用说明，未公开算法。
- [公式网：指南针的 0AMV 活跃市值转成通达信指标](https://www.gpxiazai.com/gpgs/html/36759.html)——成交额版本民间公式来源之一。
- [同花顺公式平台：OAMV 活筹指数](https://poi.10jqka.com.cn/store/formula/detail/indexid/68942)——另一份公开民间实现，用于核对公式分歧。
- [东方财富：上证指数](https://quote.eastmoney.com/zs000001.html)与[深证综指](https://quote.eastmoney.com/zs399106.html)——验证脚本的数据来源。

## License

本项目采用 [MIT License](LICENSE)。公开民间公式与第三方数据的使用仍应遵守各自来源的条款。
