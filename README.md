# 指南针 0AMV（活跃市值／活筹指数）复刻

本项目复刻指南针 0AMV 的计算结构：逐股活筹筹码池。仓库里只有这一个公式，
可部署实现、64 日官方对比表、以及把它与官方读数对齐的核验链都在这里。

> [!IMPORTANT]
> 已核验的指南针官方资料公开了 0AMV 的指标含义，但未提供完整计算规范。本项目
> 复原出的是**逐股活筹筹码池**结构，64 天官方日涨跌幅 MAE 0.0217 个百分点，
> 数值水平与官方相差一个常数因子（0.997364，变异系数 0.0006）。这是结构复原的
> 证据，不是官方算法确认；64 天全部参与了开发，不构成盲测成绩。

2026-10-30 建立的 [64 日官方显示精度核验](research/exact_parity_20261030/README.md)
是成交额代理的**反证**，不是复原结论。复原结论在 [research/autoresearch/](research/autoresearch/)。

## 研究状态

| 项目 | 状态 | 说明 |
|---|---|---|
| 官方活筹池算法复原 | 已完成，64 日日幅 MAE 0.0217 个百分点 | 见 [research/autoresearch/](research/autoresearch/)：逐股活筹筹码池公式，非成交额代理；64 日全部已曝光，属开发证据 |
| 官方数值水平对齐 | 已完成 | 模型水平 × 0.997364 即官方水平，64 天变异系数 0.0006；水平 MAPE 0.2643% → 0.0516% |
| 64 日显示精度核验 | 已建立 | [research/exact_parity_20261030/](research/exact_parity_20261030/)：可公开复算的读数、负对照、严格核验器与反证 |
| Hermes 上证代理播报 | 已保护性降级，公式仍未校准 | 线上独立的价格调整代理日幅 MAE 约 0.96 个百分点；停止发布指南针多空结论 |
| 0DMV 推导 | 不支持 | 本仓库不提供 0DMV，`0AMV + 0DMV = 流通市值` 不是这里的恒等式 |

## 官方原版复原：活筹筹码池（2026-11-02）

[research/autoresearch/](research/autoresearch/) 用
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
（`python3 -m compass_0amv --scale 0.997364`），该因子是常数、不影响日涨跌幅；
水平 MAPE 从 0.2643% 降到 0.0516%。这个常数是在那 64 天上拟合的，只有在拿到
同日期、可核验的指南针收盘序列后才谈得上标定；截图十字光标的坐标标签不能作为
收盘真值。

误差最大的 5 天（完整 64 行见 [compare_64d.md](research/autoresearch/compare_64d.md)
与 [compare_64d.csv](research/autoresearch/compare_64d.csv)）：

| 日期 | 官方日幅% | 复刻日幅% | 误差pp |
|---|--:|--:|--:|
| 2026-10-26 | -1.18 | -1.31 | -0.127 |
| 2026-10-23 | +4.95 | +5.05 | +0.099 |
| 2026-10-08 | -2.17 | -2.26 | -0.087 |
| 2026-09-15 | -1.83 | -1.91 | -0.081 |
| 2026-10-19 | -0.85 | -0.79 | +0.063 |

**限制**：64 天在搜索中全部曝光，0.0217 个百分点是开发证据而非盲测成绩，跨年份
准确率未验证。线上 Hermes 监控的保护性降级状态未改动。

## 为什么不是成交额代理

曾经流传的民间 0AMV 是全市场成交额的十日递推平滑：

```text
VAR1 = SMA(全市场 AMOUNT, 10, 1) / 1e7
```

仓库里已经把它排除掉了，而且排除得有据可查。考虑任何正平滑系数的公式
`Y_t=(1-α_t)·Y_(t−1)+α_t·c·x_t`（`x_t` 为当日两市成交额 / 1e7，`c` 为固定正数）：
`Y` 的涨跌方向必然等于 `c·x_t−Y_(t−1)` 的符号。

| 日期 | `x_t` | 前日官方点位 | 当日官方点位 |
|---|--:|--:|--:|
| 2026-08-12 | 215242.310981535 | 210839.2 | 210962.0 |
| 2026-08-13 | 255091.673935884 | 210962.0 | 208572.3 |

即使给所有点位留出 ±0.05 的舍入空间，8 月 12 日上涨仍要求 `c > 0.9795432368`，
8 月 13 日下跌仍要求 `c < 0.8270048440`。这两个条件不能同时满足。
`tests/test_exact_export.py` 用 Decimal 复核这项反证。

被排除的公式仍然以**负对照**的形式留在仓库里：`research/exact_parity_20261030/`
保存着它们算出的预测、逐日核验报告，以及生成它们的
[tools/export_exact_inputs.py](tools/export_exact_inputs.py)。两个基线在 64 天里
收盘显示一致 0/64、涨幅显示一致 0/64，是预期失败的负对照，不是已通过的复刻结果。

## 环境准备

项目要求 Python 3.11 或更高版本。计算依赖 `numpy`，研究链路另需 `pandas`。

```bash
git clone https://github.com/WdBlink/compass-0amv-imitate.git
cd compass-0amv-imitate

python3 -m venv .venv
source .venv/bin/activate
python -m pip install -e ".[dev,research]"
```

## 快速开始

### 作为库

```python
from compass_0amv import active_market_value, daily_change_pct, from_tushare_frame

# 长表：trade_date, ts_code, close, vol（手）, float_share（万股）
dates, levels = from_tushare_frame(df)

change_pct = daily_change_pct(levels)
```

也可以直接给出稠密矩阵：

```python
levels = active_market_value(vol_shares, float_shares, close)
```

### 命令行

```bash
# panel.csv 列：trade_date,ts_code,close,vol,float_share（Tushare 单位）
python3 -m compass_0amv --input panel.csv --out amv.csv

# 对齐到官方水平（常数因子，不改变日涨跌幅）
python3 -m compass_0amv --input panel.csv --out amv.csv --scale 0.997364
```

输出列为 `trade_date,amv_cny,amv_yi,change_pct`。

## 验证

```bash
# 全部公开测试（不需要私有行情；私有面板测试明确跳过）
python3 -m pytest -q

# 64 天模型 vs 官方对比表
python3 research/autoresearch/compare_table.py --csv /tmp/cmp64.csv

# 五项验证：因果性、未来尾部污染、无标签泄漏、逐日残差、预热收敛
python3 research/autoresearch/verify_model.py

# 成交额代理负对照的严格核验（预期退出码 1）
python3 tools/verify_exact_parity.py \
  --official research/exact_parity_20261030/official.csv \
  --predictions research/exact_parity_20261030/predictions_amount_only.csv \
  --calendar research/exact_parity_20261030/calendar.csv \
  --output /tmp/amount_only_parity.json
```

### 私有面板集成测试

公开测试包含非空合成面板的模型一致性和因果性检查，不把空数组比较视为模型复现成功。只有持有离线快照时才显式启用以下三项真实面板测试：

```bash
COMPASS_RUN_PRIVATE_TESTS=1 \
COMPASS_SNAPSHOT_ROOT=/path/to/offline/snapshot \
python3 -m pytest -q tests/test_amv.py
```

快照目录须包含 `raw/`。未启用时三项测试明确跳过；启用后缺目录、空数据或缺必要日期会失败，不会静默跳过。显式配置的数据根目录不复用或改写默认共享缓存。公开测试通过不代表私有样本精度、官方算法或盲测得到验证。

## 项目结构

```text
.
├── compass_0amv/                  # 可部署实现（仓库里唯一的公式）
│   ├── __init__.py                # 公开 API
│   ├── amv.py                     # 活筹筹码池公式
│   ├── cli.py                     # 命令行
│   └── __main__.py                # python3 -m compass_0amv
├── tests/                         # 全部测试，一个目录
│   ├── test_amv.py                # 实现测试，含 64 日端到端精度
│   ├── test_panel.py              # 面板加载器回归
│   ├── test_panel_amv_guards.py   # 私有面板测试门禁
│   └── ...                        # 机制研究测试
├── research/
│   ├── autoresearch/              # 官方活筹池公式复原（autoresearch 循环）
│   │   ├── program.md             # 方法、指标、规则与最终结果
│   │   ├── spec.py                # 规范定义
│   │   ├── panel.py               # 冻结快照加载
│   │   ├── model.py               # 模型族（含被否决的变体）
│   │   ├── evaluate.py            # 评分器，唯一读取官方标签处
│   │   ├── experiment.py          # 单次迭代：跑分、打分、记账
│   │   ├── cv.py                  # dev / 留出段参数选择
│   │   ├── scan.py                # 参数扫描
│   │   ├── verify_model.py        # 五项验证
│   │   ├── diagnose_*.py          # 结构探针
│   │   ├── compare_table.py       # 64 天模型 vs 官方对比表
│   │   ├── compare_64d.md / .csv  # 对比表
│   │   └── results.tsv            # 迭代日志
│   ├── mechanism_research_20261030/   # 机制研究，与本公式无关
│   └── exact_parity_20261030/         # 64 日官方读数、负对照与核验证据
├── tools/
│   ├── verify_exact_parity.py     # 严格核验器（纯标准库）
│   ├── export_exact_inputs.py     # 负对照与公开 fixture 生成
│   └── market_data.py             # 沪深成交额下载与聚合
├── assets/                        # 指南针官方图表截图
├── pyproject.toml
└── README.md
```

## 已知限制

1. 本轮未找到完整官方计算规范，不能将公式验证或有限样本拟合视为官方算法复原。
2. 活筹筹码池公式（见上文「官方原版复原」）在 64 天上达到 0.0217 个百分点日幅 MAE，水平比值恒定，指向结构复原而非拟合；但该窗口在搜索中全部曝光、只覆盖 2026 年 7—10 月，不支持跨年份准确率或所谓“拟合度 95%”。
3. 旧基线截图不足以提供逐日真值。新采集的 64 天明细支持该窗口内的误差评价，但不支持跨年份准确率或所谓“拟合度 95%”。
4. 水平对齐因子 0.997364 拟合自那 64 天，不构成跨期标定。
5. 本项目不构成投资建议，也不应用于直接生成实盘交易指令。

进一步提高精度需要更长、覆盖不同阶段的指南针 0AMV 原始序列。后续若利用本批样本修改公式，应另采未用于开发的窗口，并报告方向一致率、相关系数、MAE/MAPE 和触发日命中情况。

## 参考资料与数据来源

- [北京指南针：经典指标之 0AMV](https://www.compass.cn/shownews.php?nid=1976015)——官方指标含义与使用说明，未公开算法。
- [指南针官方 CYS 说明](https://aweb.compass.cn/article/app/news.php?newsid=33825230)——`CYS=(Close−CYC)/CYC×100%`，用于核验成本均线与乖离率的关系，不能反推 0AMV。
- [东方财富：上证指数](https://quote.eastmoney.com/zs000001.html)与[深证综指](https://quote.eastmoney.com/zs399106.html)——负对照数据来源。

## License

本项目采用 [MIT License](LICENSE)。第三方数据与公开资料的使用仍应遵守各自来源的条款。
