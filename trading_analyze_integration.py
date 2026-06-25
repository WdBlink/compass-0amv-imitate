"""
TradingAnalyze 集成片段 —— 把 0AMV 加进 technical_factors.py

================================================================
不直接修改 TradingAnalyze 代码，按 AGENTS.md 走 PR 流程
================================================================
本文件**不**直接修改你的 TradingAnalyze 项目。按 AGENTS.md 的
7 步 PR 流程，需要由你自己来执行：

  1) git checkout main && git pull --ff-only origin main
  2) git checkout -b feat/0amv-active-mkt-cap-factor
  3) 把下面 "PATCH_START" 和 "PATCH_END" 之间的代码粘进
     src/trading_analyze/factor_mining/factors/technical_factors.py
     的 _define_factors() 字典里
  4) 把下面 "TEST_START" 和 "TEST_END" 之间的代码粘进
     tests/factor_mining/factors/test_technical_factors.py
     （如不存在则新建，按现有项目风格走）
  5) poetry run ruff check src/ tests/
     poetry run black --check src/ tests/ --line-length 120
     poetry run pytest -m "not slow and not integration"
  6) 改 CHANGELOG.md
  7) commit + push + gh pr create --fill --base main
  8) 盯 CI，red 时修，最多 3 次

================================================================
不变量警告
================================================================
0AMV 在民间仿制版中是一个**相对值**（带「千万 RMB」单位），
**不**等于真实活跃市值，因此：
  - 0AMV + 0DMV = 流通市值 这个不变量在仿制版中不成立
  - 0AMV 主要用于「看趋势」（上行/下行/横盘）和「看 K 线形态」，
    而不是「算真实资金量」
  - 如果用户问「0AMV 数值多少是合理」，答案是：它没有绝对含义，
    应该看**变化方向**和**K 线形态**

================================================================
"""

# =============================================================================
# PATCH_START —— 粘进 src/trading_analyze/factor_mining/factors/technical_factors.py
# =============================================================================
#
# 在 _define_factors() 字典末尾（brick_value_ma5 那一行之后），追加：
#
#         # ----------------------------------------------------------------
#         # 0AMV (活跃市值 / 活筹指数) —— 指南针软件指标近似
#         #
#         # 仿通达信 / 同花顺民间版本。指南针原版不公开，民间复刻的核心
#         # 公式是 0AMV_close = SMA(AMOUNT, 10, 1) * CLOSE / MA(REF(CLOSE,1), 5) / 1e7
#         #
#         # qlib 没有通达信风格的 SMA(N, 1)，但 EMA(X, 2N-1) 收敛后等价。
#         # 因此用 EMA($amount, 19) 替代 SMA($amount, 10, 1)。
#         #
#         # 注意: 0AMV 是相对值，单位是「千万 RMB」；它**不**等于真实活跃市值，
#         # 0AMV + 0DMV = 流通市值 这个不变量在仿制版中不成立。
#         # ----------------------------------------------------------------
#         "0amv_amount_smooth": "EMA($amount, 19)",
#         "0amv_ref_ma5": "Mean(Ref($close, 1), 5)",
#         "0amv_close": (
#             "(EMA($amount, 19) * $close) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7"
#         ),
#         "0amv_open": (
#             "(EMA($amount, 19) * $open) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7"
#         ),
#         "0amv_high": (
#             "(EMA($amount, 19) * $high) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7"
#         ),
#         "0amv_low": (
#             "(EMA($amount, 19) * $low) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7"
#         ),
#         "0amv_life_line": (
#             "EMA("
#             "  (EMA($amount, 19) * $close) / "
#             "  (Mean(Ref($close, 1), 5) + 1e-9) / 1e7, "
#             "12)"
#         ),
#         "0amv_change_pct": (
#             "((EMA($amount, 19) * $close) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7) / "
#             "Ref((EMA($amount, 19) * $close) / "
#             "(Mean(Ref($close, 1), 5) + 1e-9) / 1e7, 1) - 1"
#         ),
#         "0amv_color": (
#             "If("
#             "  (EMA($amount, 19) * $close) / (Mean(Ref($close, 1), 5) + 1e-9) / 1e7 > "
#             "  (EMA($amount, 19) * $open) / (Mean(Ref($close, 1), 5) + 1e-9) / 1e7, "
#             "  1, 0)"
#         ),
#         "0amv_above_life_line": (
#             "If("
#             "  (EMA($amount, 19) * $close) / (Mean(Ref($close, 1), 5) + 1e-9) / 1e7 > "
#             "  EMA((EMA($amount, 19) * $close) / "
#             "      (Mean(Ref($close, 1), 5) + 1e-9) / 1e7, 12), "
#             "  1, 0)"
#         ),
#     }
#
# PATCH_END
# =============================================================================


# =============================================================================
# TEST_START —— 粘进 tests/factor_mining/factors/test_technical_factors.py
# =============================================================================
#
# @pytest.mark.unit
# class Test0AMVFactors:
#     """0AMV 因子测试 —— 验证所有 0amv_* 表达式能跑通且不变量成立。"""
#
#     def test_all_0amv_expressions_registered(self):
#         from trading_analyze.factor_mining.factors.technical_factors import (
#             TechnicalFactors,
#         )
#
#         factors = TechnicalFactors()
#         names = factors.get_factor_names()
#         for required in [
#             "0amv_amount_smooth",
#             "0amv_ref_ma5",
#             "0amv_close",
#             "0amv_open",
#             "0amv_high",
#             "0amv_low",
#             "0amv_life_line",
#             "0amv_change_pct",
#             "0amv_color",
#             "0amv_above_life_line",
#         ]:
#             assert required in names, f"missing factor: {required}"
#
#     def test_0amv_close_positive(self, sample_market_data):
#         """0AMV_close 应该是正数。"""
#         from trading_analyze.factor_mining.qlib_factor_calculator import (
#             QlibFactorCalculator,
#         )
#
#         calc = QlibFactorCalculator()
#         result = calc.compute(["0amv_close"], instruments=["000001.SZ"])
#         valid = result["0amv_close"].dropna()
#         assert (valid > 0).all(), "0AMV_close 必须为正"
#
#     def test_0amv_life_line_smoother(self, sample_market_data):
#         """生命线（EMA 12）std 应该小于 close std。"""
#         from trading_analyze.factor_mining.qlib_factor_calculator import (
#             QlibFactorCalculator,
#         )
#
#         calc = QlibFactorCalculator()
#         result = calc.compute(
#             ["0amv_close", "0amv_life_line"], instruments=["000001.SZ"]
#         )
#         close_std = result["0amv_close"].std()
#         life_std = result["0amv_life_line"].std()
#         assert life_std < close_std, "生命线 std 应小于 close std"
#
#     def test_0amv_color_signal(self, sample_market_data):
#         """0amv_color 应该在 0/1 范围内。"""
#         from trading_analyze.factor_mining.qlib_factor_calculator import (
#             QlibFactorCalculator,
#         )
#
#         calc = QlibFactorCalculator()
#         result = calc.compute(["0amv_color"], instruments=["000001.SZ"])
#         valid = result["0amv_color"].dropna().unique()
#         assert set(valid).issubset({0.0, 1.0}), f"color 应只取 0/1，实际 {valid}"
#
# TEST_END
# =============================================================================


# =============================================================================
# CHANGELOG 条目
# =============================================================================
#
# 在 CHANGELOG.md 的 ## [Unreleased] 下加：
#
# ### Added
# - `0amv_*` 因子族（指南针软件 0AMV 活跃市值指标的民间仿制版），
#   含 `0amv_close` / `0amv_open` / `0amv_high` / `0amv_low` /
#   `0amv_life_line` / `0amv_change_pct` / `0amv_color` /
#   `0amv_above_life_line` 八个因子。核心公式：
#   `0AMV_close = SMA(AMOUNT, 10, 1) * CLOSE / MA(REF(CLOSE, 1), 5) / 1e7`
#   （用 `EMA($amount, 19)` 近似通达信 `SMA($amount, 10, 1)`）。
#   **注意**: 民间仿制版的 0AMV 是带「千万 RMB」单位的相对值，
#   不等于真实活跃市值；`0AMV + 0DMV = 流通市值` 这个不变量
#   在仿制版中**不成立**。详见 `0amv/README.md`。
#
# =============================================================================


# =============================================================================
# 完整 7 步 PR 流程（按 AGENTS.md）
# =============================================================================
#
# ```bash
# # 1) 切分支
# cd /Users/wdblink/Research/trade/TradingAnalyze
# git checkout main && git pull --ff-only origin main
# git checkout -b feat/0amv-active-mkt-cap-factor
#
# # 2) 把上面 PATCH_START / TEST_START 之间的代码分别粘进对应文件
# #    改 CHANGELOG.md
#
# # 3) 本地验证
# poetry install
# poetry run ruff check src/ tests/
# poetry run black --check src/ tests/ --line-length 120
# poetry run mypy src/trading_analyze
# poetry run pytest -m "not slow and not integration"
#
# # 4) Commit (Conventional Commits + Copilot trailer)
# git add -A
# git commit -m "feat(factors): add 0AMV (active market cap) factor family
#
# 仿制指南针软件 0AMV (活跃市值 / 活筹指数) 民间版本，含 8 个因子。
# 核心公式: 0AMV_close = SMA(AMOUNT, 10, 1) * CLOSE / MA(REF(CLOSE, 1), 5) / 1e7
# qlib 表达式: 用 EMA(\$amount, 19) 近似通达信 SMA(\$amount, 10, 1)。
# 注意: 民间仿制版是相对值，0AMV + 0DMV = 流通市值 不变量不成立。
#
# Co-authored-by: Copilot <223556219+Copilot@users.noreply.github.com>"
#
# # 5) Push + PR
# git push -u origin HEAD
# gh pr create --fill --base main
# # PR body 里加 ## Verification 段，列出第 3 步的命令和结果
#
# # 6) 盯 CI
# gh run watch --exit-status
# # 红了就修，最多 3 次
#
# # 7) 报告 PR URL
# ```
#
# =============================================================================
