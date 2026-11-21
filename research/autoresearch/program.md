# autoresearch program: 指南针官方 0AMV 独立复刻

Method: [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) —
*Modify → Verify → Keep/Discard → Repeat.*

## Goal

Reproduce the official 指南针 0AMV **daily change percentage** on the
64-day official test set (2026-07-27 … 2026-10-30) with a mean absolute
error strictly below **0.01 percentage points** (2026-11-21 target).

Secondary goal (not the acceptance metric): reproduce the official 0AMV
*level* with a single constant scale factor, which is the evidence that
the recovered formula is structurally the official one rather than a
curve fit.

## Scope

Writable:

- `research/autoresearch/**` for experiments, and `autoresearch/loop-*/` for the run ledger.
- `compass_0amv/**` and related tests for integrating a verified improvement.

Read-only inputs (never modified):

- `research/exact_parity_20261030/official.csv` — the official labels
  (64 rows). **Read only inside the scorer**, never inside the model.
- `/Users/wdblink/Data/datasets/derived/compass-0amv/r2-20261030/` — the
  frozen full-market Tushare snapshot, 121 trading days
  (2026-05-06 … 2026-10-30), 5 594 stocks.

Out of scope: any change to the trading strategy, the Hermes monitor,
the deployed protection downgrade, or anything outside this directory.

## Metric

    MAE = mean |predicted_daily_change_pct - official_change_pct|

over the 64 official days, in percentage points. **Lower is better.**
Target: `< 0.01` using full-precision modeled returns and the official displayed changes.

Reported alongside, never optimised against directly:

- `mae_dev` / `mae_holdout` — first / second half of the window.  The
  split is **by index** (32 / 32 days: 2026-07-27 … 2026-09-08 and
  2026-09-09 … 2026-10-30), so `evaluate.py` and `cv.py` always agree
  on which day belongs to which half.
- `dir_hit` — sign agreement
- `level_mape_pct` — official level vs model level, **unscaled** (the
  constant offset between the two series)
- `level_mape_scaled_pct` — the same after removing **one** constant
  factor fitted on the window.  This is the structural check: the
  residual scale factor must be *constant*, i.e. have a tiny standard
  deviation.
- `level_ratio_cv` — coefficient of variation of that scale factor

## Result

The selected model has a **64-day daily-return MAE of 0.011320 pp**. The target is still open. The previous model scored 0.021689 pp on these same observations.

```text
quoted turnover     q = clip(reported_turnover_percent / 100, 0, 1)
volume fallback     q = clip(traded_shares / float_shares, 0, 1)
active fraction     f_t = q_t + (12/13) * (1 - q_t) * f_previous_quote
active market value V_t = sum(close * float_shares * f_t)
```

The aggregate covers Shanghai and Shenzhen A-shares, including the `302` ChiNext prefix. Each stock retains its last active value while unquoted. Its contribution ends on its effective delisting date. The `robust` initializer estimates the initial fraction from median turnover in the first 20 panel sessions and uses a growing prefix during warmup. The public interface accepts daily `turnover_rate` (percent) and `delist_date` as optional fields. Without reported turnover, the same panel and delisting events give 0.011340 pp.

| Frozen-window metric | Previous | Selected |
|---|---:|---:|
| Daily change MAE, pp | 0.021689 | 0.011279 |
| First 32 days, pp | 0.016813 | 0.014779 |
| Last 32 days, pp | 0.026565 | 0.007778 |
| Mean official/model level | 0.997364 | 0.999674 |
| Level ratio CV | 0.000600 | 0.000239 |

Both direction and threshold-trigger classification agree on all 64 days. The mean level ratio can be used as a retrospective scale for absolute levels. Constant rescaling cancels from daily returns. Cross-year accuracy remains unmeasured.

The [first November 21 run](../../autoresearch/loop-261121-2039/README.md) records the baseline, 25 frozen candidates, the selection criteria and how to rerun each revision. The [follow-up run](../../autoresearch/loop-261121-2318/README.md) widens the seed window from 20 to 32 sessions, locates the remaining error on three sessions and lists the mechanisms that were ruled out. The earlier `results.tsv` retains all eleven historical rows and their original meanings. `python3 research/autoresearch/evaluate.py` scores the selected model. Explicit JSON specifications still replay the earlier models.

## Direction

Search the space of per-stock active-chip market-value models, in this
order of priority:

1. **Structure** — the recursion, the turnover definition, the price
   used, the universe, the float denominator, the decay mechanism.
2. **Parameters** — window `N`, decay `lambda`, and any other constants.

Prefer simpler models that fit. Reject a change that improves the MAE
while making the level scale factor drift.

## Verification

    python3 research/autoresearch/experiment.py --spec '<json>' --desc '<text>'

must print a line

    METRIC <value>

to stderr, and append one row to `results.tsv`.

## Rules

1. One focused change per iteration.
2. Commit the candidate **before** verifying (`experiment:` prefix).
3. Keep the commit if the metric improves; `git revert` it if not.
   The `commit` column of `results.tsv` names the commit whose tree the
   iteration was evaluated against; a discarded iteration leaves no code
   behind, so its row points at the unchanged tree.
4. Never read the official labels from the model code — only the scorer
   may, and only to score.
5. Re-run `verify_model.py` after any structural change.
6. The 64 official days are already exposed, so a lower MAE here is
   development evidence, not a blind-test result. Report `mae_dev` and
   `mae_holdout` for every accepted iteration and treat a large gap
   between them as a rejection signal.
7. Every row of `results.tsv` must be reproducible from its own
   `spec`: `python3 evaluate.py '<spec>'` has to return the logged
   metric.

## Layout

    program.md      this file
    spec.py         the canonical spec
    panel.py        loads the frozen snapshot into dense matrices
    model.py        the model family (all the rejected variants live here)
    evaluate.py     the scorer; the only place that reads official.csv
    experiment.py   one iteration: run, print, log
    cv.py           honest dev/holdout parameter selection
    scan.py         landscape scan
    verify_model.py the five-check verification battery
    diagnose_*.py   structural probes (drift of the level ratio, mixes)
    amv.py          (moved) the deployable implementation now lives in
                    compass_0amv/amv.py, with its tests in tests/test_amv.py
    compare_table.py  64-day model-vs-official table, raw and aligned
    results.tsv     the iteration log

