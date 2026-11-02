# autoresearch program: 指南针官方 0AMV 独立复刻

Method: [uditgoenka/autoresearch](https://github.com/uditgoenka/autoresearch) —
*Modify → Verify → Keep/Discard → Repeat.*

## Goal

Reproduce the official 指南针 0AMV **daily change percentage** on the
64-day official test set (2026-07-27 … 2026-10-30) with a mean absolute
error below **0.1 percentage points**.

Secondary goal (not the acceptance metric): reproduce the official 0AMV
*level* with a single constant scale factor, which is the evidence that
the recovered formula is structurally the official one rather than a
curve fit.

## Scope

Writable:

- `validation/autoresearch/**`

Read-only inputs (never modified):

- `validation/exact_parity_20261030/official.csv` — the official labels
  (64 rows). **Read only inside the scorer**, never inside the model.
- `/Users/wdblink/Data/datasets/derived/compass-0amv/r2-20261030/` — the
  frozen full-market Tushare snapshot, 121 trading days
  (2026-05-06 … 2026-10-30), 5 594 stocks.

Out of scope: any change to the trading strategy, the Hermes monitor,
the deployed protection downgrade, or anything outside this directory.

## Metric

    MAE = mean |predicted_daily_change_pct - official_change_pct|

over the 64 official days, in percentage points. **Lower is better.**
Target: `< 0.10`.

Reported alongside, never optimised against directly:

- `mae_dev` / `mae_holdout` — first / second half of the window
- `dir_hit` — sign agreement
- `level_mape_pct` — official level vs model level, scaled by a single
  constant factor fitted on the same window (this is the structural
  check: the residual scale factor must be *constant*, i.e. have a
  tiny standard deviation)
- `level_ratio_cv` — coefficient of variation of that scale factor

## Direction

Search the space of per-stock active-chip market-value models, in this
order of priority:

1. **Structure** — the recursion, the turnover definition, the price
   used, the universe, the float denominator, the decay mechanism.
2. **Parameters** — window `N`, decay `lambda`, and any other constants.

Prefer simpler models that fit. Reject a change that improves the MAE
while making the level scale factor drift.

## Verification

    python3 validation/autoresearch/experiment.py --spec '<json>' --desc '<text>'

must print a line

    METRIC <value>

to stderr, and append one row to `results.tsv`.

## Rules

1. One focused change per iteration.
2. Commit the candidate **before** verifying (`experiment:` prefix).
3. Keep the commit if the metric improves; `git revert` it if not.
4. Never read the official labels from the model code — only the scorer
   may, and only to score.
5. Re-run `verify_model.py` after any structural change.
6. The 64 official days are already exposed, so a lower MAE here is
   development evidence, not a blind-test result. Report `mae_dev` and
   `mae_holdout` for every accepted iteration and treat a large gap
   between them as a rejection signal.
