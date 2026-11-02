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

The recovered structure is a per-stock **chip pool**, not an amount
proxy:

    q_i,t   = traded_shares_i,t / float_shares_i,t        turnover rate
    f_i,t   = q_i,t + lambda * (1 - q_i,t) * f_i,t-1       active float fraction
    0AMV_t  = sum_i close_i,t * float_shares_i,t * f_i,t

with `lambda = 0.9235`, every A-share except the Beijing Stock
Exchange, and the chip pool initialised at its quasi-stationary level
instead of empty (the official index has infinite history; a finite panel
needs the correction).

    MAE                 0.021689 pp      (target < 0.10)
    mae_dev             0.016813 pp      (2026-07-27 … 2026-09-08)
    mae_holdout         0.026565 pp      (2026-09-09 … 2026-10-30)
    dir_hit             64 / 64
    threshold triggers  64 / 64
    level ratio         0.997364,  CV 0.000600
    level_mape_scaled   0.0516 %

The level ratio is constant to within 0.06 % over the whole window: the
official aggregate is one constant factor times this model, which is the
evidence that the formula is structurally the official one rather than a
curve fit.

`lambda` was selected on the dev half and independently confirmed on
the holdout half — both pick 0.9235 (`cv.py`).  The mean out-of-sample
MAE over the two directions is 0.0233 pp.

**Caveat that does not go away:** all 64 official days have been
exposed during the search, so 0.0217 pp is *development* evidence, not a
blind-test score.  Cross-year accuracy is untested.


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
    amv.py          the deployable implementation (label free)
    test_amv.py     tests for the deployable implementation
    results.tsv     the iteration log

