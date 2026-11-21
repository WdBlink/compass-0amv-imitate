"""Research specs for the active-chip index. Levels are in unscaled CNY.

The current candidate uses a 13-period exponential smoothing factor, one
update per quoted session and the latest observed valuation until resumption
or a known delisting. Its seed uses each stock's expanding median turnover
through the first 30 sessions. Disclosed turnover takes precedence over the
volume/float estimate where a valid daily disclosure is available.
The model was selected in round 24 of the 2026-11-21 experiment and re-checked
in the follow-up round, which extended the seed window to 30 sessions. The
original 0.9235 model and the earlier comparisons are recorded in results.tsv.
"""

from __future__ import annotations

LEGACY_SPEC = {
    "decay": 0.9235,
    "init": "steady",
    "market": "all_a",
    "form": "recursion",
}

FINAL_SPEC = {
    "decay": 12 / 13,
    "init": "robust",
    "init_window": 32,
    "turnover_source": "reported",
    "market": "sh_sz_a",
    "form": "recursion",
    "quote_policy": "freeze",
}
