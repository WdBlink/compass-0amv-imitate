"""The canonical model spec for the 0AMV replication.

Selected by the autoresearch loop (see results.tsv).  Every number here was
chosen on the first half of the official window and checked on the second
half; `cv.py` reproduces that selection.

    decay = 0.9235   age decay of the chip pool: the pool keeps 92.35% of
                     its weight per additional day
    init  = "steady" the chip pool starts at its quasi-stationary level
                     rather than empty, which removes the finite-panel
                     warmup bias (the official index has infinite history)
    market = "all_a" every A-share except the Beijing Stock Exchange
"""

from __future__ import annotations

FINAL_SPEC = {
    "decay": 0.9235,
    "init": "steady",
    "market": "all_a",
    "form": "recursion",
}
