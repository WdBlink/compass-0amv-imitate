"""指南针 0AMV (活跃市值 / 活筹指数) 逐股活筹池复刻。

The active-chip model carries the last contribution across missing quotes.
Pass effective delisting dates with your panel to end those contributions.

    from compass_0amv import active_market_value, daily_change_pct

Command line:

    python3 -m compass_0amv --input panel.csv --out amv.csv
"""

from .amv import (
    DECAY,
    INIT_WINDOW,
    OFFICIAL_LEVEL_SCALE,
    a_share_mask,
    active_chip_fraction,
    active_market_value,
    daily_change_pct,
    from_arrays,
    from_tushare_frame,
)

__all__ = [
    "DECAY",
    "INIT_WINDOW",
    "OFFICIAL_LEVEL_SCALE",
    "a_share_mask",
    "active_chip_fraction",
    "active_market_value",
    "daily_change_pct",
    "from_arrays",
    "from_tushare_frame",
]

__version__ = "0.3.0"
