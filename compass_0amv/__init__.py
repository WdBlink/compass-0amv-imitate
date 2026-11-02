"""指南针 0AMV (活跃市值 / 活筹指数) — 官方原版公式复刻。

The package holds exactly one formula: the per-stock active-chip pool
recovered by the autoresearch loop (``research/autoresearch``).  The
retired market-amount proxy is not part of it.

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

__version__ = "0.2.0"
