"""Export a small, public parity fixture without raw provider or host records.

Official labels select scoring dates only. They never fit or initialize predictions.
Market CSV columns: date, sh_close, sh_amount, sz_amount, amount, in yuan.
The output is disclosed diagnostic data, not an unseen evaluation set.
"""
from __future__ import annotations

import argparse
from decimal import Decimal
from fractions import Fraction
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from zero_amv import compute_0amv, sma


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def dated_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    frame = pd.read_csv(path, dtype=str)
    if not {"date", *columns}.issubset(frame.columns) or frame.empty:
        raise ValueError("CSV must have dates and all required fields")
    dates = pd.to_datetime(frame["date"], format="%Y-%m-%d", errors="raise")
    if not dates.dt.strftime("%Y-%m-%d").equals(frame["date"]):
        raise ValueError("dates must use YYYY-MM-DD")
    if dates.duplicated().any() or not dates.is_monotonic_increasing:
        raise ValueError("dates must be unique and increasing")
    for column in columns:
        values = pd.to_numeric(frame[column], errors="raise")
        if not np.isfinite(values).all():
            raise ValueError(f"{column} must be finite")
        if column in {"official_close", "official_change_pct"}:
            grid = Fraction(1, 10 if column == "official_close" else 100)
            if any((Fraction(Decimal(value)) / grid).denominator != 1 for value in frame[column]):
                raise ValueError(f"{column} is not on the display grid")
        else:
            frame[column] = values
    return frame.set_index(pd.DatetimeIndex(dates, name="date"))[columns]


def export_inputs(official_path: Path, market_path: Path, output: Path) -> None:
    official = dated_csv(official_path, ["official_close", "official_change_pct"])
    market = dated_csv(market_path, ["sh_close", "sh_amount", "sz_amount", "amount"])
    if (pd.to_numeric(official["official_close"]) <= 0).any() or (pd.to_numeric(official["official_change_pct"]) <= -100).any():
        raise ValueError("official levels and changes must describe positive values")
    if (market <= 0).any().any():
        raise ValueError("market values must be positive")
    if not np.allclose(market.amount, market.sh_amount + market.sz_amount, rtol=1e-10, atol=.01):
        raise ValueError("amount must equal sh_amount + sz_amount")
    sample = market.loc[official.index[0]:official.index[-1]]
    if not sample.index.equals(official.index):
        raise ValueError("official observations must cover consecutive market dates")
    first = market.index.get_loc(official.index[0])
    if first < 6:
        raise ValueError("market needs preceding history for both proxy variants")
    market = market.loc[:official.index[-1]]
    predictions = {
        "amount_only": compute_0amv(market, fit_level="lite")["0amv_close"],
        "sse_price_adjusted": sma(market.sh_amount, 10) * market.sh_close
        / market.sh_close.shift(1).rolling(5).mean() / 1e7,
    }
    dates = market.index[first - 1:]
    output.mkdir(parents=True, exist_ok=False)
    official.to_csv(output / "official.csv")
    pd.DataFrame({"date": dates.strftime("%Y-%m-%d")}).to_csv(output / "calendar.csv", index=False)
    for name, levels in predictions.items():
        levels.loc[dates].rename("computed_close").to_csv(output / f"predictions_{name}.csv", float_format="%.17g")
    manifest = {
        "source_official_sha256": digest(official_path),
        "source_market_sha256": digest(market_path),
        "generator_sha256": digest(Path(__file__)),
        "market_source": "Tushare index_daily: 000001.SH and 399106.SZ; amount * 1000 = yuan",
        "market_warmup_start": market.index[0].strftime("%Y-%m-%d"),
        "market_rows_used": len(market),
        "official_rows": len(official),
        "prediction_rows": len(dates),
        "labels_used_for_prediction_values": False,
        "label_exposure": "all_64_observations_previously_exposed_to_research",
        "transcription_review": "ai_visual_checked; human_review_pending",
        "raw_provider_data_and_device_screenshots": "retained_locally_not_published",
        "sha256": {p.name: digest(p) for p in sorted(output.glob("*.csv"))},
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--official", type=Path, required=True)
    parser.add_argument("--market", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    export_inputs(args.official, args.market, args.output)


if __name__ == "__main__":
    main()
