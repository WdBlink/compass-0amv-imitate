from pathlib import Path
from decimal import Decimal, localcontext

import pandas as pd
import pytest

from validation.export_exact_inputs import export_inputs
from validation.verify_exact_parity import verify_exact_parity


def inputs(tmp_path: Path) -> tuple[Path, Path]:
    dates = pd.bdate_range("2026-01-01", periods=12)
    market = pd.DataFrame({
        "date": dates.strftime("%Y-%m-%d"),
        "sh_close": range(3000, 3012),
        "sh_amount": range(10000000, 10000012),
        "sz_amount": range(20000000, 20000012),
    })
    market["amount"] = market.sh_amount + market.sz_amount
    official = pd.DataFrame({
        "date": dates[8:11].strftime("%Y-%m-%d"),
        "official_close": [123.4, 124.5, 123.3],
        "official_change_pct": [1.1, .89, -.96],
    })
    market_path, official_path = tmp_path / "market.csv", tmp_path / "official.csv"
    market.to_csv(market_path, index=False)
    official.to_csv(official_path, index=False)
    return official_path, market_path


def test_official_value_poison_cannot_change_prediction_values(tmp_path):
    official, market = inputs(tmp_path)
    export_inputs(official, market, tmp_path / "original")
    labels = pd.read_csv(official)
    labels.official_close = (labels.official_close * 7).round(1)
    labels.official_change_pct = -23.45
    labels.to_csv(official, index=False)
    export_inputs(official, market, tmp_path / "poisoned")
    for name in ("amount_only", "sse_price_adjusted"):
        file = f"predictions_{name}.csv"
        assert (tmp_path / "original" / file).read_bytes() == (tmp_path / "poisoned" / file).read_bytes()


def test_future_market_poison_cannot_change_predictions(tmp_path):
    official, market = inputs(tmp_path)
    export_inputs(official, market, tmp_path / "original")
    frame = pd.read_csv(market)
    frame.loc[11, ["sh_close", "sh_amount", "sz_amount", "amount"]] *= 1000
    frame.to_csv(market, index=False)
    export_inputs(official, market, tmp_path / "poisoned")
    for name in ("amount_only", "sse_price_adjusted"):
        file = f"predictions_{name}.csv"
        assert (tmp_path / "original" / file).read_bytes() == (tmp_path / "poisoned" / file).read_bytes()


def test_missing_official_date_rejected(tmp_path):
    official, market = inputs(tmp_path)
    pd.read_csv(official).iloc[[0, 2]].to_csv(official, index=False)
    with pytest.raises(ValueError, match="consecutive"):
        export_inputs(official, market, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_existing_evidence_not_overwritten(tmp_path):
    official, market = inputs(tmp_path)
    export_inputs(official, market, tmp_path / "out")
    before = (tmp_path / "out" / "official.csv").read_bytes()
    with pytest.raises(FileExistsError):
        export_inputs(official, market, tmp_path / "out")
    assert (tmp_path / "out" / "official.csv").read_bytes() == before


@pytest.mark.parametrize("column,value", [("official_close", "123.45"), ("official_change_pct", "1.1049")])
def test_official_labels_cannot_be_silently_rounded(tmp_path, column, value):
    official, market = inputs(tmp_path)
    labels = pd.read_csv(official, dtype=str)
    labels.loc[0, column] = value
    labels.to_csv(official, index=False)
    with pytest.raises(ValueError, match="display grid"):
        export_inputs(official, market, tmp_path / "out")
    assert not (tmp_path / "out").exists()


def test_official_label_decimal_digits_are_preserved(tmp_path):
    official, market = inputs(tmp_path)
    labels = pd.read_csv(official, dtype=str)
    labels.loc[0, "official_close"] = "1000000000000000.1"
    labels.to_csv(official, index=False)
    export_inputs(official, market, tmp_path / "out")
    assert (tmp_path / "out" / "official.csv").read_bytes() == official.read_bytes()


def test_fixed_scale_positive_amount_smoothing_cannot_fit_official_signs():
    labels = pd.read_csv(Path(__file__).parent / "validation/exact_parity_20261030/official.csv", dtype=str)
    labels = labels.set_index("date")
    d = Decimal
    with localcontext() as context:
        context.prec = 60
        y11 = d(labels.loc["2026-08-11", "official_close"])
        y12 = d(labels.loc["2026-08-12", "official_close"])
        y13 = d(labels.loc["2026-08-13", "official_close"])
        assert y12 - y11 - d(".1") > 0
        assert y13 - y12 + d(".1") < 0
        lower = (y11 - d(".05")) / d("215242.310981535")
        upper = (y12 + d(".05")) / d("255091.673935884")
        assert lower > upper


@pytest.mark.parametrize("name,mismatches", [("amount_only", 16), ("sse_price_adjusted", 10)])
def test_published_baselines_fail_the_exact_gate(name, mismatches):
    root = Path(__file__).parent / "validation/exact_parity_20261030"
    result = verify_exact_parity(root / "official.csv", root / f"predictions_{name}.csv", root / "calendar.csv")
    assert result["strict_gate"] == "fail"
    assert result["official_formula_replication"] == "not_established"
    assert result["counts"]["numeric_fail"] == 64
    assert result["counts"]["signal_known_mismatches"] == mismatches
