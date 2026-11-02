"""Exact nearest-grid display compatibility, not official algorithm parity.

Call only after independently freezing predictions and a measured display
contract. Quantum is mandatory, never inferred from labels or residuals.
"""
from dataclasses import dataclass
from decimal import Decimal, InvalidOperation
from fractions import Fraction
import re
from typing import Literal


ExactNumber = str | int | Decimal | Fraction
_DECIMAL = re.compile(r"[+-]?(?:[0-9]+(?:\.[0-9]*)?|\.[0-9]+)(?:[eE][+-]?[0-9]+)?\Z")


@dataclass(frozen=True)
class DisplayComparison:
    indicator: Literal["CYF", "CYC"]
    prediction: Fraction
    displayed: Fraction
    quantum: Fraction
    lower: Fraction
    upper: Fraction
    comparison: Literal["compatible", "incompatible", "ambiguous"]
    transformed_diagnostic: Literal["censored_at_100", "interval_required", "not_applicable"]


def _exact(value: ExactNumber, name: str) -> Fraction:
    if isinstance(value, bool) or not isinstance(value, (str, int, Decimal, Fraction)):
        raise ValueError(f"{name} must be an exact decimal string, int, Decimal or Fraction; no floats")
    if isinstance(value, str):
        if not _DECIMAL.fullmatch(value):
            raise ValueError(f"{name} must be a finite decimal string")
        try:
            value = Decimal(value)
        except InvalidOperation as exc:
            raise ValueError(f"{name} must be a representable decimal") from exc
    if isinstance(value, Decimal) and not value.is_finite():
        raise ValueError(f"{name} must be finite")
    return Fraction(value)


def compare_display(
    prediction: ExactNumber, displayed: ExactNumber, *, indicator: Literal["CYF", "CYC"],
    quantum: ExactNumber,
) -> DisplayComparison:
    """Compare one frozen prediction to a zero-origin, nearest-grid display bin.

    This function assumes nearest rounding; it does not establish the App's
    display rule. Unknown quantum or non-nearest behavior cannot be scored here.
    Midpoints are ambiguous because tie direction is not assumed. Physical
    clipping at CYF 0/100 is not a rounding tie. CYC truth must be positive,
    while displayed zero may represent a small positive value. Bounds enclose
    the bin; use comparison for its tie/open-end semantics, not <= alone.

    CYF log-survival/odds diagnostics are unusable when the interval includes
    100. Other intervals require propagated uncertainty; this function never
    treats the rounded label as an exact transformed point. No I/O or state.
    """
    if indicator not in ("CYF", "CYC"):
        raise ValueError("indicator must be CYF or CYC")
    predicted = _exact(prediction, "prediction")
    shown = _exact(displayed, "displayed")
    step = _exact(quantum, "quantum")
    if step <= 0:
        raise ValueError("quantum must be positive")
    if shown < 0 or (indicator == "CYF" and shown > 100):
        raise ValueError("displayed is outside the physical display range")
    if (shown / step).denominator != 1:
        raise ValueError("displayed must be on the zero-origin quantum grid")
    if (indicator == "CYF" and not 0 <= predicted <= 100) or (indicator == "CYC" and predicted <= 0):
        raise ValueError("prediction is outside the physical domain")

    raw_lower, raw_upper = shown - step / 2, shown + step / 2
    lower = max(Fraction(0), raw_lower)
    upper = min(Fraction(100), raw_upper) if indicator == "CYF" else raw_upper
    comparison = "incompatible"
    if lower <= predicted <= upper:
        comparison = "ambiguous" if predicted in (raw_lower, raw_upper) else "compatible"
    transformed = "not_applicable"
    if indicator == "CYF":
        transformed = "censored_at_100" if upper == 100 else "interval_required"
    return DisplayComparison(indicator, predicted, shown, step, lower, upper, comparison, transformed)
