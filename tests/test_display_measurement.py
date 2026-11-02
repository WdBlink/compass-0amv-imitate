"""Synthetic measurement regressions; no official display precision is claimed."""
from decimal import Decimal, localcontext
from fractions import Fraction
from math import log, prod

import pytest

from research.mechanism_research_20261030.display_measurement import compare_display


def cyf(prediction, displayed, quantum="0.1"):
    return compare_display(prediction, displayed, indicator="CYF", quantum=quantum)


def survival_predictions(q, n=13):
    return [100 * (1 - prod(1 - x for x in q[t - n + 1:t + 1])) for t in range(n - 1, len(q))]


def test_exact_survival_saturation_retains_row_and_censors_transforms():
    truth = survival_predictions([Fraction(1, 2)] * 13)[0]
    assert truth == Fraction("99.98779296875")
    result = cyf(truth, "100.0")
    assert result.comparison == "compatible"
    assert (result.lower, result.upper) == (Fraction("99.95"), 100)
    assert result.transformed_diagnostic == "censored_at_100"
    wrong = cyf("99.9", "100.0")
    assert wrong.comparison == "incompatible"
    assert wrong.transformed_diagnostic == "censored_at_100"


def test_nonboundary_rounded_labels_break_identity_but_not_forward_truth():
    q = [Fraction(x) for x in (
        ".01", ".05", ".10", ".20", ".02", ".03", ".04", ".11", ".09", ".08",
        ".15", ".07", ".06", ".13", ".025", ".045", ".055", ".065", ".075", ".085",
    )]
    truth = survival_predictions(q)
    shown = [round(x, 1) for x in truth]
    residuals = [
        log(float((1 - shown[i] / 100) / (1 - shown[i - 1] / 100)))
        - log(float(1 - q[i + 12])) + log(float(1 - q[i - 1]))
        for i in range(1, len(shown))
    ]
    assert max(map(abs, residuals)) == pytest.approx(.0026976831569115, abs=1e-14)
    for prediction, display in zip(truth, shown):
        result = cyf(prediction, display)
        assert result.comparison == "compatible"
        assert result.transformed_diagnostic == "interval_required"
        assert cyf(prediction + 1, display).comparison == "incompatible"


@pytest.mark.parametrize("prediction,display", [("10.05", "10.0"), ("9.95", "10.0"),
                                               ("99.95", "100"), ("0.05", "0")])
def test_exact_midpoints_are_not_passes(prediction, display):
    assert cyf(prediction, display).comparison == "ambiguous"


@pytest.mark.parametrize("endpoint", [0, 100])
def test_physical_cyf_endpoints_are_valid_not_spurious_ties(endpoint):
    assert cyf(endpoint, endpoint).comparison == "compatible"
    assert cyf(endpoint, endpoint).transformed_diagnostic == (
        "censored_at_100" if endpoint == 100 else "interval_required"
    )


def test_cyc_positive_domain_including_displayed_zero_and_exact_ties():
    for prediction, display, expected in [("12.345", "12.35", "ambiguous"),
                                          ("12.346", "12.35", "compatible"),
                                          ("12.344", "12.35", "incompatible"),
                                          ("0.004", "0.00", "compatible")]:
        result = compare_display(prediction, display, indicator="CYC", quantum="0.01")
        assert result.comparison == expected
        assert result.transformed_diagnostic == "not_applicable"
    for bad in (0, -1):
        with pytest.raises(ValueError, match="physical domain"):
            compare_display(bad, "0.00", indicator="CYC", quantum="0.01")


def test_arbitrary_grid_and_precision_independent_of_decimal_context():
    assert cyf("10.1", "10.00", "0.25").comparison == "compatible"
    assert cyf("10.125", "10.00", "0.25").comparison == "ambiguous"
    with localcontext() as context:
        context.prec = 3
        inside = Decimal("10.049999999999999999999999999999999999")
        outside = Decimal("10.050000000000000000000000000000000001")
        assert cyf(inside, "10.0").comparison == "compatible"
        assert cyf(outside, "10.0").comparison == "incompatible"
        assert cyf(Decimal("10.05"), "10.0").comparison == "ambiguous"
    assert cyf(Fraction(1, 3), Fraction(1, 3), Fraction(1, 3)).comparison == "compatible"


@pytest.mark.parametrize("field", ["prediction", "displayed", "quantum"])
@pytest.mark.parametrize("bad", [True, None, [], {}, 1.0, float("nan"), float("inf"),
                                  Decimal("NaN"), Decimal("sNaN"), Decimal("Infinity"),
                                  "NaN", "inf", "", " 1", "1 ", "1/2", "1_0", "1x",
                                  "1e999999999999999999999999999999"])
def test_malformed_nonfinite_and_inexact_inputs_rejected(field, bad):
    kwargs = dict(prediction="10", displayed="10", quantum="0.1", indicator="CYF")
    kwargs[field] = bad
    with pytest.raises(ValueError):
        compare_display(**kwargs)


@pytest.mark.parametrize("prediction,display,quantum", [
    ("10", "10.01", ".1"), ("10", "10.1", ".25"), ("100", "100", ".3"),
    ("10", "10", "0"), ("10", "10", "-1"),
    ("-0.1", "0", ".1"), ("100.01", "100", ".1"),
    ("0", "-0.1", ".1"), ("100", "100.1", ".1"),
])
def test_invalid_domain_quantum_and_off_grid_labels_rejected(prediction, display, quantum):
    with pytest.raises(ValueError):
        cyf(prediction, display, quantum)


def test_contract_is_explicit_and_indicator_is_not_inferred():
    with pytest.raises(TypeError):
        compare_display("10", "10", indicator="CYF")
    with pytest.raises(ValueError, match="indicator"):
        compare_display("10", "10", indicator="other", quantum=".1")


def test_prediction_label_and_future_separation():
    q = [Fraction(".1")] * 15
    frozen = tuple(survival_predictions(q))
    prefix = tuple(survival_predictions(q[:14]))
    poisoned_future = tuple(survival_predictions(q[:14] + [Fraction(".9")]))
    assert prefix == frozen[:2] == poisoned_future[:2]
    labels = [round(value, 1) for value in frozen]
    results = [cyf(value, label) for value, label in zip(frozen, labels)]
    changed_labels = [cyf(value, "0.0") for value in frozen]
    assert all(result.comparison == "compatible" for result in results)
    assert all(result.comparison == "incompatible" for result in changed_labels)
    assert tuple(result.prediction for result in results) == frozen
    assert tuple(result.prediction for result in changed_labels) == frozen
    assert results[0] == cyf(frozen[0], labels[0])
