import pandas as pd

from validation.detect_regimes import compare_with_truth


def test_one_prediction_cannot_hit_multiple_truth_intervals():
    detected = pd.DataFrame(
        [{"start": pd.Timestamp("2024-01-01"), "end": pd.Timestamp("2024-03-31"), "type": "long"}]
    )
    result = compare_with_truth(
        detected,
        [("2024-01-10", "2024-01-20"), ("2024-03-10", "2024-03-20")],
    )
    assert len(result["hits"]) == 1
    assert len(result["misses"]) == 1
    assert result["precision"] == 1.0
    assert result["recall"] == 0.5
