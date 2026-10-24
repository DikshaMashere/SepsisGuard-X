
import numpy as np

from app.services.live_uncertainty_service import (
    estimate_live_uncertainty,
    load_bootstrap_models,
)


def sample_features():
    return {
        "Hour": 2,
        "HR": 100,
        "O2Sat": 96,
        "SBP": 115,
        "MAP": 78,
        "Resp": 20,
        "Age": 25,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": 2,
    }


def test_loads_twenty_bootstrap_models():
    models = load_bootstrap_models()

    assert len(models) == 20


def test_uncertainty_output_is_valid():
    result = estimate_live_uncertainty(sample_features())

    assert result["bootstrap_model_count"] == 20
    assert 0 <= result["mean_bootstrap_probability"] <= 1
    assert result["standard_deviation"] >= 0
    assert 0 <= result["minimum_probability"] <= 1
    assert 0 <= result["maximum_probability"] <= 1


def test_probability_range_is_consistent():
    result = estimate_live_uncertainty(sample_features())

    assert (
        result["minimum_probability"]
        <= result["mean_bootstrap_probability"]
        <= result["maximum_probability"]
    )


def test_uncertainty_is_finite():
    result = estimate_live_uncertainty(sample_features())

    assert np.isfinite(result["mean_bootstrap_probability"])
    assert np.isfinite(result["standard_deviation"])
    assert np.isfinite(result["minimum_probability"])
    assert np.isfinite(result["maximum_probability"])


def test_invalid_features_are_rejected():
    invalid_features = sample_features()
    invalid_features.pop("HR")

    try:
        estimate_live_uncertainty(invalid_features)
        assert False, "Expected ValueError for missing feature"
    except ValueError as exc:
        assert "HR" in str(exc)
