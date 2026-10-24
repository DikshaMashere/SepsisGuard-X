from app.services.uncertainty_service import (
    load_validation_uncertainty_summary,
)


def test_validation_uncertainty_summary():
    summary = load_validation_uncertainty_summary()

    assert summary["source"] == "held-out validation analysis"
    assert summary["record_count"] > 0
    assert summary["patient_count"] > 0
    assert summary["mean_prediction_std"] >= 0
    assert summary["median_prediction_std"] >= 0
    assert summary["mean_interval_width"] >= 0
    assert "not uncertainty estimates for a new prediction" in summary["warning"]