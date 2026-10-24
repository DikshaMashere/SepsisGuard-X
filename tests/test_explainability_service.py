
import numpy as np

from app.services.explainability_service import explain_sepsis_prediction


def test_explain_sepsis_prediction():
    features = {
        "Hour": 10,
        "HR": 100,
        "O2Sat": 96,
        "SBP": 120,
        "MAP": 80,
        "Resp": 18,
        "Age": 25,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": 5,
    }

    result = explain_sepsis_prediction(features)

    assert 0.0 <= result["probability"] <= 1.0
    assert np.isfinite(result["base_value"])
    assert len(result["feature_contributions"]) == 10

    assert result["explanation_method"] == (
        "SHAP with independent background masker"
    )

    for item in result["feature_contributions"]:
        assert item["feature"] in features
        assert np.isfinite(item["contribution"])
        assert np.isfinite(item["value"])

    assert "not causal effects" in result["warning"]
