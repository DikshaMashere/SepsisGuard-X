
import pytest

from app.services.trend_service import analyze_patient_trend


def make_observation(hour, hr, o2sat=98, sbp=120, map_value=80, resp=16):
    return {
        "Hour": hour,
        "HR": hr,
        "O2Sat": o2sat,
        "SBP": sbp,
        "MAP": map_value,
        "Resp": resp,
        "Age": 25,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": hour,
    }


def test_trend_returns_expected_structure():
    observations = [
        make_observation(1, 85),
        make_observation(2, 100, o2sat=96, sbp=115, map_value=78, resp=20),
    ]

    result = analyze_patient_trend(observations)

    assert result["observations_analyzed"] == 2
    assert 0 <= result["latest_probability"] <= 1
    assert 0 <= result["previous_probability"] <= 1
    assert len(result["observation_predictions"]) == 2
    assert set(result["feature_trends"]) == {
        "HR", "O2Sat", "SBP", "MAP", "Resp"
    }


def test_feature_changes_are_calculated_correctly():
    observations = [
        make_observation(1, 85),
        make_observation(2, 100, o2sat=96, sbp=115, map_value=78, resp=20),
    ]

    result = analyze_patient_trend(observations)
    trends = result["feature_trends"]

    assert trends["HR"]["change"] == 15
    assert trends["O2Sat"]["change"] == -2
    assert trends["SBP"]["change"] == -5
    assert trends["MAP"]["change"] == -2
    assert trends["Resp"]["change"] == 4


def test_increasing_probability_direction(monkeypatch):
    probabilities = iter([0.01, 0.03])

    monkeypatch.setattr(
        "app.services.trend_service.predict_sepsis_risk",
        lambda observation: next(probabilities),
    )

    result = analyze_patient_trend([
        make_observation(1, 85),
        make_observation(2, 90),
    ])

    assert result["direction"] == "increasing"
    assert result["probability_change"] == pytest.approx(0.02)


def test_decreasing_probability_direction(monkeypatch):
    probabilities = iter([0.04, 0.01])

    monkeypatch.setattr(
        "app.services.trend_service.predict_sepsis_risk",
        lambda observation: next(probabilities),
    )

    result = analyze_patient_trend([
        make_observation(1, 100),
        make_observation(2, 90),
    ])

    assert result["direction"] == "decreasing"
    assert result["probability_change"] == pytest.approx(-0.03)


def test_requires_at_least_two_observations():
    with pytest.raises(ValueError, match="At least two observations"):
        analyze_patient_trend([
            make_observation(1, 85),
        ])
