
from app.services.model_service import predict_sepsis_risk


TREND_FEATURES = ["HR", "O2Sat", "SBP", "MAP", "Resp"]


def analyze_patient_trend(observations: list[dict]) -> dict:
    if len(observations) < 2:
        raise ValueError("At least two observations are required")

    results = []

    for observation in observations:
        probability = predict_sepsis_risk(observation)

        results.append({
            "Hour": observation["Hour"],
            "ICULOS": observation["ICULOS"],
            "probability": probability,
        })

    previous_probability = results[-2]["probability"]
    latest_probability = results[-1]["probability"]

    probability_change = latest_probability - previous_probability

    # Experimental direction indicator, not a clinical classification.
    if probability_change > 0.005:
        direction = "increasing"
    elif probability_change < -0.005:
        direction = "decreasing"
    else:
        direction = "relatively stable"

    recent = results[-min(6, len(results)):]

    first = recent[0]
    last = recent[-1]
    elapsed_hours = last["Hour"] - first["Hour"]

    slope_per_hour = (
        (last["probability"] - first["probability"]) / elapsed_hours
        if elapsed_hours > 0
        else None
    )

    feature_trends = {}

    for feature in TREND_FEATURES:
        first_value = observations[0][feature]
        last_value = observations[-1][feature]

        feature_trends[feature] = {
            "first_value": first_value,
            "latest_value": last_value,
            "change": last_value - first_value,
        }

    return {
        "model_name": "Calibrated Logistic Regression",
        "latest_probability": latest_probability,
        "previous_probability": previous_probability,
        "probability_change": probability_change,
        "direction": direction,
        "recent_slope_per_hour": slope_per_hour,
        "observations_analyzed": len(observations),
        "observation_predictions": results,
        "feature_trends": feature_trends,
        "warning": (
            "Experimental trend analysis based on repeated predictions. "
            "This is not a temporal model, clinical diagnosis, or "
            "validated clinical alert."
        ),
    }
