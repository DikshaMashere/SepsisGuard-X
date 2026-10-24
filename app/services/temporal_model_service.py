
from pathlib import Path

import joblib
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    ROOT / "data" / "processed" / "temporal_model"
    / "temporal_logistic_regression_corrected.joblib"
)

BASE_FEATURES = [
    "HR", "O2Sat", "SBP", "MAP", "Resp",
    "Age", "Gender", "HospAdmTime", "ICULOS",
]


def load_temporal_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Corrected temporal model not found: {MODEL_PATH}"
        )
    return joblib.load(MODEL_PATH)


def build_latest_features(observations, feature_columns):
    if len(observations) < 7:
        raise ValueError(
            "At least 7 consecutive hourly observations are required."
        )

    df = pd.DataFrame(observations).sort_values("Hour").reset_index(drop=True)

    required = ["Hour"] + BASE_FEATURES
    missing = [col for col in required if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required fields: {missing}")

    if df["Hour"].duplicated().any():
        raise ValueError("Observation hours must be unique.")

    hour_diffs = pd.to_numeric(df["Hour"], errors="coerce").diff().dropna()
    iculos_diffs = pd.to_numeric(
        df["ICULOS"], errors="coerce"
    ).diff().dropna()

    if not np.allclose(hour_diffs, 1.0):
        raise ValueError("Observations must be exactly one hour apart.")
    if not np.allclose(iculos_diffs, 1.0):
        raise ValueError("ICULOS must increase by exactly one hour.")

    result = {
        "Hour": float(df.iloc[-1]["Hour"]),
        "HourOfStay": float(df.iloc[-1]["Hour"]),
    }

    for feature in BASE_FEATURES:
        values = pd.to_numeric(df[feature], errors="coerce")

        result[f"{feature}_current"] = values.iloc[-1]

        for lag in (1, 2, 3, 6):
            result[f"{feature}_lag{lag}"] = values.iloc[-1 - lag]

        result[f"{feature}_mean3"] = values.iloc[-3:].mean()
        result[f"{feature}_change3"] = values.iloc[-1] - values.iloc[-4]

    X = pd.DataFrame(
        [[result.get(col, np.nan) for col in feature_columns]],
        columns=feature_columns,
    )
    return X


def predict_temporal_risk(observations):
    artifact = load_temporal_model()
    X = build_latest_features(observations, artifact["feature_columns"])

    probability = float(artifact["model"].predict_proba(X)[0, 1])

    return {
        "probability": probability,
        "prediction_horizon_hours": 6,
        "model_name": "Corrected Calibrated Temporal Logistic Regression",
        "observations_used": len(observations),
        "disclaimer": (
            "Experimental research prototype, not clinically validated. "
            "Not for diagnosis or patient-care decisions."
        ),
    }
