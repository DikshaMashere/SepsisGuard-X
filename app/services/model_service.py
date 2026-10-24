from functools import lru_cache
from pathlib import Path

import joblib
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
MODEL_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "logistic_regression_calibrated.joblib"
)

FEATURES = [
    "Hour",
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "Resp",
    "Age",
    "Gender",
    "HospAdmTime",
    "ICULOS",
]


@lru_cache(maxsize=1)
def load_model():
    if not MODEL_PATH.is_file():
        raise FileNotFoundError(
            f"Trained model not found: {MODEL_PATH}"
        )

    return joblib.load(MODEL_PATH)


def predict_sepsis_risk(values: dict) -> float:
    missing_features = set(FEATURES) - set(values)
    extra_features = set(values) - set(FEATURES)

    if missing_features or extra_features:
        raise ValueError(
            f"Missing features: {sorted(missing_features)}; "
            f"unexpected features: {sorted(extra_features)}"
        )

    input_frame = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    model = load_model()
    probability = float(model.predict_proba(input_frame)[0, 1])

    return probability
