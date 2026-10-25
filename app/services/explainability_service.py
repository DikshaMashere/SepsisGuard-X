
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd
import shap

from app.services.model_service import FEATURES, load_model


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Small, reproducible SHAP background dataset committed to the repository.
BACKGROUND_DATA_PATH = (
    PROJECT_ROOT / "data" / "processed" / "shap_background.csv"
)


@lru_cache(maxsize=1)
def load_background_data():
    """Load the saved background sample used for SHAP explanations."""

    if not BACKGROUND_DATA_PATH.is_file():
        raise FileNotFoundError(
            f"SHAP background data not found: {BACKGROUND_DATA_PATH}"
        )

    df = pd.read_csv(BACKGROUND_DATA_PATH, usecols=FEATURES)

    if df.empty:
        raise ValueError("SHAP background data is empty.")

    # Keep the same reproducible sampling behavior.
    return df.sample(n=min(100, len(df)), random_state=42)


@lru_cache(maxsize=1)
def load_explainer():
    """Create and cache a SHAP explainer for model probabilities."""

    model = load_model()
    background = load_background_data()

    def predict_probability(values):
        frame = pd.DataFrame(values, columns=FEATURES)
        return model.predict_proba(frame)[:, 1]

    masker = shap.maskers.Independent(background)
    return shap.Explainer(predict_probability, masker)


def explain_sepsis_prediction(values: dict) -> dict:
    """Explain one model prediction using SHAP feature contributions."""

    missing = set(FEATURES) - set(values)
    extra = set(values) - set(FEATURES)

    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}; "
            f"unexpected features: {sorted(extra)}"
        )

    frame = pd.DataFrame(
        [[values[name] for name in FEATURES]],
        columns=FEATURES,
    )

    model = load_model()
    probability = float(model.predict_proba(frame)[0, 1])

    explanation = load_explainer()(frame)

    contributions = np.asarray(explanation.values)[0]
    base_value = float(
        np.asarray(explanation.base_values).reshape(-1)[0]
    )

    ranked = sorted(
        [
            {
                "feature": name,
                "value": float(values[name]),
                "contribution": float(contributions[index]),
            }
            for index, name in enumerate(FEATURES)
        ],
        key=lambda item: abs(item["contribution"]),
        reverse=True,
    )

    return {
        "probability": probability,
        "base_value": base_value,
        "feature_contributions": ranked,
        "explanation_method": "SHAP with independent background masker",
        "warning": (
            "Research explanation only. Contributions describe model "
            "behavior, not causal effects or clinical evidence."
        ),
    }
