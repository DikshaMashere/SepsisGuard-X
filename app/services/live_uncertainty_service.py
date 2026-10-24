
from functools import lru_cache
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from app.services.model_service import FEATURES


PROJECT_ROOT = Path(__file__).resolve().parents[2]
BOOTSTRAP_DIR = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "bootstrap_models_unweighted"
)


@lru_cache(maxsize=1)
def load_bootstrap_models():
    model_paths = sorted(
        BOOTSTRAP_DIR.glob("bootstrap_model_*.joblib")
    )

    if len(model_paths) != 20:
        raise FileNotFoundError(
            f"Expected 20 bootstrap models, found {len(model_paths)}"
        )

    return tuple(joblib.load(path) for path in model_paths)


def estimate_live_uncertainty(values: dict) -> dict:
    missing = set(FEATURES) - set(values)
    extra = set(values) - set(FEATURES)

    if missing or extra:
        raise ValueError(
            f"Missing features: {sorted(missing)}; "
            f"unexpected features: {sorted(extra)}"
        )

    frame = pd.DataFrame(
        [[values[feature] for feature in FEATURES]],
        columns=FEATURES,
    )

    models = load_bootstrap_models()

    probabilities = np.array([
        float(model.predict_proba(frame)[0, 1])
        for model in models
    ])

    return {
        "bootstrap_model_count": len(probabilities),
        "mean_bootstrap_probability": float(np.mean(probabilities)),
        "standard_deviation": float(np.std(probabilities, ddof=0)),
        "minimum_probability": float(np.min(probabilities)),
        "maximum_probability": float(np.max(probabilities)),
        "uncertainty_method": "Bootstrap ensemble disagreement",
        "warning": (
            "Experimental model-disagreement indicator, not a calibrated "
            "confidence interval or clinical uncertainty estimate."
        ),
    }
