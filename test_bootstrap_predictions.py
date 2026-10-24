
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from features import BASELINE_FEATURES


ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "data" / "processed" / "bootstrap_models"


def main():
    model_paths = sorted(MODEL_DIR.glob("bootstrap_model_*.joblib"))

    if len(model_paths) != 20:
        raise RuntimeError(
            f"Expected 20 bootstrap models, found {len(model_paths)}"
        )

    # Synthetic example only; not a real patient record.
    sample = pd.DataFrame([{
        "Hour": 5,
        "HR": 90,
        "O2Sat": 97,
        "SBP": 120,
        "MAP": 85,
        "Resp": 18,
        "Age": 55,
        "Gender": 1,
        "HospAdmTime": -10,
        "ICULOS": 5,
    }], columns=BASELINE_FEATURES)

    probabilities = []

    for path in model_paths:
        model = joblib.load(path)
        probability = float(model.predict_proba(sample)[0, 1])
        probabilities.append(probability)

    probabilities = np.asarray(probabilities)

    print("Models evaluated:", len(probabilities))
    print("Individual probabilities:", np.round(probabilities, 5))
    print("Mean probability:", round(float(probabilities.mean()), 5))
    print("Standard deviation:", round(float(probabilities.std(ddof=1)), 5))
    print(
        "2.5th–97.5th percentile interval:",
        np.round(np.percentile(probabilities, [2.5, 97.5]), 5),
    )
    print("\nSynthetic input only; not a clinical prediction.")


if __name__ == "__main__":
    main()
