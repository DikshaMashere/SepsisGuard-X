
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

ROOT = Path(__file__).resolve().parent
DATA_PATH = (
    ROOT
    / "data"
    / "processed"
    / "saved_bootstrap_validation_results_unweighted.csv"
)

THRESHOLD = 0.03


def main():
    if not DATA_PATH.is_file():
        raise FileNotFoundError(f"Validation results not found: {DATA_PATH}")

    df = pd.read_csv(DATA_PATH)

    required = {"SepsisLabel", "MeanProbability", "StdProbability"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    y_true = df["SepsisLabel"].to_numpy()
    probability = df["MeanProbability"].to_numpy()
    uncertainty = df["StdProbability"].to_numpy()

    valid = (
        np.isfinite(y_true)
        & np.isfinite(probability)
        & np.isfinite(uncertainty)
    )
    y_true = y_true[valid]
    probability = probability[valid]
    uncertainty = uncertainty[valid]

    predicted_positive = probability >= THRESHOLD
    errors = predicted_positive != y_true.astype(bool)

    print("Validation records:", len(y_true))
    print("Decision threshold:", THRESHOLD)
    print("Misclassified records:", int(errors.sum()))
    print("Misclassification rate:", round(float(errors.mean()), 5))

    if np.unique(errors).size == 2:
        print(
            "Uncertainty ROC-AUC for detecting errors:",
            round(roc_auc_score(errors.astype(int), uncertainty), 5),
        )
        print(
            "Uncertainty PR-AUC for detecting errors:",
            round(average_precision_score(errors.astype(int), uncertainty), 5),
        )
    else:
        print("Error-detection metrics unavailable: only one error class.")

    print(
        "Mean uncertainty on errors:",
        round(float(uncertainty[errors].mean()), 6)
        if errors.any() else "N/A",
    )
    print(
        "Mean uncertainty on correct predictions:",
        round(float(uncertainty[~errors].mean()), 6)
        if (~errors).any() else "N/A",
    )

    print("\nValidation analysis only; not a clinical reliability guarantee.")


if __name__ == "__main__":
    main()
