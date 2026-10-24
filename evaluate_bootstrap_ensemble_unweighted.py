
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from features import BASELINE_FEATURES, TARGET_COLUMN


ROOT = Path(__file__).resolve().parent
DATA_DIR = ROOT / "data" / "processed"
MODEL_DIR = DATA_DIR / "bootstrap_models_unweighted"


def main():
    validation = pd.read_csv(DATA_DIR / "validation.csv")
    X_val = validation[BASELINE_FEATURES]
    y_val = validation[TARGET_COLUMN].to_numpy()

    model_paths = sorted(MODEL_DIR.glob("bootstrap_model_*.joblib"))
    if len(model_paths) != 20:
        raise RuntimeError(f"Expected 20 models; found {len(model_paths)}")

    probability_matrix = np.vstack([
        joblib.load(path).predict_proba(X_val)[:, 1]
        for path in model_paths
    ])

    mean_probability = probability_matrix.mean(axis=0)
    std_probability = probability_matrix.std(axis=0, ddof=1)

    print("Validation records:", len(y_val))
    print("Positive label fraction:", round(float(y_val.mean()), 5))
    print("PR-AUC:", round(average_precision_score(y_val, mean_probability), 5))
    print("ROC-AUC:", round(roc_auc_score(y_val, mean_probability), 5))
    print("Brier score:", round(brier_score_loss(y_val, mean_probability), 5))
    print("Mean predicted probability:", round(float(mean_probability.mean()), 5))
    print("Mean bootstrap standard deviation:", round(float(std_probability.mean()), 5))
    print("Median bootstrap standard deviation:", round(float(np.median(std_probability)), 5))

    output = pd.DataFrame({
        "Patient_ID": validation["Patient_ID"],
        "SepsisLabel": y_val,
        "MeanProbability": mean_probability,
        "StdProbability": std_probability,
        "LowerPercentile": np.percentile(probability_matrix, 2.5, axis=0),
        "UpperPercentile": np.percentile(probability_matrix, 97.5, axis=0),
    })

    output_path = DATA_DIR / "saved_bootstrap_validation_results_unweighted.csv"
    output.to_csv(output_path, index=False)
    print("Saved:", output_path)


if __name__ == "__main__":
    main()

