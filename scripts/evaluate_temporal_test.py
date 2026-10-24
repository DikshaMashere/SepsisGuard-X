
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

from app.services.temporal_model_service import BASE_FEATURES
from scripts.train_temporal_model import make_history_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TEMPORAL = PROCESSED / "temporal"
MODEL_PATH = (
    PROCESSED / "temporal_model"
    / "temporal_logistic_regression_corrected.joblib"
)


def main():
    artifact = joblib.load(MODEL_PATH)
    original = pd.read_csv(PROCESSED / "test.csv")
    targets = pd.read_csv(TEMPORAL / "test.csv")

    # Build history from the full, original test timeline.
    test = make_history_features(original, targets)

    feature_columns = artifact["feature_columns"]
    X = test[feature_columns]
    y = test["Target_Sepsis_Within_6h"]

    probabilities = artifact["model"].predict_proba(X)[:, 1]

    print("Held-out test evaluation")
    print("Rows:", len(test))
    print("Patients:", test["Patient_ID"].nunique())
    print("Positive target prevalence:", round(float(y.mean()), 6))
    print(
        "PR-AUC:",
        round(float(average_precision_score(y, probabilities)), 5),
    )
    print(
        "ROC-AUC:",
        round(float(roc_auc_score(y, probabilities)), 5),
    )
    print(
        "Brier score:",
        round(float(brier_score_loss(y, probabilities)), 5),
    )
    print(
        "Mean predicted probability:",
        round(float(probabilities.mean()), 5),
    )


if __name__ == "__main__":
    main()
