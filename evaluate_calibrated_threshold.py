
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data_dir = Path("data/processed")

# Load calibrated model and existing validation data
model = joblib.load(
    data_dir / "logistic_regression_calibrated.joblib"
)

X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

# Predict probabilities
probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [0.01, 0.015, 0.02, 0.025, 0.03]

rows = []

for threshold in thresholds:
    predictions = (probabilities >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, predictions, labels=[0, 1]
    ).ravel()

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1": f1_score(
            y_val, predictions, zero_division=0
        ),
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Positives": tp,
    })

results = pd.DataFrame(rows)

print("\nCalibrated Model — Validation Threshold Results")
print(results.round(4).to_string(index=False))

output_path = data_dir / "calibrated_threshold_comparison.csv"
results.to_csv(output_path, index=False)

print("\nSaved to:", output_path.resolve())
print("Final test set has not been used.")
