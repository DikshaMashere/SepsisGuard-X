
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

# Load validation data and model trained on train_core
X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(data_dir / "y_validation.csv").squeeze("columns")

model = joblib.load(
    data_dir / "logistic_regression_validation.joblib"
)

# Predicted probability of a positive label
y_prob = model.predict_proba(X_val)[:, 1]

# Compare a few thresholds on validation data
thresholds = [0.20, 0.30, 0.40, 0.50, 0.60]

results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1": f1_score(
            y_val, y_pred, zero_division=0
        ),
        "False_Positives": int(
            confusion_matrix(y_val, y_pred).ravel()[1]
        ),
        "False_Negatives": int(
            confusion_matrix(y_val, y_pred).ravel()[2]
        ),
    })

results_df = pd.DataFrame(results)

print("Validation threshold comparison:")
print(results_df.round(4).to_string(index=False))

# Save validation threshold comparison for the project report
output_path = data_dir / "threshold_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nThreshold comparison saved to:", output_path.resolve())
