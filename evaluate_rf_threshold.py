
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    average_precision_score,
)

data_dir = Path("data/processed")

X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(data_dir / "y_validation.csv").squeeze("columns")

model = joblib.load(
    data_dir / "random_forest_validation.joblib"
)

y_prob = model.predict_proba(X_val)[:, 1]

print(
    "Random Forest validation PR-AUC:",
    round(average_precision_score(y_val, y_prob), 4),
)

thresholds = [0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50]
results = []

for threshold in thresholds:
    y_pred = (y_prob >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        y_val, y_pred, labels=[0, 1]
    ).ravel()

    results.append({
        "Threshold": threshold,
        "Precision": precision_score(
            y_val, y_pred, zero_division=0
        ),
        "Recall": recall_score(
            y_val, y_pred, zero_division=0
        ),
        "F1": f1_score(y_val, y_pred, zero_division=0),
        "False_Positives": int(fp),
        "False_Negatives": int(fn),
    })

results_df = pd.DataFrame(results)

print("\nRandom Forest threshold comparison:")
print(results_df.round(4).to_string(index=False))

output_path = data_dir / "rf_threshold_comparison.csv"
results_df.to_csv(output_path, index=False)

print("\nSaved to:", output_path.resolve())
