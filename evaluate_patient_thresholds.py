
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data_dir = Path("data/processed")

patients = pd.read_csv(
    data_dir / "patient_level_validation_results.csv"
)

# Evaluate the balanced model's patient-level scores
scores = patients["BalancedScore"]
labels = patients["SepsisLabel"]

thresholds = [0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80]

rows = []

for threshold in thresholds:
    predictions = (scores >= threshold).astype(int)

    tn, fp, fn, tp = confusion_matrix(
        labels, predictions, labels=[0, 1]
    ).ravel()

    rows.append({
        "Threshold": threshold,
        "Precision": precision_score(
            labels, predictions, zero_division=0
        ),
        "Recall": recall_score(
            labels, predictions, zero_division=0
        ),
        "F1": f1_score(
            labels, predictions, zero_division=0
        ),
        "True_Positives": tp,
        "False_Positives": fp,
        "False_Negatives": fn,
        "True_Negatives": tn,
    })

results = pd.DataFrame(rows)

print("\nPatient-level threshold analysis")
print(results.round(4).to_string(index=False))

output_path = data_dir / "patient_threshold_comparison.csv"
results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path.resolve())
print("Final test set has not been used.")
