
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data_dir = Path("data/processed")

X_train = pd.read_csv(data_dir / "X_train_core.csv")
X_val = pd.read_csv(data_dir / "X_validation.csv")

y_train = pd.read_csv(
    data_dir / "y_train_core.csv"
).squeeze("columns")

y_val = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

model = LogisticRegression(
    max_iter=1000,
    class_weight=None,
    random_state=42,
)

print("Training unweighted Logistic Regression...")
model.fit(X_train, y_train)

probabilities = model.predict_proba(X_val)[:, 1]

thresholds = [0.01, 0.015, 0.02, 0.025, 0.03, 0.04, 0.05]

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

print("\nUnweighted Logistic Regression — Threshold Results")
print(results.round(4).to_string(index=False))

output_path = data_dir / "unweighted_threshold_comparison.csv"
results.to_csv(output_path, index=False)

print("\nSaved to:", output_path.resolve())
print("Final test set has not been used.")
