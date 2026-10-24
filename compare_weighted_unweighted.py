
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    recall_score,
    precision_score,
    f1_score,
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

models = {
    "Balanced": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Unweighted": LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=42,
    ),
}

rows = []

for name, model in models.items():
    model.fit(X_train, y_train)

    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    rows.append({
        "Model": name,
        "PR_AUC": average_precision_score(y_val, probabilities),
        "ROC_AUC": roc_auc_score(y_val, probabilities),
        "Brier_Score": brier_score_loss(y_val, probabilities),
        "Precision_at_0.5": precision_score(
            y_val, predictions, zero_division=0
        ),
        "Recall_at_0.5": recall_score(
            y_val, predictions, zero_division=0
        ),
        "F1_at_0.5": f1_score(
            y_val, predictions, zero_division=0
        ),
    })

comparison = pd.DataFrame(rows).set_index("Model")

print("\nWeighted vs Unweighted — Validation Comparison")
print(comparison.round(4))

output_path = data_dir / "weighted_unweighted_comparison.csv"
comparison.to_csv(output_path)

print("\nSaved to:", output_path.resolve())
print("Final test set has not been used.")
