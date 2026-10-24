
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.calibration import calibration_curve

import matplotlib.pyplot as plt

data_dir = Path("data/processed")

# Load validation data; do not use the final test set
X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

# Load the calibrated model
calibrated_model = joblib.load(
    data_dir / "logistic_regression_calibrated.joblib"
)

# Train an unweighted reference model on the original training core
# for comparison with the same validation set.
X_train = pd.read_csv(data_dir / "X_train_core.csv")
y_train = pd.read_csv(
    data_dir / "y_train_core.csv"
).squeeze("columns")

baseline = LogisticRegression(
    max_iter=1000,
    class_weight=None,
    random_state=42,
)
baseline.fit(X_train, y_train)

models = {
    "Unweighted baseline": baseline,
    "Sigmoid calibrated": calibrated_model,
}

probabilities_by_model = {}

print("Validation comparison\n")

for name, model in models.items():
    probabilities = model.predict_proba(X_val)[:, 1]
    predictions = (probabilities >= 0.5).astype(int)

    probabilities_by_model[name] = probabilities

    print(name)
    print(f"  PR-AUC: {average_precision_score(y_val, probabilities):.5f}")
    print(f"  ROC-AUC: {roc_auc_score(y_val, probabilities):.5f}")
    print(f"  Brier score: {brier_score_loss(y_val, probabilities):.5f}")
    print(f"  Mean predicted probability: {probabilities.mean():.5f}")
    print(f"  Precision at 0.5: {precision_score(y_val, predictions, zero_division=0):.5f}")
    print(f"  Recall at 0.5: {recall_score(y_val, predictions, zero_division=0):.5f}")
    print(f"  F1 at 0.5: {f1_score(y_val, predictions, zero_division=0):.5f}")
    print()

# Plot validation calibration curves
plt.figure(figsize=(8, 6))

for name, probabilities in probabilities_by_model.items():
    observed, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    plt.plot(
        predicted,
        observed,
        marker="o",
        label=name,
    )

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title("Validation Calibration: Baseline vs Sigmoid")
plt.legend()
plt.tight_layout()

output_path = data_dir / "calibrated_validation_curve.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Calibration plot saved to:", output_path.resolve())
print("Final test set has not been used.")
