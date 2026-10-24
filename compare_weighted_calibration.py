
import pandas as pd

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss
from sklearn.calibration import calibration_curve

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
    "Balanced Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    ),
    "Unweighted Logistic Regression": LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=42,
    ),
}

plt_data = {}

print("Comparing weighted and unweighted models...\n")

for name, model in models.items():
    model.fit(X_train, y_train)
    probabilities = model.predict_proba(X_val)[:, 1]

    brier = brier_score_loss(y_val, probabilities)

    observed, predicted = calibration_curve(
        y_val,
        probabilities,
        n_bins=10,
        strategy="quantile",
    )

    plt_data[name] = (predicted, observed)

    print(name)
    print(f"  Brier score: {brier:.5f}")
    print(f"  Mean predicted probability: {probabilities.mean():.5f}")
    print()

import matplotlib.pyplot as plt

plt.figure(figsize=(8, 6))

for name, (predicted, observed) in plt_data.items():
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
plt.title("Weighted vs Unweighted Logistic Regression")
plt.legend()
plt.tight_layout()

output_path = data_dir / "weighted_vs_unweighted_calibration.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Chart saved to:", output_path.resolve())
print("Final test set has not been used.")
