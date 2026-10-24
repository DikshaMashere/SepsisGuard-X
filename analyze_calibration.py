
import pandas as pd
import joblib
import matplotlib.pyplot as plt

from pathlib import Path
from sklearn.calibration import calibration_curve
from sklearn.metrics import brier_score_loss

data_dir = Path("data/processed")

# Load the saved baseline model and validation data
model = joblib.load(
    data_dir / "logistic_regression_validation.joblib"
)

X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

# Predicted probability for each validation record
probabilities = model.predict_proba(X_val)[:, 1]

# Brier score: lower is better, but compare against a baseline too
brier = brier_score_loss(y_val, probabilities)

# Compare predicted probabilities with observed event frequencies
observed_fraction, mean_predicted = calibration_curve(
    y_val,
    probabilities,
    n_bins=10,
    strategy="quantile",
)

plt.figure(figsize=(7, 6))

plt.plot(
    mean_predicted,
    observed_fraction,
    marker="o",
    label="Baseline Logistic Regression",
)

plt.plot(
    [0, 1],
    [0, 1],
    linestyle="--",
    label="Perfect calibration",
)

plt.xlabel("Mean predicted probability")
plt.ylabel("Observed positive fraction")
plt.title("SepsisGuard-X: Validation Calibration Curve")
plt.legend()
plt.tight_layout()

output_path = data_dir / "calibration_curve.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Validation Brier score:", round(brier, 5))
print("Calibration curve saved to:", output_path.resolve())
print("Final test set has not been used.")
