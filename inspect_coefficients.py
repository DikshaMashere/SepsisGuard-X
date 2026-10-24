
import pandas as pd
import joblib

from pathlib import Path

from features import BASELINE_FEATURES

data_dir = Path("data/processed")

# Load model trained on train_core
model = joblib.load(
    data_dir / "logistic_regression_validation.joblib"
)

# Inspect Logistic Regression coefficients
coefficients = pd.DataFrame({
    "Feature": BASELINE_FEATURES,
    "Coefficient": model.coef_[0],
})

coefficients["Absolute_Coefficient"] = (
    coefficients["Coefficient"].abs()
)

coefficients = coefficients.sort_values(
    "Absolute_Coefficient",
    ascending=False,
)

print("Logistic Regression coefficients:")
print(coefficients.to_string(index=False))

output_path = data_dir / "feature_coefficients.csv"
coefficients.to_csv(output_path, index=False)

print("\nCoefficient report saved to:", output_path.resolve())
print(
    "\nNote: Coefficients depend on feature scales; "
    "they are not causal effects."
)
