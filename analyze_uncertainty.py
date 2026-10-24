
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import average_precision_score, roc_auc_score

data_dir = Path("data/processed")

# Load the saved baseline model and validation data
model_path = data_dir / "logistic_regression_validation.joblib"

X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(data_dir / "y_validation.csv").squeeze("columns")

model = joblib.load(model_path)

print("Loaded baseline Logistic Regression model.")

# Predicted probability of sepsis for each hourly record
probabilities = model.predict_proba(X_val)[:, 1]

# Create an uncertainty analysis table
results = pd.DataFrame({
    "SepsisLabel": y_val,
    "SepsisProbability": probabilities,
})

# Distance from 0.5: smaller distance means closer to the
# default classification boundary, not necessarily calibrated uncertainty.
results["DistanceFromThreshold"] = (
    results["SepsisProbability"] - 0.5
).abs()

# Define an exploratory uncertainty band around 0.5
results["UncertaintyBand"] = (
    results["SepsisProbability"].between(0.4, 0.6)
)

print("\nValidation probability summary:")
print(results["SepsisProbability"].describe().round(4))

print("\nRecords inside probability band [0.4, 0.6]:")
print(int(results["UncertaintyBand"].sum()))

print(
    "Percentage inside band:",
    round(results["UncertaintyBand"].mean() * 100, 2),
    "%",
)

print("\nAverage predicted probability by true label:")
print(
    results.groupby("SepsisLabel")["SepsisProbability"]
    .mean()
    .round(4)
)

print("\nValidation PR-AUC:",
      round(average_precision_score(y_val, probabilities), 4))
print("Validation ROC-AUC:",
      round(roc_auc_score(y_val, probabilities), 4))

# Save results for further research analysis
output_path = data_dir / "uncertainty_analysis.csv"
results.to_csv(output_path, index=False)

print("\nAnalysis saved to:", output_path.resolve())
print("Final test set has not been used.")
