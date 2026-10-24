
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.metrics import confusion_matrix

data_dir = Path("data/processed")

# Load bootstrap uncertainty results
df = pd.read_csv(
    data_dir / "bootstrap_uncertainty_validation.csv"
)

# Use the same validation features and labels
X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

assert len(df) == len(X_val) == len(y_val), (
    "Validation row counts do not match."
)

# Mean probability across bootstrap models
probabilities = df["MeanProbability"].to_numpy()
labels = y_val.to_numpy()

# Exploratory classification threshold
threshold = 0.5
predictions = (probabilities >= threshold).astype(int)

df["Prediction"] = predictions
df["ActualLabel"] = labels
df["PredictionCorrect"] = (
    df["Prediction"] == df["ActualLabel"]
)

df["ErrorType"] = np.where(
    df["PredictionCorrect"],
    "Correct",
    "Incorrect",
)

# Compare uncertainty for correct and incorrect predictions
summary = df.groupby("ErrorType")["StdProbability"].agg(
    ["count", "mean", "median", "std"]
).reset_index()

print("\nUncertainty by prediction correctness")
print(summary.round(5).to_string(index=False))

# Compare uncertainty at the 90th percentile cutoff
cutoff = df["StdProbability"].quantile(0.90)
df["HighUncertainty"] = df["StdProbability"] >= cutoff

print(f"\n90th-percentile uncertainty cutoff: {cutoff:.5f}")

print("\nError rate by uncertainty group")
error_summary = df.groupby("HighUncertainty")[
    "PredictionCorrect"
].apply(lambda x: 1 - x.mean()).reset_index(name="ErrorRate")

print(error_summary.to_string(index=False))

# Save row-level analysis and summaries
df.to_csv(
    data_dir / "uncertainty_error_analysis.csv",
    index=False,
)

summary.to_csv(
    data_dir / "uncertainty_correct_incorrect_summary.csv",
    index=False,
)

print("\nConfusion matrix at threshold 0.5:")
print(confusion_matrix(labels, predictions, labels=[0, 1]))

print("\nResults saved to data/processed/")
print("Final test set has not been used.")
