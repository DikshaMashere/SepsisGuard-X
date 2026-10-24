
import pandas as pd
from pathlib import Path
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
)

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "uncertainty_error_analysis.csv"
)

# Incorrect prediction = positive class for this meta-evaluation
y_error = (~df["PredictionCorrect"].astype(bool)).astype(int)
uncertainty = df["StdProbability"]

print("Uncertainty-Based Error Detection")
print("Records:", len(df))
print("Incorrect predictions:", int(y_error.sum()))
print("Correct predictions:", int((y_error == 0).sum()))

# Higher uncertainty should ideally rank incorrect predictions higher
error_roc_auc = roc_auc_score(y_error, uncertainty)
error_pr_auc = average_precision_score(y_error, uncertainty)

print(f"\nError-detection ROC-AUC: {error_roc_auc:.4f}")
print(f"Error-detection PR-AUC:  {error_pr_auc:.4f}")

# Compare high-uncertainty vs lower-uncertainty error rates
cutoff = uncertainty.quantile(0.90)
high = uncertainty >= cutoff

high_error_rate = y_error[high].mean()
low_error_rate = y_error[~high].mean()

print(f"\n90th-percentile cutoff: {cutoff:.5f}")
print(f"High-uncertainty error rate: {high_error_rate:.4f}")
print(f"Lower-uncertainty error rate: {low_error_rate:.4f}")

print("\nExploratory validation analysis only.")
print("Final test set has not been used.")
