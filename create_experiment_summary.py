
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

# Load saved experiment reports
model_comparison = pd.read_csv(
    data_dir / "weighted_unweighted_comparison.csv"
)

unweighted_thresholds = pd.read_csv(
    data_dir / "unweighted_threshold_comparison.csv"
)

calibrated_thresholds = pd.read_csv(
    data_dir / "calibrated_threshold_comparison.csv"
)

# Summary of model-level validation metrics
summary_rows = []

for _, row in model_comparison.iterrows():
    summary_rows.append({
        "Experiment": "Weighted vs Unweighted",
        "Model": row["Model"],
        "Threshold": 0.5,
        "PR_AUC": row["PR_AUC"],
        "ROC_AUC": row["ROC_AUC"],
        "Brier_Score": row["Brier_Score"],
        "Precision": row["Precision_at_0.5"],
        "Recall": row["Recall_at_0.5"],
        "F1": row["F1_at_0.5"],
    })

summary = pd.DataFrame(summary_rows)

# Save threshold results separately in one long-format file
unweighted_thresholds["Model"] = "Unweighted Logistic Regression"
calibrated_thresholds["Model"] = "Sigmoid-calibrated Logistic Regression"

threshold_summary = pd.concat(
    [unweighted_thresholds, calibrated_thresholds],
    ignore_index=True,
)

summary.to_csv(
    data_dir / "experiment_model_summary.csv",
    index=False,
)

threshold_summary.to_csv(
    data_dir / "experiment_threshold_summary.csv",
    index=False,
)

print("Model summary:")
print(summary.round(4).to_string(index=False))

print("\nThreshold summary saved.")
print("Model summary saved.")
print("Final test set has not been used.")
