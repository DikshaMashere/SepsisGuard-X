
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

data_dir = Path("data/processed")

# Load original validation records and processed model inputs
raw_val = pd.read_csv(data_dir / "validation.csv")
X_val = pd.read_csv(data_dir / "X_validation.csv")
y_val = pd.read_csv(data_dir / "y_validation.csv").squeeze("columns")

# Check row alignment before evaluating
assert len(raw_val) == len(X_val) == len(y_val), (
    "Validation row counts do not match."
)
assert "Patient_ID" in raw_val.columns
assert "SepsisLabel" in raw_val.columns

# Load existing baseline and calibrated models
balanced_model = joblib.load(
    data_dir / "logistic_regression_validation.joblib"
)
calibrated_model = joblib.load(
    data_dir / "logistic_regression_calibrated.joblib"
)

# Predict hourly probabilities
balanced_prob = balanced_model.predict_proba(X_val)[:, 1]
calibrated_prob = calibrated_model.predict_proba(X_val)[:, 1]

# Create a row-aligned evaluation table
hourly = pd.DataFrame({
    "Patient_ID": raw_val["Patient_ID"].values,
    "SepsisLabel": y_val.values,
    "BalancedProbability": balanced_prob,
    "CalibratedProbability": calibrated_prob,
})

# Patient-level outcome: at least one positive SepsisLabel
# in the patient's validation records.
patient_labels = hourly.groupby("Patient_ID")["SepsisLabel"].max()

# Exploratory patient score: maximum hourly predicted probability
patient_scores = hourly.groupby("Patient_ID").agg(
    BalancedScore=("BalancedProbability", "max"),
    CalibratedScore=("CalibratedProbability", "max"),
)

patient_data = patient_scores.join(patient_labels).reset_index()

print("Validation patients:", len(patient_data))
print(
    "Patients with a positive label:",
    int(patient_data["SepsisLabel"].sum()),
)

# Evaluate ranking at patient level
for name, score_column in [
    ("Balanced Logistic Regression", "BalancedScore"),
    ("Sigmoid-calibrated Logistic Regression", "CalibratedScore"),
]:
    scores = patient_data[score_column]
    labels = patient_data["SepsisLabel"]

    print(f"\n{name}")
    print(
        "Patient-level PR-AUC:",
        round(average_precision_score(labels, scores), 4),
    )
    print(
        "Patient-level ROC-AUC:",
        round(roc_auc_score(labels, scores), 4),
    )

    # Exploratory threshold; not a clinically selected threshold
    predictions = (scores >= 0.5).astype(int)

    print(
        "Precision at 0.5:",
        round(precision_score(labels, predictions, zero_division=0), 4),
    )
    print(
        "Recall at 0.5:",
        round(recall_score(labels, predictions, zero_division=0), 4),
    )
    print(
        "F1 at 0.5:",
        round(f1_score(labels, predictions, zero_division=0), 4),
    )
    print("Confusion matrix:")
    print(confusion_matrix(labels, predictions, labels=[0, 1]))

# Save patient-level results for analysis
output_path = data_dir / "patient_level_validation_results.csv"
patient_data.to_csv(output_path, index=False)

print("\nPatient-level results saved to:", output_path.resolve())
print("Final test set has not been used.")
