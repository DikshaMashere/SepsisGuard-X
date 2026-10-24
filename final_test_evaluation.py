
import pandas as pd
import joblib

from pathlib import Path
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

from features import BASELINE_FEATURES, TARGET_COLUMN, GROUP_COLUMN


# ==================================================
# 1. FILE PATHS
# ==================================================

data_dir = Path("data/processed")
model_path = data_dir / "logistic_regression_calibrated.joblib"
test_path = data_dir / "test.csv"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

if not test_path.exists():
    raise FileNotFoundError(f"Test dataset not found: {test_path}")


# ==================================================
# 2. LOAD MODEL AND TEST DATA
# ==================================================

print("Loading calibrated model...")
model = joblib.load(model_path)

print("Loading untouched test dataset...")
test_df = pd.read_csv(test_path)

required_columns = BASELINE_FEATURES + [TARGET_COLUMN, GROUP_COLUMN]
missing_columns = sorted(set(required_columns) - set(test_df.columns))

if missing_columns:
    raise ValueError(f"Missing required columns: {missing_columns}")

X_test = test_df[BASELINE_FEATURES].copy()
y_test = test_df[TARGET_COLUMN].astype(int).copy()

if y_test.nunique() != 2:
    raise ValueError("Test target must contain both classes.")

print("\nTest rows:", len(test_df))
print("Test patients:", test_df[GROUP_COLUMN].nunique())
print("Positive hourly records:", int(y_test.sum()))
print("Missing feature values:", int(X_test.isna().sum().sum()))


# ==================================================
# 3. HOURLY-RECORD EVALUATION
# ==================================================

print("\nGenerating predictions using the saved model...")
# The saved pipeline's fitted imputer handles missing values.
# Do not fit a new imputer or retrain the model on test data.
probabilities = model.predict_proba(X_test)[:, 1]

if not pd.Series(probabilities).map(pd.notna).all():
    raise ValueError("Predictions contain non-finite values.")

hourly_metrics = {
    "Rows": len(y_test),
    "Positive_Records": int(y_test.sum()),
    "PR_AUC": average_precision_score(y_test, probabilities),
    "ROC_AUC": roc_auc_score(y_test, probabilities),
    "Brier_Score": brier_score_loss(y_test, probabilities),
    "Mean_Predicted_Probability": float(probabilities.mean()),
    "Observed_Positive_Fraction": float(y_test.mean()),
}

# Reference threshold examined previously on validation data.
# This is not claimed to be clinically optimal.
hourly_threshold = 0.03
hourly_predictions = (probabilities >= hourly_threshold).astype(int)

tn, fp, fn, tp = confusion_matrix(
    y_test,
    hourly_predictions,
    labels=[0, 1],
).ravel()

hourly_metrics.update({
    "Threshold": hourly_threshold,
    "Precision": precision_score(
        y_test, hourly_predictions, zero_division=0
    ),
    "Recall": recall_score(
        y_test, hourly_predictions, zero_division=0
    ),
    "F1": f1_score(
        y_test, hourly_predictions, zero_division=0
    ),
    "TN": int(tn),
    "FP": int(fp),
    "FN": int(fn),
    "TP": int(tp),
})

print("\n========== HOURLY-RECORD TEST METRICS ==========")
for name, value in hourly_metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.6f}")
    else:
        print(f"{name}: {value}")


# ==================================================
# 4. PATIENT-LEVEL EXPLORATORY EVALUATION
# ==================================================

# Patient outcome: any positive hourly SepsisLabel.
# Patient score: maximum probability across that patient's records.
# This is retrospective aggregation, not an early-warning-time metric.

patient_df = pd.DataFrame({
    GROUP_COLUMN: test_df[GROUP_COLUMN].values,
    "Actual_Label": y_test.values,
    "Predicted_Probability": probabilities,
})

patient_df = patient_df.groupby(GROUP_COLUMN).agg(
    Actual_Label=("Actual_Label", "max"),
    Predicted_Probability=("Predicted_Probability", "max"),
).reset_index()

patient_y = patient_df["Actual_Label"].astype(int)
patient_prob = patient_df["Predicted_Probability"]

patient_metrics = {
    "Patients": len(patient_df),
    "Positive_Patients": int(patient_y.sum()),
    "PR_AUC": average_precision_score(patient_y, patient_prob),
    "ROC_AUC": roc_auc_score(patient_y, patient_prob),
    "Brier_Score": brier_score_loss(patient_y, patient_prob),
}

# Separate reference threshold for patient-level aggregated scores.
# It is not the hourly threshold and is not claimed to be optimal.
patient_threshold = 0.50
patient_predictions = (patient_prob >= patient_threshold).astype(int)

ptn, pfp, pfn, ptp = confusion_matrix(
    patient_y,
    patient_predictions,
    labels=[0, 1],
).ravel()

patient_metrics.update({
    "Reference_Threshold": patient_threshold,
    "Precision": precision_score(
        patient_y, patient_predictions, zero_division=0
    ),
    "Recall": recall_score(
        patient_y, patient_predictions, zero_division=0
    ),
    "F1": f1_score(
        patient_y, patient_predictions, zero_division=0
    ),
    "TN": int(ptn),
    "FP": int(pfp),
    "FN": int(pfn),
    "TP": int(ptp),
})

print("\n========== PATIENT-LEVEL TEST METRICS ==========")
for name, value in patient_metrics.items():
    if isinstance(value, float):
        print(f"{name}: {value:.6f}")
    else:
        print(f"{name}: {value}")


# ==================================================
# 5. SAVE RESULTS
# ==================================================

hourly_output = data_dir / "final_test_hourly_metrics.csv"
patient_output = data_dir / "final_test_patient_metrics.csv"

pd.DataFrame([hourly_metrics]).to_csv(hourly_output, index=False)
pd.DataFrame([patient_metrics]).to_csv(patient_output, index=False)

print("\nHourly metrics saved to:", hourly_output.resolve())
print("Patient metrics saved to:", patient_output.resolve())

print("\nFinal evaluation complete.")
print("The saved model was not retrained.")
print("Thresholds were not selected using test results.")
print("Academic retrospective evaluation only; not clinical validation.")
