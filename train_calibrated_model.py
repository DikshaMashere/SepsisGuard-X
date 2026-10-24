
import pandas as pd
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)

data_dir = Path("data/processed")

# Load separate patient-level groups
train_df = pd.read_csv(data_dir / "model_train.csv")
cal_df = pd.read_csv(data_dir / "calibration.csv")

from features import BASELINE_FEATURES, TARGET_COLUMN

X_train = train_df[BASELINE_FEATURES].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_cal = cal_df[BASELINE_FEATURES].copy()
y_cal = cal_df[TARGET_COLUMN].copy()

# Train imputation and unweighted Logistic Regression
# using model-training patients only.
base_model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight=None,
        random_state=42,
    )),
])

print("Training unweighted baseline on model-training patients...")
base_model.fit(X_train, y_train)

# Freeze the fitted model so calibration data is used
# only to learn the sigmoid calibration mapping.
calibrated_model = CalibratedClassifierCV(
    estimator=FrozenEstimator(base_model),
    method="sigmoid",
)

print("Fitting sigmoid calibrator on calibration patients...")
calibrated_model.fit(X_cal, y_cal)

# Save the calibrated model
model_path = data_dir / "logistic_regression_calibrated.joblib"
joblib.dump(calibrated_model, model_path)

# Evaluate calibration data only as a basic sanity check
cal_prob = calibrated_model.predict_proba(X_cal)[:, 1]

print("\nCalibration-set results (not independent evaluation):")
print(
    "Brier score:",
    round(brier_score_loss(y_cal, cal_prob), 5),
)
print(
    "Mean predicted probability:",
    round(cal_prob.mean(), 5),
)
print(
    "Observed positive fraction:",
    round(y_cal.mean(), 5),
)

print("\nCalibrated model saved to:", model_path.resolve())
print("Existing validation and final test sets were not used.")
