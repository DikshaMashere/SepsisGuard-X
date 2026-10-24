
import pandas as pd
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
)

# Project paths
data_dir = Path("data/processed")

# Load original training and validation datasets
X_train = pd.read_csv(data_dir / "train_core.csv")
X_validation = pd.read_csv(data_dir / "validation.csv")

from features import BASELINE_FEATURES

# Select baseline features
X_train = X_train[BASELINE_FEATURES].copy()
X_validation = X_validation[BASELINE_FEATURES].copy()

# Load missingness indicators created before imputation
M_train = pd.read_csv(data_dir / "X_train_missingness.csv")
M_validation = pd.read_csv(data_dir / "X_validation_missingness.csv")

# Combine original features and missingness indicators
X_train_enhanced = pd.concat(
    [X_train.reset_index(drop=True), M_train.reset_index(drop=True)],
    axis=1,
)

X_validation_enhanced = pd.concat(
    [X_validation.reset_index(drop=True), M_validation.reset_index(drop=True)],
    axis=1,
)

# Load target labels
y_train = pd.read_csv(
    data_dir / "y_train_core.csv"
).squeeze("columns")

y_validation = pd.read_csv(
    data_dir / "y_validation.csv"
).squeeze("columns")

# Verify dataset dimensions
assert len(X_train_enhanced) == len(y_train), (
    "Training feature and label row counts do not match."
)
assert len(X_validation_enhanced) == len(y_validation), (
    "Validation feature and label row counts do not match."
)
assert list(X_train_enhanced.columns) == list(
    X_validation_enhanced.columns
), "Training and validation feature columns do not match."

print("Training enhanced Logistic Regression...")
print("Training rows:", len(X_train_enhanced))
print("Validation rows:", len(X_validation_enhanced))
print("Total features:", X_train_enhanced.shape[1])

# Imputation and scaling are fitted on training data only
model = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=3000,
        class_weight="balanced",
        random_state=42,
    )),
])

# Train model
model.fit(X_train_enhanced, y_train)

# Predict on validation data
probabilities = model.predict_proba(X_validation_enhanced)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

# Evaluate validation performance
print("\nEnhanced Model — Validation Results")
print(f"PR-AUC:    {average_precision_score(y_validation, probabilities):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_validation, probabilities):.4f}")
print(f"Precision: {precision_score(y_validation, predictions, zero_division=0):.4f}")
print(f"Recall:    {recall_score(y_validation, predictions, zero_division=0):.4f}")
print(f"F1-score:  {f1_score(y_validation, predictions, zero_division=0):.4f}")

print("\nConfusion Matrix:")
print(confusion_matrix(y_validation, predictions))

# Save trained model
model_path = data_dir / "logistic_regression_missingness.joblib"
joblib.dump(model, model_path)

print("\nEnhanced model saved successfully:")
print(model_path.resolve())
print("\nFinal test data has not been used.")
