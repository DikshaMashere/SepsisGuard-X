
import pandas as pd
import joblib

from pathlib import Path
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    average_precision_score,
    roc_auc_score,
)

data_dir = Path("data/processed")

X_train = pd.read_csv(data_dir / "X_train_core.csv")
X_val = pd.read_csv(data_dir / "X_validation.csv")

y_train = pd.read_csv(data_dir / "y_train_core.csv").squeeze("columns")
y_val = pd.read_csv(data_dir / "y_validation.csv").squeeze("columns")

# Scaling is fitted on training data only
model = Pipeline([
    ("scaler", StandardScaler()),
    ("classifier", LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        random_state=42,
    )),
])

print("Training scaled Logistic Regression...")
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
y_prob = model.predict_proba(X_val)[:, 1]

print("\nValidation classification report:")
print(classification_report(y_val, y_pred, zero_division=0))

print("Validation confusion matrix:")
print(confusion_matrix(y_val, y_pred))

print(
    "\nValidation PR-AUC:",
    round(average_precision_score(y_val, y_prob), 4),
)

print(
    "Validation ROC-AUC:",
    round(roc_auc_score(y_val, y_prob), 4),
)

model_path = data_dir / "logistic_regression_scaled.joblib"
joblib.dump(model, model_path)

print("\nScaled model saved to:", model_path.resolve())
