
import pandas as pd
import joblib

from pathlib import Path
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    classification_report,
    average_precision_score,
    roc_auc_score,
    confusion_matrix,
)

# Load processed data
data_dir = Path("data/processed")

X_train = pd.read_csv(data_dir / "X_train.csv")
X_test = pd.read_csv(data_dir / "X_test.csv")

y_train = pd.read_csv(data_dir / "y_train.csv").squeeze("columns")
y_test = pd.read_csv(data_dir / "y_test.csv").squeeze("columns")

# Create baseline model
model = LogisticRegression(
    max_iter=1000,
    class_weight="balanced",
    random_state=42,
)

print("Training Logistic Regression...")
model.fit(X_train, y_train)

# Generate predictions
y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

# Evaluate model
print("\nClassification report:")
print(classification_report(y_test, y_pred, zero_division=0))

print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))

print("\nPR-AUC (Average Precision):",
      round(average_precision_score(y_test, y_prob), 4))

print("ROC-AUC:",
      round(roc_auc_score(y_test, y_prob), 4))

# Save trained model
model_path = data_dir / "logistic_regression.joblib"
joblib.dump(model, model_path)

print("\nModel saved to:", model_path.resolve())
