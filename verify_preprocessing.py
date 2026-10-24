
import pandas as pd
import joblib
from pathlib import Path

data_dir = Path("data/processed")

# Load saved processed files
X_train = pd.read_csv(data_dir / "X_train.csv")
X_test = pd.read_csv(data_dir / "X_test.csv")
y_train = pd.read_csv(data_dir / "y_train.csv")
y_test = pd.read_csv(data_dir / "y_test.csv")
imputer = joblib.load(data_dir / "median_imputer.joblib")

print("Saved files loaded successfully.")

print("\nTraining features:", X_train.shape)
print("Testing features:", X_test.shape)
print("Training labels:", y_train.shape)
print("Testing labels:", y_test.shape)

print("\nMissing values in X_train:", X_train.isnull().sum().sum())
print("Missing values in X_test:", X_test.isnull().sum().sum())

print("\nFeature columns match:",
      list(X_train.columns) == list(X_test.columns))

print("Training and label row counts match:",
      len(X_train) == len(y_train))

print("Testing and label row counts match:",
      len(X_test) == len(y_test))

print("Saved imputer type:", type(imputer).__name__)
