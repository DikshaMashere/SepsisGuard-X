
import pandas as pd
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer

from features import BASELINE_FEATURES, TARGET_COLUMN

data_dir = Path("data/processed")

# Load training-core and validation data
train_df = pd.read_csv(data_dir / "train_core.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

# Separate features and labels
X_train = train_df[BASELINE_FEATURES].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_val = val_df[BASELINE_FEATURES].copy()
y_val = val_df[TARGET_COLUMN].copy()

# Fit imputer ONLY on training-core data
imputer = SimpleImputer(strategy="median")

X_train_processed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=BASELINE_FEATURES,
    index=X_train.index,
)

# Transform validation with the same imputer
X_val_processed = pd.DataFrame(
    imputer.transform(X_val),
    columns=BASELINE_FEATURES,
    index=X_val.index,
)

# Save processed data
X_train_processed.to_csv(data_dir / "X_train_core.csv", index=False)
X_val_processed.to_csv(data_dir / "X_validation.csv", index=False)

y_train.to_csv(data_dir / "y_train_core.csv", index=False)
y_val.to_csv(data_dir / "y_validation.csv", index=False)

joblib.dump(imputer, data_dir / "validation_imputer.joblib")

print("Training features:", X_train_processed.shape)
print("Validation features:", X_val_processed.shape)

print("Training missing values:",
      int(X_train_processed.isnull().sum().sum()))
print("Validation missing values:",
      int(X_val_processed.isnull().sum().sum()))

print("Validation preprocessing completed.")
