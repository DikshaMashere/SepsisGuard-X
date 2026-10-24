
import pandas as pd
from pathlib import Path

# Project paths
data_dir = Path("data/processed")

train_path = data_dir / "train_core.csv"
validation_path = data_dir / "validation.csv"

# Load raw training and validation data
train_df = pd.read_csv(train_path)
validation_df = pd.read_csv(validation_path)

# Initial baseline features
from features import BASELINE_FEATURES

print("\nCreating missingness indicators...")

# Copy feature columns and create indicators BEFORE imputation.
X_train = train_df[BASELINE_FEATURES].copy()
X_validation = validation_df[BASELINE_FEATURES].copy()

# Indicator = 1 if the original measurement is missing, otherwise 0.
X_train_missing = X_train.isna().astype("int8")
X_validation_missing = X_validation.isna().astype("int8")

X_train_missing.columns = [
    f"{column}_missing" for column in X_train_missing.columns
]
X_validation_missing.columns = [
    f"{column}_missing" for column in X_validation_missing.columns
]

# Keep indicators only for features that actually have missing values
# in the training data. This decision is based on training data alone.
missing_columns = X_train.columns[X_train.isna().any()].tolist()

X_train_missing = X_train_missing[
    [f"{column}_missing" for column in missing_columns]
]
X_validation_missing = X_validation_missing[
    [f"{column}_missing" for column in missing_columns]
]

# Save indicators separately; existing model inputs remain unchanged.
X_train_missing.to_csv(
    data_dir / "X_train_missingness.csv", index=False
)
X_validation_missing.to_csv(
    data_dir / "X_validation_missingness.csv", index=False
)

print("\nFeatures with missing values in training data:")
print(missing_columns)

print("\nTraining missingness indicators shape:", X_train_missing.shape)
print("Validation missingness indicators shape:", X_validation_missing.shape)

print("\nMissing values per indicator in training data:")
print(X_train_missing.sum().sort_values(ascending=False))

print("\nMissingness indicator files saved successfully.")
