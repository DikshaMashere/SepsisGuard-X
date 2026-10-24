
import pandas as pd
from sklearn.impute import SimpleImputer

from features import BASELINE_FEATURES, TARGET_COLUMN, GROUP_COLUMN

# Load the saved train and test datasets
train_df = pd.read_csv("data/processed/train.csv")
test_df = pd.read_csv("data/processed/test.csv")

# Separate features and labels
X_train = train_df[BASELINE_FEATURES].copy()
y_train = train_df[TARGET_COLUMN].copy()

X_test = test_df[BASELINE_FEATURES].copy()
y_test = test_df[TARGET_COLUMN].copy()

# Fit only on training data
imputer = SimpleImputer(strategy="median")
X_train_processed = pd.DataFrame(
    imputer.fit_transform(X_train),
    columns=BASELINE_FEATURES,
    index=X_train.index,
)

# Apply the same fitted imputer to test data
X_test_processed = pd.DataFrame(
    imputer.transform(X_test),
    columns=BASELINE_FEATURES,
    index=X_test.index,
)

# Verify results
print("Training features:", X_train_processed.shape)
print("Testing features:", X_test_processed.shape)

print("\nMissing values in processed training features:",
      int(X_train_processed.isnull().sum().sum()))

print("Missing values in processed testing features:",
      int(X_test_processed.isnull().sum().sum()))

print("\nTraining labels:", y_train.shape)
print("Testing labels:", y_test.shape)

# Verify patient-level separation
train_patient_ids = set(train_df[GROUP_COLUMN])
test_patient_ids = set(test_df[GROUP_COLUMN])

print("\nPatient overlap:", len(train_patient_ids & test_patient_ids))

print("\nPreprocessing verification complete.")
print("No datasets have been overwritten.")

from pathlib import Path
import joblib

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

# Save processed features and labels
X_train_processed.to_csv(
    output_dir / "X_train.csv", index=False
)
X_test_processed.to_csv(
    output_dir / "X_test.csv", index=False
)

y_train.to_csv(output_dir / "y_train.csv", index=False)
y_test.to_csv(output_dir / "y_test.csv", index=False)

# Save fitted imputer for reuse
joblib.dump(imputer, output_dir / "median_imputer.joblib")

print("\nProcessed files saved successfully!")
print("X_train.csv:", (output_dir / "X_train.csv").resolve())
print("X_test.csv:", (output_dir / "X_test.csv").resolve())
print("y_train.csv:", (output_dir / "y_train.csv").resolve())
print("y_test.csv:", (output_dir / "y_test.csv").resolve())
print("median_imputer.joblib saved.")
