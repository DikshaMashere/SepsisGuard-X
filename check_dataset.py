
import pandas as pd
from sklearn.model_selection import train_test_split

# Load cleaned dataset
df = pd.read_csv("data/processed/sepsis_clean.csv")

# One row per patient for splitting
patient_summary = df.groupby("Patient_ID")["SepsisLabel"].max()

# Stratified patient-level split: 80% train, 20% test
train_patients, test_patients = train_test_split(
    patient_summary.index,
    test_size=0.20,
    random_state=42,
    stratify=patient_summary.values
)

# Assign all rows of each patient to only one set
train_df = df[df["Patient_ID"].isin(train_patients)].copy()
test_df = df[df["Patient_ID"].isin(test_patients)].copy()

# Verify no patient overlap
overlap = set(train_df["Patient_ID"]) & set(test_df["Patient_ID"])

print("Training patients:", train_df["Patient_ID"].nunique())
print("Testing patients:", test_df["Patient_ID"].nunique())

print("\nTraining rows:", len(train_df))
print("Testing rows:", len(test_df))

print("\nTraining sepsis-positive patients:",
      patient_summary.loc[train_patients].sum())

print("Testing sepsis-positive patients:",
      patient_summary.loc[test_patients].sum())

print("\nPatient overlap:", len(overlap))
print("No patient leakage:", len(overlap) == 0)

from pathlib import Path

output_dir = Path("data/processed")
output_dir.mkdir(parents=True, exist_ok=True)

train_df.to_csv(output_dir / "train.csv", index=False)
test_df.to_csv(output_dir / "test.csv", index=False)

print("\nTrain and test datasets saved successfully!")
print("Training file:", (output_dir / "train.csv").resolve())
print("Testing file:", (output_dir / "test.csv").resolve())


import pandas as pd

train_df = pd.read_csv("data/processed/train.csv")

print("Training dataset shape:", train_df.shape)

print("\nMissing values in each column:")
missing = train_df.isnull().sum()
missing_percent = (missing / len(train_df) * 100).round(2)

missing_report = pd.DataFrame({
    "Missing_Count": missing,
    "Missing_Percent": missing_percent
})

print(
    missing_report[missing_report["Missing_Count"] > 0]
    .sort_values("Missing_Percent", ascending=False)
)

print("\nTotal missing cells:", int(missing.sum()))


import pandas as pd

train_df = pd.read_csv("data/processed/train.csv")

# Columns we want to investigate first
candidate_features = [
    "Hour",
    "HR",
    "O2Sat",
    "Temp",
    "SBP",
    "MAP",
    "DBP",
    "Resp",
    "Age",
    "Gender",
    "HospAdmTime",
    "ICULOS",
]

report = pd.DataFrame({
    "Missing_Percent": (
        train_df[candidate_features].isnull().mean() * 100
    ).round(2),
    "Unique_Values": train_df[candidate_features].nunique(),
})

print("\nCandidate feature analysis:")
print(report.sort_values("Missing_Percent"))

print("\nTarget distribution:")
print(train_df["SepsisLabel"].value_counts())

print("\nPatient ID will be used for grouping, not as a model feature.")
print("Target: SepsisLabel")
