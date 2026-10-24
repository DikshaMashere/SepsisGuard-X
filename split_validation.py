
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("data/processed")

# Load the existing training set only
df = pd.read_csv(data_dir / "train.csv")

# One row per patient
patient_summary = df.groupby("Patient_ID")["SepsisLabel"].max()

# Split patients into model-training and validation groups
train_patients, val_patients = train_test_split(
    patient_summary.index,
    test_size=0.20,
    random_state=123,
    stratify=patient_summary.values,
)

# Keep all records from each patient together
train_core = df[df["Patient_ID"].isin(train_patients)].copy()
validation = df[df["Patient_ID"].isin(val_patients)].copy()

# Verify patient separation
overlap = (
    set(train_core["Patient_ID"])
    & set(validation["Patient_ID"])
)

train_core.to_csv(data_dir / "train_core.csv", index=False)
validation.to_csv(data_dir / "validation.csv", index=False)

print("Training patients:", train_core["Patient_ID"].nunique())
print("Validation patients:", validation["Patient_ID"].nunique())

print("Training rows:", len(train_core))
print("Validation rows:", len(validation))

print(
    "Training sepsis-positive patients:",
    patient_summary.loc[train_patients].sum(),
)
print(
    "Validation sepsis-positive patients:",
    patient_summary.loc[val_patients].sum(),
)

print("Patient overlap:", len(overlap))
print("No patient leakage:", len(overlap) == 0)
print("Validation split saved successfully.")
