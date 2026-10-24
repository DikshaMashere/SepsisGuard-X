
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

data_dir = Path("data/processed")

# Use only the existing training-core dataset
df = pd.read_csv(data_dir / "train_core.csv")

# One summary row per patient
patient_summary = df.groupby("Patient_ID")["SepsisLabel"].max()

# Reserve 20% of training-core patients for calibration
model_patients, calibration_patients = train_test_split(
    patient_summary.index,
    test_size=0.20,
    random_state=42,
    stratify=patient_summary.values,
)

# Keep all hourly records from each patient together
model_train = df[df["Patient_ID"].isin(model_patients)].copy()
calibration = df[
    df["Patient_ID"].isin(calibration_patients)
].copy()

# Verify no patient overlap
overlap = (
    set(model_train["Patient_ID"])
    & set(calibration["Patient_ID"])
)

# Save the raw splits; do not overwrite existing train_core.csv
model_train.to_csv(
    data_dir / "model_train.csv", index=False
)
calibration.to_csv(
    data_dir / "calibration.csv", index=False
)

print("Model-training patients:", model_train["Patient_ID"].nunique())
print("Calibration patients:", calibration["Patient_ID"].nunique())

print("Model-training rows:", len(model_train))
print("Calibration rows:", len(calibration))

print(
    "Model-training sepsis-positive patients:",
    patient_summary.loc[model_patients].sum(),
)
print(
    "Calibration sepsis-positive patients:",
    patient_summary.loc[calibration_patients].sum(),
)

print("Patient overlap:", len(overlap))
print("No patient leakage:", len(overlap) == 0)
print("Calibration split saved successfully.")
