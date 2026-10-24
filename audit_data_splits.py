
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

files = {
    "train": data_dir / "train.csv",
    "train_core": data_dir / "train_core.csv",
    "validation": data_dir / "validation.csv",
    "test": data_dir / "test.csv",
}

print("=" * 60)
print("SEPSISGUARD-X DATA SPLIT AUDIT")
print("=" * 60)

required_columns = {"Patient_ID", "SepsisLabel"}
frames = {}

for name, path in files.items():
    if not path.exists():
        print(f"\n{name}: FILE MISSING - {path}")
        continue

    df = pd.read_csv(path)
    frames[name] = df

    print(f"\n{name.upper()}")
    print("Rows:", len(df))
    print("Unique patients:", df["Patient_ID"].nunique())
    print("Required columns present:", required_columns.issubset(df.columns))
    print("Missing Patient_ID:", int(df["Patient_ID"].isna().sum()))
    print("Missing SepsisLabel:", int(df["SepsisLabel"].isna().sum()))
    print("Positive hourly labels:", int((df["SepsisLabel"] == 1).sum()))
    print(
        "Patients with a positive label:",
        int(df.groupby("Patient_ID")["SepsisLabel"].max().sum()),
    )

# Check patient overlap across train_core, validation and test.
print("\n" + "=" * 60)
print("PATIENT OVERLAP CHECK")
print("=" * 60)

independent_splits = ["train_core", "validation", "test"]

for i, name_a in enumerate(independent_splits):
    for name_b in independent_splits[i + 1:]:
        if name_a not in frames or name_b not in frames:
            continue

        patients_a = set(frames[name_a]["Patient_ID"].dropna())
        patients_b = set(frames[name_b]["Patient_ID"].dropna())
        overlap = patients_a.intersection(patients_b)

        print(f"{name_a} vs {name_b}: {len(overlap)} overlapping patients")

# Verify train_core + validation reconstruct train.csv by patient membership.
if all(name in frames for name in ["train", "train_core", "validation"]):
    train_patients = set(frames["train"]["Patient_ID"].dropna())
    core_patients = set(frames["train_core"]["Patient_ID"].dropna())
    val_patients = set(frames["validation"]["Patient_ID"].dropna())

    print("\n" + "=" * 60)
    print("TRAIN / VALIDATION RECONSTRUCTION")
    print("=" * 60)
    print("Train patients:", len(train_patients))
    print("Core + validation patient union:", len(core_patients | val_patients))
    print("Same patient set:", train_patients == (core_patients | val_patients))
    print("Core/validation overlap:", len(core_patients & val_patients))

print("\nAudit completed. No files were modified.")
print("No model was trained and the test set was not used for evaluation.")
