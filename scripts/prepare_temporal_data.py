from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "temporal"

HORIZON_HOURS = 6

FEATURES = [
    "Hour", "HR", "O2Sat", "SBP", "MAP",
    "Resp", "Age", "Gender", "HospAdmTime", "ICULOS",
]


def prepare_temporal_dataset(input_path: Path) -> pd.DataFrame:
    df = pd.read_csv(input_path)

    required = set(FEATURES + ["Patient_ID", "SepsisLabel"])
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    df = df.sort_values(
        ["Patient_ID", "Hour", "ICULOS"], kind="stable"
    ).reset_index(drop=True)

    if df.duplicated(["Patient_ID", "Hour"]).any():
        raise ValueError("Duplicate patient-hour observations found")

    df["SepsisLabel"] = pd.to_numeric(df["SepsisLabel"], errors="raise")
    if not df["SepsisLabel"].isin([0, 1]).all():
        raise ValueError("SepsisLabel must contain only 0 and 1")

    groups = df.groupby("Patient_ID", sort=False)
    first_positive_hour = (
        df["Hour"].where(df["SepsisLabel"].eq(1))
        .groupby(df["Patient_ID"]).transform("min")
    )
    last_hour = groups["Hour"].transform("max")

    # Predict only while the patient has not yet become positive.
    at_risk = first_positive_hour.isna() | (
        df["Hour"] < first_positive_hour
    )

    # Require a full six-hour follow-up window.
    eligible = at_risk & (last_hour >= df["Hour"] + HORIZON_HOURS)

    # Hourly continuity has been checked for model_train.csv.
    # For each eligible row, identify onset strictly after the current hour
    # and within the next six hours, inclusive.
    target = (
        first_positive_hour.notna()
        & (first_positive_hour > df["Hour"])
        & (first_positive_hour <= df["Hour"] + HORIZON_HOURS)
    ).astype("int8")

    output = df.loc[eligible, FEATURES + ["Patient_ID"]].copy()
    output["Target_Sepsis_Within_6h"] = target.loc[eligible].to_numpy()

    return output.reset_index(drop=True)


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    input_files = {
        "train": DATA_DIR / "model_train.csv",
        "calibration": DATA_DIR / "calibration.csv",
        "validation": DATA_DIR / "validation.csv",
        "test": DATA_DIR / "test.csv",
    }

    for split_name, input_path in input_files.items():
        if not input_path.is_file():
            raise FileNotFoundError(f"Required split not found: {input_path}")

        temporal_df = prepare_temporal_dataset(input_path)
        output_path = OUTPUT_DIR / f"{split_name}.csv"
        temporal_df.to_csv(output_path, index=False)

        print(f"\n{split_name.upper()}")
        print(f"Saved: {output_path}")
        print(f"Rows: {len(temporal_df):,}")
        print(f"Patients: {temporal_df['Patient_ID'].nunique():,}")
        print(f"Positive targets: {int(temporal_df['Target_Sepsis_Within_6h'].sum()):,}")
        print(f"Target prevalence: {temporal_df['Target_Sepsis_Within_6h'].mean() * 100:.3f}%")
        print("Missing feature values:", int(temporal_df[FEATURES].isna().sum().sum()))


if __name__ == "__main__":
    main()
