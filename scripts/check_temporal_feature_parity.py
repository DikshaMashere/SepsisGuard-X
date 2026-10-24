
from pathlib import Path

import numpy as np
import pandas as pd

from app.services.temporal_model_service import (
    load_temporal_model,
    build_latest_features,
)
from scripts.train_temporal_model import make_history_features


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TEMPORAL = PROCESSED / "temporal"

BASE_FEATURES = [
    "HR", "O2Sat", "SBP", "MAP", "Resp",
    "Age", "Gender", "HospAdmTime", "ICULOS",
]


def main():
    original = pd.read_csv(PROCESSED / "model_train.csv")
    targets = pd.read_csv(TEMPORAL / "train.csv")

    training_features = make_history_features(original, targets)
    artifact = load_temporal_model()
    columns = artifact["feature_columns"]

    # Select an eligible row with six preceding hourly observations.
    candidates = training_features.sort_values(
        ["Patient_ID", "Hour"]
    )

    for _, row in candidates.iterrows():
        patient_id = row["Patient_ID"]
        hour = row["Hour"]

        history = original.loc[
            original["Patient_ID"].eq(patient_id)
            & original["Hour"].between(hour - 6, hour)
        ].sort_values("Hour")

        if len(history) != 7:
            continue

        if not np.allclose(history["Hour"].diff().dropna(), 1):
            continue

        observations = history[["Hour"] + BASE_FEATURES].to_dict(
            orient="records"
        )
        api_features = build_latest_features(observations, columns)

        train_row = training_features.loc[
            training_features["Patient_ID"].eq(patient_id)
            & training_features["Hour"].eq(hour)
        ]

        if len(train_row) != 1:
            continue

        expected = train_row.iloc[0][columns].astype(float)
        actual = api_features.iloc[0][columns].astype(float)

        matches = np.isclose(
            expected.to_numpy(),
            actual.to_numpy(),
            equal_nan=True,
            atol=1e-8,
        )

        mismatches = [
            (name, expected[name], actual[name])
            for name, match in zip(columns, matches)
            if not match
        ]

        print("Patient history selected: 7 consecutive hours")
        print("Feature count:", len(columns))
        print("Feature values match:", len(mismatches) == 0)

        if mismatches:
            print("\nFirst mismatches:")
            for name, expected_value, actual_value in mismatches[:15]:
                print(
                    f"{name}: training={expected_value}, "
                    f"API={actual_value}"
                )
            raise SystemExit(1)

        print("PASS: Training and API features match.")
        return

    raise RuntimeError("No suitable seven-hour patient history found.")


if __name__ == "__main__":
    main()
