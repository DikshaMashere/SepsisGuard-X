
from pathlib import Path

import joblib
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV
from sklearn.frozen import FrozenEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    brier_score_loss,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler


ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"
TEMPORAL = PROCESSED / "temporal"
OUT = PROCESSED / "temporal_model"
OUT.mkdir(parents=True, exist_ok=True)

BASE_FEATURES = [
    "HR", "O2Sat", "SBP", "MAP", "Resp",
    "Age", "Gender", "HospAdmTime", "ICULOS",
]
TARGET = "Target_Sepsis_Within_6h"
SPLITS = ["train", "calibration", "validation"]


def make_history_features(original, targets):
    original = original.sort_values(
        ["Patient_ID", "Hour"]
    ).copy()

    if original.duplicated(["Patient_ID", "Hour"]).any():
        raise ValueError("Duplicate patient-hour rows found.")

    gaps = original.groupby("Patient_ID")["Hour"].diff().dropna()
    if not gaps.eq(1).all():
        raise ValueError(
            "Non-hourly gap found. Fix continuity before training."
        )

    result = original[["Patient_ID", "Hour"]].copy()
    result["HourOfStay"] = original["Hour"]

    for feature in BASE_FEATURES:
        values = pd.to_numeric(
            original[feature], errors="coerce"
        )
        history = values.groupby(
            original["Patient_ID"], sort=False
        )

        result[f"{feature}_current"] = values

        for lag in (1, 2, 3, 6):
            result[f"{feature}_lag{lag}"] = history.shift(lag)

        result[f"{feature}_mean3"] = history.transform(
            lambda s: s.rolling(3, min_periods=1).mean()
        )
        result[f"{feature}_change3"] = (
            values - history.shift(3)
        )

    # Attach target labels only after computing history on the
    # complete original hourly timeline.
    labels = targets[
        ["Patient_ID", "Hour", TARGET]
    ].copy()

    result = result.merge(
        labels,
        on=["Patient_ID", "Hour"],
        how="inner",
        validate="one_to_one",
    )

    # Six preceding hourly observations must exist.
    history_count = (
        original.groupby("Patient_ID").cumcount()
    )
    result["_history_count"] = history_count.to_numpy()[
        result.index
    ] if result.index.equals(original.index) else -1

    # Derive the count safely from the original timeline.
    counts = original[["Patient_ID", "Hour"]].copy()
    counts["_history_count"] = (
        original.groupby("Patient_ID").cumcount().to_numpy()
    )
    result = result.drop(columns=["_history_count"], errors="ignore")
    result = result.merge(
        counts,
        on=["Patient_ID", "Hour"],
        how="left",
        validate="one_to_one",
    )
    result = result.loc[result["_history_count"] >= 6].drop(
        columns="_history_count"
    )

    return result.reset_index(drop=True)


def main():
    datasets = {}

    for split in SPLITS:
        original_path = PROCESSED / (
            "model_train.csv" if split == "train"
            else f"{split}.csv"
        )
        target_path = TEMPORAL / f"{split}.csv"

        original = pd.read_csv(original_path)
        targets = pd.read_csv(target_path)

        datasets[split] = make_history_features(
            original, targets
        )
        print(f"{split}: {len(datasets[split]):,} rows")

    train = datasets["train"]
    calibration = datasets["calibration"]
    validation = datasets["validation"]

    feature_cols = [
        c for c in train.columns
        if c not in {"Patient_ID", TARGET}
    ]

    X_train, y_train = train[feature_cols], train[TARGET]
    X_cal, y_cal = calibration[feature_cols], calibration[TARGET]
    X_val, y_val = validation[feature_cols], validation[TARGET]

    if y_train.nunique() != 2 or y_cal.nunique() != 2:
        raise ValueError(
            "Training and calibration splits must each contain "
            "both target classes."
        )

    base = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median", add_indicator=True),
        ),
        ("scaler", StandardScaler()),
        (
            "classifier",
            LogisticRegression(max_iter=1000, random_state=42),
        ),
    ])
    base.fit(X_train, y_train)

    calibrated = CalibratedClassifierCV(
        estimator=FrozenEstimator(base),
        method="sigmoid",
    )
    calibrated.fit(X_cal, y_cal)

    probabilities = calibrated.predict_proba(X_val)[:, 1]

    print("\nValidation results")
    print("Prevalence:", round(float(y_val.mean()), 5))
    print(
        "PR-AUC:",
        round(float(average_precision_score(y_val, probabilities)), 5),
    )
    print(
        "ROC-AUC:",
        round(float(roc_auc_score(y_val, probabilities)), 5),
    )
    print(
        "Brier score:",
        round(float(brier_score_loss(y_val, probabilities)), 5),
    )
    print("Mean predicted risk:", round(float(probabilities.mean()), 5))

    model_path = OUT / "temporal_logistic_regression_corrected.joblib"
    joblib.dump(
        {
            "model": calibrated,
            "feature_columns": feature_cols,
            "target": TARGET,
            "horizon_hours": 6,
            "calibration_method": "sigmoid",
            "training_note": (
                "Experimental six-hour prediction using original "
                "hourly history. Not clinically validated."
            ),
        },
        model_path,
    )
    print("\nSaved new model:", model_path)


if __name__ == "__main__":
    main()
