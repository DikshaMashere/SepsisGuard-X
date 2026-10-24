
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import BASELINE_FEATURES, TARGET_COLUMN


PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data" / "processed"
OUTPUT_DIR = DATA_DIR / "bootstrap_models_unweighted"

N_BOOTSTRAP = 20
RANDOM_SEED = 42


def main():
    train_df = pd.read_csv(DATA_DIR / "train_core.csv")

    required = {
        "Patient_ID",
        TARGET_COLUMN,
        *BASELINE_FEATURES,
    }
    missing = required - set(train_df.columns)

    if missing:
        raise ValueError(f"Training data missing columns: {sorted(missing)}")

    patient_ids = train_df["Patient_ID"].dropna().unique()

    if len(patient_ids) < 2:
        raise ValueError("Insufficient unique patients for bootstrap training.")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(RANDOM_SEED)

    print("Unique training patients:", len(patient_ids))
    print("Bootstrap models:", N_BOOTSTRAP)
    print("Output folder:", OUTPUT_DIR)

    for iteration in range(N_BOOTSTRAP):
        sampled_ids = rng.choice(
            patient_ids,
            size=len(patient_ids),
            replace=True,
        )

        # Keep repeated draws as repeated patient contributions.
        patient_groups = {
            patient_id: train_df.loc[
                train_df["Patient_ID"] == patient_id
            ]
            for patient_id in np.unique(sampled_ids)
        }

        bootstrap_parts = []

        for draw_id, patient_id in enumerate(sampled_ids):
            part = patient_groups[patient_id].copy()
            part["_bootstrap_draw"] = draw_id
            bootstrap_parts.append(part)

        bootstrap_df = pd.concat(
            bootstrap_parts,
            ignore_index=True,
        )

        X_boot = bootstrap_df[BASELINE_FEATURES]
        y_boot = bootstrap_df[TARGET_COLUMN]

        if y_boot.nunique() < 2:
            raise ValueError(
                f"Bootstrap iteration {iteration + 1} has only one class."
            )

        model = Pipeline([
            ("imputer", SimpleImputer(strategy="median")),
            ("classifier", LogisticRegression(
                max_iter=1000,
                class_weight=None,
                random_state=RANDOM_SEED + iteration,
            )),
        ])

        model.fit(X_boot, y_boot)

        model_path = OUTPUT_DIR / f"bootstrap_model_{iteration + 1:02d}.joblib"
        joblib.dump(model, model_path)

        print(
            f"Saved model {iteration + 1}/{N_BOOTSTRAP}: "
            f"{model_path.name}"
        )

    metadata = {
        "n_bootstrap": N_BOOTSTRAP,
        "random_seed": RANDOM_SEED,
        "features": BASELINE_FEATURES,
        "target": TARGET_COLUMN,
        "sampling_unit": "Patient_ID",
        "training_source": "train_core.csv",
        "class_weight": None,
        "purpose": (
            "Exploratory bootstrap ensemble; requires separate validation "
            "before use for live prediction."
        ),
    }

    joblib.dump(metadata, OUTPUT_DIR / "metadata.joblib")

    print("\nBootstrap training complete.")
    print("Models saved:", N_BOOTSTRAP)
    print("Final test set was not loaded or used.")


if __name__ == "__main__":
    main()

