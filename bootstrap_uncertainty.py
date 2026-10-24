
import pandas as pd
import numpy as np
import joblib

from pathlib import Path
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from features import BASELINE_FEATURES, TARGET_COLUMN

data_dir = Path("data/processed")

# Load training-core and validation data only
train_df = pd.read_csv(data_dir / "train_core.csv")
val_df = pd.read_csv(data_dir / "validation.csv")

X_val = pd.read_csv(data_dir / "X_validation.csv")

# Confirm row alignment before predictions
assert len(val_df) == len(X_val), (
    "Raw validation and processed validation rows do not match."
)

# Use training-core patient IDs as bootstrap sampling units
patient_ids = train_df["Patient_ID"].drop_duplicates().to_numpy()

rng = np.random.default_rng(42)
n_bootstrap = 20

# Store each model's prediction for every validation record
all_probabilities = []

print("Training patient-level bootstrap models...")
print("Unique training patients:", len(patient_ids))
print("Bootstrap iterations:", n_bootstrap)

for iteration in range(n_bootstrap):
    # Sample patients with replacement
    sampled_patients = rng.choice(
        patient_ids,
        size=len(patient_ids),
        replace=True,
    )

    # Repeated patients represent repeated bootstrap draws.
    # Assign a unique temporary group ID to each draw.
    bootstrap_parts = []

    for draw_id, patient_id in enumerate(sampled_patients):
        patient_rows = train_df.loc[
            train_df["Patient_ID"] == patient_id
        ].copy()

        patient_rows["_bootstrap_group"] = draw_id
        bootstrap_parts.append(patient_rows)

    bootstrap_df = pd.concat(
        bootstrap_parts,
        ignore_index=True,
    )

    X_boot = bootstrap_df[BASELINE_FEATURES]
    y_boot = bootstrap_df[TARGET_COLUMN]

    # Fit preprocessing and model on this bootstrap sample only
    model = Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("classifier", LogisticRegression(
            max_iter=1000,
            class_weight="balanced",
            random_state=42,
        )),
    ])

    model.fit(X_boot, y_boot)

    # Same validation records for every bootstrap model
    probabilities = model.predict_proba(X_val)[:, 1]
    all_probabilities.append(probabilities)

    print(f"Completed bootstrap model {iteration + 1}/{n_bootstrap}")

# Shape: (number of bootstrap models, validation records)
probability_matrix = np.vstack(all_probabilities)

# Record-level prediction summaries
results = pd.DataFrame({
    "Patient_ID": val_df["Patient_ID"].to_numpy(),
    "SepsisLabel": val_df[TARGET_COLUMN].to_numpy(),
    "MeanProbability": probability_matrix.mean(axis=0),
    "StdProbability": probability_matrix.std(axis=0, ddof=1),
    "LowerPercentile": np.percentile(
        probability_matrix, 2.5, axis=0
    ),
    "UpperPercentile": np.percentile(
        probability_matrix, 97.5, axis=0
    ),
})

output_path = data_dir / "bootstrap_uncertainty_validation.csv"
results.to_csv(output_path, index=False)

print("\nBootstrap uncertainty summary")
print("Validation records:", len(results))
print(
    "Mean prediction standard deviation:",
    round(results["StdProbability"].mean(), 5),
)
print(
    "Median prediction standard deviation:",
    round(results["StdProbability"].median(), 5),
)
print(
    "Mean interval width:",
    round(
        (results["UpperPercentile"] - results["LowerPercentile"]).mean(),
        5,
    ),
)

print("\nResults saved to:", output_path.resolve())
print("Final test set has not been used.")
