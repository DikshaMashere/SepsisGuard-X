
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_RESULTS_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "saved_bootstrap_validation_results_unweighted.csv"
)


def load_validation_uncertainty_summary() -> dict:
    """
    Summarize previously computed validation-set uncertainty.
    This is research analysis, not live-patient uncertainty.
    """
    if not UNCERTAINTY_RESULTS_PATH.is_file():
        raise FileNotFoundError(
            f"Validation uncertainty results not found: "
            f"{UNCERTAINTY_RESULTS_PATH}"
        )

    results = pd.read_csv(UNCERTAINTY_RESULTS_PATH)

    required_columns = {
        "Patient_ID",
        "SepsisLabel",
        "MeanProbability",
        "StdProbability",
        "LowerPercentile",
        "UpperPercentile",
    }

    missing = required_columns - set(results.columns)
    if missing:
        raise ValueError(
            f"Uncertainty results are missing columns: {sorted(missing)}"
        )

    uncertainty = results["StdProbability"].dropna()

    if uncertainty.empty:
        raise ValueError("No valid uncertainty values were found.")

    return {
        "source": "held-out validation analysis",
        "record_count": int(len(results)),
        "patient_count": int(results["Patient_ID"].nunique()),
        "mean_prediction_std": float(uncertainty.mean()),
        "median_prediction_std": float(uncertainty.median()),
        "mean_interval_width": float(
            (
                results["UpperPercentile"]
                - results["LowerPercentile"]
            ).mean()
        ),
        "warning": (
            "Historical validation analysis only. "
            "These values are not uncertainty estimates for a new prediction."
        ),
    }

