import json
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
UNCERTAINTY_SUMMARY_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "validation_uncertainty_summary.json"
)


def load_validation_uncertainty_summary() -> dict:
    """
    Load aggregate statistics from previously computed held-out
    validation uncertainty analysis. This is research analysis,
    not live-patient uncertainty.
    """
    if not UNCERTAINTY_SUMMARY_PATH.is_file():
        raise FileNotFoundError(
            f"Validation uncertainty summary not found: "
            f"{UNCERTAINTY_SUMMARY_PATH}"
        )

    with UNCERTAINTY_SUMMARY_PATH.open("r", encoding="utf-8-sig") as file:
        summary = json.load(file)

    required_fields = {
        "record_count",
        "patient_count",
        "mean_prediction_std",
        "median_prediction_std",
        "mean_interval_width",
    }

    missing = required_fields - set(summary)
    if missing:
        raise ValueError(
            f"Uncertainty summary is missing fields: {sorted(missing)}"
        )

    return {
        "source": "held-out validation analysis",
        "record_count": int(summary["record_count"]),
        "patient_count": int(summary["patient_count"]),
        "mean_prediction_std": float(summary["mean_prediction_std"]),
        "median_prediction_std": float(summary["median_prediction_std"]),
        "mean_interval_width": float(summary["mean_interval_width"]),
        "warning": (
            "Historical validation analysis only. "
            "These values are not uncertainty estimates for a new prediction."
        ),
    }
