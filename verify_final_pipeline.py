
import joblib
from pathlib import Path
from sklearn.calibration import CalibratedClassifierCV

data_dir = Path("data/processed")

model_path = data_dir / "logistic_regression_calibrated.joblib"

if not model_path.exists():
    raise FileNotFoundError(f"Model not found: {model_path}")

model = joblib.load(model_path)

print("Saved model loaded successfully.")
print("Model type:", type(model).__name__)

if isinstance(model, CalibratedClassifierCV):
    print("Calibration method:", model.method)
    print("Number of calibrated classifiers:", len(model.calibrated_classifiers_))

    for index, calibrated_classifier in enumerate(
        model.calibrated_classifiers_
    ):
        print(f"\nCalibrated classifier {index + 1}")
        print(
            "Frozen estimator type:",
            type(calibrated_classifier.estimator).__name__,
        )
        print(
            "Estimator pipeline steps:",
            list(calibrated_classifier.estimator.named_steps.keys()),
        )
else:
    print("WARNING: Loaded object is not CalibratedClassifierCV.")

print("\nVerification complete.")
print("No data was loaded and no evaluation was performed.")
