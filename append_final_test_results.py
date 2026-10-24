
from pathlib import Path
import pandas as pd

data_dir = Path("data/processed")
report_path = Path("research_experiment_report_updated.md")

hourly_path = data_dir / "final_test_hourly_metrics.csv"
patient_path = data_dir / "final_test_patient_metrics.csv"

if not report_path.exists():
    raise FileNotFoundError(f"Report not found: {report_path}")

if not hourly_path.exists() or not patient_path.exists():
    raise FileNotFoundError(
        "Final test metrics files were not found. "
        "Run final_test_evaluation.py first."
    )

report = report_path.read_text(encoding="utf-8")

section_title = "## 9. Final Independent Test Evaluation"

if section_title in report:
    raise RuntimeError(
        "Final test evaluation section already exists. "
        "Report was not changed."
    )

hourly = pd.read_csv(hourly_path).iloc[0].to_dict()
patient = pd.read_csv(patient_path).iloc[0].to_dict()

section = f"""

{section_title}

### Evaluation protocol

The saved sigmoid-calibrated Logistic Regression pipeline was evaluated
on the held-out test set after model fitting, calibration, and validation
experiments were completed. The test set was not used to fit the model
or select thresholds. The pipeline's existing imputer handled missing
feature values.

The test set contained {int(hourly['Rows']):,} hourly records from
{int(patient['Patients']):,} patients. There were
{int(hourly['Positive_Records']):,} positive hourly records and
{int(patient['Positive_Patients']):,} patients with at least one positive
hourly label.

### Hourly-record results

| Metric | Test result |
|---|---:|
| PR-AUC | {hourly['PR_AUC']:.6f} |
| ROC-AUC | {hourly['ROC_AUC']:.6f} |
| Brier score | {hourly['Brier_Score']:.6f} |
| Mean predicted probability | {hourly['Mean_Predicted_Probability']:.6f} |
| Observed positive fraction | {hourly['Observed_Positive_Fraction']:.6f} |
| Reference threshold | {hourly['Threshold']:.3f} |
| Precision | {hourly['Precision']:.6f} |
| Recall | {hourly['Recall']:.6f} |
| F1-score | {hourly['F1']:.6f} |
| True positives | {int(hourly['TP']):,} |
| False positives | {int(hourly['FP']):,} |
| True negatives | {int(hourly['TN']):,} |
| False negatives | {int(hourly['FN']):,} |

The hourly reference threshold of 0.03 had been examined on validation
data before final test evaluation. It is not a clinically validated
threshold. At this threshold, precision was low and many positive hourly
records were missed.

### Exploratory patient-level aggregation

For each patient, the outcome was defined as whether any observed hourly
label was positive, and the patient score was the maximum predicted
hourly probability across the available sequence.

| Metric | Test result |
|---|---:|
| PR-AUC | {patient['PR_AUC']:.6f} |
| ROC-AUC | {patient['ROC_AUC']:.6f} |
| Brier score | {patient['Brier_Score']:.6f} |
| Reference threshold | {patient['Reference_Threshold']:.2f} |
| Precision | {patient['Precision']:.6f} |
| Recall | {patient['Recall']:.6f} |
| F1-score | {patient['F1']:.6f} |
| True positives | {int(patient['TP']):,} |
| False positives | {int(patient['FP']):,} |
| True negatives | {int(patient['TN']):,} |
| False negatives | {int(patient['FN']):,} |

At the patient-level reference threshold of 0.50, only
{int(patient['TP'])} positive patient was detected among
{int(patient['Positive_Patients'])} positive patients. This threshold
therefore performed poorly for binary patient-level classification.
It was included as a reference, not as an optimized patient-level
decision threshold.

The maximum-probability aggregation uses the complete observed patient
sequence. It does not measure whether the model predicts sepsis before
onset, and its results should not be interpreted as prospective
early-warning performance.

### Limitations

- This is a retrospective evaluation of an academic prototype, not
  clinical validation.
- Hourly observations within a patient are correlated.
- Precision-recall results must be interpreted in light of the low
  positive-label prevalence.
- The patient-level maximum score can depend on the length of the
  observed sequence.
- Thresholds were not optimized using test results.
- No confidence intervals or external-cohort validation are reported
  in this experiment.
- The validation-only bootstrap uncertainty and abstention experiments
  are separate from this final test evaluation; their results should
  not be presented as test-set uncertainty performance.

### Saved metric files

- `data/processed/final_test_hourly_metrics.csv`
- `data/processed/final_test_patient_metrics.csv`

These results complete the current frozen-model test evaluation. Any
future model or threshold changes should be evaluated as a new
experiment and must not be described as part of this untouched-test
evaluation.
"""

with report_path.open("a", encoding="utf-8", newline="\n") as file:
    file.write(section)

print("Final test results appended successfully.")
print("Updated report:", report_path.resolve())
print("Original research_experiment_report.md was not changed.")
