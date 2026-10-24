
from pathlib import Path
import pandas as pd

root = Path(".")
data_dir = root / "data" / "processed"

report_path = root / "research_experiment_report.md"
output_path = root / "research_experiment_report_updated.md"

if not report_path.exists():
    raise FileNotFoundError(
        "Existing research_experiment_report.md was not found."
    )

# Preserve the existing report.
original_report = report_path.read_text(encoding="utf-8")

uncertainty_file = data_dir / "uncertainty_correct_incorrect_summary.csv"
abstention_file = data_dir / "abstention_validation_results.csv"

if not uncertainty_file.exists() or not abstention_file.exists():
    raise FileNotFoundError(
        "Required uncertainty or abstention result files are missing."
    )

uncertainty = pd.read_csv(uncertainty_file)
abstention = pd.read_csv(abstention_file)

section = """

## Additional Experiment: Bootstrap Uncertainty and Error Detection

### Objective
Investigate whether variability across bootstrap model predictions
is associated with incorrect validation predictions.

### Error Detection Results
- Error-detection ROC-AUC: 0.5805
- Error-detection PR-AUC: 0.2963
- Mean uncertainty for correct predictions: 0.02357
- Mean uncertainty for incorrect predictions: 0.02581

The ROC-AUC indicates weak error-ranking ability. Although incorrect
predictions had higher mean uncertainty, uncertainty alone did not
reliably identify all errors.

### Uncertainty-Aware Abstention

The validation experiment retained records below different uncertainty
quantiles and flagged the remaining records for hypothetical review.

| Coverage | Flagged | Error rate among retained records |
|---:|---:|---:|
"""

for _, row in abstention.iterrows():
    coverage = row["Coverage"] * 100
    flagged = 100 - coverage
    error_rate = row["Error_Rate_Retained"] * 100

    section += (
        f"| {coverage:.0f}% | {flagged:.0f}% | "
        f"{error_rate:.2f}% |\\n"
    )

section += """

### Interpretation and Limitations
The retained-record error rate was lower at lower coverage levels.
However, errors remained even when only the least-uncertain records
were retained. These exploratory results do not establish clinical
safety or prospective usefulness.

The thresholds were examined on the validation set, so performance
may be optimistic. Records from the same patient can also be correlated.
An independent patient-level evaluation of the finalized pipeline is
required. The final test set has not been used in these experiments.

Bootstrap prediction variability is an experimental uncertainty proxy;
it should not be interpreted as a validated measure of clinical risk.
"""

output_path.write_text(
    original_report.rstrip() + "\n\n" + section,
    encoding="utf-8",
)

print("Updated report created successfully:")
print(output_path.resolve())
print("Original report was preserved.")
