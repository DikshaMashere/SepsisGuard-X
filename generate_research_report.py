
import pandas as pd

from pathlib import Path
from datetime import datetime

data_dir = Path("data/processed")
output_path = Path("research_experiment_report.md")

report = []

def add_heading(text, level=2):
    report.append(f"{'#' * level} {text}\n")

def add_paragraph(text):
    report.append(f"{text}\n")

def add_csv_table(filename, title):
    file_path = data_dir / filename

    add_heading(title, level=3)

    if not file_path.exists():
        add_paragraph(f"Report unavailable: `{filename}` not found.")
        return

    try:
        df = pd.read_csv(file_path)

        if df.empty:
            add_paragraph("The CSV file contains no rows.")
        else:
            report.append(df.round(4).to_markdown(index=False))
            report.append("\n")

    except Exception as error:
        add_paragraph(f"Could not read `{filename}`: {error}")

# Report title and metadata
add_heading(
    "SepsisGuard-X: Research Experiment Report",
    level=1,
)

add_paragraph(
    f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
)

add_paragraph(
    "**Project objective:** Explore uncertainty-aware, "
    "trend-adaptive and explainable machine learning for "
    "sepsis risk prediction."
)

add_paragraph(
    "**Evaluation status:** Validation experiments only. "
    "The final test set has not been used in the reported experiments."
)

# 1. Dataset and methodology
add_heading("1. Dataset and Methodology")

add_paragraph(
    "- Dataset: Sepsis Prediction Dataset based on the "
    "PhysioNet Computing in Cardiology Challenge 2019."
)
add_paragraph(
    "- Data cleaning removed the fully missing `EtCO2` column "
    "and the redundant `Unnamed: 0` column."
)
add_paragraph(
    "- Patients were separated between training, validation, "
    "and final test groups."
)
add_paragraph(
    "- Median imputation was fitted on training data only "
    "for the original baseline preprocessing."
)
add_paragraph(
    "- Primary baseline features: Hour, HR, O2Sat, SBP, MAP, "
    "Resp, Age, Gender, HospAdmTime and ICULOS."
)

# 2. Model comparison
add_heading("2. Baseline Model Comparison")
add_csv_table(
    "model_threshold_comparison.csv",
    "Threshold comparison results",
)
add_csv_table(
    "weighted_unweighted_comparison.csv",
    "Weighted vs unweighted Logistic Regression",
)

# 3. Threshold experiments
add_heading("3. Hourly Record-Level Threshold Experiments")
add_csv_table(
    "unweighted_threshold_comparison.csv",
    "Unweighted Logistic Regression",
)
add_csv_table(
    "calibrated_threshold_comparison.csv",
    "Sigmoid-calibrated Logistic Regression",
)

# 4. Patient-level experiments
add_heading("4. Exploratory Patient-Level Evaluation")
add_paragraph(
    "Patient-level scores were defined as the maximum hourly "
    "predicted probability within each patient's validation records. "
    "Patient labels were positive if any validation record had "
    "`SepsisLabel = 1`."
)
add_paragraph(
    "This is an exploratory aggregation and does not establish "
    "prospective early-warning performance."
)
add_csv_table(
    "patient_threshold_comparison.csv",
    "Patient-level threshold comparison",
)

# 5. Calibration
add_heading("5. Probability Calibration")
add_paragraph(
    "A sigmoid calibrator was fitted using a separate "
    "patient-level calibration subset. The existing validation "
    "set was then used for evaluation."
)
add_paragraph(
    "Calibration quality should be assessed using the validation "
    "calibration curve and proper scoring rules. Matching mean "
    "predicted probability to prevalence alone does not prove "
    "individual-level calibration."
)

# 6. Missingness experiment
add_heading("6. Missingness Indicator Experiment")
add_paragraph(
    "Adding missingness indicators did not improve validation "
    "performance over the original baseline Logistic Regression "
    "in the reported experiment. The baseline PR-AUC was 0.0941; "
    "the scaled missingness model PR-AUC was 0.0904."
)

# 7. Limitations
add_heading("7. Limitations")

add_paragraph(
    "- Positive labels are rare, so accuracy alone is not a "
    "sufficient performance metric."
)
add_paragraph(
    "- Results are exploratory and are not clinical validation."
)
add_paragraph(
    "- Hourly records from the same patient are correlated."
)
add_paragraph(
    "- Maximum-probability patient aggregation uses the complete "
    "observed validation sequence and does not prove early warning "
    "before sepsis onset."
)
add_paragraph(
    "- Thresholds were explored on validation data and should not "
    "be interpreted as clinically optimal."
)
add_paragraph(
    "- The final test set must remain untouched until the modelling "
    "pipeline and evaluation protocol are finalized."
)

# 8. Next steps
add_heading("8. Recommended Next Steps")

add_paragraph(
    "1. Implement and validate a clearly defined uncertainty-estimation method."
)
add_paragraph(
    "2. Define a prospective prediction target and lead-time evaluation protocol."
)
add_paragraph(
    "3. Assess calibration and discrimination with appropriate uncertainty intervals."
)
add_paragraph(
    "4. Finalize the modelling pipeline and threshold-selection criteria."
)
add_paragraph(
    "5. Evaluate the final frozen pipeline once on the untouched test set."
)

# Write report
output_path.write_text(
    "\n".join(report),
    encoding="utf-8",
)

print("Research report generated successfully.")
print("Saved to:", output_path.resolve())
print("Final test data was not loaded or used by this script.")
