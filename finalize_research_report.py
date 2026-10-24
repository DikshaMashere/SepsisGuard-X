
from pathlib import Path

report_path = Path("research_experiment_report_updated.md")

text = report_path.read_text(encoding="utf-8")

# Correct the opening evaluation status.
text = text.replace(
    "**Evaluation status:** Validation experiments only. "
    "The final test set has not been used in the reported experiments.",
    "**Evaluation status:** Validation experiments and a final "
    "held-out test evaluation are complete. The test set was used "
    "once for the frozen calibrated model; it was not used for "
    "model fitting, calibration, or threshold selection."
)

# Fix the known typo if it is still present.
text = text.replace("wasthen", "was then")

# Replace the outdated next-steps section.
start_marker = "## 8. Recommended Next Steps"
end_marker = "## Additional Experiment: Bootstrap Uncertainty and Error Detection"

if start_marker in text and end_marker in text:
    start = text.index(start_marker)
    end = text.index(end_marker, start)

    replacement = """## 8. Current Status and Future Work

The frozen sigmoid-calibrated Logistic Regression pipeline has been
evaluated on the held-out test set. The reported test results are final
for this experiment and must not be used to tune the current model.

Future work should focus on:

1. Define a clinically meaningful prediction target and a prospective
   lead-time evaluation protocol.
2. Investigate uncertainty-estimation methods using validation data
   and assess them on a new independent test set or external cohort.
3. Report confidence intervals and patient-level evaluation metrics
   using an appropriate resampling design.
4. Improve probability calibration and assess it on independent data.
5. Compare candidate models only within a documented experimental
   protocol, reserving a new test cohort for unbiased final evaluation.

These are future research directions, not completed experiments.

"""

    text = text[:start] + replacement + text[end:]

# Write only if expected report sections are present.
required_sections = [
    "## 1. Dataset and Methodology",
    "## 9. Final Independent Test Evaluation",
]

missing = [section for section in required_sections if section not in text]

if missing:
    raise ValueError(
        f"Report structure check failed; missing sections: {missing}"
    )

report_path.write_text(text, encoding="utf-8")

print("Report updated:", report_path.resolve())
print("Verified required sections are present.")
print("Final test results and metric tables were preserved.")
