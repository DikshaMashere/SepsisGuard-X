# SepsisGuard-X: Research Experiment Report

Generated: 2026-10-24 00:16:49

**Project objective:** Explore uncertainty-aware, trend-adaptive and explainable machine learning for sepsis risk prediction.

**Evaluation status:** Validation experiments and a final held-out test evaluation are complete. The test set was used once for the frozen calibrated model; it was not used for model fitting, calibration, or threshold selection.

## 1. Dataset and Methodology

- Dataset: Sepsis Prediction Dataset based on the PhysioNet Computing in Cardiology Challenge 2019.

- Data cleaning removed the fully missing `EtCO2` column and the redundant `Unnamed: 0` column.

- Patients were separated between training, validation, and final test groups.

- Median imputation was fitted on training data only for the original baseline preprocessing.

- Primary baseline features: Hour, HR, O2Sat, SBP, MAP, Resp, Age, Gender, HospAdmTime and ICULOS.

## 2. Baseline Model Comparison

### Threshold comparison results

| Model               |   Threshold |   Precision |   Recall |     F1 |   False_Positives |   False_Negatives |
|:--------------------|------------:|------------:|---------:|-------:|------------------:|------------------:|
| Logistic Regression |        0.2  |      0.0219 |   0.9942 | 0.0428 |             84616 |                11 |
| Logistic Regression |        0.3  |      0.024  |   0.9454 | 0.0467 |             73334 |               104 |
| Logistic Regression |        0.4  |      0.0317 |   0.7658 | 0.0609 |             44555 |               446 |
| Logistic Regression |        0.5  |      0.053  |   0.5966 | 0.0974 |             20289 |               768 |
| Logistic Regression |        0.6  |      0.0945 |   0.4333 | 0.1551 |              7909 |              1079 |
| Random Forest       |        0.1  |      0.0221 |   0.9989 | 0.0431 |             84353 |                 2 |
| Random Forest       |        0.15 |      0.0234 |   0.9842 | 0.0458 |             78135 |                30 |
| Random Forest       |        0.2  |      0.0261 |   0.937  | 0.0507 |             66621 |               120 |
| Random Forest       |        0.25 |      0.03   |   0.8666 | 0.0579 |             53433 |               254 |
| Random Forest       |        0.3  |      0.0363 |   0.7841 | 0.0693 |             39671 |               411 |
| Random Forest       |        0.4  |      0.0611 |   0.5898 | 0.1108 |             17244 |               781 |
| Random Forest       |        0.5  |      0.1037 |   0.4359 | 0.1676 |              7172 |              1074 |


### Weighted vs unweighted Logistic Regression

| Model      |   PR_AUC |   ROC_AUC |   Brier_Score |   Precision_at_0.5 |   Recall_at_0.5 |   F1_at_0.5 |
|:-----------|---------:|----------:|--------------:|-------------------:|----------------:|------------:|
| Balanced   |   0.0941 |    0.7206 |        0.2007 |              0.053 |          0.5966 |      0.0974 |
| Unweighted |   0.0921 |    0.7244 |        0.0207 |              0     |          0      |      0      |


## 3. Hourly Record-Level Threshold Experiments

### Unweighted Logistic Regression

|   Threshold |   Precision |   Recall |     F1 |   False_Positives |   False_Negatives |   True_Positives |
|------------:|------------:|---------:|-------:|------------------:|------------------:|-----------------:|
|       0.01  |      0.0231 |   0.9664 | 0.0451 |             77913 |                64 |             1840 |
|       0.015 |      0.0299 |   0.813  | 0.0577 |             50169 |               356 |             1548 |
|       0.02  |      0.0435 |   0.6513 | 0.0815 |             27283 |               664 |             1240 |
|       0.025 |      0.0638 |   0.5305 | 0.1139 |             14817 |               894 |             1010 |
|       0.03  |      0.0882 |   0.4438 | 0.1472 |              8733 |              1059 |              845 |
|       0.04  |      0.1187 |   0.2957 | 0.1694 |              4180 |              1341 |              563 |
|       0.05  |      0.1436 |   0.229  | 0.1765 |              2601 |              1468 |              436 |


### Sigmoid-calibrated Logistic Regression

|   Threshold |   Precision |   Recall |     F1 |   False_Positives |   False_Negatives |   True_Positives |
|------------:|------------:|---------:|-------:|------------------:|------------------:|-----------------:|
|       0.01  |      0.0229 |   0.9701 | 0.0447 |             78923 |                57 |             1847 |
|       0.015 |      0.0303 |   0.8078 | 0.0583 |             49285 |               366 |             1538 |
|       0.02  |      0.0473 |   0.6339 | 0.0881 |             24297 |               697 |             1207 |
|       0.025 |      0.072  |   0.49   | 0.1256 |             12024 |               971 |              933 |
|       0.03  |      0.0982 |   0.3829 | 0.1563 |              6696 |              1175 |              729 |


## 4. Exploratory Patient-Level Evaluation

Patient-level scores were defined as the maximum hourly predicted probability within each patient's validation records. Patient labels were positive if any validation record had `SepsisLabel = 1`.

This is an exploratory aggregation and does not establish prospective early-warning performance.

### Patient-level threshold comparison

|   Threshold |   Precision |   Recall |     F1 |   True_Positives |   False_Positives |   False_Negatives |   True_Negatives |
|------------:|------------:|---------:|-------:|-----------------:|------------------:|------------------:|-----------------:|
|         0.1 |      0.0876 |   0.9949 | 0.1611 |              197 |              2051 |                 1 |                0 |
|         0.2 |      0.0876 |   0.9949 | 0.1611 |              197 |              2051 |                 1 |                0 |
|         0.3 |      0.0875 |   0.9899 | 0.1608 |              196 |              2044 |                 2 |                7 |
|         0.4 |      0.0876 |   0.9091 | 0.1599 |              180 |              1874 |                18 |              177 |
|         0.5 |      0.1052 |   0.7626 | 0.1848 |              151 |              1285 |                47 |              766 |
|         0.6 |      0.1594 |   0.5556 | 0.2477 |              110 |               580 |                88 |             1471 |
|         0.7 |      0.3391 |   0.3939 | 0.3645 |               78 |               152 |               120 |             1899 |
|         0.8 |      0.6533 |   0.2475 | 0.359  |               49 |                26 |               149 |             2025 |


## 5. Probability Calibration

A sigmoid calibrator was fitted using a separate patient-level calibration subset. The existing validation set was then used for evaluation.

Calibration quality should be assessed using the validation calibration curve and proper scoring rules. Matching mean predicted probability to prevalence alone does not prove individual-level calibration.

## 6. Missingness Indicator Experiment

Adding missingness indicators did not improve validation performance over the original baseline Logistic Regression in the reported experiment. The baseline PR-AUC was 0.0941; the scaled missingness model PR-AUC was 0.0904.

## 7. Limitations

- Positive labels are rare, so accuracy alone is not a sufficient performance metric.

- Results are exploratory and are not clinical validation.

- Hourly records from the same patient are correlated.

- Maximum-probability patient aggregation uses the complete observed validation sequence and does not prove early warning before sepsis onset.

- Thresholds were explored on validation data and should not be interpreted as clinically optimal.

- The final test evaluation is complete for this experiment. Test results must not be used to tune the current model or its thresholds.

## 8. Current Status and Future Work

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
| 50% | 50% | 19.08% |
| 70% | 30% | 21.23% |
| 80% | 20% | 22.10% |
| 90% | 10% | 22.84% |
| 95% | 5% | 23.19% |


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


## 9. Final Independent Test Evaluation

### Evaluation protocol

The saved sigmoid-calibrated Logistic Regression pipeline was evaluated
on the held-out test set after model fitting, calibration, and validation
experiments were completed. The test set was not used to fit the model
or select thresholds. The pipeline's existing imputer handled missing
feature values.

The test set contained 108,704 hourly records from
2,812 patients. There were
2,375 positive hourly records and
248 patients with at least one positive
hourly label.

### Hourly-record results

| Metric | Test result |
|---|---:|
| PR-AUC | 0.078437 |
| ROC-AUC | 0.709050 |
| Brier score | 0.020986 |
| Mean predicted probability | 0.020607 |
| Observed positive fraction | 0.021848 |
| Reference threshold | 0.030 |
| Precision | 0.088995 |
| Recall | 0.384421 |
| F1-score | 0.144531 |
| True positives | 913 |
| False positives | 9,346 |
| True negatives | 96,983 |
| False negatives | 1,462 |

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
| PR-AUC | 0.371766 |
| ROC-AUC | 0.678794 |
| Brier score | 0.079871 |
| Reference threshold | 0.50 |
| Precision | 0.500000 |
| Recall | 0.004032 |
| F1-score | 0.008000 |
| True positives | 1 |
| False positives | 1 |
| True negatives | 2,563 |
| False negatives | 247 |

At the patient-level reference threshold of 0.50, only
1 positive patient was detected among
248 positive patients. This threshold
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

