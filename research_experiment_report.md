# SepsisGuard-X: Research Experiment Report

Generated: 2026-10-24 00:16:49

**Project objective:** Explore uncertainty-aware, trend-adaptive and explainable machine learning for sepsis risk prediction.

**Evaluation status:** Validation experiments only. The final test set has not been used in the reported experiments.

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

- The final test set must remain untouched until the modelling pipeline and evaluation protocol are finalized.

## 8. Recommended Next Steps

1. Implement and validate a clearly defined uncertainty-estimation method.

2. Define a prospective prediction target and lead-time evaluation protocol.

3. Assess calibration and discrimination with appropriate uncertainty intervals.

4. Finalize the modelling pipeline and threshold-selection criteria.

5. Evaluate the final frozen pipeline once on the untouched test set.
