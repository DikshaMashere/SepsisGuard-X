
# SepsisGuard-X: Initial feature configuration

# Target variable: what the model predicts
TARGET_COLUMN = "SepsisLabel"

# Patient identifier: grouping and leakage prevention only
GROUP_COLUMN = "Patient_ID"

# Initial baseline features
BASELINE_FEATURES = [
    "Hour",
    "HR",
    "O2Sat",
    "SBP",
    "MAP",
    "Resp",
    "Age",
    "Gender",
    "HospAdmTime",
    "ICULOS",
]

# Additional features to evaluate later
OPTIONAL_FEATURES = [
    "Temp",
    "DBP",
]

# Columns excluded from model inputs
EXCLUDED_COLUMNS = [
    "Patient_ID",
]

print("Feature configuration loaded.")
print("Baseline features:", len(BASELINE_FEATURES))
print("Optional features:", OPTIONAL_FEATURES)
print("Target:", TARGET_COLUMN)
print("Patient grouping column:", GROUP_COLUMN)
