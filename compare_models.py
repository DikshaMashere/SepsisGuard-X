
import pandas as pd
from pathlib import Path

data_dir = Path("data/processed")

# Load threshold comparison reports
lr = pd.read_csv(data_dir / "threshold_comparison.csv")
rf = pd.read_csv(data_dir / "rf_threshold_comparison.csv")

# Add model names
lr["Model"] = "Logistic Regression"
rf["Model"] = "Random Forest"

# Combine results
comparison = pd.concat([lr, rf], ignore_index=True)

# Keep key metrics together
comparison = comparison[
    [
        "Model",
        "Threshold",
        "Precision",
        "Recall",
        "F1",
        "False_Positives",
        "False_Negatives",
    ]
]

print("Validation model and threshold comparison:")
print(comparison.round(4).to_string(index=False))

output_path = data_dir / "model_threshold_comparison.csv"
comparison.to_csv(output_path, index=False)

print("\nComparison report saved to:", output_path.resolve())
