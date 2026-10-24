
import pandas as pd
from pathlib import Path

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
)

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "uncertainty_error_analysis.csv"
)

# Records already contain mean-probability predictions
# and bootstrap uncertainty from the validation experiment.
df["Error"] = (
    ~df["PredictionCorrect"].astype(bool)
).astype(int)

# Evaluate several uncertainty cutoffs.
# A record is retained when uncertainty <= cutoff.
quantiles = [0.50, 0.70, 0.80, 0.90, 0.95]

rows = []

for q in quantiles:
    cutoff = df["StdProbability"].quantile(q)
    retained = df["StdProbability"] <= cutoff

    subset = df.loc[retained]

    rows.append({
        "Uncertainty_Quantile": q,
        "Uncertainty_Cutoff": cutoff,
        "Coverage": retained.mean(),
        "Records_Retained": int(retained.sum()),
        "Records_Flagged": int((~retained).sum()),
        "Error_Rate_Retained": subset["Error"].mean(),
        "Accuracy_Retained": accuracy_score(
            subset["ActualLabel"],
            subset["Prediction"],
        ),
        "Precision_Retained": precision_score(
            subset["ActualLabel"],
            subset["Prediction"],
            zero_division=0,
        ),
        "Recall_Retained": recall_score(
            subset["ActualLabel"],
            subset["Prediction"],
            zero_division=0,
        ),
    })

results = pd.DataFrame(rows)

print("\nUncertainty-aware abstention results")
print(results.round(4).to_string(index=False))

output_path = data_dir / "abstention_validation_results.csv"
results.to_csv(output_path, index=False)

print("\nResults saved to:", output_path.resolve())
print("Final test set has not been used.")
