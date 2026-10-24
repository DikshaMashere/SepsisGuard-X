
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "bootstrap_uncertainty_validation.csv"
)

print("Bootstrap uncertainty summary")
print(df["StdProbability"].describe().round(5))

# Exploratory high-uncertainty band:
# This is a descriptive cutoff, not a clinical threshold.
cutoff = df["StdProbability"].quantile(0.90)
high_uncertainty = df["StdProbability"] >= cutoff

print(f"\n90th-percentile uncertainty cutoff: {cutoff:.5f}")
print("High-uncertainty records:", int(high_uncertainty.sum()))
print(
    "Percentage of records:",
    round(high_uncertainty.mean() * 100, 2),
)

# Plot uncertainty distribution
plt.figure(figsize=(9, 5))
plt.hist(df["StdProbability"], bins=40, edgecolor="black")
plt.axvline(
    cutoff,
    linestyle="--",
    label="90th-percentile cutoff",
)
plt.xlabel("Bootstrap prediction standard deviation")
plt.ylabel("Number of validation records")
plt.title("SepsisGuard-X: Bootstrap Uncertainty Distribution")
plt.legend()
plt.tight_layout()

output_path = data_dir / "bootstrap_uncertainty_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

# Save summary for reproducibility
summary = pd.DataFrame([{
    "Bootstrap_Models": 20,
    "Validation_Records": len(df),
    "Mean_StdProbability": df["StdProbability"].mean(),
    "Median_StdProbability": df["StdProbability"].median(),
    "Mean_Interval_Width": (
        df["UpperPercentile"] - df["LowerPercentile"]
    ).mean(),
    "P90_Uncertainty_Cutoff": cutoff,
    "High_Uncertainty_Records": int(high_uncertainty.sum()),
}])

summary.to_csv(
    data_dir / "bootstrap_uncertainty_summary.csv",
    index=False,
)

print("\nGraph saved to:", output_path.resolve())
print("Summary saved to:", (data_dir / "bootstrap_uncertainty_summary.csv").resolve())
print("Final test set has not been used.")
