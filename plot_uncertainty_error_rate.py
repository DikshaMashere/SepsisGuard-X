
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "uncertainty_error_analysis.csv"
)

# Create uncertainty groups using validation-data quantiles
df["UncertaintyBin"] = pd.qcut(
    df["StdProbability"],
    q=10,
    duplicates="drop",
)

df["Error"] = (
    ~df["PredictionCorrect"].astype(bool)
).astype(int)

summary = (
    df.groupby("UncertaintyBin", observed=True)
    .agg(
        Records=("Error", "size"),
        ErrorRate=("Error", "mean"),
        MeanUncertainty=("StdProbability", "mean"),
    )
    .reset_index()
)

print("\nError rate by uncertainty bin")
print(summary.round(5).to_string(index=False))

summary.to_csv(
    data_dir / "uncertainty_error_rate_by_bin.csv",
    index=False,
)

plt.figure(figsize=(9, 5))

plt.plot(
    summary["MeanUncertainty"],
    summary["ErrorRate"],
    marker="o",
)

plt.xlabel("Mean bootstrap prediction standard deviation")
plt.ylabel("Classification error rate")
plt.title("SepsisGuard-X: Error Rate vs Uncertainty")
plt.grid(True, alpha=0.3)
plt.tight_layout()

output_path = data_dir / "uncertainty_error_rate_by_bin.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("\nGraph saved to:", output_path.resolve())
print("Final test set has not been used.")
