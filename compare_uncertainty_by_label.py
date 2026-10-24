
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "bootstrap_uncertainty_validation.csv"
)

# Compare bootstrap uncertainty by actual hourly label
summary = df.groupby("SepsisLabel")["StdProbability"].agg(
    ["count", "mean", "median", "std"]
).reset_index()

summary["SepsisLabel"] = summary["SepsisLabel"].map({
    0: "Negative label (0)",
    1: "Positive label (1)",
})

print("\nBootstrap uncertainty by actual label")
print(summary.round(5).to_string(index=False))

summary.to_csv(
    data_dir / "uncertainty_by_label_summary.csv",
    index=False,
)

# Plot uncertainty distributions
plt.figure(figsize=(9, 5))

for label, name in [
    (0, "Negative label (0)"),
    (1, "Positive label (1)"),
]:
    values = df.loc[
        df["SepsisLabel"] == label,
        "StdProbability",
    ]

    plt.hist(
        values,
        bins=40,
        density=True,
        alpha=0.5,
        label=name,
    )

plt.xlabel("Bootstrap prediction standard deviation")
plt.ylabel("Density")
plt.title("SepsisGuard-X: Uncertainty by Actual Label")
plt.legend()
plt.tight_layout()

output_path = data_dir / "uncertainty_by_label.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("\nSummary saved to:",
      (data_dir / "uncertainty_by_label_summary.csv").resolve())
print("Graph saved to:", output_path.resolve())
print("Final test set has not been used.")
