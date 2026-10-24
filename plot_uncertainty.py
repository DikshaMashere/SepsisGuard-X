
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data/processed")
results = pd.read_csv(data_dir / "uncertainty_analysis.csv")

plt.figure(figsize=(10, 6))

negative = results.loc[
    results["SepsisLabel"] == 0, "SepsisProbability"
]
positive = results.loc[
    results["SepsisLabel"] == 1, "SepsisProbability"
]

plt.hist(
    negative,
    bins=50,
    range=(0, 1),
    density=True,
    alpha=0.6,
    label="Actual label: 0",
)

plt.hist(
    positive,
    bins=50,
    range=(0, 1),
    density=True,
    alpha=0.6,
    label="Actual label: 1",
)

plt.axvline(
    0.4,
    linestyle="--",
    label="Exploratory band: 0.4–0.6",
)
plt.axvline(0.6, linestyle="--")

plt.xlabel("Predicted probability")
plt.ylabel("Density")
plt.title("SepsisGuard-X: Validation Probability Distribution")
plt.legend()
plt.tight_layout()

output_path = data_dir / "uncertainty_distribution.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Chart saved to:", output_path.resolve())
