
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

results = pd.read_csv(
    data_dir / "patient_threshold_comparison.csv"
)

plt.figure(figsize=(9, 6))

plt.plot(
    results["Threshold"],
    results["Precision"],
    marker="o",
    label="Precision",
)

plt.plot(
    results["Threshold"],
    results["Recall"],
    marker="o",
    label="Recall",
)

plt.plot(
    results["Threshold"],
    results["F1"],
    marker="o",
    label="F1-score",
)

plt.xlabel("Patient-level threshold")
plt.ylabel("Metric score")
plt.title("SepsisGuard-X: Patient-Level Threshold Trade-off")
plt.ylim(0, 1)
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = data_dir / "patient_threshold_tradeoff.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Graph saved to:", output_path.resolve())
