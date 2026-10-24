
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

unweighted = pd.read_csv(
    data_dir / "unweighted_threshold_comparison.csv"
)
calibrated = pd.read_csv(
    data_dir / "calibrated_threshold_comparison.csv"
)

plt.figure(figsize=(9, 6))

plt.plot(
    unweighted["Threshold"],
    unweighted["Recall"],
    marker="o",
    label="Unweighted Logistic Regression",
)

plt.plot(
    calibrated["Threshold"],
    calibrated["Recall"],
    marker="o",
    label="Sigmoid-calibrated Logistic Regression",
)

plt.xlabel("Classification threshold")
plt.ylabel("Recall")
plt.title("SepsisGuard-X: Recall Comparison")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = data_dir / "calibrated_recall_comparison.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Comparison graph saved to:", output_path.resolve())
