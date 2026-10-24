
import pandas as pd
import matplotlib.pyplot as plt

from pathlib import Path

data_dir = Path("data/processed")

df = pd.read_csv(
    data_dir / "abstention_validation_results.csv"
)

plt.figure(figsize=(9, 5))

plt.plot(
    df["Coverage"] * 100,
    df["Error_Rate_Retained"] * 100,
    marker="o",
    label="Error rate on retained records",
)

plt.xlabel("Coverage (%)")
plt.ylabel("Error rate on retained records (%)")
plt.title("SepsisGuard-X: Coverage vs Error Rate")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()

output_path = data_dir / "abstention_coverage_error.png"
plt.savefig(output_path, dpi=150, bbox_inches="tight")
plt.show()

print("Graph saved to:", output_path.resolve())
print("Final test set has not been used.")
