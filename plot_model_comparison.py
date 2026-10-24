
import matplotlib.pyplot as plt
from pathlib import Path

data_dir = Path("data/processed")
data_dir.mkdir(parents=True, exist_ok=True)

models = ["Logistic Regression", "Random Forest"]
pr_auc = [0.0941, 0.0922]
roc_auc = [0.7206, 0.7534]

fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# PR-AUC comparison
bars_pr = axes[0].bar(models, pr_auc)
axes[0].set_title("Validation PR-AUC")
axes[0].set_ylabel("Average Precision")
axes[0].set_ylim(0, 0.11)
axes[0].bar_label(bars_pr, fmt="%.4f", padding=4)
axes[0].tick_params(axis="x", rotation=12)
axes[0].grid(axis="y", linestyle="--", alpha=0.3)
axes[0].set_axisbelow(True)

# ROC-AUC comparison
bars_roc = axes[1].bar(models, roc_auc)
axes[1].set_title("Validation ROC-AUC")
axes[1].set_ylabel("ROC-AUC")
axes[1].set_ylim(0, 0.85)
axes[1].bar_label(bars_roc, fmt="%.4f", padding=4)
axes[1].tick_params(axis="x", rotation=12)
axes[1].grid(axis="y", linestyle="--", alpha=0.3)
axes[1].set_axisbelow(True)

fig.suptitle(
    "SepsisGuard-X: Baseline Model Comparison",
    fontsize=14,
    fontweight="bold",
)
fig.tight_layout()

output_path = data_dir / "model_comparison.png"
fig.savefig(output_path, dpi=300, bbox_inches="tight")
print("Chart saved to:", output_path.resolve())

plt.show()
plt.close(fig)
