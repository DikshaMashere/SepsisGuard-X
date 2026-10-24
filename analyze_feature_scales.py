
import pandas as pd
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from features import BASELINE_FEATURES

data_dir = Path("data/processed")

# Use only the model-training partition
X_train = pd.read_csv(data_dir / "X_train_core.csv")

scaler = StandardScaler()
scaler.fit(X_train)

scale_report = pd.DataFrame({
    "Feature": BASELINE_FEATURES,
    "Training_Mean": scaler.mean_,
    "Training_Std": scaler.scale_,
})

print("Training feature scales:")
print(scale_report.round(4).to_string(index=False))

output_path = data_dir / "feature_scales.csv"
scale_report.to_csv(output_path, index=False)

print("\nFeature scale report saved to:", output_path.resolve())
