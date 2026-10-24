
import pandas as pd
from pathlib import Path

input_path = Path("data/Dataset.csv")
output_path = Path("data/processed/sepsis_clean.csv")

print("Loading original dataset...")
df = pd.read_csv(input_path)

print("Original shape:", df.shape)

# Missing patient IDs aur labels wale records remove karo
df = df.dropna(subset=["Patient_ID", "SepsisLabel"]).copy()

# Unnecessary columns remove karo
columns_to_drop = ["Unnamed: 0", "EtCO2"]
df = df.drop(
    columns=[col for col in columns_to_drop if col in df.columns]
)

# Cleaned dataset save karo
output_path.parent.mkdir(parents=True, exist_ok=True)
df.to_csv(output_path, index=False)

print("\nCleaned dataset saved successfully!")
print("Final shape:", df.shape)
print("EtCO2 present:", "EtCO2" in df.columns)
print("Saved at:", output_path.resolve())
