import pandas as pd

# Load the current CSV
df = pd.read_csv("data/results.csv")

# Pop the ground_truth column out and append it to the very end
gt_col = df.pop('ground_truth')
df['ground_truth'] = gt_col

# Save it back
df.to_csv("data/results.csv", index=False)
print("Column moved to the end!")