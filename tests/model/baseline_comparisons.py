import pandas as pd
import numpy as np
from collections import Counter

# Load data
df = pd.read_csv("../../training/training_data_v1.csv")
df = df[df['accepted'] == 1].copy()
df['chosen_hw_type'] = df['chosen_hw_type'] - 1

y_true = df['chosen_hw_type'].values

print("=" * 60)
print("BASELINE COMPARISON")
print("=" * 60)

# Baseline 1: Random Selection
np.random.seed(42)
y_random = np.random.randint(0, 4, size=len(y_true))
acc_random = (y_random == y_true).mean()
print(f"\n1. Random Selection: {acc_random:.4f} ({acc_random*100:.2f}%)")

# Baseline 2: Most Frequent Class (Majority Vote)
most_frequent = Counter(y_true).most_common(1)[0][0]
y_majority = np.full(len(y_true), most_frequent)
acc_majority = (y_majority == y_true).mean()
print(f"2. Majority Class (HW {most_frequent+1}): {acc_majority:.4f} ({acc_majority*100:.2f}%)")

# Baseline 3: Rule-based (e.g., largest tasks → HW with accelerators)
# Assuming HW type 4 (index 3) has accelerators
def rule_based_selection(row):
    if row['cpu_req'] > 10 and row['mem_req'] > 6:
        return 3  # HW type 4 (with accelerators)
    elif row['cpu_req'] > 7:
        return 2  # HW type 3
    elif row['cpu_req'] > 4:
        return 1  # HW type 2
    else:
        return 0  # HW type 1

y_rule = df.apply(rule_based_selection, axis=1).values
acc_rule = (y_rule == y_true).mean()
print(f"3. Rule-Based: {acc_rule:.4f} ({acc_rule*100:.2f}%)")

# Baseline 4: Load from your "traditional", "sosm", "improved sosm" brokers
# (You mentioned these in dataset_merger.py - if you have their accuracy, add here)

print("\n" + "=" * 60)
print("Your NN model should outperform these baselines!")
print("=" * 60)
