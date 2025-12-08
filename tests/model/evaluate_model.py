import pandas as pd
import torch
import pickle
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, confusion_matrix
from models.nn.neural_network import NeuralNetwork
import numpy as np

# Load test data
df = pd.read_csv("../../training/training_data_v1.csv")
df = df[df['accepted'] == 1].copy()
df['chosen_hw_type'] = df['chosen_hw_type'] - 1

# Load scaler and model
with open("../../models/nn/scaler.pkl", "rb") as f:
    scaler = pickle.load(f)

model = NeuralNetwork(input_dim=75, num_classes=4)  # Adjust input_dim
model.load_state_dict(torch.load("../../models/nn/model.pth"))
model.eval()

# Prepare test set (same split as training)
from sklearn.model_selection import train_test_split
feature_cols = ['num_vms', 'cpu_req', 'mem_req'] + \
               [c for c in df.columns if c.startswith('cell')]
X = df[feature_cols].values
y_true = df['chosen_hw_type'].values
y_energy_true = df['energy_cost'].values

_, X_test, _, y_test, _, ye_test = train_test_split(
    X, y_true, y_energy_true, test_size=0.2, random_state=42
)

X_test_scaled = scaler.transform(X_test)
X_test_t = torch.FloatTensor(X_test_scaled)

# Predict
with torch.no_grad():
    logits, energy_pred = model(X_test_t)
    y_pred = torch.argmax(logits, dim=1).numpy()
    energy_pred = energy_pred.squeeze().numpy()

# Metrics
print("=" * 60)
print("BASELINE MODEL PERFORMANCE")
print("=" * 60)

# 1. Classification Accuracy
acc = accuracy_score(y_test, y_pred)
print(f"\n[HW Type Selection Accuracy]: {acc:.4f} ({acc*100:.2f}%)")

# 2. Per-class Performance
precision, recall, f1, support = precision_recall_fscore_support(
    y_test, y_pred, average=None, labels=[0,1,2,3]
)
print("\nPer-HW Type Performance:")
for hw in range(4):
    print(f"  HW Type {hw+1}: Precision={precision[hw]:.3f}, "
          f"Recall={recall[hw]:.3f}, F1={f1[hw]:.3f}, "
          f"Support={support[hw]}")

# 3. Confusion Matrix
cm = confusion_matrix(y_test, y_pred, labels=[0,1,2,3])
print("\nConfusion Matrix:")
print(cm)

# 4. Energy Prediction Error (if energy data is available)
if ye_test.sum() > 0:  # Check if energy data exists
    mae = np.mean(np.abs(energy_pred - ye_test))
    rmse = np.sqrt(np.mean((energy_pred - ye_test)**2))
    print(f"\n[Energy Prediction]: MAE={mae:.4f}, RMSE={rmse:.4f}")
else:
    print("\n[Energy Prediction]: No ground truth energy data available")

print("=" * 60)
