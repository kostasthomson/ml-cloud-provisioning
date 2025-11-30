import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from models.nn.neural_network import NeuralNetwork
import pickle

print("Loading dataset...")
df = pd.read_csv("training/training_data_v1.csv")

# Filter to accepted tasks only
df = df[df['accepted'] == 1].copy()

# REMAP: 1,2,3,4 → 0,1,2,3
print(f"Original HW Type range: {df['chosen_hw_type'].min()}-{df['chosen_hw_type'].max()}")
df['chosen_hw_type'] = df['chosen_hw_type'] - 1
print(f"Remapped for PyTorch: {df['chosen_hw_type'].min()}-{df['chosen_hw_type'].max()}")

# Features
feature_cols = ['num_vms', 'cpu_req', 'mem_req'] + \
               [c for c in df.columns if c.startswith('cell')]

X = df[feature_cols].values
y = df['chosen_hw_type'].values  # 0-3
y_energy = df['energy_cost'].values

print(f"Input Features: {X.shape[1]}, Output Classes: {len(df['chosen_hw_type'].unique())}")

# Split
X_train, X_test, y_train, y_test, ye_train, ye_test = train_test_split(
    X, y, y_energy, test_size=0.2, random_state=42
)

# Scale
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convert to PyTorch tensors
X_train_t = torch.FloatTensor(X_train)
y_train_t = torch.LongTensor(y_train)
ye_train_t = torch.FloatTensor(ye_train).unsqueeze(1)

X_test_t = torch.FloatTensor(X_test)
y_test_t = torch.LongTensor(y_test)
ye_test_t = torch.FloatTensor(ye_test).unsqueeze(1)

train_ds = TensorDataset(X_train_t, y_train_t, ye_train_t)
test_ds = TensorDataset(X_test_t, y_test_t, ye_test_t)

train_loader = DataLoader(train_ds, batch_size=64, shuffle=True)
test_loader = DataLoader(test_ds, batch_size=64, shuffle=False)

# Model
num_classes = len(df['chosen_hw_type'].unique())
model = NeuralNetwork(input_dim=X.shape[1], num_classes=num_classes)

# Training
crit_class = nn.CrossEntropyLoss()
crit_energy = nn.MSELoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

EPOCHS = 50
print("Starting training...")
for epoch in range(EPOCHS):
    model.train()
    for xb, yb, yeb in train_loader:
        optimizer.zero_grad()
        pred_logits, pred_energy = model(xb)

        loss_cls = crit_class(pred_logits, yb)
        loss_energy = crit_energy(pred_energy, yeb)
        loss = loss_cls + 0.1 * loss_energy

        loss.backward()
        optimizer.step()

    # Validation
    if epoch % 5 == 0:
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for xb, yb, yeb in test_loader:
                pred_logits, _ = model(xb)
                preds = torch.argmax(pred_logits, dim=1)
                correct += (preds == yb).sum().item()
                total += yb.size(0)

        val_acc = 100.0 * correct / total
        print(f"Epoch {epoch}: Loss={loss.item():.4f}, Val Acc={val_acc:.2f}%")

print("Saving model...")
torch.save(model.state_dict(), f"{model.directory}/model.pth")
with open(f"{model.directory}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

print("Done! Ready for integration.")
