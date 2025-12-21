import numpy as np
import torch


def train_epoch_regression(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    total_samples = 0

    for features, targets in dataloader:
        features = features.to(device)
        targets = targets.to(device).float().unsqueeze(1)

        outputs = model(features)
        loss = criterion(outputs, targets)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        total_samples += features.size(0)

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(avg_loss)
    return avg_loss, rmse


def validate_regression(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    total_samples = 0
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in dataloader:
            features = features.to(device)
            targets = targets.to(device).float().unsqueeze(1)

            outputs = model(features)
            loss = criterion(outputs, targets)

            total_loss += loss.item() * features.size(0)
            total_samples += features.size(0)

            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.cpu().numpy().flatten())

    avg_loss = total_loss / total_samples
    rmse = np.sqrt(avg_loss)

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)
    mae = np.mean(np.abs(all_preds - all_targets))
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

    return avg_loss, rmse, mae, mape


def analyze_regression_predictions(model, test_loader, device, df_test=None):
    model.eval()
    all_preds = []
    all_targets = []

    with torch.no_grad():
        for features, targets in test_loader:
            features = features.to(device)
            outputs = model(features)
            all_preds.extend(outputs.cpu().numpy().flatten())
            all_targets.extend(targets.numpy())

    all_preds = np.array(all_preds)
    all_targets = np.array(all_targets)

    print("\nREGRESSION ANALYSIS")
    print("=" * 60)

    rmse = np.sqrt(np.mean((all_preds - all_targets) ** 2))
    mae = np.mean(np.abs(all_preds - all_targets))
    mape = np.mean(np.abs((all_targets - all_preds) / (all_targets + 1e-8))) * 100

    ss_res = np.sum((all_targets - all_preds) ** 2)
    ss_tot = np.sum((all_targets - np.mean(all_targets)) ** 2)
    r2 = 1 - (ss_res / (ss_tot + 1e-8))

    print(f"RMSE: {rmse:.6f} kWh")
    print(f"MAE: {mae:.6f} kWh")
    print(f"MAPE: {mape:.2f}%")
    print(f"R2 Score: {r2:.4f}")

    print(f"\nTarget range: {all_targets.min():.6f} to {all_targets.max():.6f}")
    print(f"Prediction range: {all_preds.min():.6f} to {all_preds.max():.6f}")

    if df_test is not None and 'chosen_hw_type' in df_test.columns:
        hw_types = df_test['chosen_hw_type'].values[:len(all_preds)]
        print("\nPer HW Type Performance:")
        for t in range(1, 5):
            mask = hw_types == t
            if mask.sum() > 0:
                t_rmse = np.sqrt(np.mean((all_preds[mask] - all_targets[mask]) ** 2))
                t_mae = np.mean(np.abs(all_preds[mask] - all_targets[mask]))
                print(f"  Type {t}: RMSE={t_rmse:.6f}, MAE={t_mae:.6f}, n={mask.sum()}")

    print("=" * 60)

    return rmse, mae, r2


def train_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)
        outputs = model(features)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    model.eval()
    total_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)
            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total
    return avg_loss, accuracy
