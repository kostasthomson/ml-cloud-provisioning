import numpy as np
import torch
from sklearn.metrics import confusion_matrix, classification_report


def train_epoch(model, dataloader, criterion, optimizer, device):
    """Train for one epoch"""

    model.train()
    total_loss = 0.0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        # Forward pass
        outputs = model(features)
        loss = criterion(outputs, labels)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # Metrics
        total_loss += loss.item()
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

    avg_loss = total_loss / len(dataloader)
    accuracy = 100 * correct / total

    return avg_loss, accuracy


def validate(model, dataloader, criterion, device):
    """Validate the model"""

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


def analyze_feature_importance(model, X_test, y_test, feature_names, device):
    """Analyze which features the model relies on"""

    print("\n" + "=" * 80)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("=" * 80)

    # Convert test data to tensor
    X_tensor = torch.FloatTensor(X_test).to(device)
    y_tensor = torch.LongTensor(y_test).to(device)

    # Get model predictions
    model.eval()
    with torch.no_grad():
        outputs = model(X_tensor)
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()

    # Check perfect features (features that perfectly predict output)
    for i, feature_name in enumerate(feature_names):
        print(i, feature_name)
        feature_values = X_test[:, i]

        # Check if this feature alone predicts the output
        unique_combos = {}
        for val, pred in zip(feature_values, predictions):
            val_rounded = round(val, 4)  # Round to avoid float precision issues
            if val_rounded not in unique_combos:
                unique_combos[val_rounded] = pred
            elif unique_combos[val_rounded] != pred:
                unique_combos[val_rounded] = -1  # Multiple predictions for same value

        # If all unique values map to consistent predictions, this feature is perfect
        perfect_predictor = all(v != -1 for v in unique_combos.values())

        if perfect_predictor and len(unique_combos) <= 10:
            print(f"⚠️  {feature_name} might be a PERFECT PREDICTOR!")
            print(f"   Unique values → HW types mapping:")
            for val, hw_type in sorted(unique_combos.items())[:10]:
                print(f"     {val:.4f} → HW Type {hw_type + 1}")

    print("=" * 80)


def analyze_predictions(model, test_loader, device):
    """Detailed prediction analysis"""

    model.eval()
    all_predictions = []
    all_labels = []

    with torch.no_grad():
        for features, labels in test_loader:
            features = features.to(device)
            outputs = model(features)
            predictions = torch.argmax(outputs, dim=1)

            all_predictions.extend(predictions.cpu().numpy())
            all_labels.extend(labels.numpy())

    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)

    print("\n" + "=" * 80)
    print("PREDICTION ANALYSIS")
    print("=" * 80)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    print("\nConfusion Matrix:")
    print("(Rows: True labels, Columns: Predicted labels)")
    print("\n       Type1  Type2  Type3  Type4")
    for i, row in enumerate(cm):
        print(f"Type{i + 1}: {row}")

    # Classification report
    print("\nClassification Report:")
    print(classification_report(
        all_labels,
        all_predictions,
        target_names=['HW Type 1', 'HW Type 2', 'HW Type 3', 'HW Type 4']
    ))

    # Check if model always predicts same class
    unique_predictions = np.unique(all_predictions)
    print(f"\nUnique predictions: {unique_predictions + 1}")  # +1 for 1-indexed

    if len(unique_predictions) == 1:
        print(f"⚠️  WARNING: Model only predicts HW Type {unique_predictions[0] + 1}!")

    print("=" * 80 + "\n")
