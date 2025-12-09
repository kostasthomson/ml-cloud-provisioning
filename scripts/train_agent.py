import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from scripts.utility_functions import *
from config import model_configuration as model_config
from entities import CloudTaskDataset
from models import EnergyAwareNN
from torch.utils.data import WeightedRandomSampler


def main(training_data_path: str = ""):
    """Main training function"""

    print("=" * 60)
    print("ENERGY-AWARE CLOUD RESOURCE ALLOCATION TRAINING")
    print("=" * 60)

    # Load and preprocess data
    X, y, df = load_and_preprocess_data(training_data_path if training_data_path != "" else model_config.DATA_PATH)

    # Compute sample weights (prefer low-energy decisions)
    df['sample_weight'] = 1.0 / (df['energy_kwh'] + 1e-6)
    df['sample_weight'] /= df['sample_weight'].sum()

    # Create weighted sampler
    sampler = WeightedRandomSampler(
        weights=df['sample_weight'].values,
        num_samples=len(df),
        replacement=True
    )

    # Split data
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )

    print(f"\\nDataset splits:")
    print(f"  Training:   {len(X_train)} samples")
    print(f"  Validation: {len(X_val)} samples")
    print(f"  Test:       {len(X_test)} samples")

    # Normalize features
    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    # Create datasets and dataloaders
    train_dataset = CloudTaskDataset(X_train, y_train)
    val_dataset = CloudTaskDataset(X_val, y_val)
    test_dataset = CloudTaskDataset(X_test, y_test)

    train_loader = DataLoader(train_dataset, sampler=sampler, batch_size=model_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=model_config.BATCH_SIZE)

    # Initialize model
    model = EnergyAwareNN(
        model_config.INPUT_SIZE,
        model_config.HIDDEN_SIZE,
        model_config.NUM_CLASSES
    ).to(model_config.DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)

    print(f"\\nModel architecture:")
    print(model)
    print(f"\\nTraining on: {model_config.DEVICE}")

    # Training loop
    train_losses = []
    val_losses = []
    train_accs = []
    val_accs = []

    best_val_acc = 0.0

    print(f"\\nStarting training for {model_config.NUM_EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(model_config.NUM_EPOCHS):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, model_config.DEVICE)
        val_loss, val_acc = validate(model, val_loader, criterion, model_config.DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)
        train_accs.append(train_acc)
        val_accs.append(val_acc)

        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'val_acc': val_acc
            }, 'best_energy_aware_model.pth')

        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{model_config.NUM_EPOCHS}] "
                  f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}% | "
                  f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")

    # Test evaluation
    test_loss, test_acc = validate(model, test_loader, criterion, model_config.DEVICE)
    print(f"\\nTest Accuracy: {test_acc:.2f}%")

    # Plot training curves
    plt.figure(figsize=(12, 4))

    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training and Validation Loss')

    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Acc')
    plt.plot(val_accs, label='Val Acc')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.title('Training and Validation Accuracy')

    plt.tight_layout()
    plt.savefig('training_curves.png')
    print(f"\\nTraining curves saved to training_curves.png")

    print(f"\\nBest validation accuracy: {best_val_acc:.2f}%")
    print("Model saved to best_energy_aware_model.pth")


if __name__ == "__main__":
    main()
