import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import numpy as np
from utility_functions.data_loading_preprocessing import load_and_preprocess_data, normalize_features
from utility_functions.training_functions import train_epoch_regression, validate_regression, analyze_regression_predictions
from config import model_configuration as model_config
from models import EnergyAwareNN


def main(training_data_path: str = "training/training_data.csv"):
    print("=" * 60)
    print("ENERGY PREDICTION MODEL TRAINING")
    print("=" * 60)

    X, y, df, feature_names = load_and_preprocess_data(
        training_data_path if training_data_path != "" else model_config.DATA_PATH
    )

    input_size = X.shape[1]
    print(f"Input features: {input_size}")
    print(f"Features: {feature_names}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)

    print(f"Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")

    X_train, X_val, X_test, scaler = normalize_features(X_train, X_val, X_test)

    train_dataset = TensorDataset(
        torch.FloatTensor(X_train),
        torch.FloatTensor(y_train)
    )
    val_dataset = TensorDataset(
        torch.FloatTensor(X_val),
        torch.FloatTensor(y_val)
    )
    test_dataset = TensorDataset(
        torch.FloatTensor(X_test),
        torch.FloatTensor(y_test)
    )

    train_loader = DataLoader(train_dataset, batch_size=model_config.BATCH_SIZE, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=model_config.BATCH_SIZE)
    test_loader = DataLoader(test_dataset, batch_size=model_config.BATCH_SIZE)

    model = EnergyAwareNN(input_size, model_config.HIDDEN_SIZE, output_size=1).to(model_config.DEVICE)

    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=model_config.LEARNING_RATE)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=10)

    print(f"Model: {model}")
    print(f"Device: {model_config.DEVICE}")

    train_losses = []
    val_losses = []
    best_val_loss = float('inf')

    print(f"Training for {model_config.NUM_EPOCHS} epochs...")
    print("-" * 60)

    for epoch in range(model_config.NUM_EPOCHS):
        train_loss, train_rmse = train_epoch_regression(model, train_loader, criterion, optimizer, model_config.DEVICE)
        val_loss, val_rmse, val_mae, val_mape = validate_regression(model, val_loader, criterion, model_config.DEVICE)

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        scheduler.step(val_loss)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scaler': scaler,
                'val_loss': val_loss,
                'input_size': input_size,
                'hidden_size': model_config.HIDDEN_SIZE,
                'feature_names': feature_names
            }, f"{model.directory}/model.pth")

        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch+1}/{model_config.NUM_EPOCHS} | Train RMSE: {train_rmse:.6f} | Val RMSE: {val_rmse:.6f} | Val MAE: {val_mae:.6f}")

    print("-" * 60)

    test_loss, test_rmse, test_mae, test_mape = validate_regression(model, test_loader, criterion, model_config.DEVICE)
    print(f"Test RMSE: {test_rmse:.6f} kWh")
    print(f"Test MAE: {test_mae:.6f} kWh")
    print(f"Test MAPE: {test_mape:.2f}%")

    df_test = df.iloc[-len(y_test):].reset_index(drop=True)
    analyze_regression_predictions(model, test_loader, model_config.DEVICE, df_test)

    plt.figure(figsize=(10, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train')
    plt.plot(val_losses, label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.title('Training Loss')

    plt.subplot(1, 2, 2)
    plt.plot(np.sqrt(train_losses), label='Train')
    plt.plot(np.sqrt(val_losses), label='Val')
    plt.xlabel('Epoch')
    plt.ylabel('RMSE (kWh)')
    plt.legend()
    plt.title('RMSE over Training')

    plt.tight_layout()
    plt.savefig(f"{model.directory}/training_curves.png")
    print(f"Saved training curves to {model.directory}/training_curves.png")

    print(f"Best model saved to {model.directory}/model.pth")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('data_path', nargs='?', default='../training/training_data.csv')
    args = parser.parse_args()
    main(args.data_path)
