import torch


class ModelConfiguration:
    # Data
    DATA_PATH = "../training/training_data.csv"

    # Model architecture
    # INPUT_SIZE is determined dynamically based on available features
    # Base features: 5 task + 8 state + derived = ~13 (old format)
    # Enhanced features: 5 task + 22 state + derived = ~30 (new format)
    INPUT_SIZE = None  # Set dynamically during training
    HIDDEN_SIZE = 128  # Increased for more features
    NUM_CLASSES = 4  # HW types 1-4

    # Training
    BATCH_SIZE = 64
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    VALIDATION_SPLIT = 0.2

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_configuration = ModelConfiguration()
