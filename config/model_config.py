import torch


class ModelConfiguration:
    # Data
    DATA_PATH = "../training/training_data.csv"
    # Model architecture
    INPUT_SIZE = 12  # Task features (5) + Cell state features (6) + timing (1)
    HIDDEN_SIZE = 64
    NUM_CLASSES = 4  # HW types 1-4

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 100
    VALIDATION_SPLIT = 0.2

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_configuration = ModelConfiguration()
