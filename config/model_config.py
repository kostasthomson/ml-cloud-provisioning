import torch


class ModelConfiguration:
    # Data
    DATA_PATH = "../training/training_data.csv"
    # Model architecture
    INPUT_SIZE = 13
    HIDDEN_SIZE = 64
    NUM_CLASSES = 4  # HW types 1-4

    # Training
    BATCH_SIZE = 32
    LEARNING_RATE = 0.001
    NUM_EPOCHS = 10
    VALIDATION_SPLIT = 0.2

    # Hardware
    DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


model_configuration = ModelConfiguration()
