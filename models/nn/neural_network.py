from pathlib import Path
import torch.nn as nn


class NeuralNetwork(nn.Module):
    parent_directory: str = '/'.join(str(Path(__file__).parent).split('\\')[-2:])

    def __init__(self, input_dim, num_classes):
        super(NeuralNetwork, self).__init__()
        self.directory = NeuralNetwork.parent_directory

        # Shared Layers (Feature Extractor)
        self.shared = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Linear(128, 64),
            nn.ReLU()
        )

        # Head 1: Classification (Which HW Type?)
        self.classifier = nn.Linear(64, num_classes)

        # Head 2: Energy Prediction (Value Function)
        self.estimator = nn.Linear(64, 1)

    def forward(self, x):
        features = self.shared(x)
        logits = self.classifier(features)
        energy = self.estimator(features)
        return logits, energy
