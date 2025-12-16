from pathlib import Path

import torch.nn as nn


class EnergyAwareNN(nn.Module):
    """Neural network for energy-aware HW type prediction"""
    parent_directory: str = '/'.join(str(Path(__file__).parent).split('\\'))

    def __init__(self, input_size, hidden_size, num_classes):
        super(EnergyAwareNN, self).__init__()
        self.directory = EnergyAwareNN.parent_directory

        self.network = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(0.3),

            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(0.2),

            nn.Linear(hidden_size // 2, num_classes)
        )

    def forward(self, x):
        return self.network(x)
