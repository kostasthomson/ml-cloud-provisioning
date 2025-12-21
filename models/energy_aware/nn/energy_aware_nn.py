from pathlib import Path
import torch.nn as nn


class EnergyAwareNN(nn.Module):
    parent_directory: str = '/'.join(str(Path(__file__).parent).split('\\'))

    def __init__(self, input_size, hidden_size, output_size=1):
        super(EnergyAwareNN, self).__init__()
        self.directory = EnergyAwareNN.parent_directory
        self.input_size = input_size
        self.is_regression = (output_size == 1)

        if input_size >= 20:
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),

                nn.Linear(hidden_size, hidden_size),
                nn.BatchNorm1d(hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),

                nn.Linear(hidden_size, hidden_size // 2),
                nn.BatchNorm1d(hidden_size // 2),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),

                nn.Linear(hidden_size // 2, hidden_size // 4),
                nn.LeakyReLU(0.1),

                nn.Linear(hidden_size // 4, output_size)
            )
        else:
            self.network = nn.Sequential(
                nn.Linear(input_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),

                nn.Linear(hidden_size, hidden_size),
                nn.LeakyReLU(0.1),
                nn.Dropout(0.1),

                nn.Linear(hidden_size, hidden_size // 2),
                nn.LeakyReLU(0.1),

                nn.Linear(hidden_size // 2, output_size)
            )

    def forward(self, x):
        return self.network(x)
