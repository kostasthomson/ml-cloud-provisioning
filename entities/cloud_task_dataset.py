import torch
from torch.utils.data import Dataset
import numpy as np


class CloudTaskDataset(Dataset):
    """PyTorch Dataset for cloud task allocation"""

    def __init__(self, features, labels):
        """
        Args:
            features: numpy array of shape (N, F) or torch tensor
            labels: numpy array of shape (N,) or torch tensor
        """
        # Ensure inputs are numpy arrays first
        if isinstance(features, torch.Tensor):
            features = features.numpy()
        if isinstance(labels, torch.Tensor):
            labels = labels.numpy()

        # Force contiguous copies
        features = np.ascontiguousarray(features, dtype=np.float32)
        labels = np.ascontiguousarray(labels, dtype=np.int64)

        # Convert to tensors (creates new memory)
        self.features = torch.from_numpy(features)
        self.labels = torch.from_numpy(labels)

        # Sanity check
        assert len(self.features) == len(self.labels), \
            f"Features ({len(self.features)}) and labels ({len(self.labels)}) must have same length"

        print(f"  CloudTaskDataset created: {len(self)} samples, "
              f"{self.features.shape[1]} features")

    def __len__(self):
        return len(self.features)

    def __getitem__(self, idx):
        """Get single sample by index"""
        if idx >= len(self):
            raise IndexError(f"Index {idx} out of bounds for dataset of size {len(self)}")

        return self.features[idx], self.labels[idx]
