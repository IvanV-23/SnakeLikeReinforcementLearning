import torch
import torch.nn as nn

class NavNet(nn.Module):
    """A simple feedforward neural network for navigation tasks.

    Args:
        nn (Module): PyTorch neural network module.
    """
    def __init__(self):
        super().__init__()

        self.fc1 = nn.Linear(6, 16)
        self.fc2 = nn.Linear(16, 4)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x
