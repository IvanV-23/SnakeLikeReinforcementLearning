import torch
import torch.nn as nn

class NavNet(nn.Module):
    """A simple feedforward neural network for navigation tasks."""
    def __init__(self):
        super().__init__()
        
        # Input (10 state features) → Hidden 1
        self.fc1 = nn.Linear(10, 64)      # Increased from 16
        # Hidden 1 → Hidden 2  
        self.fc2 = nn.Linear(64, 32)     # NEW middle layer
        # Hidden 2 → Output (4 actions)
        self.fc3 = nn.Linear(32, 4)      # Renamed from fc2
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))      # Hidden 1 activation
        x = torch.relu(self.fc2(x))      # NEW: Hidden 2 activation
        x = self.fc3(x)                  # Output (Q-values)
        return x
