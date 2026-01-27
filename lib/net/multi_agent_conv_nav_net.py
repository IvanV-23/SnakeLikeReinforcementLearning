
import torch.nn as nn
import torch.nn.functional as F


class ConvNavNet(nn.Module):
    def __init__(self, n_actions=4):
        super().__init__()
        # Optimized for small grids (e.g., 84x84 scaled from 10x10)
        self.conv = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 42x42
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2), # 21x21
            nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
        )
        
        # Dueling Head
        self.fc_adv = nn.Sequential(nn.Linear(64 * 21 * 21, 512), nn.ReLU(), nn.Linear(512, n_actions))
        self.fc_val = nn.Sequential(nn.Linear(64 * 21 * 21, 512), nn.ReLU(), nn.Linear(512, 1))

    def forward(self, x):
        x = self.conv(x)
        x = x.view(x.size(0), -1)
        adv = self.fc_adv(x)
        val = self.fc_val(x)
        return val + adv - adv.mean(dim=1, keepdim=True)
