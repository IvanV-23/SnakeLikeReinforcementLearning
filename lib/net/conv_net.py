

import torch.nn as nn

class ConvNavNet(nn.Module):
    def __init__(self, frame_stack=4):
        super().__init__()
        self.frame_stack = frame_stack
        
        self.conv = nn.Sequential(
            nn.Conv2d(3 * frame_stack, 32, 8, stride=4),  # 84→20x20
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),               # 20→9x9 (not 8x8!)
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),               # 9→7x7 (not 6x6!)
            nn.ReLU(),
            nn.Flatten()
        )
        
        # FIXED: Correct conv output size
        # Conv math: 84→20→9→7 → 7*7*64 = 3136
        self.fc = nn.Sequential(
            nn.Linear(3136, 512),  # ← 2304 → 3136
            nn.ReLU(),
            nn.Linear(512, 4)
        )
    
    def forward(self, x):
        conv_out = self.conv(x)      # (B, 3136)
        q_values = self.fc(conv_out) # (B, 4)
        return q_values
