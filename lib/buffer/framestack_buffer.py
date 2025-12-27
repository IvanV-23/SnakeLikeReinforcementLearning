

from pyparsing import deque
import torch

from collections import deque
import torch

class FrameStackBuffer:
    def __init__(self, frame_stack=4):
        self.frame_stack = frame_stack
        self.frames = deque(maxlen=frame_stack)
    
    def reset(self):
        self.frames.clear()
    
    def push(self, frame):
        # Ensure frame is always (3,84,84) - remove batch dim if present
        if frame.dim() == 4:  # (1,3,84,84) → squeeze batch dim
            frame = frame.squeeze(0)
        assert frame.shape == (3, 84, 84), f"Expected (3,84,84), got {frame.shape}"
        self.frames.append(frame)
    
    def get_state(self, device):
        # Pad with black frames ON THE CORRECT DEVICE
        while len(self.frames) < self.frame_stack:
            black_frame = torch.zeros(3, 84, 84, device=device)
            self.frames.appendleft(black_frame)
        
        # Stack: (4, 3, 84, 84)
        stacked = torch.stack(list(self.frames), dim=0)
        # Channel stack: (12, 84, 84) ← NO BATCH DIM HERE
        stacked = stacked.transpose(0, 1).flatten(0, 1)
        # NO .unsqueeze(0)! Return (12,84,84) for single state
        return stacked.to(device)

