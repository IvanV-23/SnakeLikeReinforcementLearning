import random
from collections import deque
import torch

class CNNReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((
            state.cpu(),           # ← CPU! (was GPU)
            action, 
            reward, 
            next_state.cpu(),      # ← CPU! (was GPU)
            done
        ))

    def sample(self, batch_size,device='cuda'):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Stack images
        states = torch.stack([s[0] for s in batch]).to(device)  # Fresh GPU each time ✅
        next_states = torch.stack([s[3] for s in batch]).to(device)  # Fresh GPU ✅
        
        # Scalars → CPU first → then to device
        actions = torch.tensor(actions, dtype=torch.long).to(states.device)
        rewards = torch.tensor(rewards, dtype=torch.float32).to(states.device)
        dones = torch.tensor(dones, dtype=torch.float32).to(states.device)
        
        return states, actions, rewards, next_states, dones

    def __len__(self):
        return len(self.buffer)
