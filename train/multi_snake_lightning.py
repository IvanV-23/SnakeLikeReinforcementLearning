import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
import numpy as np
import copy
import random
from collections import deque, namedtuple
from torch.utils.data import IterableDataset, DataLoader

# 1. Experience Structure
Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

# 2. Replay Memory
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# 3. RL Dataset Bridge
class RLDataset(IterableDataset):
    def __init__(self, buffer, sample_size=128):
        self.buffer = buffer
        self.sample_size = sample_size

    def __iter__(self):
        # Continuous stream for the Lightning trainer
        while True:
            if len(self.buffer) >= self.sample_size:
                batch = self.buffer.sample(self.sample_size)
                for i in range(self.sample_size):
                    # Yielding as float32 tensors for the CNN
                    yield (
                        torch.tensor(batch[i].state, dtype=torch.float32),
                        torch.tensor(batch[i].action, dtype=torch.long),
                        torch.tensor(batch[i].next_state, dtype=torch.float32),
                        torch.tensor(batch[i].reward, dtype=torch.float32),
                        torch.tensor(batch[i].done, dtype=torch.float32)
                    )

# 4. Lightning System
class MultiSnakeLightning(pl.LightningModule):
    def __init__(self, model, lr=1e-4, gamma=0.99, batch_size=128, target_update_freq=1000):
        super().__init__()
        # ignore=['model'] prevents Lightning from trying to serialize the whole network into a config file
        self.save_hyperparameters(ignore=['model'])
        
        self.model = model
        self.target_net = copy.deepcopy(model)
        self.buffer = ReplayBuffer(100_000)
        
        # Metrics trackers
        self.epsilon = 1.0
        self.total_reward = 0

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        states, actions, next_states, rewards, dones = batch
        
        # Q(s, a) - Current predictions
        # gather(1, ...) selects the Q-value for the action that was actually taken
        current_q_values = self.model(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        
        # max Q(s', a') - Target predictions from the frozen network
        with torch.no_grad():
            next_q_values = self.target_net(next_states).max(1)[0]
            # Bellman Equation: r + gamma * max Q(s') * (1 - done)
            expected_q_values = rewards + (self.hparams.gamma * next_q_values * (1 - dones))
            
        # Compute Huber loss (SmoothL1) which is more robust to outliers in RL
        loss = nn.functional.smooth_l1_loss(current_q_values, expected_q_values)
        
        # Log to TensorBoard/MLFlow
        self.log("train/loss", loss, prog_bar=True)
        self.log("train/epsilon", self.epsilon)
        
        return loss

    def on_train_batch_end(self, outputs, batch, batch_idx):
        # Periodic update of the target network to stabilize training
        if self.global_step % self.hparams.target_update_freq == 0:
            self.target_net.load_state_dict(self.model.state_dict())

    def train_dataloader(self):
        dataset = RLDataset(self.buffer, self.hparams.batch_size)
        return DataLoader(
            dataset=dataset,
            batch_size=self.hparams.batch_size,
            num_workers=0, # Mandatory for RL to avoid buffer synchronization issues
            pin_memory=True # Speeds up data transfer to GPU
        )

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.hparams.lr)