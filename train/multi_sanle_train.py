import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from collections import deque, namedtuple
import pygame


from lib.enviroment.multy_snake_env import MultiSnakeEnv
from lib.net.multi_agent_conv_nav_net import ConvNavNet


# ---------- Training setup ----------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

BATCH_SIZE = 128
GAMMA = 0.99
EPS_START = 1.0
EPS_END = 0.05
EPS_DECAY = 800_000  # ~5000 episodes * 200 steps
TARGET_UPDATE = 500
TRAIN_FREQ = 2
REPLAY_SIZE = 100_000
LEARNING_RATE = 1e-4

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward', 'done'))

class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, *args):
        self.buffer.append(Transition(*args))
    
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)

# ---------- Training functions ----------
def select_actions(q_net, obs_dict, epsilon, device):
    """Select actions for all agents using shared Q-network."""
    obs_list = []
    agent_keys = []
    
    for key, obs in obs_dict.items():
        # obs is already (3,H,W) numpy -> make (1,3,H,W) tensor
        obs_tensor = torch.FloatTensor(obs).unsqueeze(0).to(device)
        obs_list.append(obs_tensor)
        agent_keys.append(key)
    
    if not obs_list:  # no alive agents
        return {}
    
    batch_obs = torch.cat(obs_list, dim=0)  # (n_agents, 3, H, W)
    
    actions = {}
    with torch.no_grad():
        q_values = q_net(batch_obs)  # (n_agents, 4)
    
    for i, key in enumerate(agent_keys):
        if random.random() < epsilon:
            actions[key] = random.randrange(4)
        else:
            actions[key] = q_values[i].argmax().item()
    
    return actions

def optimize_dqn(q_net, target_net, replay_buffer, optimizer, batch_size, gamma):
    if len(replay_buffer) < batch_size:
        return
    
    transitions = replay_buffer.sample(batch_size)
    batch = Transition(*zip(*transitions))
    
    # Convert states to proper (B, 3, H, W) tensors
    state_batch = torch.cat([
        torch.FloatTensor(s).unsqueeze(0).to(device)  # (1,3,H,W)
        for s in batch.state
    ])  # (B, 3, H, W)
    
    next_state_batch = []
    done_batch = []
    for ns, d in zip(batch.next_state, batch.done):
        if ns is not None:
            next_state_batch.append(torch.FloatTensor(ns).unsqueeze(0).to(device))
            done_batch.append(float(d))
        else:
            next_state_batch.append(None)
            done_batch.append(float(d))
    
    action_batch = torch.cat([
        torch.tensor([[a]], dtype=torch.int64, device=device)
        for a in batch.action
    ])
    
    reward_batch = torch.tensor([float(r) for r in batch.reward], device=device, dtype=torch.float32)
    done_tensor = torch.tensor(done_batch, device=device, dtype=torch.float32)
    
    # Q(s, a)
    state_action_values = q_net(state_batch).gather(1, action_batch)
    
    # Q(s', a') from target net
    next_state_values = torch.zeros(batch_size, device=device)
    non_final_next_states = torch.cat([ns for ns in next_state_batch if ns is not None])
    
    if len(non_final_next_states) > 0:
        with torch.no_grad():
            next_state_values[:len(non_final_next_states)] = target_net(non_final_next_states).max(1)[0]
    
    # TD target: r + γ max Q(s', a') * (1-done)
    expected_state_action_values = reward_batch + (next_state_values * gamma * (1 - done_tensor))
    
    # Loss
    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values.squeeze(1), expected_state_action_values)
    
    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(q_net.parameters(), 1.0)
    optimizer.step()

# ---------- Main training loop ----------
def train_multi_snake():
    env = MultiSnakeEnv(n_agents=2, width=10, height=10, training=True)
    
    q_net = ConvNavNet().to(device)
    target_net = ConvNavNet().to(device)
    target_net.load_state_dict(q_net.state_dict())
    
    optimizer = optim.Adam(q_net.parameters(), lr=LEARNING_RATE)
    replay_buffer = ReplayBuffer(REPLAY_SIZE)
    
    global_step = 0
    episode_rewards = []
    episode_lengths = []
    
    #plt.ion()
    #fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    for episode in range(1000):
        obs = env.reset_multi()
        episode_reward = 0
        done = False
        
        while not done:
            epsilon = EPS_END + (EPS_START - EPS_END) * \
                     np.exp(-1. * global_step / EPS_DECAY)
            
            # Select actions for all agents
            actions = select_actions(q_net, obs, epsilon, device)
            
            # Environment step
            next_obs, reward, done, info = env.step_multi(actions)
            episode_reward += reward
            
            # Store transitions for each agent
            # In training loop, replace the storage block:
            for i in range(env.n_agents):
                agent_key = f"agent_{i}"
                if agent_key in obs:  # agent still exists in obs dict
                    replay_buffer.push(
                        obs[agent_key].copy(),           # numpy array
                        actions[agent_key],
                        next_obs[agent_key].copy(),      # numpy array
                        float(reward),                   # scalar
                        float(done)
                    )
            
            obs = next_obs
            global_step += 1
            
            # Training step
            if global_step % TRAIN_FREQ == 0:
                optimize_dqn(q_net, target_net, replay_buffer, 
                           optimizer, BATCH_SIZE, GAMMA)
            
            # Target network update
            if global_step % TARGET_UPDATE == 0:
                target_net.load_state_dict(q_net.state_dict())
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(env.steps)
        
        # Logging
        if episode % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode},Score {env.score:.2f}, Avg Reward: {avg_reward:.2f}, "
                  f"Avg Length: {avg_length:.1f}, Epsilon: {epsilon:.3f}, "
                  f"Buffer: {len(replay_buffer)}")
            
            # Plotting
            #ax1.clear()
            #ax1.plot(episode_rewards)
            #ax1.set_title('Episode Rewards')
            #ax1.set_xlabel('Episode')
            
            #ax2.clear()
            #ax2.plot(episode_lengths)
            #ax2.set_title('Episode Lengths')
            #ax2.set_xlabel('Episode')
            
            #plt.pause(0.01)
    
    #plt.ioff()
    #plt.show()
    torch.save(q_net.state_dict(), 'multi_snake_dqn.pth')
    print("Training complete! Model saved as 'multi_snake_dqn.pth'")

if __name__ == "__main__":
    pygame.init()  # Needed for rendering
    train_multi_snake()
