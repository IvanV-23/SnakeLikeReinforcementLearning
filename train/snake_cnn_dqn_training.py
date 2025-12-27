import math
import datetime

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from lib.evaluation.evaluate_cnn import evaluate_cnn
from lib.enviroment.navigation import NavigationEnv
from lib.net.conv_net import ConvNavNet

from lib.buffer.framestack_buffer import FrameStackBuffer
from lib.buffer.cnn_replay_buffer import CNNReplayBuffer
import random



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
env = NavigationEnv(training=True)
frame_stack = 4
policy_net = ConvNavNet(frame_stack).to(device)
target_net = ConvNavNet(frame_stack).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
writer = SummaryWriter("runs/snake_cnn_dqn/10000/2")

replay_buffer = CNNReplayBuffer(1000)
batch_size = 32  # Smaller batch for CNN
gamma = 0.99
target_update = 1000
epsilon_start = 1.0
epsilon_end   = 0.1
epsilon_decay = 20_000  # instead of 100_000

N_EPISODES = 10000
SAVE_FILTER_EVERY = 1000  # episodes
global_step = 0
frame_buffer = FrameStackBuffer(frame_stack)

for episode in range(N_EPISODES):

    state = env.reset_cnn()
    frame = torch.tensor(env.get_image_state()).to(device)      
    frame_buffer.reset()
    frame_buffer.push(frame)
    



    stacked_state = frame_buffer.get_state(device)  # (1,12,84,84)
    
    episode_reward = 0
    episode_length = 0
    done = False

    while not done:
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                math.exp(-global_step / epsilon_decay)
        # Epsilon-greedy
        if random.random() < epsilon:
            action = random.randint(0, 3)
        else:
            with torch.no_grad():
                q_values = policy_net(stacked_state.unsqueeze(0))[0]  # ← ONE LINE FIX
                action = torch.argmax(q_values).item()
        # Step env
        next_state_img, reward, done, info = env.cnn_step(action)

        episode_reward += reward
        episode_length += 1

        # Get next frame & stack
        next_frame = torch.tensor(next_state_img).to(device)
        frame_buffer.push(next_frame)
        next_stacked_state = frame_buffer.get_state(device)

        # Store TRANSITION (stacked states!)
        replay_buffer.push(stacked_state, action, reward, 
                            next_stacked_state, done)

        state = next_stacked_state
        global_step += 1

        # DQN Update (Double DQN!)
        if len(replay_buffer) >= batch_size:
            states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)
            
            # Current Q(s,a)
            q_values = policy_net(states)
            q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

            # DOUBLE DQN Target
            with torch.no_grad():
                next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                target = rewards + gamma * next_q_values * (1 - dones)

            loss = loss_fn(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            

            torch.cuda.empty_cache()
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/epsilon", epsilon, global_step)
      

        if global_step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

        if global_step % 5000 == 0:
            img = next_state_img  # (3,84,84) numpy
            print("Frame stats:", img.min(), img.max())
        if (episode + 1) % SAVE_FILTER_EVERY == 0:
            # conv[0] is nn.Conv2d(3*frame_stack, 32, 8, stride=4)
            conv1_weights = policy_net.conv[0].weight.data.cpu().clone()  # (32, C, 8, 8)
            torch.save(conv1_weights, f"conv1_filters_ep{episode+1}.pt")

        

    # Episode logging
    writer.add_scalar("episode/reward", episode_reward, episode)
    writer.add_scalar("episode/length", episode_length, episode)
    print(f"Ep {episode}: reward={episode_reward:.1f}, len={episode_length}, eps={epsilon:.3f}")
    # Maintain a Python list of episode rewards
    # At top of file, before loop:
    episode_rewards_history = []

    # Inside the loop, after computing episode_reward:
    episode_rewards_history.append(episode_reward)
    if len(episode_rewards_history) >= 50:
        avg_last_50 = sum(episode_rewards_history[-50:]) / 50.0
        writer.add_scalar("episode/reward_last_50", avg_last_50, episode)
    
    if (episode + 1) % 500 == 0:
        avg_eval = evaluate_cnn(policy_net, env, device, n_episodes=10)
        writer.add_scalar("eval/avg_reward_greedy", avg_eval, episode)
        print(f"[EVAL] Ep {episode}: greedy avg_reward={avg_eval:.2f}")
    
    print(f"Peak: {torch.cuda.max_memory_allocated()/1e6:.1f}MB")

# Save
torch.save(policy_net.state_dict(), "snake_cnn_dqn.pth")
print("Saved CNN-DQN model!")
writer.close()
