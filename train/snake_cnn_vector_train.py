import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random

from lib.net.conv_net import ConvNavNet
from lib.buffer.cnn_replay_buffer import CNNReplayBuffer
from lib.enviroment.vec_snale_cnn_env import VecSnakeCNNEnv

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Hyperparameters
frame_stack = 4
n_envs = 4
vec_env = VecSnakeCNNEnv(n_envs=n_envs, frame_stack=frame_stack)

policy_net = ConvNavNet(frame_stack).to(device)
target_net = ConvNavNet(frame_stack).to(device)
target_net.load_state_dict(policy_net.state_dict())

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
writer = SummaryWriter("runs/snake_cnn_dqn_vec")

replay_buffer = CNNReplayBuffer(20_000)
batch_size = 16
gamma = 0.99
target_update = 100
epsilon_start, epsilon_end, epsilon_decay = 1.0, 0.05, 50_000

N_EPISODES = 5000
global_step = 0
episode_rewards_history = []

state = vec_env.reset(device)          # (N,12,84,84)

for episode in range(N_EPISODES):
    try:
        ep_rewards = torch.zeros(n_envs, device=device)

        frac = episode / N_EPISODES  # goes 0 → 1 over training
        epsilon = epsilon_start + (epsilon_end - epsilon_start) * frac 
        # run a fixed horizon per "meta-episode"
        for t in range(200):               # you can use MAX_STEPS here
            #epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
            #        torch.exp(torch.tensor(-global_step / epsilon_decay)).item()

            # epsilon-greedy per env
            if random.random() < epsilon:
                actions = torch.randint(0, 4, (n_envs,), device=device)
            else:
                with torch.no_grad():
                    q_batch = policy_net(state)        # (N,4)
                    actions = q_batch.argmax(dim=1)    # (N,)

            next_state, reward, done, infos = vec_env.step(actions, device)

            # store transitions from each env
            for i in range(n_envs):
                replay_buffer.push(
                    state[i].detach(),
                    int(actions[i].item()),
                    float(reward[i].item()),
                    next_state[i].detach(),
                    bool(done[i].item())
                )

            ep_rewards += reward
            state = next_state
            global_step += 1

            # DQN update
            if len(replay_buffer) >= batch_size:
                states_b, actions_b, rewards_b, next_states_b, dones_b = \
                    replay_buffer.sample(batch_size)

                q_values = policy_net(states_b)                       # (B,4)
                q_value = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

                with torch.no_grad():
                    next_actions = policy_net(next_states_b).max(1)[1].unsqueeze(1)
                    next_q_values = target_net(next_states_b).gather(1, next_actions).squeeze(1)
                    target = rewards_b + gamma * next_q_values * (1 - dones_b)

                loss = loss_fn(q_value, target)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/epsilon", epsilon, global_step)

            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # log mean reward across envs
        mean_ep_reward = ep_rewards.mean().item()
        episode_rewards_history.append(mean_ep_reward)
        writer.add_scalar("episode/reward_vec", mean_ep_reward, episode)
        print(f"Ep {episode}: mean_reward={mean_ep_reward:.2f}, eps={epsilon:.3f}")

        if len(episode_rewards_history) >= 50:
            avg_last_50 = sum(episode_rewards_history[-50:]) / 50.0
            writer.add_scalar("episode/reward_last_50_vec", avg_last_50, episode)
    except Exception as exc:
        print(f"Exception during training at episode {episode}: {exc}")
        break
# save model
torch.save(policy_net.state_dict(), "snake_cnn_dqn_vec.pth")
writer.close()
print("Saved vectorized CNN-DQN model!")
