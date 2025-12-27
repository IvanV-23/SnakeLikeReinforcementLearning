import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
import random
import numpy as np

from lib.enviroment.vector_enviroment import VecNavigationEnv
from lib.net.nav_net import NavNet
from lib.buffer.replay_buffer import ReplayBuffer
from lib.evaluation.single_evaluation import single_evaluate

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

n_envs = 4
vec_env = VecNavigationEnv(n_envs=n_envs)

policy_net = NavNet().to(device)
target_net = NavNet().to(device)
target_net.load_state_dict(policy_net.state_dict())
target_net.eval()

optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
loss_fn = nn.MSELoss()
writer = SummaryWriter("runs/snake_nav_vec")

buffer_capacity = 100_000
batch_size = 128
gamma = 0.99
target_update = 2000
epsilon_start = 1.0
epsilon_end = 0.05
epsilon_decay = 100_000
N_EPISODES = 1000
EVAL_EVERY = 50   # episodes

replay_buffer = ReplayBuffer(buffer_capacity)
global_step = 0

for episode in range(N_EPISODES):
    state = vec_env.reset()  # (n_envs, obs_dim)
    episode_rewards = np.zeros(n_envs, dtype=float)
    episode_lengths = np.zeros(n_envs, dtype=int)

    # Max steps per "meta-episode"
    for _ in range(500):
        epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                    torch.exp(torch.tensor(-global_step / epsilon_decay)).item()

        state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
        with torch.no_grad():
            q_batch = policy_net(state_tensor)  # (B, 4)

        actions = []
        for i in range(n_envs):
            if random.random() < epsilon:
                a = random.randint(0, 3)
            else:
                a = torch.argmax(q_batch[i]).item()
            actions.append(a)

        next_state, reward, done, infos = vec_env.step(actions)

        # store transitions for each env
        for i in range(n_envs):
            replay_buffer.push(state[i], actions[i], reward[i], next_state[i], done[i])
            episode_rewards[i] += reward[i]
            episode_lengths[i] += 1

        state = next_state
        global_step += 1

        # --- DQN update ---
        if len(replay_buffer) >= batch_size:
            states_b, actions_b, rewards_b, next_states_b, dones_b = replay_buffer.sample(batch_size)

            states_b      = states_b.to(device)
            actions_b     = actions_b.to(device)
            rewards_b     = rewards_b.to(device)
            next_states_b = next_states_b.to(device)
            dones_b       = dones_b.to(device)

            q_values = policy_net(states_b)                      # (B,4)
            q_value = q_values.gather(1, actions_b.unsqueeze(1)).squeeze(1)

            #with torch.no_grad():
            #    next_q_values = target_net(next_states_b)
            #    max_next_q = next_q_values.max(1)[0]
            #    target = rewards_b + gamma * max_next_q * (1 - dones_b)

            with torch.no_grad():
                next_actions = policy_net(next_states_b).max(1)[1].unsqueeze(1)
                next_q_values = target_net(next_states_b).gather(1, next_actions).squeeze(1)
                target = rewards_b + gamma * next_q_values * (1 - dones_b)


            loss = loss_fn(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/reward_step", rewards_b.mean().item(), global_step)

        if global_step % target_update == 0:
            target_net.load_state_dict(policy_net.state_dict())

    # log mean over parallel envs
    avg_reward = episode_rewards.mean()
    avg_length = episode_lengths.mean()
    writer.add_scalar("episode/reward", avg_reward, episode)
    writer.add_scalar("episode/length", avg_length, episode)
    print(f"Episode {episode} | avg_reward={avg_reward:.2f} | avg_length={avg_length:.1f} | eps={epsilon:.3f}")

    # periodic evaluation (single-env, greedy)
    #if (episode + 1) % EVAL_EVERY == 0:
    #    single_evaluate(policy_net, writer, global_step, n_episodes=5, device=device)

# -------- Save model --------
torch.save(policy_net.state_dict(), "snake_model_dqn_vec_double_dqn.pth")
print("Saved model to snake_model_dqn_vec.pth")

writer.close()
