import random
from sympy import evaluate
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



from lib.enviroment.navigation import NavigationEnv
from lib.net.nav_net import NavNet
from lib.buffer.replay_buffer import ReplayBuffer
from lib.evaluation import evaluate

if __name__ == "__main__":
    env = NavigationEnv(training=True)
    
    # env.init_pygame()

    # Online / policy network
    policy_net = NavNet()
    # Target network
    target_net = NavNet()
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval()

    optimizer = optim.Adam(policy_net.parameters(), lr=1e-4)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter("runs/snake_nav")

    # DQN hyperparameters
    gamma = 0.99
    buffer_capacity = 50_000
    batch_size = 64
    target_update = 1000      # steps between target sync
    epsilon_start = 1.0
    epsilon_end = 0.05
    epsilon_decay = 50_000    # slower decay for RL
    N_EPISODES = 10000

    replay_buffer = ReplayBuffer(buffer_capacity)
    global_step = 0

    EVAL_EVERY = 50  # episodes

    for episode in range(N_EPISODES):
        state = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0

        while not done:
            # OPTIONAL: text or pygame render (disable for faster training)
            # env.render()
            # env.render_pygame()

            state_tensor = torch.tensor(state, dtype=torch.float32)

            # ----- Epsilon schedule -----
            epsilon = epsilon_end + (epsilon_start - epsilon_end) * \
                      torch.exp(torch.tensor(-global_step / epsilon_decay)).item()

            # Optional: reduce exploration more when snake is long
            length = 1 + len(env.body)
            if length >= 4:
                effective_epsilon = max(epsilon * 0.5, 0.02)
            else:
                effective_epsilon = epsilon

            # ----- Epsilon-greedy action selection -----
            if random.random() < effective_epsilon:
                action = random.randint(0, 3)  # explore
            else:
                with torch.no_grad():
                    q_values = policy_net(state_tensor)
                    action = torch.argmax(q_values).item()  # exploit

            # ----- Step environment -----
            next_state, reward, done, info = env.step(action)
            episode_reward += reward
            episode_length += 1

            # Store transition
            replay_buffer.push(state, action, reward, next_state, done)

            state = next_state
            global_step += 1

            # ----- DQN update from replay buffer -----
            if len(replay_buffer) >= batch_size:
                states, actions, rewards, next_states, dones = replay_buffer.sample(batch_size)

                # Current Q(s,a) from policy_net
                q_values = policy_net(states)                      # (B, 4)
                q_value = q_values.gather(1, actions.unsqueeze(1)).squeeze(1)

                # Target Q-values from target_net
                #with torch.no_grad():
                #    next_q_values = target_net(next_states)        # (B, 4)
                #    max_next_q = next_q_values.max(1)[0] # ← OLD: target picks AND evaluates
                #    target = rewards + gamma * max_next_q * (1 - dones)

                with torch.no_grad():
                    # 1. policy_net picks best actions for next states
                    next_actions = policy_net(next_states).max(1)[1].unsqueeze(1)
                    # 2. target_net evaluates those actions
                    next_q_values = target_net(next_states).gather(1, next_actions).squeeze(1)
                    # 3. Double DQN target
                    target = rewards + gamma * next_q_values * (1 - dones)

                loss = loss_fn(q_value, target)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # ----- TensorBoard logging (per update) -----
                writer.add_scalar("train/loss", loss.item(), global_step)
                writer.add_scalar("train/reward_step", reward, global_step)
                # Log mean Q per action for the batch
                with torch.no_grad():
                    mean_q = q_values.mean(dim=0)
                    for a in range(mean_q.shape[0]):
                        writer.add_scalar(f"Q/action_{a}", mean_q[a].item(), global_step)

                if global_step % 100 == 0:
                    for name, param in policy_net.named_parameters():
                        writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), global_step)

            # ----- Periodically update target network -----
            if global_step % target_update == 0:
                target_net.load_state_dict(policy_net.state_dict())

        # Log episode-level stats
        writer.add_scalar("episode/reward", episode_reward, episode)
        writer.add_scalar("episode/length", episode_length, episode)
        print(f"Episode {episode} | reward={episode_reward:.2f} | length={episode_length} | eps={epsilon:.3f}")

        if (episode + 1) % EVAL_EVERY == 0:
            evaluate(policy_net, env, writer, eval_episodes=5, global_step=global_step)

    torch.save(policy_net.state_dict(), "snake_model_dqn.pth")
    print("Model saved!")

    # env.close_pygame()
    writer.close()