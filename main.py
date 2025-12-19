import random
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter



from lib.navigation import NavigationEnv
from lib.nav_net import NavNet

if __name__ == "__main__":
    env = NavigationEnv()  # Your environment
    env.init_pygame()  # initialize window once
    model = NavNet()
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    loss_fn = nn.MSELoss()
    writer = SummaryWriter("runs/snake_nav")

    gamma = 0.9          # discount factor
    epsilon = 0.1        # exploration probability (10% random moves)

    state = env.reset()
    done = False

    N_EPISODES = 1000  # number of episodes to train
    epsilon = 1.0       # start with high exploration
    epsilon_min = 0.05
    epsilon_decay = 0.995


    global_step = 0
    
    for episode in range(N_EPISODES):
        state = env.reset()
        done = False
        total_reward = 0

        while not done:
            env.render_pygame()  # Print the board

            state_tensor = torch.tensor(state, dtype=torch.float32)

            # ----- Epsilon-greedy action selection -----
            if random.random() < epsilon:
                action = random.randint(0, 3)  # explore: random action
            else:
                q_values = model(state_tensor)
                action = torch.argmax(q_values).item()  # exploit: best action

            # ----- Take action in environment -----
            next_state, reward, done, info = env.step(action)
            #if info["ate_food"]:
            #    print(" Food eaten!")
            next_state_tensor = torch.tensor(next_state, dtype=torch.float32)

            # ----- Compute Q-learning target -----
            if done:
                max_next_q = 0
            else:
                with torch.no_grad():
                    next_q_values = model(next_state_tensor)
                    max_next_q = torch.max(next_q_values)

            target = torch.tensor([reward + gamma * max_next_q], dtype=torch.float32)

            # ----- Get current Q-value -----
            q_values = model(state_tensor)
            q_value = q_values[action]

            # ----- Compute loss and update network -----
            loss = loss_fn(q_value, target)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            state = next_state

            print(f"Action: {action}, Reward: {reward}, Loss: {loss.item():.4f}")

                # ================== TENSORBOARD LOGGING (NEW) ==================
            # 1) log loss and reward
            writer.add_scalar("train/loss", loss.item(), global_step)
            writer.add_scalar("train/reward", reward, global_step)

            # 2) log Q-values per action
            with torch.no_grad():
                for a in range(q_values.shape[0]):  # assuming q_values is 1D over actions
                    writer.add_scalar(f"Q/action_{a}", q_values[a].item(), global_step)

            # 3) (optional) log weight histograms every N steps
            if global_step % 100 == 0:
                for name, param in model.named_parameters():
                    writer.add_histogram(f"weights/{name}", param.data.cpu().numpy(), global_step)

            global_step += 1
            # ===============================================================

    torch.save(model.state_dict(), "snake_model.pth")
    print("Model saved!")
    env.close_pygame()  # when loop ends
    writer.close()
