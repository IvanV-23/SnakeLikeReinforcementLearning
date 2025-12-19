

# 1. Recreate the network structure
import torch
import time
from lib.nav_net import NavNet
from lib.navigation import NavigationEnv

env = NavigationEnv(training=False)  # Your environment
env.init_pygame()  # initialize window once
model = NavNet()  # must match the original network definition

# 2. Load the saved weights
model.load_state_dict(torch.load("snake_model_dqn.pth"))
model.eval()  # set to evaluation mode (important if using dropout/batchnorm)

# 3. Use the model to choose actions
state = env.reset()
done = False

while not done:
    env.render_pygame(fps=10)  
    state_tensor = torch.tensor(state, dtype=torch.float32)
    with torch.no_grad():
        q_values = model(state_tensor)
        action = torch.argmax(q_values).item()
    state, reward, done, info = env.step(action)

env.close_pygame()  # when loop ends
