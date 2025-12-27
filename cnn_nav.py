import torch
import numpy as np
import pygame
from lib.enviroment.navigation import NavigationEnv
from lib.net.conv_net import ConvNavNet  # ← CNN VERSION
from lib.buffer.framestack_buffer import FrameStackBuffer

# 1. Initialize environment + pygame
env = NavigationEnv(training=False)
env.init_pygame()
frame_stack = 4
frame_buffer = FrameStackBuffer(frame_stack)

# 2. Load CNN MODEL (NOT NavNet!)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = ConvNavNet(frame_stack).to(device)
model.load_state_dict(torch.load("models/snake_cnn_dqn_vec_v2.pth", map_location=device))
model.eval()

# 3. Test loop
obs = env.reset_cnn()  # ← CNN reset
frame = torch.tensor(obs).to(device)
frame_buffer.reset()
frame_buffer.push(frame)
stacked_state = frame_buffer.get_state(device)

done = False
while not done:
    env.render_pygame(fps=10)
    
    # CNN inference (needs batch dim)
    with torch.no_grad():
        q_values = model(stacked_state.unsqueeze(0))[0]  # (1,B,C,H,W) → [0]
        action = torch.argmax(q_values).item()
    
    # CNN step
    next_obs, reward, done, info = env.cnn_step(action)
    
    # Update frame stack
    next_frame = torch.tensor(next_obs).to(device)
    frame_buffer.push(next_frame)
    stacked_state = frame_buffer.get_state(device)

env.close_pygame()
print("CNN-DQN Demo Complete!")
