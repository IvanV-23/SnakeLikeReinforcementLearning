import torch
import numpy as np
import pygame

from lib.enviroment.multy_snake_env import MultiSnakeEnv
#from lib.enviroment.navigation import NavigationEnv  # Your single-snake env (for MultiSnakeEnv import)
from lib.net.multi_agent_conv_nav_net import ConvNavNet  # Your CNN
from lib.buffer.framestack_buffer import FrameStackBuffer

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Load YOUR exact model (3 channels)
model = ConvNavNet().to(device)  # No frame_stack param!
model.load_state_dict(torch.load("models/multi_snake_dqn.pth", map_location=device))
model.eval()

# Initialize environment
env = MultiSnakeEnv(n_agents=2, width=20, height=20, training=False)
env.init_pygame()

# Single episode evaluation
obs = env.reset_multi()
done = False
total_reward = 0
step_count = 0

print("Multi-Snake CNN-DQN Demo (Press ESC or close window to quit)")
print("Single frames (3 channels) - matches your trained model!")

while not done:
    # Events
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE):
            env.close_pygame()
            print(f"\nDemo ended! reward={total_reward:.1f}, steps={step_count}")
            exit()
    
    env.render_pygame(fps=8)
    
    # ========== FIXED INFERENCE (3 channels only) ==========
    actions = {}
    obs_list = []
    agent_keys = []
    
    with torch.no_grad():
        for key, obs_img in obs.items():
            obs_tensor = torch.FloatTensor(obs_img).unsqueeze(0).to(device)  # (1,3,84,84)
            obs_list.append(obs_tensor)
            agent_keys.append(key)
        
        batch_states = torch.cat(obs_list, dim=0)  # (2, 3, 84, 84) ← PERFECT MATCH
        q_values = model(batch_states)  # (2, 4)
        
        for i, key in enumerate(agent_keys):
            actions[key] = torch.argmax(q_values[i]).item()
    
    # Step
    next_obs, reward, done, info = env.step_multi(actions)
    total_reward += reward
    step_count += 1
    
    # Print
    action_names = ["↑", "↓", "←", "→"]
    print(f"Step {step_count:4d} | R:{reward:+.2f} | Total:{total_reward:.1f} | Score:{env.score:.1f} | "
          f"Actions: {dict((k, action_names[v]) for k,v in actions.items())}")
    
    obs = next_obs

env.close_pygame()
print(f"\n=== COMPLETE === reward={total_reward:.1f}, steps={step_count}")