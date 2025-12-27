# test_ppo.py
from stable_baselines3 import PPO
from lib.enviroment.ppo_env import SnakeEnv

model = PPO.load("ppo_snake_model")
env = SnakeEnv()
env.init_pygame()  # initialize window once

obs, _ = env.reset()
while True:
    action, _ = model.predict(obs, deterministic=True)
    obs, reward, done, _, _ = env.step(action)
    env.inner_env.render_pygame()  # Watch PPO snake!
    if done: 
        obs, _ = env.reset()
        break
