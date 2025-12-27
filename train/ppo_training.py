from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env

from lib.enviroment.ppo_env import SnakeEnv

env = make_vec_env(SnakeEnv, n_envs=4)
model = PPO("MlpPolicy", env, verbose=1, device='cpu', tensorboard_log="./ppo_snake_tensorboard/")
model.learn(total_timesteps=50_000)
model.save("ppo_snake_model")