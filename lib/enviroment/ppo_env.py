import gymnasium as gym
import pygame
from gymnasium import spaces

from lib.enviroment.navigation import NavigationEnv

MAX_STEPS = 200
CELL_SIZE = 30

class SnakeEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.inner_env = NavigationEnv(training=True)
        self.action_space = spaces.Discrete(4)
        self.observation_space = spaces.Box(low=0, high=1, shape=(43,))

    def reset(self, seed=None, options=None):
        obs = self.inner_env.reset()
        return obs, {}

    def step(self, action):
        # FIXED: unpack 4 → split done into terminated/truncated
        obs, reward, done, info = self.inner_env.step(action)
        terminated = done  # Snake: done = crashed/timeout
        truncated = False  # Snake: never time-truncated
        return obs, reward, terminated, truncated, info
    
    def init_pygame(self):
        """Call this once before using render_pygame in a loop."""
        self.inner_env.init_pygame()