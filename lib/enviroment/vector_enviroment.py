import numpy as np
from lib.enviroment.navigation import NavigationEnv

class VecNavigationEnv:
    def __init__(self, n_envs=4):
        self.n_envs = n_envs
        self.envs = [NavigationEnv(training=True) for _ in range(n_envs)]

        self.width = self.envs[0].width
        self.height = self.envs[0].height

    def reset(self):
        states = [env.reset() for env in self.envs]
        return np.array(states, dtype=np.float32)   # (n_envs, obs_dim)

    def step(self, actions):
        """actions: array/list of length n_envs"""
        next_states = []
        rewards = []
        dones = []
        infos = []

        for env, action in zip(self.envs, actions):
            ns, r, d, info = env.step(int(action))
            if d:
                ns = env.reset()   
            next_states.append(ns)
            rewards.append(r)
            dones.append(d)
            infos.append(info)

        return (np.array(next_states, dtype=np.float32),
                np.array(rewards, dtype=np.float32),
                np.array(dones, dtype=np.float32),
                infos)
