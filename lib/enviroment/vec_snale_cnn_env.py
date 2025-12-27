# lib/vec_cnn_env.py
import torch
from lib.enviroment.navigation import NavigationEnv
from lib.buffer.framestack_buffer import FrameStackBuffer

class VecSnakeCNNEnv:
    def __init__(self, n_envs=4, frame_stack=4):
        self.n_envs = n_envs
        self.frame_stack = frame_stack
        self.envs = [NavigationEnv(training=True) for _ in range(n_envs)]
        self.buffers = [FrameStackBuffer(frame_stack=frame_stack) for _ in range(n_envs)]

    def reset(self, device):
        states = []
        for env, buf in zip(self.envs, self.buffers):
            obs = env.reset_cnn()                          # (3,84,84)
            buf.reset()
            buf.push(torch.tensor(obs).to(device))
            state = buf.get_state(device)                  # (12,84,84)
            states.append(state)
        return torch.stack(states)                         # (N,12,84,84)

    def step(self, actions, device):
        """
        actions: tensor (N,) on CPU or CUDA
        returns: next_states (N,12,84,84), rewards (N,), dones (N,), infos list
        """
        next_states, rewards, dones, infos = [], [], [], []
        for i, (env, buf) in enumerate(zip(self.envs, self.buffers)):
            a = int(actions[i].item())
            obs, r, done, info = env.cnn_step(a)

            rewards.append(r)
            dones.append(done)
            infos.append(info)

            frame = torch.tensor(obs).to(device)
            buf.push(frame)
            state = buf.get_state(device)                  # (12,84,84)
            next_states.append(state)

            if done:
                # auto-reset
                obs = env.reset_cnn()
                buf.reset()
                buf.push(torch.tensor(obs).to(device))
                next_states[-1] = buf.get_state(device)

        next_states = torch.stack(next_states).to(device)
        rewards = torch.tensor(rewards, dtype=torch.float32, device=device)
        dones = torch.tensor(dones, dtype=torch.float32, device=device)
        return next_states, rewards, dones, infos
