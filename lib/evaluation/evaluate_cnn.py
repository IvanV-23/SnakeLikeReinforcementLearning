from lib.buffer.framestack_buffer import FrameStackBuffer
import torch



def evaluate_cnn(policy_net, env, device, n_episodes=10):
    policy_net.eval()
    total_reward = 0.0

    for _ in range(n_episodes):
        obs = env.reset_cnn()
        frame_buffer = FrameStackBuffer(frame_stack=4)
        frame = torch.tensor(obs).to(device)
        frame_buffer.reset()
        frame_buffer.push(frame)
        stacked_state = frame_buffer.get_state(device)  # (12,84,84)

        done = False
        ep_reward = 0.0

        while not done:
            # GREEDY policy: no epsilon
            with torch.no_grad():
                q_values = policy_net(stacked_state.unsqueeze(0))[0]
                action = torch.argmax(q_values).item()

            next_obs, reward, done, _ = env.cnn_step(action)
            ep_reward += reward

            next_frame = torch.tensor(next_obs).to(device)
            frame_buffer.push(next_frame)
            stacked_state = frame_buffer.get_state(device)

        total_reward += ep_reward

    policy_net.train()
    return total_reward / n_episodes
