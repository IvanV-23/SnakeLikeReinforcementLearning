import torch

from lib.enviroment.navigation import NavigationEnv


def single_evaluate(policy_net, writer, global_step, n_episodes=5, device=None):
    if device is None:
        device = next(policy_net.parameters()).device  # infer device

    eval_env = NavigationEnv(training=False)
    policy_net.eval()

    total_reward = 0.0
    total_length = 0

    for _ in range(n_episodes):
        state = eval_env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32, device=device)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()

            state, reward, done, _ = eval_env.step(action)
            ep_reward += reward
            ep_length += 1

        total_reward += ep_reward
        total_length += ep_length

    avg_reward = total_reward / n_episodes
    avg_length = total_length / n_episodes
    writer.add_scalar("eval/avg_reward", avg_reward, global_step)
    writer.add_scalar("eval/avg_length", avg_length, global_step)
    print(f"[EVAL] step={global_step} avg_reward={avg_reward:.2f} avg_length={avg_length:.1f}")

    policy_net.train()