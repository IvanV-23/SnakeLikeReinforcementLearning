
import torch

def evaluate(policy_net, env, writer, eval_episodes=5, global_step=0):
    """Run greedy episodes (epsilon=0) and log average reward/length."""
    policy_net.eval()
    total_reward = 0.0
    total_length = 0

    for _ in range(eval_episodes):
        state = env.reset()
        done = False
        ep_reward = 0.0
        ep_length = 0

        while not done:
            state_tensor = torch.tensor(state, dtype=torch.float32)
            with torch.no_grad():
                q_values = policy_net(state_tensor)
                action = torch.argmax(q_values).item()  # greedy: epsilon = 0

            next_state, reward, done, info = env.step(action)
            state = next_state
            ep_reward += reward
            ep_length += 1

        total_reward += ep_reward
        total_length += ep_length

    avg_reward = total_reward / eval_episodes
    avg_length = total_length / eval_episodes

    # Log to TensorBoard
    writer.add_scalar("eval/avg_reward", avg_reward, global_step)
    writer.add_scalar("eval/avg_length", avg_length, global_step)

    print(f"[EVAL] avg_reward={avg_reward:.2f}, avg_length={avg_length:.1f}")

    policy_net.train()
