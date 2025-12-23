# demo_grid.py
import numpy as np
import matplotlib.pyplot as plt

from train_grid import train_grid

def greedy_rollout(env, agent, max_steps=200):
    s = env.reset()
    path_coords = [env.state_to_coord[s]]
    total_r = 0.0

    for _ in range(max_steps):
        valid = env.valid_actions(s)
        a = agent.predict_action(s, valid)
        s, r, done, info = env.step(a)
        total_r += r
        path_coords.append(info["coord"])
        if done:
            break

    return path_coords, total_r

if __name__ == "__main__":
    env, agent, returns, steps = train_grid(num_episodes=2000)

    path, total_r = greedy_rollout(env, agent)
    print("Greedy-Path coords:", path)
    print("Greedy-Return:", total_r)

    # Return-Plot speichern (berichtstauglich)
    plt.figure()
    window = 50
    if len(returns) >= window:
        smooth = np.convolve(returns, np.ones(window)/window, mode="valid")
        plt.plot(returns, label="Episode Return")
        plt.plot(range(window-1, len(returns)), smooth, label=f"Moving Avg ({window})")
    else:
        plt.plot(returns, label="Episode Return")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Gridworld Training Returns (Cost-aware)")
    plt.legend()
    plt.tight_layout()
    plt.savefig("grid_training_returns.png")
    plt.show()
