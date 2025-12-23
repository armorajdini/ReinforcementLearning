import numpy as np
import matplotlib.pyplot as plt
from warehouse.reward_matrix import build_reward_matrix
from warehouse.env import WarehouseEnv
from q_learning.agent import QLearningAgent

import matplotlib
matplotlib.use("TkAgg")

def train(num_episodes=1000, max_steps=100, alpha=0.9, gamma=0.75,
          eps_start=1.0, eps_min=0.05, eps_decay=0.995, seed=42):
    np.random.seed(seed)

    R = build_reward_matrix()
    env = WarehouseEnv(R, start_state=0, goal_state=11, max_steps=max_steps)
    agent = QLearningAgent(env.observation_space_n, env.action_space_n,
                           alpha=alpha, gamma=gamma,
                           epsilon_start=eps_start, epsilon_min=eps_min, epsilon_decay=eps_decay)

    episode_rewards = []
    steps_to_goal = []

    for ep in range(num_episodes):
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            valid_actions = env.valid_actions(s)
            a = agent.choose_action(s, valid_actions)
            s_next, r, done, _ = env.step(a)
            total_r += r
            agent.update(s, a, r, s_next, env.valid_actions(s_next))
            s = s_next
            if done:
                steps_to_goal.append(t + 1)
                break

        agent.decay_epsilon()
        episode_rewards.append(total_r)

    return agent, episode_rewards, steps_to_goal, R

if __name__ == "__main__":
    agent, rewards, steps, R = train()

    import matplotlib.pyplot as plt
    import seaborn as sns

    # Plot: Return-Verlauf
    plt.figure()
    window = 25
    smooth = np.convolve(rewards, np.ones(window)/window, mode='valid')
    plt.plot(rewards, label="Episode Return")
    plt.plot(range(window-1, len(rewards)), smooth, label=f"Moving Avg ({window})")
    plt.xlabel("Episode")
    plt.ylabel("Return")
    plt.title("Q-Learning Training Returns")
    plt.legend()
    plt.tight_layout()
    plt.savefig("training_returns.png")

    # 🔥 Q-Table Heatmap
    plt.figure(figsize=(8,6))
    sns.heatmap(agent.Q, annot=False, cmap="viridis")
    plt.title("Q-Table Heatmap")
    plt.xlabel("Actions")
    plt.ylabel("States")
    plt.tight_layout()
    plt.savefig("q_heatmap.png")

    plt.show()

    solved_eps = np.where(np.array(steps) < 100)[0]
    print(f"Episoden: {len(rewards)}, Erreichtes Ziel in {len(solved_eps)} Episoden.")

