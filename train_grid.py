# train_grid.py
import numpy as np
from gridworld.maps import default_map_6x6
from gridworld.env import GridWorldEnv
from q_learning.agent import QLearningAgent

def train_grid(
    num_episodes=2000,
    max_steps=200,
    alpha=0.9,
    gamma=0.75,
    eps_start=1.0,
    eps_min=0.05,
    eps_decay=0.995,
    seed=42,
    start=(0, 0),
    goal=(5, 5),
    step_cost=-1.0,
    goal_reward=100.0,
    invalid_move_penalty=-10.0,
    bottleneck_penalty=-10.0
):
    np.random.seed(seed)

    grid = default_map_6x6()
    env = GridWorldEnv(
        grid=grid,
        start=start,
        goal=goal,
        max_steps=max_steps,
        step_cost=step_cost,
        goal_reward=goal_reward,
        invalid_move_penalty=invalid_move_penalty,
        bottleneck_penalty=bottleneck_penalty
    )

    agent = QLearningAgent(
        n_states=env.observation_space_n,
        n_actions=env.action_space_n,
        alpha=alpha,
        gamma=gamma,
        epsilon_start=eps_start,
        epsilon_min=eps_min,
        epsilon_decay=eps_decay
    )

    episode_returns = []
    steps_to_goal = []
    bottleneck_hits = []
    reached_goal_flags = []

    for ep in range(num_episodes):
        s = env.reset()
        total_r = 0.0
        bn_count = 0
        reached_goal = False

        for t in range(max_steps):
            valid = env.valid_actions(s)
            a = agent.choose_action(s, valid)

            s_next, r, done, info = env.step(a)
            total_r += r

            # Bottleneck hit tracking
            if info.get("coord") in env.grid.bottlenecks:
                bn_count += 1

            agent.update(s, a, r, s_next, env.valid_actions(s_next))
            s = s_next

            if info.get("coord") == env.goal:
                reached_goal = True

            if done:
                steps_to_goal.append(t + 1)
                break

        agent.decay_epsilon()
        episode_returns.append(total_r)
        bottleneck_hits.append(bn_count)
        reached_goal_flags.append(1 if reached_goal else 0)

    metrics = {
        "episode_returns": episode_returns,
        "steps_to_goal": steps_to_goal,
        "bottleneck_hits": bottleneck_hits,
        "reached_goal_flags": reached_goal_flags,
        "final_epsilon": agent.epsilon,
        "params": {
            "alpha": alpha,
            "gamma": gamma,
            "eps_start": eps_start,
            "eps_min": eps_min,
            "eps_decay": eps_decay,
            "step_cost": step_cost,
            "goal_reward": goal_reward,
            "invalid_move_penalty": invalid_move_penalty,
            "bottleneck_penalty": bottleneck_penalty,
            "num_episodes": num_episodes,
            "max_steps": max_steps,
        }
    }

    return env, agent, metrics

if __name__ == "__main__":
    env, agent, metrics = train_grid()
    print("Train done.")
    print("Final epsilon:", metrics["final_epsilon"])
