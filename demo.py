import numpy as np
from warehouse.reward_matrix import build_reward_matrix
from warehouse.env import WarehouseEnv
from q_learning.agent import QLearningAgent
from train import train

def greedy_path_from_Q(Q, start=0, goal=11, max_hops=50):
    s = start
    path = [s]
    for _ in range(max_hops):
        a = np.argmax(Q[s])
        if a == s:
            # Schutz gegen Selbstschleifen
            valid = np.where(Q[s] == np.max(Q[s]))[0]
            a = valid[0]
        path.append(a)
        s = a
        if s == goal:
            break
    return path

if __name__ == "__main__":
    agent, rewards, steps, R = train(num_episodes=1000)
    path = greedy_path_from_Q(agent.Q, start=0, goal=11)
    print("Greedy-Policy Pfad (Start→Ziel):", " -> ".join(map(str, path)))
    print("Q-Table (gerundet):")
    with np.printoptions(precision=1, suppress=True):
        print(agent.Q)
