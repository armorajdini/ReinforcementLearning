# q_learning/agent.py
import numpy as np

class QLearningAgent:
    def __init__(self, n_states, n_actions,
                 alpha=0.9, gamma=0.75,
                 epsilon_start=1.0, epsilon_min=0.05, epsilon_decay=0.995):
        self.Q = np.zeros((n_states, n_actions), dtype=float)
        self.alpha = alpha
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay

    def choose_action(self, state, valid_actions):
        if len(valid_actions) == 0:
            return 0
        # epsilon-greedy
        if np.random.rand() < self.epsilon:
            return int(np.random.choice(valid_actions))
        return self.predict_action(state, valid_actions)

    def predict_action(self, state, valid_actions):
        """Greedy action unter valid_actions."""
        q_vals = self.Q[state, valid_actions]
        return int(valid_actions[int(np.argmax(q_vals))])

    def update(self, s, a, r, s_next, valid_actions_next):
        best_next = 0.0
        if len(valid_actions_next) > 0:
            best_next = np.max(self.Q[s_next, valid_actions_next])
        td_target = r + self.gamma * best_next
        td_error = td_target - self.Q[s, a]
        self.Q[s, a] += self.alpha * td_error

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
