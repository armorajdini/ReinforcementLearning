import numpy as np

class WarehouseEnv:
    """
    Minimal-Gym-ähnliche API:
    - reset() -> state
    - step(action) -> next_state, reward, done, info
    - state_space_n, action_space_n
    """
    def __init__(self, reward_matrix: np.ndarray, start_state: int = 0, goal_state: int = 11, max_steps=100):
        assert reward_matrix.shape[0] == reward_matrix.shape[1]
        self.R = reward_matrix
        self.n_states = reward_matrix.shape[0]
        self.n_actions = reward_matrix.shape[1]
        self.start_state = start_state
        self.goal_state = goal_state
        self.max_steps = max_steps

        self.state = None
        self.steps = 0

    @property
    def observation_space_n(self):
        return self.n_states

    @property
    def action_space_n(self):
        return self.n_actions

    def valid_actions(self, s):
        # Erlaubt sind Aktionen mit R[s,a] > 0 oder Zielprämie
        return np.where(self.R[s] > 0)[0]

    def reset(self, *, seed=None):
        if seed is not None:
            np.random.seed(seed)
        self.state = self.start_state
        self.steps = 0
        return self.state

    def step(self, action: int):
        assert 0 <= action < self.n_actions, "Ungültige Aktion"
        reward = self.R[self.state, action]
        if reward <= 0:
            # Ungültiger Zug → große negative Strafe
            reward = -10.0
            next_state = self.state
        else:
            next_state = action

        self.state = next_state
        self.steps += 1

        done = (self.state == self.goal_state) or (self.steps >= self.max_steps)
        info = {}
        return next_state, float(reward), bool(done), info
