# gridworld/env.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional

import numpy as np
from gridworld.maps import GridMap, Coord

Action = int
# 0=UP, 1=RIGHT, 2=DOWN, 3=LEFT
DELTAS = {
    0: (-1, 0),
    1: (0, 1),
    2: (1, 0),
    3: (0, -1),
}


@dataclass
class StepResult:
    next_state: int
    reward: float
    done: bool
    info: Dict


class GridWorldEnv:
    """
    Diskretes Gridworld-Environment, gym-ähnlich.
    Bottleneck levels (1/2/3) add extra cost: bottleneck_base_penalty * level.
    """

    def __init__(
        self,
        grid: GridMap,
        start: Coord = (0, 0),
        goal: Coord = (5, 5),
        max_steps: int = 200,
        step_cost: float = -1.0,
        goal_reward: float = 100.0,
        invalid_move_penalty: float = -10.0,
        bottleneck_base_penalty: float = -6.0,
    ):
        self.grid = grid
        self.start = start
        self.goal = goal
        self.max_steps = max_steps

        self.step_cost = step_cost
        self.goal_reward = goal_reward
        self.invalid_move_penalty = invalid_move_penalty
        self.bottleneck_base_penalty = bottleneck_base_penalty

        self._state: Optional[Coord] = None
        self._steps = 0

        # Precompute state mapping for all non-wall cells
        self.coord_to_state: Dict[Coord, int] = {}
        self.state_to_coord: Dict[int, Coord] = {}
        sid = 0
        for r in range(grid.rows):
            for c in range(grid.cols):
                if (r, c) in grid.walls:
                    continue
                self.coord_to_state[(r, c)] = sid
                self.state_to_coord[sid] = (r, c)
                sid += 1

        if start in grid.walls or goal in grid.walls:
            raise ValueError("Start oder Goal darf nicht auf einer Wall liegen.")

        self.n_states = len(self.coord_to_state)
        self.n_actions = 4

    @property
    def observation_space_n(self) -> int:
        return self.n_states

    @property
    def action_space_n(self) -> int:
        return self.n_actions

    def set_start_goal(self, start: Coord, goal: Coord) -> None:
        if start in self.grid.walls or goal in self.grid.walls:
            raise ValueError("Start/Goal darf nicht auf einer Wall liegen.")
        self.start = start
        self.goal = goal

    def reset(self, *, seed: Optional[int] = None) -> int:
        if seed is not None:
            np.random.seed(seed)
        self._state = self.start
        self._steps = 0
        return self.coord_to_state[self._state]

    def _in_bounds(self, r: int, c: int) -> bool:
        return 0 <= r < self.grid.rows and 0 <= c < self.grid.cols

    def valid_actions(self, state_id: int):
        coord = self.state_to_coord[state_id]
        valid = []
        for a, (dr, dc) in DELTAS.items():
            nr, nc = coord[0] + dr, coord[1] + dc
            if not self._in_bounds(nr, nc):
                continue
            if (nr, nc) in self.grid.walls:
                continue
            valid.append(a)
        return valid

    def step(self, action: Action) -> Tuple[int, float, bool, Dict]:
        if self._state is None:
            raise RuntimeError("Call reset() before step().")

        self._steps += 1
        r, c = self._state
        dr, dc = DELTAS[action]
        nr, nc = r + dr, c + dc

        # Invalid move: out of bounds or wall
        if (not self._in_bounds(nr, nc)) or ((nr, nc) in self.grid.walls):
            reward = self.invalid_move_penalty
            next_coord = self._state
            level = 0
        else:
            next_coord = (nr, nc)
            reward = self.step_cost

            # Bottleneck severity (0 if not a bottleneck)
            level = self.grid.bottlenecks.get(next_coord, 0)
            if level > 0:
                reward += self.bottleneck_base_penalty * level

            if next_coord == self.goal:
                reward += self.goal_reward

        self._state = next_coord
        done = (self._state == self.goal) or (self._steps >= self.max_steps)

        next_state_id = self.coord_to_state[self._state]
        info = {"coord": self._state, "steps": self._steps, "bottleneck_level": level}
        return next_state_id, float(reward), bool(done), info
