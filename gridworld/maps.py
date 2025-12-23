# gridworld/maps.py
from dataclasses import dataclass
from typing import Set, Tuple, List, Optional, Dict
import random

Coord = Tuple[int, int]  # (row, col)


@dataclass(frozen=True)
class GridMap:
    rows: int
    cols: int
    walls: Set[Coord]
    # Bottlenecks as severity levels: 1 (low), 2 (medium), 3 (high)
    bottlenecks: Dict[Coord, int]  # coord -> level


def all_cells(rows: int, cols: int) -> List[Coord]:
    return [(r, c) for r in range(rows) for c in range(cols)]


def random_map(
    rows: int = 12,
    cols: int = 12,
    wall_ratio: float = 0.18,
    bottleneck_ratio: float = 0.12,
    seed: Optional[int] = None
) -> GridMap:
    """
    Random map generator with guaranteed connectivity from start (0,0) to goal (rows-1, cols-1).
    - Walls are non-traversable.
    - Bottlenecks are traversable but add extra cost based on severity level (1/2/3).
    """
    rng = random.Random(seed)
    start = (0, 0)
    goal = (rows - 1, cols - 1)

    # 1) Carve guaranteed corridor (randomized R/D path)
    corridor: Set[Coord] = set()
    r, c = start
    corridor.add((r, c))

    while r < goal[0] or c < goal[1]:
        moves = []
        if r < goal[0]:
            moves.append("D")
        if c < goal[1]:
            moves.append("R")
        move = rng.choice(moves)
        if move == "D":
            r += 1
        else:
            c += 1
        corridor.add((r, c))

    # 2) Slightly widen corridor to ensure "enough free paths"
    expanded = set(corridor)
    for (rr, cc) in list(corridor):
        for dr, dc in [(-1, 0), (1, 0), (0, -1), (0, 1)]:
            nr, nc = rr + dr, cc + dc
            if 0 <= nr < rows and 0 <= nc < cols and rng.random() < 0.25:
                expanded.add((nr, nc))
    corridor = expanded

    allc = all_cells(rows, cols)
    candidates_for_walls = [x for x in allc if x not in corridor and x != start and x != goal]

    target_walls = int(rows * cols * wall_ratio)
    walls = set(rng.sample(candidates_for_walls, k=min(target_walls, len(candidates_for_walls))))

    # 3) Place bottlenecks on free cells (not walls, not start/goal)
    free_cells = [x for x in allc if x not in walls and x != start and x != goal]
    target_bottlenecks = int(rows * cols * bottleneck_ratio)
    chosen = rng.sample(free_cells, k=min(target_bottlenecks, len(free_cells)))

    # Severity distribution: more mild than severe (realistic)
    # 60% level 1, 30% level 2, 10% level 3
    bottlenecks: Dict[Coord, int] = {}
    for cell in chosen:
        p = rng.random()
        if p < 0.60:
            level = 1
        elif p < 0.90:
            level = 2
        else:
            level = 3
        bottlenecks[cell] = level

    return GridMap(rows=rows, cols=cols, walls=walls, bottlenecks=bottlenecks)
