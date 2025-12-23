# visualization/app.py
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button, TextBox, RadioButtons
from matplotlib.patches import Patch
from matplotlib.lines import Line2D

from gridworld.maps import random_map
from gridworld.env import GridWorldEnv
from q_learning.agent import QLearningAgent

# ---------- Helper: Parsing ----------
def parse_coord(text: str):
    parts = text.strip().split(",")
    if len(parts) != 2:
        raise ValueError("Format muss 'row,col' sein, z.B. 0,0")
    r = int(parts[0].strip())
    c = int(parts[1].strip())
    return (r, c)

# ---------- Drawing: Grid / Policy / Values ----------
ACTION_TO_DELTA = {0: (-1, 0), 1: (0, 1), 2: (1, 0), 3: (0, -1)}

def draw_grid(ax, env: GridWorldEnv, start=None, goal=None, path=None, title=None):
    g = env.grid
    ax.clear()
    ax.set_xlim(-0.5, g.cols - 0.5)
    ax.set_ylim(g.rows - 0.5, -0.5)
    ax.set_xticks(range(g.cols))
    ax.set_yticks(range(g.rows))
    ax.grid(True)
    ax.set_aspect("equal", adjustable="box")

    # walls
    for (r, c) in g.walls:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True))

    # bottlenecks with level text
    for (r, c), level in g.bottlenecks.items():
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///"))
        ax.text(c, r, str(level), ha="center", va="center", fontsize=9)

    if start is not None:
        ax.scatter([start[1]], [start[0]], marker="s", s=200)
    if goal is not None:
        ax.scatter([goal[1]], [goal[0]], marker="X", s=220)

    if path:
        xs = [c for (r, c) in path]
        ys = [r for (r, c) in path]
        ax.plot(xs, ys, linewidth=3)

    if title:
        ax.set_title(title)

def compute_value_grid(env: GridWorldEnv, agent: QLearningAgent):
    g = env.grid
    V = np.full((g.rows, g.cols), np.nan, dtype=float)
    for coord, sid in env.coord_to_state.items():
        r, c = coord
        valid = env.valid_actions(sid)
        if len(valid) == 0:
            V[r, c] = np.nan
        else:
            V[r, c] = np.max(agent.Q[sid, valid])
    return V

def draw_value_and_policy(ax, env: GridWorldEnv, agent: QLearningAgent, start=None, goal=None):
    g = env.grid
    ax.clear()
    V = compute_value_grid(env, agent)

    ax.imshow(V, origin="upper")
    ax.set_xticks(range(g.cols))
    ax.set_yticks(range(g.rows))
    ax.set_title("Value Heatmap + Policy Arrows")

    # walls overlay
    for (r, c) in g.walls:
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=True))

    # bottlenecks overlay with level
    for (r, c), level in g.bottlenecks.items():
        ax.add_patch(plt.Rectangle((c - 0.5, r - 0.5), 1, 1, fill=False, hatch="///"))
        ax.text(c, r, str(level), ha="center", va="center", fontsize=8)

    # policy arrows
    for coord, sid in env.coord_to_state.items():
        r, c = coord
        valid = env.valid_actions(sid)
        if len(valid) == 0:
            continue
        a = agent.predict_action(sid, valid)
        dr, dc = ACTION_TO_DELTA[a]
        ax.arrow(c, r, 0.3 * dc, 0.3 * dr, head_width=0.15, length_includes_head=True)

    if start is not None:
        ax.scatter([start[1]], [start[0]], marker="s", s=120)
    if goal is not None:
        ax.scatter([goal[1]], [goal[0]], marker="X", s=140)

def draw_returns(ax, returns):
    ax.clear()
    ax.set_title("Training Returns (Cost-aware)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("Return")
    ax.plot(returns, label="Return")

    window = 50
    if len(returns) >= window:
        smooth = np.convolve(returns, np.ones(window) / window, mode="valid")
        ax.plot(range(window - 1, len(returns)), smooth, label=f"Moving Avg ({window})")
    ax.legend()

# ---------- Training / Rollout ----------
def train_env(env: GridWorldEnv, num_episodes=2500, max_steps=250, seed=42):
    np.random.seed(seed)
    agent = QLearningAgent(
        n_states=env.observation_space_n,
        n_actions=env.action_space_n,
        alpha=0.9,
        gamma=0.75,
        epsilon_start=1.0,
        epsilon_min=0.05,
        epsilon_decay=0.995,
    )

    returns = []
    steps_to_goal = []

    for _ in range(num_episodes):
        s = env.reset()
        total_r = 0.0
        for t in range(max_steps):
            valid = env.valid_actions(s)
            a = agent.choose_action(s, valid)
            s_next, r, done, _ = env.step(a)
            total_r += r
            agent.update(s, a, r, s_next, env.valid_actions(s_next))
            s = s_next
            if done:
                steps_to_goal.append(t + 1)
                break
        agent.decay_epsilon()
        returns.append(total_r)

    return agent, returns, steps_to_goal

def greedy_rollout(env: GridWorldEnv, agent: QLearningAgent, max_steps=250):
    s = env.reset()
    path = [env.state_to_coord[s]]
    total_r = 0.0

    bottleneck_hits = 0
    bottleneck_level_sum = 0

    for _ in range(max_steps):
        valid = env.valid_actions(s)
        a = agent.predict_action(s, valid)
        s, r, done, info = env.step(a)
        total_r += r

        coord = info["coord"]
        path.append(coord)

        level = info.get("bottleneck_level", 0)
        if level > 0:
            bottleneck_hits += 1
            bottleneck_level_sum += level

        if done:
            break

    return path, total_r, bottleneck_hits, bottleneck_level_sum

# ---------- App ----------
def main():
    MAP_ROWS = 12
    MAP_COLS = 12
    WALL_RATIO = 0.18
    BOTTLENECK_RATIO = 0.12

    # Cost-aware config: bottlenecks add bottleneck_base_penalty * level
    REWARD_CONFIG = dict(
        max_steps=250,
        step_cost=-1.0,
        goal_reward=100.0,
        invalid_move_penalty=-10.0,
        bottleneck_base_penalty=-6.0,  # Level 1=-6, Level 2=-12, Level 3=-18
    )

    seed_counter = {"seed": 42}

    def make_env_with_new_map(seed):
        g = random_map(
            rows=MAP_ROWS,
            cols=MAP_COLS,
            wall_ratio=WALL_RATIO,
            bottleneck_ratio=BOTTLENECK_RATIO,
            seed=seed
        )
        e = GridWorldEnv(
            grid=g,
            start=(0, 0),
            goal=(g.rows - 1, g.cols - 1),
            **REWARD_CONFIG
        )
        return g, e

    grid, env = make_env_with_new_map(seed_counter["seed"])

    fig = plt.figure(figsize=(13, 7))
    gs = fig.add_gridspec(2, 3, width_ratios=[1.5, 1, 1])
    fig.subplots_adjust(left=0.05, right=0.98, top=0.88, bottom=0.20, wspace=0.25, hspace=0.35)

    ax_grid = fig.add_subplot(gs[:, 0])
    ax_returns = fig.add_subplot(gs[0, 1:])
    ax_value = fig.add_subplot(gs[1, 1:])

    # Bottom controls
    ax_mode      = fig.add_axes([0.05, 0.03, 0.15, 0.12])
    ax_startbox  = fig.add_axes([0.23, 0.06, 0.12, 0.06])
    ax_goalbox   = fig.add_axes([0.36, 0.06, 0.12, 0.06])
    ax_trainbtn  = fig.add_axes([0.50, 0.06, 0.12, 0.06])
    ax_resetbtn  = fig.add_axes([0.64, 0.06, 0.12, 0.06])
    ax_changemap = fig.add_axes([0.78, 0.06, 0.18, 0.06])

    mode = RadioButtons(ax_mode, ("Set Start (click)", "Set Goal (click)"), active=0)
    for t in mode.labels:
        t.set_fontsize(9)

    start_box = TextBox(ax_startbox, "Start (r,c): ", initial="0,0")
    goal_box = TextBox(ax_goalbox, "Goal (r,c): ", initial=f"{grid.rows-1},{grid.cols-1}")
    train_btn = Button(ax_trainbtn, "Train + Show")
    reset_btn = Button(ax_resetbtn, "Reset View")
    change_btn = Button(ax_changemap, "Change Map")

    state = {
        "grid": grid,
        "env": env,
        "start": (0, 0),
        "goal": (grid.rows - 1, grid.cols - 1),
        "agent": None,
        "returns": None,
        "path": None,
    }

    legend_elements = [
        Patch(facecolor="black", label="Wall (blocked)"),
        Patch(facecolor="none", hatch="///", label="Bottleneck (extra cost; level 1-3)"),
        Line2D([0], [0], marker="s", linestyle="None", markersize=10, label="Start"),
        Line2D([0], [0], marker="X", linestyle="None", markersize=10, label="Goal"),
        Line2D([0], [0], linewidth=3, label="Greedy path"),
    ]

    fig.legend(
        handles=legend_elements,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.97),
        ncol=5,
        frameon=True
    )

    def refresh_title():
        ax_grid.set_title(
            "Random Gridworld Warehouse (cost-aware)\n"
            "Set Start/Goal (click or text) → Train + Show"
        )

    def redraw():
        draw_grid(ax_grid, state["env"], start=state["start"], goal=state["goal"], path=state["path"])
        refresh_title()

        if state["returns"] is not None:
            draw_returns(ax_returns, state["returns"])
        else:
            ax_returns.clear()
            ax_returns.set_title("Training Returns (Cost-aware)")

        if state["agent"] is not None:
            draw_value_and_policy(ax_value, state["env"], state["agent"], start=state["start"], goal=state["goal"])
        else:
            ax_value.clear()
            ax_value.set_title("Value Heatmap + Policy Arrows")

        fig.canvas.draw_idle()

    def sync_env_start_goal():
        state["env"].set_start_goal(state["start"], state["goal"])

    # Click to set start/goal
    def on_click(event):
        if event.inaxes != ax_grid or event.xdata is None or event.ydata is None:
            return

        g = state["grid"]
        r = int(round(event.ydata))
        c = int(round(event.xdata))
        if not (0 <= r < g.rows and 0 <= c < g.cols):
            return
        if (r, c) in g.walls:
            return

        selected_mode = mode.value_selected
        if "Start" in selected_mode:
            state["start"] = (r, c)
            start_box.set_val(f"{r},{c}")
        else:
            state["goal"] = (r, c)
            goal_box.set_val(f"{r},{c}")

        sync_env_start_goal()
        state["agent"] = None
        state["returns"] = None
        state["path"] = None
        redraw()

    fig.canvas.mpl_connect("button_press_event", on_click)

    # Text submit
    def on_start_submit(text):
        try:
            coord = parse_coord(text)
            if coord in state["grid"].walls:
                return
            state["start"] = coord
            sync_env_start_goal()
            state["agent"] = None
            state["returns"] = None
            state["path"] = None
            redraw()
        except Exception:
            pass

    def on_goal_submit(text):
        try:
            coord = parse_coord(text)
            if coord in state["grid"].walls:
                return
            state["goal"] = coord
            sync_env_start_goal()
            state["agent"] = None
            state["returns"] = None
            state["path"] = None
            redraw()
        except Exception:
            pass

    start_box.on_submit(on_start_submit)
    goal_box.on_submit(on_goal_submit)

    # Buttons
    def on_train(_):
        sync_env_start_goal()

        agent, returns, _steps = train_env(state["env"], num_episodes=2500, max_steps=250, seed=42)
        path, greedy_return, bottleneck_hits, bottleneck_level_sum = greedy_rollout(state["env"], agent, max_steps=250)

        state["agent"] = agent
        state["returns"] = returns
        state["path"] = path

        draw_grid(
            ax_grid,
            state["env"],
            start=state["start"],
            goal=state["goal"],
            path=state["path"],
            title=f"Greedy path | steps={len(path)-1}, return={greedy_return:.1f}, bottleneck_hits={bottleneck_hits}, bottleneck_level_sum={bottleneck_level_sum}"
        )
        refresh_title()
        draw_returns(ax_returns, returns)
        draw_value_and_policy(ax_value, state["env"], agent, start=state["start"], goal=state["goal"])
        fig.canvas.draw_idle()

    def on_reset(_):
        state["agent"] = None
        state["returns"] = None
        state["path"] = None
        redraw()

    def on_change_map(_):
        seed_counter["seed"] += 1
        new_grid, new_env = make_env_with_new_map(seed_counter["seed"])

        state["grid"] = new_grid
        state["env"] = new_env

        state["start"] = (0, 0)
        state["goal"] = (new_grid.rows - 1, new_grid.cols - 1)
        start_box.set_val("0,0")
        goal_box.set_val(f"{new_grid.rows-1},{new_grid.cols-1}")

        state["agent"] = None
        state["returns"] = None
        state["path"] = None
        redraw()

    train_btn.on_clicked(on_train)
    reset_btn.on_clicked(on_reset)
    change_btn.on_clicked(on_change_map)

    redraw()
    plt.show()

if __name__ == "__main__":
    main()
