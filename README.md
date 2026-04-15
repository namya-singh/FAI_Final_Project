# Adversarial Maze Navigation

**By Namya Singh & Vikramaditya Sogani**

An AI maze game where an agent tries to reach the goal while being chased by a pursuer. You can watch different AI algorithms play, or take control yourself. The project also includes a batch experiment runner that tests all agents across hundreds of configurations and saves the results to CSV files.

---

## What You Need

- Python 3.8 or higher
- pygame
- numpy (optional, used for sound effects)

Install dependencies:

```bash
pip install pygame numpy
```

---

## How to Launch

### Play the visual game (recommended starting point)

```bash
python3 main.py --visual
```

This opens a game window where you can watch AI agents play and switch settings on the fly.

### Run a quick terminal simulation

```bash
python3 main.py
```

Runs a series of games in the terminal and prints results.

### Run a single game with a specific setup

```bash
python3 main.py --algo alpha_beta --pursuer astar --size 13 --density 0.25 --dynamic
```

---

## The Visual Game — Step by Step

### Starting Screen

When the game opens, you land on a setup screen. Use the following keys to configure your game before starting:

| Key | What it does |
|-----|--------------|
| `←` / `→` arrow keys | Cycle through AI agents |
| `↑` / `↓` arrow keys | Change maze size |
| `1` / `2` / `3` | Set the number of pursuers (1, 2, or 3) |
| `D` | Toggle dynamic walls on/off (walls shift during the game) |
| `F` | Toggle fog of war on/off (the agent can only see nearby cells) |
| `Space` or `Enter` | Start the game |

### In-Game Controls

Once the game is running, use these keys:

| Key | What it does |
|-----|--------------|
| `Space` | Pause / unpause |
| `R` | Generate a new maze and restart |
| `Enter` | Go back to the start menu |
| `ESC` or `Q` | Quit the game |
| `+` / `-` | Speed the game up or slow it down |
| `F` | Toggle fog of war on/off mid-game |
| `D` | Toggle dynamic walls on/off (resets the current game) |
| `B` | Toggle split view — shows the agent's belief map next to the true maze |

### Switch the AI Agent Mid-Game

Press a number key to instantly switch which algorithm the agent uses:

| Key | Agent |
|-----|-------|
| `1` | LRTA* |
| `2` | Minimax |
| `3` | Alpha-Beta |
| `4` | Expectimax |
| `5` | Hill Climbing |
| `6` | Beam Search |
| `7` | **Manual** — you control the agent with the arrow keys |

### Switch the Pursuer Mid-Game

| Key | Pursuer strategy |
|-----|-----------------|
| `Z` | Random (easiest — moves unpredictably) |
| `X` | Greedy (medium — always moves toward you) |
| `C` | Beam Search (medium-hard — navigates around walls) |
| `V` | A* (hardest — finds the optimal path to catch you) |

---

## Playing Manually

Press `7` to switch to manual mode. Then use the **arrow keys** to move the agent yourself. Try to reach the goal (the green cell) before the pursuer catches you.

Tips:
- Watch the pursuer's color: it flashes when it is about to move.
- Power-ups on the map can give you an edge (see the Power-Ups section below).
- If you step on a trap (red-orange cell), you freeze for 3 turns.

---

## Understanding the Map

### What Each Color Means

| Color | Cell type | Effect |
|-------|-----------|--------|
| Dark navy | Open cell | Normal movement, costs 1 step |
| Slate/dark | Wall | Cannot walk through |
| Red-orange | Trap | Freezes you for 3 turns when stepped on |
| Purple | Power-up | Gives a temporary ability (see below) |
| Brown | Mud | Slower to cross, costs 3 steps |
| Blue | Water | Costs 2 steps to cross |
| Dark green | Road | Looks different but costs the same as a normal cell |

### Characters on the Map

- **Agent** — the blue character you are watching (or controlling). Trying to reach the goal.
- **Pursuer(s)** — the red/orange ghost(s) chasing the agent.
- **Goal** — the bright green cell. The agent wins by reaching it.

### Power-Ups

When the agent steps on a purple cell, it picks up a power-up. There are four types:

| Power-up | What it does | Duration |
|----------|-------------|----------|
| Reveal (R) | Lifts the fog of war so the agent can see the entire maze | 10 steps |
| Speed (S) | Agent moves twice per turn | 8 steps |
| Shield (X) | Makes the agent immune to traps | 6 steps |
| Freeze (F) | Freezes the pursuer in place | 5 steps |

---

## Game Modes Explained

### Full Visibility
The agent can see the entire maze at all times. All six AI agents work in this mode. The game tree agents (Minimax, Alpha-Beta, Expectimax) perform best here.

### Fog of War (press `F` to enable)
The agent starts with a blank map and can only see cells within a short radius around itself. Unknown cells appear dark. The three fog-of-war agents (Online A*, Online LRTA*, Exploration) are designed for this mode. Online A* performs best here.

### Dynamic Walls (press `D` to enable)
Some walls randomly shift position every 5 steps. Agents that plan one step at a time (LRTA*, Online LRTA*) handle this naturally. Agents that plan full paths ahead may get caught off-guard when a wall moves.

### Split View (press `B` to enable)
When fog of war is active, pressing `B` shows two panels side by side — the left shows what the agent believes the maze looks like, and the right shows the true maze. Useful for understanding how the agent reasons.

---

## AI Agents — Quick Guide

### Full Visibility Agents

| Agent | How it works | Best for |
|-------|-------------|----------|
| **Alpha-Beta** | Looks 6 moves ahead, skips branches that can't change the outcome | Best overall win rate, fast |
| **Minimax** | Looks 4 moves ahead, assumes the pursuer plays perfectly | Strong but slower than Alpha-Beta |
| **Expectimax** | Like Minimax, but models the pursuer as partially random (50%) | Good when the pursuer is unpredictable |
| **LRTA*** | Makes exactly one move at a time using a learned score table | Extremely fast, handles shifting walls |
| **Beam Search** | Keeps the 3 most promising paths open at each step | Good balance of speed and win rate |
| **Hill Climbing** | Always moves toward the nearest-looking cell to the goal | Fast but can get stuck |

### Fog of War Agents (active when fog is on)

| Agent | How it works | Best for |
|-------|-------------|----------|
| **Online A*** | Replans a full path every step based on what it has seen so far | Best win rate in fog of war |
| **Online LRTA*** | One-step lookahead with a learned score table, fog-of-war version | Very fast, low compute |
| **Exploration** | Maps the maze first, then navigates to the goal | Dense mazes where the goal location is completely unknown |

---

## Pursuer Strategies

| Pursuer | Behavior | Difficulty |
|---------|----------|------------|
| **Random** | Picks any legal move at random | Easy |
| **Greedy** | Always moves to whichever neighboring cell is closest to the agent | Medium |
| **Beam** | Uses beam search to navigate around walls toward the agent | Medium-Hard |
| **A*** | Runs a full optimal pathfinding search to reach the agent every turn | Hard |

---

## Command Line Options

If you want to run a specific setup from the terminal:

```bash
python3 main.py --algo <agent> --pursuer <pursuer> --size <N> --density <0.0–1.0> --seed <number> --dynamic --visual
```

| Option | What it does | Default |
|--------|-------------|---------|
| `--algo` | Agent algorithm: `minimax`, `alpha_beta`, `expectimax`, `lrta`, `hill_climb`, `beam_search` | None (runs all) |
| `--pursuer` | Pursuer strategy: `random`, `greedy`, `beam`, `astar` | `greedy` |
| `--size` | Maze grid size (N×N) | 13 |
| `--density` | Fraction of cells that are walls (0.0 to 1.0) | 0.25 |
| `--seed` | Random seed for reproducible mazes | 7 |
| `--dynamic` | Enable shifting walls | Off |
| `--visual` | Launch the pygame window | Off |
| `--pursuers` | Number of pursuers (visual mode) | 2 |

**Examples:**

```bash
# Watch Alpha-Beta vs the hardest pursuer
python3 main.py --visual --algo alpha_beta --pursuer astar

# Test LRTA* on a dynamic maze in the terminal
python3 main.py --algo lrta --pursuer greedy --dynamic

# Play manually
python3 main.py --visual --algo manual
```

---

## Running Experiments

To run a full batch of experiments across all agents, pursuer types, maze sizes, and densities:

```bash
python3 experiments.py
```

Results are saved to `experiment_outputs/` as CSV files. Each run gets its own folder named by timestamp (e.g. `run_20260328_173816/`). Inside each folder:

| File | Contents |
|------|----------|
| `raw_results.csv` | One row per individual game |
| `summary_results.csv` | Averages grouped by agent, pursuer, maze type, and settings |
| `summary_by_agent.csv` | Overall win rate, steps, and runtime per agent |

You can open any of these in Excel or import them into Python with pandas.

---

## File Overview

| File | What it does |
|------|-------------|
| `main.py` | Main entry point — terminal and visual modes |
| `visual_game.py` | The pygame visual game |
| `maze.py` | Static and dynamic maze generation |
| `game_state.py` | Tracks the game state for adversarial search |
| `game_objects.py` | Traps, power-ups, and terrain types |
| `adversarial_search.py` | Minimax, Alpha-Beta, Expectimax, LRTA*, and pursuer AI |
| `search.py` | Hill Climbing, Beam Search, A*, BFS, DFS, UCS, Weighted A*, IDA* |
| `online_agent.py` | Online A*, Online LRTA*, and Exploration agents |
| `partial_maze.py` | Belief map and visibility logic for fog of war |
| `partial_game.py` | Fog of war game runner |
| `experiments.py` | Batch experiment runner and CSV export |
| `node.py` | Search node used by pathfinding algorithms |

---

## Troubleshooting

**pygame not found**
```bash
pip install pygame
```

**Game window is too small or too large**
The window size adjusts to your maze size. Try changing `--size` to 11 or 13.

**Game runs too fast or too slow**
Press `+` to speed up or `-` to slow down while the game is running.

**Agent seems stuck**
Some agents (Hill Climbing in particular) can get stuck in local dead-ends. Press `R` to generate a new maze, or press a number key to switch to a different agent.