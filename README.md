# Maze Navigator - Adversarial AI

Foundations of Artificial Intelligence project on **adversarial search, partial observability, dynamic environments, and experimental evaluation**.

Authors :- ***Vikramaditya Sogani & Namya Singh***

This project studies how different AI agents navigate a maze while avoiding one or more pursuers under several increasingly difficult settings:

- **Full observability**: the entire maze is known
- **Partial observability**: the agent sees only a limited radius and must act under uncertainty
- **Dynamic mazes**: walls can shift during gameplay
- **Adversarial pursuit**: the agent must avoid a pursuer that uses different chase strategies

In addition to terminal simulations, the project includes:
- a **Pygame visual simulator**
- a **batch experiment runner**
- **CSV result export**
- **summary tables and plots**

---

## Project Goals

This project was designed to compare multiple AI paradigms in a single environment:

- **Adversarial search**: Minimax, Alpha-Beta, Expectimax
- **Online / real-time search**: LRTA*
- **Approximate / local search**: Hill Climbing, Beam Search
- **Partial-observability agents**: Online A*, Online LRTA*, Exploration agent
- **Pursuer strategies**: Random, Greedy, Beam Search, A*

The central question is:

> How do different decision-making methods behave under pursuit, uncertainty, and dynamic maze changes?

---

## Repository Structure

### Core files

- `main.py`  
  Full-observability game runner, demos, and terminal comparison tables

- `visual_game.py`  
  Pygame-based visual simulator with in-game controls, algorithm switching, fog of war, split view, and dynamic walls

- `search.py`  
  Classical and approximate search methods used by the project

- `adversarial_search.py`  
  Adversarial algorithms and pursuer AI

- `game_state.py`  
  Shared turn-based game state used by adversarial search

- `maze.py`  
  Static and dynamic maze generation and utilities

- `game_objects.py`  
  Terrain, traps, power-ups, pursuer/agent status effects, and object placement

### Partial observability

- `partial_game.py`  
  Partial-observability simulator and comparison runner

- `partial_maze.py`  
  Belief-map / visibility logic for limited observability

- `online_agent.py`  
  Online A*, Online LRTA*, and exploration-based agents

### Experiments

- `experiments.py`  
  Batch experiment runner for:
  - full mode
  - partial mode
  - static vs dynamic
  - repeated trials
  - CSV export
  - summaries
  - plots / heatmaps

### Other

- `node.py`  
  Search node representation

- `test_partial.py`  
  Partial-observability testing utilities

---

## Implemented Algorithms

## Agent algorithms (full / visual mode)

- **Minimax**
- **Alpha-Beta**
- **Expectimax**
- **LRTA***
- **Hill Climbing**
- **Beam Search**
- **Manual control** (visual mode only)

`main.py` currently runs the full terminal simulation with: Minimax, Alpha-Beta, Expectimax, LRTA*, Hill Climbing, and Beam Search. 

## Partial-observability agent algorithms

- **Online A\***
- **Online LRTA\***
- **Exploration agent**

These are used by `partial_game.py` and `experiments.py` for belief-state / limited-visibility runs. :contentReference[oaicite:2]{index=2}

## Pursuer strategies

- **Random**
- **Greedy**
- **Beam Search**
- **A\***

Both `main.py` and the experiment pipeline include all four pursuer strategies. 

---

## Environment Features

The maze environment includes more than simple walls and open cells.

### Dynamic environment
- Walls can shift during play
- Dynamic settings are supported in terminal mode, visual mode, and experiments

### Partial observability
- Fog of war
- Limited visibility radius
- Belief-state reasoning
- Split-view visual mode to compare visible/believed world vs true maze state

### Gameplay objects / terrain
- Traps
- Power-ups
- Mud
- Water
- Road

These enrich the environment and make the visual mode more than a plain pathfinding demo. :contentReference[oaicite:4]{index=4}

---

## Running the Project

## 1. Visual mode

Launch the visual simulator:

```bash
python main.py --visual