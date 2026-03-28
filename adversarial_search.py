

import math
import time
from dataclasses import dataclass, field
from typing import Optional

from game_state import GameState, Turn, _manhattan
from search import astar, beam_search



#  result container


@dataclass
class AdversarialResult:
    algorithm      : str
    best_action    : Optional[str]  = None
    best_move_pos  : Optional[tuple]= None
    value          : float          = 0
    nodes_expanded : int            = 0
    depth_reached  : int            = 0
    runtime_ms     : float          = 0
    pruned         : int            = 0    # alpha-beta only

    def summary(self):
        return (
            f"\n{'─'*42}\n"
            f"  algorithm     : {self.algorithm}\n"
            f"  best action   : {self.best_action} → {self.best_move_pos}\n"
            f"  value (agent) : {self.value:.2f}\n"
            f"  nodes expanded: {self.nodes_expanded}\n"
            f"  depth reached : {self.depth_reached}\n"
            + (f"  branches pruned: {self.pruned}\n" if self.pruned else "") +
            f"  runtime       : {self.runtime_ms:.3f} ms\n"
            f"{'─'*42}"
        )



#  minimax


def minimax(game_state, depth_limit=4):
    """
    Minimax search from the agent's perspective.

    - MAX node : agent's turn  (maximize utility)
    - MIN node : pursuer's turn (minimize utility)
    - Depth-limited with heuristic evaluation at leaves.

    Args:
        game_state  : current GameState
        depth_limit : how many plies to look ahead (1 ply = one agent OR pursuer move)

    Returns:
        AdversarialResult with best action for the agent
    """
    t0       = time.perf_counter()
    expanded = [0]

    def _max(state, depth):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        best = -math.inf
        for action, new_pos in state.get_agent_moves():
            child = state.apply_agent_move(new_pos)
            val   = _min(child, depth - 1)
            best  = max(best, val)
        return best

    def _min(state, depth):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        best = math.inf
        for action, new_pos in state.get_pursuer_moves():
            child = state.apply_pursuer_move(new_pos)
            val   = _max(child, depth - 1)
            best  = min(best, val)
        return best

    # Root: agent is MAX
    best_val    = -math.inf
    best_action = None
    best_pos    = None

    for action, new_pos in game_state.get_agent_moves():
        child = game_state.apply_agent_move(new_pos)
        val   = _min(child, depth_limit - 1)
        if val > best_val:
            best_val    = val
            best_action = action
            best_pos    = new_pos

    return AdversarialResult(
        algorithm      = "Minimax",
        best_action    = best_action,
        best_move_pos  = best_pos,
        value          = best_val,
        nodes_expanded = expanded[0],
        depth_reached  = depth_limit,
        runtime_ms     = (time.perf_counter() - t0) * 1000,
    )



#  alpha-beta pruning


def alpha_beta(game_state, depth_limit=6):
    """
    Alpha-Beta pruning — same as Minimax but prunes branches that cannot
    affect the final decision.

    Typically allows 2× deeper search than plain Minimax in same time budget.

    Args:
        game_state  : current GameState
        depth_limit : ply depth limit (can be higher than Minimax due to pruning)

    Returns:
        AdversarialResult with best action + pruning count
    """
    t0       = time.perf_counter()
    expanded = [0]
    pruned   = [0]

    def _max(state, depth, alpha, beta):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        val = -math.inf
        for action, new_pos in state.get_agent_moves():
            child = state.apply_agent_move(new_pos)
            val   = max(val, _min(child, depth - 1, alpha, beta))
            if val >= beta:
                pruned[0] += 1
                return val              # β cut-off
            alpha = max(alpha, val)
        return val

    def _min(state, depth, alpha, beta):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        val = math.inf
        for action, new_pos in state.get_pursuer_moves():
            child = state.apply_pursuer_move(new_pos)
            val   = min(val, _max(child, depth - 1, alpha, beta))
            if val <= alpha:
                pruned[0] += 1
                return val              # α cut-off
            beta = min(beta, val)
        return val

    best_val    = -math.inf
    best_action = None
    best_pos    = None
    alpha       = -math.inf
    beta        = math.inf

    for action, new_pos in game_state.get_agent_moves():
        child = game_state.apply_agent_move(new_pos)
        val   = _min(child, depth_limit - 1, alpha, beta)
        if val > best_val:
            best_val    = val
            best_action = action
            best_pos    = new_pos
        alpha = max(alpha, best_val)

    return AdversarialResult(
        algorithm      = "Alpha-Beta",
        best_action    = best_action,
        best_move_pos  = best_pos,
        value          = best_val,
        nodes_expanded = expanded[0],
        depth_reached  = depth_limit,
        runtime_ms     = (time.perf_counter() - t0) * 1000,
        pruned         = pruned[0],
    )



#  expectimax


def expectimax(game_state, depth_limit=4, pursuer_randomness=0.5):
    """
    Expectimax — pursuer is modeled as a probabilistic agent, not perfectly rational.

    The pursuer node is now a CHANCE node:
      - With probability (1 - pursuer_randomness) it picks the best move (greedy A*)
      - With probability pursuer_randomness it picks uniformly at random

    This models a realistic pursuer that isn't perfectly optimal.

    Args:
        game_state         : current GameState
        depth_limit        : ply depth
        pursuer_randomness : 0.0 = pursuer fully rational (→ same as Minimax)
                             1.0 = pursuer fully random

    Returns:
        AdversarialResult with best action for the agent
    """
    t0       = time.perf_counter()
    expanded = [0]

    def _greedy_pursuer_action(state):
        """Pursuer's 'best' move: the one that minimizes distance to agent."""
        moves = state.get_pursuer_moves()
        if not moves:
            return None, state.pursuer_pos
        return min(moves, key=lambda m: _manhattan(m[1], state.agent_pos))

    def _max_node(state, depth):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        best = -math.inf
        for action, new_pos in state.get_agent_moves():
            child = state.apply_agent_move(new_pos)
            val   = _chance_node(child, depth - 1)
            best  = max(best, val)
        return best

    def _chance_node(state, depth):
        expanded[0] += 1
        if state.is_terminal() or depth == 0:
            return state.utility() if state.is_terminal() else state.heuristic_eval()

        moves = state.get_pursuer_moves()
        if not moves:
            return _max_node(state.apply_pursuer_move(state.pursuer_pos), depth - 1)

        _, greedy_pos = _greedy_pursuer_action(state)
        n             = len(moves)
        expected      = 0.0

        for action, new_pos in moves:
            # Probability: greedy move gets extra weight
            if new_pos == greedy_pos:
                prob = (1 - pursuer_randomness) + pursuer_randomness / n
            else:
                prob = pursuer_randomness / n

            child    = state.apply_pursuer_move(new_pos)
            expected += prob * _max_node(child, depth - 1)

        return expected

    best_val    = -math.inf
    best_action = None
    best_pos    = None

    for action, new_pos in game_state.get_agent_moves():
        child = game_state.apply_agent_move(new_pos)
        val   = _chance_node(child, depth_limit - 1)
        if val > best_val:
            best_val    = val
            best_action = action
            best_pos    = new_pos

    return AdversarialResult(
        algorithm      = f"Expectimax(r={pursuer_randomness})",
        best_action    = best_action,
        best_move_pos  = best_pos,
        value          = best_val,
        nodes_expanded = expanded[0],
        depth_reached  = depth_limit,
        runtime_ms     = (time.perf_counter() - t0) * 1000,
    )



#  4. LRTA* (Learning Real-Time A*)


class LRTAStar:
    """
    Learning Real-Time A* — an online search algorithm that:
      - Makes ONE move per step (real-time, no full lookahead)
      - Maintains an H-table of learned heuristic values updated after each move
      - Naturally handles shifting walls because it replans every single step

    This is the RIGHT algorithm for DynamicMaze: classical A* would plan a
    full path upfront that becomes invalid when walls shift.

    Usage:
        agent = LRTAStar(maze)
        action, new_pos = agent.step(current_pos)   # call once per game step
    """

    def __init__(self, maze):
        self.maze   = maze
        self.H      = {}    # learned heuristic table: pos → h-value
        self.moves  = 0
        self.total_runtime_ms = 0

    def _h(self, pos):
        """Returns learned h-value, defaulting to manhattan if not yet learned."""
        if pos not in self.H:
            self.H[pos] = _manhattan(pos, self.maze.goal)
        return self.H[pos]

    def step(self, current_pos):
        """
        Compute the next move from current_pos.

        LRTA* update rule:
          H[current] ← max(H[current],  min over neighbors n of: cost(current→n) + H[n])
          Then move to the neighbor that minimizes cost + H[neighbor]

        Returns:
            (action, new_pos)
        """
        t0 = time.perf_counter()

        if self.maze.is_goal(current_pos):
            return ("STAY", current_pos)

        neighbors = self.maze.get_neighbors(current_pos)

        if not neighbors:
            return ("STAY", current_pos)

        # LRTA* update: raise H[current] to be consistent with best neighbor
        min_neighbor_cost = min(cost + self._h(npos) for _, npos, cost in neighbors)
        self.H[current_pos] = max(self._h(current_pos), min_neighbor_cost)

        # Move to best neighbor (lowest cost + learned h)
        best_action, best_pos, _ = min(neighbors, key=lambda t: t[2] + self._h(t[1]))

        self.moves += 1
        self.total_runtime_ms += (time.perf_counter() - t0) * 1000
        return (best_action, best_pos)

    def reset(self, maze=None):
        """Reset H-table (use when maze changes drastically). Optionally swap maze."""
        self.H     = {}
        self.moves = 0
        if maze:
            self.maze = maze





class PursuerAI:
    """
    Controls the pursuer's movement each turn.
    Four difficulty levels:
      'random'  — moves randomly (easy)
      'greedy'  — always moves toward agent via manhattan (medium)
      'beam'    — bounded beam search pursuit toward agent (medium-hard)
      'astar'   — full A* toward agent position (hard)
    """

    def __init__(self, strategy="greedy",beam_width=3):
        assert strategy in ("random", "greedy", "astar", "beam"), \
            "strategy must be 'random', 'greedy', 'astar' or 'beam'"
        self.strategy = strategy
        self.beam_width = beam_width
        import random as _r
        self._rng = _r

    def choose_move(self, game_state):
        """
        Returns (action, new_pursuer_pos) given the current game state.
        """
        moves = game_state.get_pursuer_moves()
        # Filter out STAY unless it's the only option
        moving = [(a, p) for a, p in moves if a != "STAY"]
        if not moving:
            return ("STAY", game_state.pursuer_pos)

        if self.strategy == "random":
            return self._rng.choice(moving)

        elif self.strategy == "greedy":
            return min(moving, key=lambda m: _manhattan(m[1], game_state.agent_pos))


        elif self.strategy == "astar":
            from maze import Maze
            pursuer_pos = tuple(game_state.pursuer_pos)
            agent_pos = tuple(game_state.agent_pos)
            grid_copy = [row[:] for row in game_state.maze.grid]
            grid_copy[pursuer_pos[0]][pursuer_pos[1]] = Maze.OPEN
            grid_copy[agent_pos[0]][agent_pos[1]] = Maze.OPEN
            temp = Maze(grid_copy, pursuer_pos, agent_pos)
            result = astar(temp, "manhattan")
            if result.success and len(result.actions) > 0:
                first_action = result.actions[0]
                for a, p in moving:
                    if a == first_action:
                        return (a, p)
            return min(moving, key=lambda m: _manhattan(m[1], game_state.agent_pos))


        elif self.strategy == "beam":
            from maze import Maze
            pursuer_pos = tuple(game_state.pursuer_pos)
            agent_pos = tuple(game_state.agent_pos)
            grid_copy = [row[:] for row in game_state.maze.grid]
            grid_copy[pursuer_pos[0]][pursuer_pos[1]] = Maze.OPEN
            grid_copy[agent_pos[0]][agent_pos[1]] = Maze.OPEN
            temp = Maze(grid_copy, pursuer_pos, agent_pos)
            result = beam_search(
                maze=temp,
                beam_width=self.beam_width,
                heuristic_name="manhattan",
                start=pursuer_pos
            )
            if result.success and len(result.actions) > 0:
                first_action = result.actions[0]
                for a, p in moving:
                    if a == first_action:
                        return (a, p)
            return min(moving, key=lambda m: _manhattan(m[1], game_state.agent_pos))
