

import copy
from enum import Enum


class Turn(Enum):
    AGENT   = "agent"
    PURSUER = "pursuer"


class GameState:
    """
    Full state of one moment in the adversarial maze game.

    Win conditions:
      - Agent wins  : agent reaches maze.goal
      - Pursuer wins: pursuer occupies same cell as agent (caught)
      - Draw/timeout: step_limit exceeded

    Args:
        maze       : Maze or DynamicMaze instance
        agent_pos  : (row, col) of the agent
        pursuer_pos: (row, col) of the pursuer
        turn       : whose move it is (Turn.AGENT or Turn.PURSUER)
        step       : current step number
        step_limit : maximum steps before timeout (default 200)
    """

    def __init__(self, maze, agent_pos, pursuer_pos,
                 turn=Turn.AGENT, step=0, step_limit=200):
        self.maze        = maze
        self.agent_pos   = agent_pos
        self.pursuer_pos = pursuer_pos
        self.turn        = turn
        self.step        = step
        self.step_limit  = step_limit

    # terminal checks 

    def is_terminal(self):
        return (
            self.agent_pos == self.maze.goal        or   # agent wins
            self.agent_pos == self.pursuer_pos      or   # pursuer catches agent
            self.step >= self.step_limit                  # timeout
        )

    def agent_won(self):
        return self.agent_pos == self.maze.goal and self.agent_pos != self.pursuer_pos

    def pursuer_won(self):
        return self.agent_pos == self.pursuer_pos

    def is_timeout(self):
        return self.step >= self.step_limit and not self.agent_won() and not self.pursuer_won()

    # utility 

    def utility(self):
        """
        Terminal utility from the AGENT's perspective.
          +1000 : agent wins
          -1000 : pursuer wins (agent caught)
              0 : timeout
        """
        if self.agent_won():
            return 1000
        if self.pursuer_won():
            return -1000
        return 0

    def heuristic_eval(self):
        """
        Non-terminal heuristic evaluation from agent's perspective.
        Used by Minimax with depth limit and Expectimax.

        Score = (pursuer_dist_to_goal - agent_dist_to_goal) * 10
                - agent_dist_to_pursuer * 5

        Higher is better for the agent.
        """
        ad = _manhattan(self.agent_pos,   self.maze.goal)
        pd = _manhattan(self.pursuer_pos, self.maze.goal)
        sd = _manhattan(self.agent_pos,   self.pursuer_pos)

        # Agent wants: small ad (close to goal), large sd (far from pursuer)
        return (pd - ad) * 10 + sd * 5

    #  move generation 

    def get_agent_moves(self):
        """Returns list of (action, new_agent_pos) for all legal agent moves."""
        moves = []
        for action, pos, _ in self.maze.get_neighbors(self.agent_pos):
            moves.append((action, pos))
        # Agent can also STAY (useful in adversarial settings)
        moves.append(("STAY", self.agent_pos))
        return moves

    def get_pursuer_moves(self):
        """Returns list of (action, new_pursuer_pos) for all legal pursuer moves."""
        moves = []
        for action, pos, _ in self.maze.get_neighbors(self.pursuer_pos):
            moves.append((action, pos))
        moves.append(("STAY", self.pursuer_pos))
        return moves

    #  state transitions 

    def apply_agent_move(self, new_agent_pos):
        """Returns a new GameState after agent moves. Does NOT mutate self."""
        new = GameState(
            maze        = self.maze,
            agent_pos   = new_agent_pos,
            pursuer_pos = self.pursuer_pos,
            turn        = Turn.PURSUER,
            step        = self.step + 1,
            step_limit  = self.step_limit,
        )
        return new

    def apply_pursuer_move(self, new_pursuer_pos):
        """Returns a new GameState after pursuer moves. Does NOT mutate self."""
        new = GameState(
            maze        = self.maze,
            agent_pos   = self.agent_pos,
            pursuer_pos = new_pursuer_pos,
            turn        = Turn.AGENT,
            step        = self.step + 1,
            step_limit  = self.step_limit,
        )
        return new

    # display 

    def display(self, label=""):
        self.maze.display(
            agent   = self.agent_pos,
            pursuer = self.pursuer_pos,
            label   = label or f"Step {self.step} | Turn: {self.turn.value}"
        )

    def __repr__(self):
        return (f"GameState(agent={self.agent_pos}, pursuer={self.pursuer_pos}, "
                f"turn={self.turn.value}, step={self.step})")



# utility helpers


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])