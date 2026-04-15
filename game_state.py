import copy
from enum import Enum

# Keeps track of whose turn it is: the agent moves first, then the pursuer.
# This alternates back and forth every step throughout the game.
class Turn(Enum):
    AGENT   = "agent"
    PURSUER = "pursuer"

# A snapshot of everything happening in the game at one moment in time.
# Every time someone moves, we create a brand new GameState instead of changing the old one.
# This is important for the AI algorithms as they need to explore "what if" scenarios
# without accidentally messing up the real game state.
class GameState:
    

    def __init__(self, maze, agent_pos, pursuer_pos,
                 turn=Turn.AGENT, step=0, step_limit=200):
        self.maze        = maze
        self.agent_pos   = agent_pos
        self.pursuer_pos = pursuer_pos
        self.turn        = turn
        self.step        = step
        self.step_limit  = step_limit

  

    def is_terminal(self):
        return (
            self.agent_pos == self.maze.goal        or   
            self.agent_pos == self.pursuer_pos      or   
            self.step >= self.step_limit                  
        )

    def agent_won(self):
        return self.agent_pos == self.maze.goal and self.agent_pos != self.pursuer_pos

    def pursuer_won(self):
        return self.agent_pos == self.pursuer_pos

    def is_timeout(self):
        return self.step >= self.step_limit and not self.agent_won() and not self.pursuer_won()

    

    def utility(self):
        
        if self.agent_won():
            return 1000
        if self.pursuer_won():
            return -1000
        return 0

    def heuristic_eval(self):
        
        ad = _manhattan(self.agent_pos,   self.maze.goal)
        pd = _manhattan(self.pursuer_pos, self.maze.goal)
        sd = _manhattan(self.agent_pos,   self.pursuer_pos)

       
        return (pd - ad) * 10 + sd * 5



    def get_agent_moves(self):
        
        moves = []
        for action, pos, _ in self.maze.get_neighbors(self.agent_pos):
            moves.append((action, pos))
        
        moves.append(("STAY", self.agent_pos))
        return moves

    def get_pursuer_moves(self):
        
        moves = []
        for action, pos, _ in self.maze.get_neighbors(self.pursuer_pos):
            moves.append((action, pos))
        moves.append(("STAY", self.pursuer_pos))
        return moves



    def apply_agent_move(self, new_agent_pos):
       
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
        
        new = GameState(
            maze        = self.maze,
            agent_pos   = self.agent_pos,
            pursuer_pos = new_pursuer_pos,
            turn        = Turn.AGENT,
            step        = self.step + 1,
            step_limit  = self.step_limit,
        )
        return new

  

    def display(self, label=""):
        self.maze.display(
            agent   = self.agent_pos,
            pursuer = self.pursuer_pos,
            label   = label or f"Step {self.step} | Turn: {self.turn.value}"
        )

    def __repr__(self):
        return (f"GameState(agent={self.agent_pos}, pursuer={self.pursuer_pos}, "
                f"turn={self.turn.value}, step={self.step})")






def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])