"""
maze.py — Static + Dynamic Maze Environments
Project: Adversarial Maze Navigation
Authors: VikramAditya Sogani & Namya Singh

Classes:
  Maze        — original static 2D grid maze
  DynamicMaze — extends Maze; walls shift every N steps
"""

import random
import copy
from collections import deque



#  BASE MAZE (static)


class Maze:
    OPEN = 0
    WALL = 1

    def __init__(self, grid, start, goal):
        self.grid  = [row[:] for row in grid]   # deep copy
        self.rows  = len(grid)
        self.cols  = len(grid[0])
        self.start = start
        self.goal  = goal

        assert self._in_bounds(start), "Start is out of bounds"
        assert self._in_bounds(goal),  "Goal is out of bounds"
        assert grid[start[0]][start[1]] != self.WALL, "Start is a wall"
        assert grid[goal[0]][goal[1]]   != self.WALL, "Goal is a wall"

    def _in_bounds(self, pos):
        r, c = pos
        return 0 <= r < self.rows and 0 <= c < self.cols

    def is_walkable(self, pos):
        r, c = pos
        return self._in_bounds(pos) and self.grid[r][c] == self.OPEN

    def get_neighbors(self, pos):
        """Returns list of (action, neighbor_pos, step_cost)."""
        r, c = pos
        directions = {
            "UP":    (r - 1, c),
            "DOWN":  (r + 1, c),
            "LEFT":  (r, c - 1),
            "RIGHT": (r, c + 1),
        }
        result = []
        for action, neighbor in directions.items():
            if self.is_walkable(neighbor):
                result.append((action, neighbor, 1))
        return result

    def is_goal(self, pos):
        return pos == self.goal

    def display(self, agent=None, pursuer=None, path=None, visited=None, label=""):
        """
        Prints maze to terminal.
        Legend:  # wall  . open  S start  G goal
                 A agent  P pursuer  * path  o visited
        """
        path_set    = set(path)    if path    else set()
        visited_set = set(visited) if visited else set()

        if label:
            print(f"\n  [{label}]")
        print("+" + "──" * self.cols + "+")
        for r in range(self.rows):
            row_str = "│"
            for c in range(self.cols):
                pos = (r, c)
                if pos == agent:
                    row_str += " A"
                elif pos == pursuer:
                    row_str += " P"
                elif pos == self.start:
                    row_str += " S"
                elif pos == self.goal:
                    row_str += " G"
                elif pos in path_set:
                    row_str += " *"
                elif pos in visited_set:
                    row_str += " o"
                elif self.grid[r][c] == self.WALL:
                    row_str += " #"
                else:
                    row_str += "  "
            row_str += " │"
            print(row_str)
        print("+" + "──" * self.cols + "+")
        print(f"  Start:{self.start}  Goal:{self.goal}  Size:{self.rows}×{self.cols}\n")

    # ── Factory: load from string ────────────

    @classmethod
    def from_string(cls, text):
        """
        Load maze from multi-line string.
        '#'=wall, ' '/'.'=open, 'S'=start, 'G'=goal
        """
        lines = text.strip().splitlines()
        grid  = []
        start = goal = None
        for r, line in enumerate(lines):
            row = []
            for c, ch in enumerate(line):
                if ch == 'S':
                    start = (r, c); row.append(cls.OPEN)
                elif ch == 'G':
                    goal  = (r, c); row.append(cls.OPEN)
                elif ch == '#':
                    row.append(cls.WALL)
                else:
                    row.append(cls.OPEN)
            grid.append(row)
        assert start and goal, "Maze string must contain 'S' and 'G'"
        return cls(grid, start, goal)

    # ── Factory: random generation ───────────

    @classmethod
    def generate_random(cls, rows=15, cols=15, obstacle_density=0.28, seed=None):
        """Generates a random maze guaranteed to have a path from start to goal."""
        if seed is not None:
            random.seed(seed)
        start = (0, 0)
        goal  = (rows - 1, cols - 1)
        while True:
            grid = []
            for r in range(rows):
                row = []
                for c in range(cols):
                    if (r, c) in (start, goal):
                        row.append(cls.OPEN)
                    else:
                        row.append(cls.WALL if random.random() < obstacle_density else cls.OPEN)
                grid.append(row)
            if cls._path_exists(grid, start, goal, rows, cols):
                return cls(grid, start, goal)

    @staticmethod
    def _path_exists(grid, start, goal, rows, cols):
        visited = {start}
        queue   = deque([start])
        while queue:
            r, c = queue.popleft()
            if (r, c) == goal:
                return True
            for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
                nr, nc = r+dr, c+dc
                if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]==0 and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return False


#  DYNAMIC MAZE (walls shift every N steps)


class DynamicMaze(Maze):
    """
    Extends Maze: a fixed number of walls shift positions every `shift_interval` steps.
    
    Rules:
      - Only interior non-border cells can become walls or open up
      - Start, goal, agent position, and pursuer position are always protected
      - Connectivity (start ↔ goal) is re-verified after every shift; if broken, shift is rolled back
    
    Args:
        grid             : 2D list of 0s and 1s
        start            : (row, col)
        goal             : (row, col)
        shift_interval   : how many game steps between wall shifts (default: 5)
        num_shifts       : how many wall cells move per shift event (default: 2)
        seed             : random seed for reproducibility
    """

    def __init__(self, grid, start, goal, shift_interval=5, num_shifts=2, seed=None):
        super().__init__(grid, start, goal)
        self.shift_interval = shift_interval
        self.num_shifts     = num_shifts
        self.step_count     = 0
        self.shift_history  = []   # list of shift events for analysis
        if seed is not None:
            random.seed(seed)

    def step(self, protected_positions=None):
        """
        Advance the maze by one game step.
        If step_count hits shift_interval, trigger a wall shift.
        
        Args:
            protected_positions : set of (row,col) that must stay open (agent + pursuer)
        
        Returns:
            shifted (bool) : True if a wall shift occurred this step
        """
        self.step_count += 1
        if self.step_count % self.shift_interval == 0:
            return self._shift_walls(protected_positions or set())
        return False

    def _shift_walls(self, protected):
        """
        Moves `num_shifts` walls to new random open positions.
        Rolls back if connectivity breaks.
        """
        always_open = {self.start, self.goal} | set(protected)

        # Candidate walls that can be removed (interior, not protecting anything)
        removable_walls = [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if self.grid[r][c] == self.WALL
        ]

        # Candidate open cells that can become walls
        fillable_cells = [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if self.grid[r][c] == self.OPEN and (r, c) not in always_open
        ]

        if not removable_walls or not fillable_cells:
            return False

        n = min(self.num_shifts, len(removable_walls), len(fillable_cells))
        walls_to_remove = random.sample(removable_walls, n)
        cells_to_fill   = random.sample(fillable_cells,  n)

        # Apply tentative shift
        for (r, c) in walls_to_remove:
            self.grid[r][c] = self.OPEN
        for (r, c) in cells_to_fill:
            self.grid[r][c] = self.WALL

        # Verify connectivity — rollback if broken
        if not self._path_exists(self.grid, self.start, self.goal, self.rows, self.cols):
            for (r, c) in walls_to_remove:
                self.grid[r][c] = self.WALL
            for (r, c) in cells_to_fill:
                self.grid[r][c] = self.OPEN
            return False

        event = {"step": self.step_count, "removed": walls_to_remove, "added": cells_to_fill}
        self.shift_history.append(event)
        return True

    def clone(self):
        """Returns a deep copy of this DynamicMaze (used by search algorithms for lookahead)."""
        new = DynamicMaze(
            self.grid, self.start, self.goal,
            self.shift_interval, self.num_shifts
        )
        new.step_count    = self.step_count
        new.shift_history = self.shift_history[:]
        return new

    @classmethod
    def from_static(cls, maze, shift_interval=5, num_shifts=2, seed=None):
        """Upgrade a static Maze into a DynamicMaze."""
        return cls(maze.grid, maze.start, maze.goal, shift_interval, num_shifts, seed)