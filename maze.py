import random
import copy
from collections import deque

# A standard grid maze where each cell is either open floor or a wall.
# The grid is stored as a 2D list of numbers: 0 means open, 1 means wall.
class Maze:
    OPEN = 0
    WALL = 1

    def __init__(self, grid, start, goal):
        self.grid  = [row[:] for row in grid]   
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
    
     # Returns every cell the agent can legally step into from the current position.
    def get_neighbors(self, pos):

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
 
    @classmethod
    def from_string(cls, text):
        
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

    @classmethod
    def generate_random(cls, rows=15, cols=15, obstacle_density=0.28, seed=None):
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
                if 0<=nr<rows and 0<=nc<cols and grid[nr][nc]!=1 and (nr,nc) not in visited:
                    visited.add((nr,nc))
                    queue.append((nr,nc))
        return False

# A maze that changes while the game is being played.
# Every few steps, a handful of walls swap positions: some walls disappear and new ones appear elsewhere.
# This forces agents that planned a full route ahead of time to adapt 
class DynamicMaze(Maze):
    

    def __init__(self, grid, start, goal, shift_interval=5, num_shifts=2, seed=None):
        super().__init__(grid, start, goal)
        self.shift_interval = shift_interval
        self.num_shifts     = num_shifts
        self.step_count     = 0
        self.shift_history  = []   
        if seed is not None:
            random.seed(seed)

    def step(self, protected_positions=None):
        
        self.step_count += 1
        if self.step_count % self.shift_interval == 0:
            return self._shift_walls(protected_positions or set())
        return False

    def _shift_walls(self, protected):
        
        always_open = {self.start, self.goal} | set(protected)

        
        removable_walls = [
            (r, c)
            for r in range(1, self.rows - 1)
            for c in range(1, self.cols - 1)
            if self.grid[r][c] == self.WALL
        ]

        
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

        
        for (r, c) in walls_to_remove:
            self.grid[r][c] = self.OPEN
        for (r, c) in cells_to_fill:
            self.grid[r][c] = self.WALL

        
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
        new = DynamicMaze(
            [row[:] for row in self.grid],  
            self.start, self.goal,
            self.shift_interval, self.num_shifts
        )
        new.step_count    = self.step_count
        new.shift_history = self.shift_history[:]
        return new

    @classmethod
    def from_static(cls, maze, shift_interval=5, num_shifts=2, seed=None):

        return cls([row[:] for row in maze.grid], maze.start, maze.goal, shift_interval, num_shifts, seed)