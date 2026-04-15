from maze import Maze, DynamicMaze

UNKNOWN = -1
OPEN    =  0
WALL    =  1

class PartialMaze:
 

    def __init__(self, true_maze, visibility_radius=4):
        self.true_maze         = true_maze
        self.visibility_radius = visibility_radius
        self.rows              = true_maze.rows
        self.cols              = true_maze.cols
        self.start             = true_maze.start
        self.goal              = true_maze.goal

      
        self.belief_map = [
            [UNKNOWN] * self.cols
            for _ in range(self.rows)
        ]

        
        self.seen = set()

        self.update_belief(self.start)

    def get_visible_cells(self, pos):
    
        r, c = pos
        visible = []
        for dr in range(-self.visibility_radius, self.visibility_radius + 1):
            for dc in range(-self.visibility_radius, self.visibility_radius + 1):
                if abs(dr) + abs(dc) <= self.visibility_radius:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < self.rows and 0 <= nc < self.cols:
                        visible.append((nr, nc))
        return visible

    def update_belief(self, agent_pos):

        newly_revealed = set()
        for (r, c) in self.get_visible_cells(agent_pos):
            if self.belief_map[r][c] == UNKNOWN:
                newly_revealed.add((r, c))
            
            self.belief_map[r][c] = self.true_maze.grid[r][c]
            self.seen.add((r, c))
        return newly_revealed

    def cells_explored(self):
        return len(self.seen) / (self.rows * self.cols)

    

    def believed_walkable(self, pos):
        
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.belief_map[r][c] != WALL

    def get_believed_neighbors(self, pos):
       
        r, c = pos
        directions = {
            "UP":    (r - 1, c),
            "DOWN":  (r + 1, c),
            "LEFT":  (r, c - 1),
            "RIGHT": (r, c + 1),
        }
        result = []
        for action, neighbor in directions.items():
            if self.believed_walkable(neighbor):
                result.append((action, neighbor, 1))
        return result

    def is_goal(self, pos):
        return pos == self.goal

    def is_goal_known(self):
        
        gr, gc = self.goal
        return self.belief_map[gr][gc] != UNKNOWN

    def is_known_open(self, pos):
        r, c = pos
        return self.belief_map[r][c] == OPEN

    def is_unknown(self, pos):
        r, c = pos
        return self.belief_map[r][c] == UNKNOWN

   

    def display(self, agent=None, pursuer=None, label=""):
        
        if label:
            print(f"\n  [{label}]")

        
        visible_cells = set(self.get_visible_cells(agent)) if agent else set()
        pursuer_visible = pursuer and pursuer in visible_cells

        print("+" + "──" * self.cols + "+")
        for r in range(self.rows):
            row_str = "│"
            for c in range(self.cols):
                pos = (r, c)
                if pos == agent:
                    row_str += " A"
                elif pos == pursuer and pursuer_visible:
                    row_str += " P"
                elif pos == self.goal:
                    row_str += " G"
                elif pos == self.start:
                    row_str += " S"
                elif self.belief_map[r][c] == UNKNOWN:
                    row_str += " ?"
                elif self.belief_map[r][c] == WALL:
                    row_str += " #"
                else:
                    row_str += "  "
            row_str += " │"
            print(row_str)
        print("+" + "──" * self.cols + "+")
        explored_pct = self.cells_explored() * 100
        print(f"  explored: {explored_pct:.1f}%  |  radius: {self.visibility_radius}  |  "
              f"start:{self.start}  goal:{self.goal}\n")