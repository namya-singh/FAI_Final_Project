"""
partial_maze.py — partially observable maze environment
project: adversarial maze navigation
authors: vikramaditya sogani & namya singh

the agent cannot see the full maze. it only perceives cells within
a visibility radius, and maintains a belief map of what it has
discovered so far. everything outside the radius is unknown.

cell states in the belief map:
  UNKNOWN = -1  (never seen)
  OPEN    =  0  (seen and walkable)
  WALL    =  1  (seen and blocked)

key idea: the agent makes decisions using belief_map, not the true maze.
the true maze is only used to resolve what the agent actually sees
when it steps into a new area.
"""

from maze import Maze, DynamicMaze


# ─────────────────────────────────────────────
#  cell state constants
# ─────────────────────────────────────────────

UNKNOWN = -1
OPEN    =  0
WALL    =  1


# ─────────────────────────────────────────────
#  partial maze
# ─────────────────────────────────────────────

class PartialMaze:
    """
    wraps a true Maze (or DynamicMaze) and exposes only what the agent
    has observed so far via a belief map.

    args:
        true_maze       : the actual Maze or DynamicMaze (ground truth)
        visibility_radius : how many cells the agent can see in each
                           direction (manhattan distance). default 4.

    the agent interacts with this class instead of the true maze.
    it calls update_belief(pos) each step to reveal nearby cells.
    """

    def __init__(self, true_maze, visibility_radius=4):
        self.true_maze         = true_maze
        self.visibility_radius = visibility_radius
        self.rows              = true_maze.rows
        self.cols              = true_maze.cols
        self.start             = true_maze.start
        self.goal              = true_maze.goal

        # belief map: what the agent thinks the maze looks like
        # starts fully unknown
        self.belief_map = [
            [UNKNOWN] * self.cols
            for _ in range(self.rows)
        ]

        # track which cells have ever been seen (for display)
        self.seen = set()

        # reveal cells around the start position immediately
        self.update_belief(self.start)

    # ── visibility ───────────────────────────

    def get_visible_cells(self, pos):
        """
        returns all cells within visibility_radius manhattan distance
        of pos that are within bounds.
        """
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
        """
        reveal all cells within visibility_radius of agent_pos.
        updates belief_map from the true maze.
        call this every time the agent moves.

        returns:
            newly_revealed : set of cells revealed this step (useful for analysis)
        """
        newly_revealed = set()
        for (r, c) in self.get_visible_cells(agent_pos):
            if self.belief_map[r][c] == UNKNOWN:
                newly_revealed.add((r, c))
            # always update — walls may have shifted in dynamic maze
            self.belief_map[r][c] = self.true_maze.grid[r][c]
            self.seen.add((r, c))
        return newly_revealed

    def cells_explored(self):
        """returns fraction of maze cells that have been seen (0.0 to 1.0)."""
        return len(self.seen) / (self.rows * self.cols)

    # ── belief-based navigation ──────────────

    def believed_walkable(self, pos):
        """
        returns True if the agent believes pos is walkable.
        unknown cells are treated as walkable (optimistic assumption —
        encourages exploration rather than freezing at unknown borders).
        """
        r, c = pos
        if not (0 <= r < self.rows and 0 <= c < self.cols):
            return False
        return self.belief_map[r][c] != WALL

    def get_believed_neighbors(self, pos):
        """
        returns (action, neighbor_pos, cost) for neighbors the agent
        believes are walkable based on its current belief map.
        unknown cells are included (optimistic).
        """
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
        """returns True if the agent has seen the goal cell."""
        gr, gc = self.goal
        return self.belief_map[gr][gc] != UNKNOWN

    def is_known_open(self, pos):
        r, c = pos
        return self.belief_map[r][c] == OPEN

    def is_unknown(self, pos):
        r, c = pos
        return self.belief_map[r][c] == UNKNOWN

    # ── display ──────────────────────────────

    def display(self, agent=None, pursuer=None, label=""):
        """
        prints the belief map from the agent's perspective.

        legend:
          #  = known wall
          .  = known open
          ?  = unknown (never seen)
          A  = agent
          P  = pursuer (only shown if within visibility radius)
          S  = start
          G  = goal (always shown — agent knows goal location)
        """
        if label:
            print(f"\n  [{label}]")

        # pursuer is only visible if within agent's radius
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