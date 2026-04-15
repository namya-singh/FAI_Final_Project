import heapq
import math
from collections import deque
from partial_maze import PartialMaze, UNKNOWN, WALL, OPEN


def _manhattan(a, b):
    return abs(a[0] - b[0]) + abs(a[1] - b[1])

# The smartest fog of war agent. It can't see the full maze but it doesn't let that stop it.
# Every single step it replans its entire path from scratch using only what it has discovered so far.
class OnlineAstarAgent:
    

    def __init__(self, partial_maze):
        self.pm        = partial_maze
        self.plan      = []   
        self.replans   = 0    
        self.steps     = 0

    def step(self, current_pos):
        
        self.steps += 1

        self.plan  = self._astar_on_belief(current_pos)
        if self.plan and len(self.plan) > 1:
            self.replans += 1
            next_pos = self.plan[1]
            action   = self._direction(current_pos, next_pos)
            return (action, next_pos)
        fallback = self._explore_step(current_pos)
        return fallback

    def _astar_on_belief(self, start):
        
        goal    = self.pm.goal
        counter = 0
        heap    = [(_manhattan(start, goal), counter, start, [start])]
        visited = {}

        while heap:
            f, _, pos, path = heapq.heappop(heap)
            if pos == goal:
                return path
            g = len(path) - 1
            if pos in visited and visited[pos] <= g:
                continue
            visited[pos] = g

            for action, neighbor, cost in self.pm.get_believed_neighbors(pos):
                new_g = g + cost
                new_f = new_g + _manhattan(neighbor, goal)
                counter += 1
                heapq.heappush(heap, (new_f, counter, neighbor, path + [neighbor]))

        return []

    def _explore_step(self, current_pos):
        
        target = self._nearest_unknown(current_pos)
        if target:
            path = self._bfs_on_belief(current_pos, target)
            if path and len(path) > 1:
                next_pos = path[1]
                return (self._direction(current_pos, next_pos), next_pos)

        
        neighbors = self.pm.get_believed_neighbors(current_pos)
        if neighbors:
            action, next_pos, _ = neighbors[0]
            return (action, next_pos)
        return ("STAY", current_pos)

    def _nearest_unknown(self, pos):
        
        visited = {pos}
        queue   = deque([pos])
        while queue:
            curr = queue.popleft()
            for _, neighbor, _ in self.pm.get_believed_neighbors(curr):
                if neighbor not in visited:
                    if self.pm.is_unknown(neighbor):
                        return neighbor
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None

    def _bfs_on_belief(self, start, goal_pos):
        
        visited = {start}
        queue   = deque([(start, [start])])
        while queue:
            pos, path = queue.popleft()
            if pos == goal_pos:
                return path
            for _, neighbor, _ in self.pm.get_believed_neighbors(pos):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        return []

    def _direction(self, from_pos, to_pos):
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if dr == -1: return "UP"
        if dr ==  1: return "DOWN"
        if dc == -1: return "LEFT"
        if dc ==  1: return "RIGHT"
        return "STAY"

    def reset(self, partial_maze):
        self.pm      = partial_maze
        self.plan    = []
        self.replans = 0
        self.steps   = 0

# The lightweight fog of war agent. No full path planning, just one step at a time.
# It keeps a personal notebook of danger scores for every cell it has visited
# and uses those to avoid getting stuck in loops or dead ends.
class OnlineLRTAAgent:
   
    def __init__(self, partial_maze):
        self.pm    = partial_maze
        self.H     = {}   
        self.steps = 0
        self.total_runtime_ms = 0

    def _h(self, pos):
        if pos not in self.H:
            self.H[pos] = _manhattan(pos, self.pm.goal)
        return self.H[pos]

    def step(self, current_pos):
    
        import time
        t0 = time.perf_counter()
        self.steps += 1

        if self.pm.is_goal(current_pos):
            return ("STAY", current_pos)

        neighbors = self.pm.get_believed_neighbors(current_pos)
        if not neighbors:
            return ("STAY", current_pos)

        live_neighbors = []
        for action, npos, cost in neighbors:
            if self.pm.belief_map[npos[0]][npos[1]] != WALL:
                live_neighbors.append((action, npos, cost))

        if not live_neighbors:
            return ("STAY", current_pos)

        min_cost = min(cost + self._h(npos) for _, npos, cost in live_neighbors)
        self.H[current_pos] = max(self._h(current_pos), min_cost)

        best_action, best_pos, _ = min(
            live_neighbors,
            key=lambda t: t[2] + self._h(t[1])
        )

        self.total_runtime_ms += (time.perf_counter() - t0) * 1000
        return (best_action, best_pos)

    def reset(self, partial_maze=None):
        self.H     = {}
        self.steps = 0
        if partial_maze:
            self.pm = partial_maze

# A two phase agent that maps the maze before going for the goal.
# Phase 1 — explore: deliberately wanders toward the edges of what it knows to build up a map.
# Phase 2 — navigate: once the goal is spotted, switches to running A star straight toward it.
class ExplorationAgent:
    

    def __init__(self, partial_maze):
        self.pm           = partial_maze
        self.phase        = "explore"   
        self.current_plan = []
        self.steps        = 0
        self.phase_switch_step = None

    def step(self, current_pos):
        self.steps += 1

        if self.phase == "explore" and self.pm.is_goal_known():
            self.phase             = "navigate"
            self.phase_switch_step = self.steps
            self.current_plan      = []

        if self.phase == "navigate":
            return self._navigate_step(current_pos)
        else:
            return self._explore_step(current_pos)

    def _navigate_step(self, current_pos):
        
        if self.pm.is_goal(current_pos):
            return ("STAY", current_pos)

        if (not self.current_plan or len(self.current_plan) < 2 or
                self.pm.belief_map[self.current_plan[1][0]][self.current_plan[1][1]] == WALL):
            self.current_plan = self._astar_belief(current_pos, self.pm.goal)

        if self.current_plan and len(self.current_plan) > 1:
            next_pos = self.current_plan.pop(0)   
            next_pos = self.current_plan[0] if self.current_plan else current_pos
            return (self._direction(current_pos, next_pos), next_pos)

       
        neighbors = self.pm.get_believed_neighbors(current_pos)
        if neighbors:
            action, next_pos, _ = min(
                neighbors,
                key=lambda t: _manhattan(t[1], self.pm.goal)
            )
            return (action, next_pos)
        return ("STAY", current_pos)

    def _explore_step(self, current_pos):
        
        if (not self.current_plan or len(self.current_plan) < 2 or
                self.pm.belief_map[self.current_plan[1][0]][self.current_plan[1][1]] == WALL):
            frontier_target = self._nearest_frontier(current_pos)
            if frontier_target:
                self.current_plan = self._astar_belief(current_pos, frontier_target)
            else:

                self.phase        = "navigate"
                self.current_plan = []
                return self._navigate_step(current_pos)

        if self.current_plan and len(self.current_plan) > 1:
            self.current_plan.pop(0)
            next_pos = self.current_plan[0]
            return (self._direction(current_pos, next_pos), next_pos)

        return ("STAY", current_pos)

    def _nearest_frontier(self, pos):
        
        visited = {pos}
        queue   = deque([pos])
        while queue:
            curr = queue.popleft()
            if self._is_frontier(curr):
                return curr
            for _, neighbor, _ in self.pm.get_believed_neighbors(curr):
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append(neighbor)
        return None

    def _is_frontier(self, pos):
       
        if self.pm.belief_map[pos[0]][pos[1]] != OPEN:
            return False
        r, c = pos
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < self.pm.rows and 0 <= nc < self.pm.cols:
                if self.pm.belief_map[nr][nc] == UNKNOWN:
                    return True
        return False

    def _astar_belief(self, start, goal_pos):
        
        counter = 0
        heap    = [(_manhattan(start, goal_pos), counter, start, [start])]
        visited = {}
        while heap:
            f, _, pos, path = heapq.heappop(heap)
            if pos == goal_pos:
                return path
            g = len(path) - 1
            if pos in visited and visited[pos] <= g:
                continue
            visited[pos] = g
            for _, neighbor, cost in self.pm.get_believed_neighbors(pos):
                new_g = g + cost
                new_f = new_g + _manhattan(neighbor, goal_pos)
                counter += 1
                heapq.heappush(heap, (new_f, counter, neighbor, path + [neighbor]))
        return []

    def _direction(self, from_pos, to_pos):
        dr = to_pos[0] - from_pos[0]
        dc = to_pos[1] - from_pos[1]
        if dr == -1: return "UP"
        if dr ==  1: return "DOWN"
        if dc == -1: return "LEFT"
        if dc ==  1: return "RIGHT"
        return "STAY"

    def reset(self, partial_maze):
        self.pm                = partial_maze
        self.phase             = "explore"
        self.current_plan      = []
        self.steps             = 0
        self.phase_switch_step = None