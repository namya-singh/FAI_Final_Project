import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

from node import Node

# stores everything worth knowing after an algorithm finishes: did it find a path, how long did it take, how many cells did it look at.
@dataclass
class SearchResult:
    algorithm      : str
    heuristic      : str        = "none"
    success        : bool       = False
    path           : List       = field(default_factory=list)
    actions        : List       = field(default_factory=list)
    path_cost      : float      = 0
    nodes_expanded : int        = 0
    max_frontier   : int        = 0
    runtime_ms     : float      = 0

    def summary(self):
        h_str  = f"  heuristic     : {self.heuristic}\n" if self.heuristic != "none" else ""
        status = "✓ SOLVED" if self.success else "✗ NO PATH"
        return (
            f"\n{'─'*42}\n"
            f"  algorithm     : {self.algorithm}\n"
            + h_str +
            f"  status        : {status}\n"
            f"  path cost     : {self.path_cost}\n"
            f"  nodes expanded: {self.nodes_expanded}\n"
            f"  peak frontier : {self.max_frontier}\n"
            f"  runtime       : {self.runtime_ms:.3f} ms\n"
            f"{'─'*42}"
        )


def heuristic_manhattan(pos, goal):
    return abs(pos[0] - goal[0]) + abs(pos[1] - goal[1])

def heuristic_euclidean(pos, goal):
    return math.sqrt((pos[0] - goal[0])**2 + (pos[1] - goal[1])**2)

def heuristic_zero(pos, goal):
    return 0

HEURISTICS = {
    "manhattan" : heuristic_manhattan,
    "euclidean" : heuristic_euclidean,
    "zero"      : heuristic_zero,
}

# Takes a node and generates all the nodes you can reach from it in one step.
# Each child remembers its parent so we can trace the full path back later.
def _expand(node, maze):
    for action, next_state, step_cost in maze.get_neighbors(node.state):
        yield Node(state=next_state, parent=node, action=action,
                   path_cost=node.path_cost + step_cost)



def bfs(maze, start=None):
    
    t0       = time.perf_counter()
    root     = Node(start or maze.start)
    frontier = deque([root])
    explored = set()
    expanded = 0; max_f = 1

    while frontier:
        max_f = max(max_f, len(frontier))
        node  = frontier.popleft()

        if maze.is_goal(node.state):
            return SearchResult("BFS", success=True, path=node.path(),
                                actions=node.solution(), path_cost=node.path_cost,
                                nodes_expanded=expanded, max_frontier=max_f,
                                runtime_ms=(time.perf_counter()-t0)*1000)

        if node.state in explored:
            continue
        explored.add(node.state); expanded += 1

        for child in _expand(node, maze):
            if child.state not in explored:
                frontier.append(child)

    return SearchResult("BFS", runtime_ms=(time.perf_counter()-t0)*1000, nodes_expanded=expanded)


# Uses much less memory than BFS but can wander far down dead ends
# and doesn't guarantee the shortest path.
def dfs(maze, start=None):
    t0       = time.perf_counter()
    root     = Node(start or maze.start)
    frontier = [root]
    explored = set()
    expanded = 0; max_f = 1

    while frontier:
        max_f = max(max_f, len(frontier))
        node  = frontier.pop()

        if maze.is_goal(node.state):
            return SearchResult("DFS", success=True, path=node.path(),
                                actions=node.solution(), path_cost=node.path_cost,
                                nodes_expanded=expanded, max_frontier=max_f,
                                runtime_ms=(time.perf_counter()-t0)*1000)

        if node.state in explored:
            continue
        explored.add(node.state); expanded += 1

        for child in reversed(list(_expand(node, maze))):
            if child.state not in explored:
                frontier.append(child)

    return SearchResult("DFS", runtime_ms=(time.perf_counter()-t0)*1000, nodes_expanded=expanded)

# On a maze where some cells cost more to walk through, this finds the truly cheapest route
# rather than just the one with the fewest steps.
def ucs(maze, start=None):
    t0      = time.perf_counter()
    root    = Node(start or maze.start)
    counter = 0
    heap    = [(0, counter, root)]
    explored = {}
    expanded = 0; max_f = 1

    while heap:
        max_f = max(max_f, len(heap))
        cost, _, node = heapq.heappop(heap)

        if maze.is_goal(node.state):
            return SearchResult("UCS", success=True, path=node.path(),
                                actions=node.solution(), path_cost=node.path_cost,
                                nodes_expanded=expanded, max_frontier=max_f,
                                runtime_ms=(time.perf_counter()-t0)*1000)

        if node.state in explored and explored[node.state] <= cost:
            continue
        explored[node.state] = cost; expanded += 1

        for child in _expand(node, maze):
            if child.state not in explored:
                counter += 1
                heapq.heappush(heap, (child.path_cost, counter, child))

    return SearchResult("UCS", runtime_ms=(time.perf_counter()-t0)*1000, nodes_expanded=expanded)

# It combines the actual cost to get somewhere with a guess of how far the goal still is.
# By always processing the node that looks most promising overall, it finds the shortest
# path while exploring far fewer cells than BFS or UCS.
def astar(maze, heuristic_name="manhattan", start=None):
   
    h       = HEURISTICS[heuristic_name]
    t0      = time.perf_counter()
    s       = start or maze.start
    root    = Node(s)
    counter = 0
    heap    = [(h(s, maze.goal), counter, root)]
    explored = {}
    expanded = 0; max_f = 1

    while heap:
        max_f = max(max_f, len(heap))
        _, _, node = heapq.heappop(heap)

        if maze.is_goal(node.state):
            return SearchResult("A*", heuristic=heuristic_name, success=True,
                                path=node.path(), actions=node.solution(),
                                path_cost=node.path_cost, nodes_expanded=expanded,
                                max_frontier=max_f,
                                runtime_ms=(time.perf_counter()-t0)*1000)

        if node.state in explored and explored[node.state] <= node.path_cost:
            continue
        explored[node.state] = node.path_cost; expanded += 1

        for child in _expand(node, maze):
            if child.state not in explored:
                f = child.path_cost + h(child.state, maze.goal)
                counter += 1
                heapq.heappush(heap, (f, counter, child))

    return SearchResult("A*", heuristic=heuristic_name,
                        runtime_ms=(time.perf_counter()-t0)*1000, nodes_expanded=expanded)




def hill_climb(maze, heuristic_name="manhattan", start=None,
               allow_sideways=False, max_sideways=10):
    h = HEURISTICS[heuristic_name]
    t0 = time.perf_counter()
    current = Node(start or maze.start)
    expanded = 0
    sideways = 0
    visited = {current.state}

    while True:
        if maze.is_goal(current.state):
            return SearchResult("Hill Climb", heuristic=heuristic_name,success=True,
                                path=current.path(), actions=current.solution(), path_cost=current.path_cost,
                                nodes_expanded=expanded, max_frontier=1,
                                runtime_ms=(time.perf_counter()-t0)*1000
            )

        neighbors = list(_expand(current, maze))
        if not neighbors:
            break

        expanded += 1
        current_h = h(current.state, maze.goal)

        neighbors.sort(key=lambda n: h(n.state, maze.goal))
        best = neighbors[0]
        best_h = h(best.state, maze.goal)

        
        if best_h < current_h:
            current = best
            sideways = 0
            visited.add(current.state)
            continue

       
        if (allow_sideways and best_h == current_h
                and sideways < max_sideways
                and best.state not in visited):
            current = best
            sideways += 1
            visited.add(current.state)
            continue

        
        break

    return SearchResult("Hill Climb", heuristic=heuristic_name, runtime_ms=(time.perf_counter()-t0)*1000,
                        nodes_expanded=expanded, max_frontier=1)



def beam_search(maze, beam_width=3, heuristic_name="manhattan", start=None):
    h = HEURISTICS[heuristic_name]
    t0 = time.perf_counter()
    root = Node(start or maze.start)

    frontier = [root]
    explored = set()
    expanded = 0
    max_f = 1

    while frontier:
        max_f = max(max_f, len(frontier))

        
        for node in frontier:
            if maze.is_goal(node.state):
                return SearchResult(f"Beam Search(k={beam_width})", heuristic=heuristic_name,
                                    success=True, path=node.path(), actions=node.solution(),
                                    path_cost=node.path_cost, nodes_expanded=expanded,
                                    max_frontier=max_f, runtime_ms=(time.perf_counter()-t0)*1000
                                    )

        next_level = []

        for node in frontier:
            if node.state in explored:
                continue

            explored.add(node.state)
            expanded += 1

            for child in _expand(node, maze):
                if child.state not in explored:
                    next_level.append(child)

        if not next_level:
            break

        next_level.sort(key=lambda n: h(n.state, maze.goal))
        frontier = next_level[:beam_width]

    return SearchResult(f"Beam Search(k={beam_width})", heuristic=heuristic_name,
                        runtime_ms=(time.perf_counter()-t0)*1000, nodes_expanded=expanded,
                        max_frontier=max_f
                        )


# Weighted A star: a faster but slightly less perfect version of A star.
# It multiplies the distance guess by a weight greater than 1, making the algorithm
# more aggressive about heading toward the goal and less careful about finding the absolute shortest path.
# The tradeoff: explores far fewer cells and runs much faster, but the path might be a little longer.
def weighted_astar(maze, heuristic_name="manhattan", weight=1.5, start=None):
    
    h        = HEURISTICS[heuristic_name]
    t0       = time.perf_counter()
    s        = start or maze.start
    root     = Node(s)
    counter  = 0
    heap     = [(weight * h(s, maze.goal), counter, root)]
    explored = {}
    expanded = 0
    max_f    = 1

    while heap:
        max_f      = max(max_f, len(heap))
        _, _, node = heapq.heappop(heap)

        if maze.is_goal(node.state):
            return SearchResult(
                f"Weighted A*(w={weight})", heuristic=heuristic_name,
                success=True, path=node.path(), actions=node.solution(),
                path_cost=node.path_cost, nodes_expanded=expanded,
                max_frontier=max_f,
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )

        if node.state in explored and explored[node.state] <= node.path_cost:
            continue
        explored[node.state] = node.path_cost
        expanded += 1

        for child in _expand(node, maze):
            if child.state not in explored:
                f_val   = child.path_cost + weight * h(child.state, maze.goal)
                counter += 1
                heapq.heappush(heap, (f_val, counter, child))

    return SearchResult(
        f"Weighted A*(w={weight})", heuristic=heuristic_name,
        runtime_ms=(time.perf_counter() - t0) * 1000,
        nodes_expanded=expanded,
    )



# IDA star: finds the same optimal path as A star but uses almost no memory.
# Instead of storing every node it has ever visited, it runs repeated depth first searches
# with a growing cost limit. Each pass only follows paths where the total estimated cost
# stays within the current threshold, then raises the threshold and tries again.
# The right choice when the maze is so large that A star would run out of memory.
def idastar(maze, heuristic_name="manhattan", start=None):
   
    h       = HEURISTICS[heuristic_name]
    t0      = time.perf_counter()
    s       = start or maze.start
    expanded= [0]

    def search(path, g, threshold):
        node    = path[-1]
        f       = g + h(node.state, maze.goal)
        if f > threshold:
            return f, None
        if maze.is_goal(node.state):
            return -1, path[:]
        minimum = math.inf
        for child in _expand(path[-1], maze):
            if any(p.state == child.state for p in path):
                continue
            expanded[0] += 1
            path.append(child)
            result, solution = search(path, g + child.path_cost - node.path_cost, threshold)
            if result == -1:
                return -1, solution
            if result < minimum:
                minimum = result
            path.pop()
        return minimum, None

    threshold = h(s, maze.goal)
    root      = Node(s)
    path      = [root]

    while True:
        result, solution = search(path, 0, threshold)
        if result == -1 and solution:
            goal_node = solution[-1]
            return SearchResult(
                "IDA*", heuristic=heuristic_name,
                success=True,
                path=[n.state for n in solution],
                actions=[n.action for n in solution[1:]],
                path_cost=goal_node.path_cost,
                nodes_expanded=expanded[0],
                max_frontier=1,
                runtime_ms=(time.perf_counter() - t0) * 1000,
            )
        if result == math.inf:
            return SearchResult(
                "IDA*", heuristic=heuristic_name,
                runtime_ms=(time.perf_counter() - t0) * 1000,
                nodes_expanded=expanded[0],
            )
        threshold = result