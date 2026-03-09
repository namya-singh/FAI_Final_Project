"""
search.py — Classical Search Algorithms (used for baseline + pursuer AI)
Project: Adversarial Maze Navigation
Authors: VikramAditya Sogani & Namya Singh

Algorithms: BFS, DFS, UCS, A* (manhattan / euclidean heuristics)
Each returns a SearchResult.
"""

import heapq
import math
import time
from collections import deque
from dataclasses import dataclass, field
from typing import List

from node import Node



#  RESULT CONTAINER


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



#  HEURISTICS


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



#  SHARED HELPER


def _expand(node, maze):
    for action, next_state, step_cost in maze.get_neighbors(node.state):
        yield Node(state=next_state, parent=node, action=action,
                   path_cost=node.path_cost + step_cost)



#  BFS


def bfs(maze, start=None):
    """BFS from start (defaults to maze.start) to maze.goal."""
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



#  DFS


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



#  UCS


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



#  A*


def astar(maze, heuristic_name="manhattan", start=None):
    """A* from start (defaults to maze.start) to maze.goal."""
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