from maze import Maze
from partial_game import simulate_partial_game, run_partial_comparison

maze = Maze.generate_random(13, 13, 0.25, seed=7)
simulate_partial_game(maze, agent_type="online_astar", pursuer_strategy="greedy", verbose=True)
run_partial_comparison(maze)