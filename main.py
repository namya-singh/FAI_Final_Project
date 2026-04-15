import argparse
import time
from maze              import Maze, DynamicMaze
from game_state        import GameState, Turn
from search            import bfs, dfs, ucs, astar as static_astar, hill_climb, beam_search
from adversarial_search import (
    minimax, alpha_beta, expectimax,
    LRTAStar, PursuerAI
)





SAMPLE_MAZE = """
#############
#S          #
# ######### #
#         # #
######### # #
#           #
# ######### #
#         # #
######### # #
#           #
# ######### #
#          G#
#############
""".strip()

# The main game loop: runs one full game from start to finish and returns a summary of what happened.
# You tell it which agent algorithm to use, which pursuer strategy to fight, and whether
# the walls should shift mid-game. It handles everything in between.
def simulate_game(maze, agent_algo, pursuer_strategy,
                  depth_limit=4, dynamic=False, shift_interval=6,
                  verbose=True, step_limit=150):
   

  
    if dynamic:
        dyn_maze = DynamicMaze.from_static(maze, shift_interval=shift_interval,
                                           num_shifts=2, seed=42)
        active_maze = dyn_maze
    else:
        active_maze = maze

    pursuer_start = (active_maze.rows - 1, 0) \
        if active_maze.start == (0, 0) else (0, active_maze.cols - 1)
    # pursuer start is walkable
    if not active_maze.is_walkable(pursuer_start):
        pursuer_start = (1, active_maze.cols - 2)

    state   = GameState(active_maze, active_maze.start, pursuer_start,
                        turn=Turn.AGENT, step_limit=step_limit)
    pursuer = PursuerAI(strategy=pursuer_strategy, beam_width=3)
    lrta    = LRTAStar(active_maze) if agent_algo == "lrta" else None

    stats = {
        "algorithm"     : agent_algo,
        "pursuer"       : pursuer_strategy,
        "dynamic"       : dynamic,
        "wall_shifts"   : 0,
        "steps"         : 0,
        "outcome"       : None,
        "total_nodes"   : 0,
        "total_time_ms" : 0,
        "path"          : [active_maze.start],
    }

    if verbose:
        label = f"{agent_algo.upper()} agent vs {pursuer_strategy} pursuer"
        label += " [DYNAMIC MAZE]" if dynamic else " [STATIC MAZE]"
        state.display(label=label)

    while not state.is_terminal():

        
        shifted = False
        if dynamic:
            protected = {state.agent_pos, state.pursuer_pos}
            shifted = active_maze.step(protected_positions=protected)
            if shifted:
                stats["wall_shifts"] += 1

    
        t0 = time.perf_counter()

        if agent_algo == "minimax":
            result = minimax(state, depth_limit=depth_limit)
            action, new_agent_pos = result.best_action, result.best_move_pos
            stats["total_nodes"] += result.nodes_expanded
            stats["total_time_ms"] += result.runtime_ms

        elif agent_algo == "alpha_beta":
            result = alpha_beta(state, depth_limit=depth_limit)
            action, new_agent_pos = result.best_action, result.best_move_pos
            stats["total_nodes"] += result.nodes_expanded
            stats["total_time_ms"] += result.runtime_ms

        elif agent_algo == "expectimax":
            result = expectimax(state, depth_limit=depth_limit)
            action, new_agent_pos = result.best_action, result.best_move_pos
            stats["total_nodes"] += result.nodes_expanded
            stats["total_time_ms"] += result.runtime_ms

        elif agent_algo == "lrta":
            action, new_agent_pos = lrta.step(state.agent_pos)
            stats["total_time_ms"] += (time.perf_counter() - t0) * 1000

        elif agent_algo == "hill_climb":
            result = hill_climb(active_maze, "manhattan", start=state.agent_pos)
            if result.success and len(result.path) > 1:
                new_agent_pos = result.path[1]
            else:
                new_agent_pos = greedy_fallback(active_maze, state.agent_pos)
            stats["total_nodes"] += result.nodes_expanded
            stats["total_time_ms"] += result.runtime_ms

        elif agent_algo == "beam_search":
            result = beam_search(active_maze, 3, heuristic_name="manhattan", start=state.agent_pos)
            if result.success and len(result.path) > 1:
                new_agent_pos = result.path[1]
            else:
                new_agent_pos = greedy_fallback(active_maze, state.agent_pos)
            stats["total_nodes"] += result.nodes_expanded
            stats["total_time_ms"] += result.runtime_ms

        else:
            raise ValueError(f"unknown agent_algo: {agent_algo}")

     
        if new_agent_pos is None or not active_maze.is_walkable(new_agent_pos):
            new_agent_pos = state.agent_pos

        state = state.apply_agent_move(new_agent_pos)
        stats["path"].append(new_agent_pos)

        if state.is_terminal():
            break

    
        p_action, new_pursuer_pos = pursuer.choose_move(state)
        state = state.apply_pursuer_move(new_pursuer_pos)

        stats["steps"] += 1

        if verbose and stats["steps"] % 10 == 0:
            wall_note = " [walls shifted]" if shifted else ""
            state.display(label=f"Step {stats['steps']}{wall_note}")

    if state.agent_won():
        stats["outcome"] = "agent_win"
    elif state.pursuer_won():
        stats["outcome"] = "pursuer_win"
    else:
        stats["outcome"] = "timeout"

    if verbose:
        state.display(label=f"GAME OVER — {stats['outcome'].upper().replace('_',' ')}")
        _print_game_stats(stats)

    return stats


def _print_game_stats(stats):
    icon = {"agent_win": "✓ AGENT WINS", "pursuer_win": "✗ CAUGHT", "timeout": "⏱ TIMEOUT"}
    print(f"\n{'═'*44}")
    print(f"  {icon[stats['outcome']]}")
    print(f"  Agent algo    : {stats['algorithm']}")
    print(f"  Pursuer       : {stats['pursuer']}")
    print(f"  Dynamic walls : {stats['dynamic']}")
    print(f" wall shifts    : {stats['wall_shifts']}")
    print(f"  Steps taken   : {stats['steps']}")
    print(f"  Nodes expanded: {stats['total_nodes']}")
    print(f"  Total time    : {stats['total_time_ms']:.1f} ms")
    print(f"{'═'*44}\n")

# Runs every agent against every pursuer and prints the results as a side by side table.
def run_comparison(maze, dynamic=False):

    algorithms = ["minimax", "alpha_beta", "expectimax", "lrta", "hill_climb", "beam_search"]
    pursuers   = ["random", "greedy", "beam", "astar"]

    print(f"\n{'═'*70}")
    print(f"  COMPARISON {'[DYNAMIC MAZE]' if dynamic else '[STATIC MAZE]'}")
    print(f"{'═'*70}")
    print(f"{'Agent':<16} {'Pursuer':<10} {'Outcome':<14} {'Steps':>6} "
          f"{'Nodes':>8} {'Time(ms)':>10}")
    print(f"{'─'*70}")

    for algo in algorithms:
        for pursuer in pursuers:
            stats = simulate_game(
                maze, algo, pursuer,
                depth_limit=4, dynamic=dynamic,
                verbose=False, step_limit=100
            )
            outcome_str = stats["outcome"].replace("_", " ").upper()
            print(f"{algo:<16} {pursuer:<10} {outcome_str:<14} "
                  f"{stats['steps']:>6} {stats['total_nodes']:>8} "
                  f"{stats['total_time_ms']:>10.1f}")
    print(f"{'─'*70}\n")


def greedy_fallback(maze, current_position):
   
    neighbor = maze.get_neighbors(current_position)
    if not neighbor:
        return current_position
    _, best_position, _ = min(
        neighbor,
        key=lambda x: abs(x[1][0] - maze.goal[0]) + abs(x[1][1] - maze.goal[1])
    )
    return best_position




def main():
    parser = argparse.ArgumentParser(description="Adversarial Maze Navigation")
    parser.add_argument("--size",     type=int,   default=13)
    parser.add_argument("--density",  type=float, default=0.25)
    parser.add_argument("--seed",     type=int,   default=7)
    parser.add_argument("--algo",     type=str,   default=None,
                        choices=["minimax","alpha_beta","expectimax","lrta",
                                 "hill_climb","beam_search","alpha_beta","manual"])
    parser.add_argument("--pursuer",  type=str,   default="greedy",
                        choices=["random","greedy","astar", "beam"])
    parser.add_argument("--dynamic",  action="store_true")
    parser.add_argument("--compare",  action="store_true",
                        help="run full comparison table")
    parser.add_argument("--visual",   action="store_true",
                        help="launch pygame visual game")
    parser.add_argument("--pursuers", type=int,   default=2,
                        help="number of pursuers in visual mode (1-3)")
    parser.add_argument("--no-fog",   action="store_true",
                        help="disable fog of war in visual mode")
    args = parser.parse_args()

    
    if args.visual:
        from visual_game import launch
        launch(
            agent_algo       = args.algo or "lrta",
            pursuer_strategy = args.pursuer,
            num_pursuers     = min(3, max(1, args.pursuers)),
            dynamic          = args.dynamic,
            fog              = not args.no_fog,
        )
        return

    
    maze = Maze.generate_random(args.size, args.size, args.density, seed=args.seed)

    if args.compare:
        run_comparison(maze, dynamic=False)
        run_comparison(maze, dynamic=True)

    elif args.algo:
        simulate_game(maze, args.algo, args.pursuer,
                      dynamic=args.dynamic, verbose=True)
    else:
        
        print("\n" + "═"*50)
        print("  DEMO: Alpha-Beta agent vs Greedy pursuer (Static)")
        print("═"*50)
        simulate_game(maze, "alpha_beta", "greedy", dynamic=False, verbose=True)

        print("\n" + "═"*50)
        print("  DEMO: LRTA* agent vs Greedy pursuer (Dynamic Maze)")
        print("═"*50)
        simulate_game(maze, "lrta", "greedy", dynamic=True, verbose=True)

        print("\n" + "="*50)
        print("  DEMO: Hill-Climbing agent vs Greedy pursuer (Static Maze)")
        print("="*50)
        simulate_game(maze, "hill_climb", "greedy", dynamic=False, verbose=True)

        print("\n" + "="*50)
        print("  DEMO: Beam search agent vs Greedy pursuer (Dynamic Maze)")
        print("="*50)
        simulate_game(maze, "beam_search", "greedy", dynamic=True, verbose=True)

        print("\n" + "═"*50)
        print("  FULL COMPARISON TABLE")
        print("═"*50)
        run_comparison(maze, dynamic=False)
        run_comparison(maze, dynamic=True)


if __name__ == "__main__":
    main()