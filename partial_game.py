
import time
from maze            import Maze, DynamicMaze
from partial_maze    import PartialMaze
from online_agent    import OnlineAstarAgent, OnlineLRTAAgent, ExplorationAgent
from game_state      import GameState, Turn
from adversarial_search import PursuerAI


def simulate_partial_game(maze, agent_type="online_astar", pursuer_strategy="greedy",
                           visibility_radius=4, dynamic=False, shift_interval=6,
                           verbose=True, step_limit=200):
  
    if dynamic:
        active_maze = DynamicMaze.from_static(maze, shift_interval=shift_interval,
                                               num_shifts=2, seed=42)
    else:
        active_maze = maze

    pm = PartialMaze(active_maze, visibility_radius=visibility_radius)

    pursuer_start = (active_maze.rows - 1, 0) \
        if active_maze.start == (0, 0) else (0, active_maze.cols - 1)
    if not active_maze.is_walkable(pursuer_start):
        pursuer_start = (1, active_maze.cols - 2)

    if agent_type == "online_astar":
        agent = OnlineAstarAgent(pm)
    elif agent_type == "online_lrta":
        agent = OnlineLRTAAgent(pm)
    elif agent_type == "exploration":
        agent = ExplorationAgent(pm)
    else:
        raise ValueError(f"unknown agent_type: {agent_type}")

    pursuer    = PursuerAI(strategy=pursuer_strategy, beam_width=3)
    agent_pos  = active_maze.start
    pursuer_pos = pursuer_start

    stats = {
        "agent_type"       : agent_type,
        "pursuer"          : pursuer_strategy,
        "dynamic"          : dynamic,
        "wall_shifts"      : 0,
        "visibility_radius": visibility_radius,
        "steps"            : 0,
        "outcome"          : None,
        "cells_explored"   : 0.0,
        "total_time_ms"    : 0.0,
        "path"             : [agent_pos],
        "phase_switch_step": None,   
    }

    if verbose:
        label = f"{agent_type} vs {pursuer_strategy} pursuer"
        label += " [DYNAMIC]" if dynamic else " [STATIC]"
        label += f" | visibility r={visibility_radius}"
        pm.display(agent=agent_pos, pursuer=pursuer_pos, label=label)

    step = 0
    while step < step_limit:

        
        if agent_pos == active_maze.goal:
            stats["outcome"] = "agent_win"
            break
        if agent_pos == pursuer_pos:
            stats["outcome"] = "pursuer_win"
            break

        if dynamic:
            protected = {agent_pos, pursuer_pos}
            shifted = active_maze.step(protected_positions=protected)
            if shifted:
                stats["wall_shifts"] += 1
            pm.update_belief(agent_pos)

        t0 = time.perf_counter()
        _, new_agent_pos = agent.step(agent_pos)
        stats["total_time_ms"] += (time.perf_counter() - t0) * 1000
        if new_agent_pos != agent_pos and not active_maze.is_walkable(new_agent_pos):
         
            new_agent_pos = agent_pos

        agent_pos = new_agent_pos
        stats["path"].append(agent_pos)

        
        pm.update_belief(agent_pos)

        
        if agent_pos == active_maze.goal:
            stats["outcome"] = "agent_win"
            break

        
        game_state_for_pursuer = _make_pursuer_state(active_maze, agent_pos, pursuer_pos)
        _, new_pursuer_pos = pursuer.choose_move(game_state_for_pursuer)
        pursuer_pos = new_pursuer_pos

        
        if pursuer_pos == agent_pos:
            stats["outcome"] = "pursuer_win"
            break

        step += 1
        stats["steps"] = step

        if verbose and step % 15 == 0:
            wall_note = " [walls shifted]" if dynamic else ""
            pm.display(agent=agent_pos, pursuer=pursuer_pos,
                       label=f"step {step}{wall_note} | explored {pm.cells_explored()*100:.0f}%")

    
    if stats["outcome"] is None:
        stats["outcome"] = "timeout"

    stats["cells_explored"] = pm.cells_explored()

    if hasattr(agent, "phase_switch_step"):
        stats["phase_switch_step"] = agent.phase_switch_step

    if verbose:
        pm.display(agent=agent_pos, pursuer=pursuer_pos,
                   label=f"game over — {stats['outcome'].upper().replace('_', ' ')}")
        _print_partial_stats(stats)

    return stats


def _make_pursuer_state(maze, agent_pos, pursuer_pos):

    gs             = object.__new__(GameState)
    gs.maze        = maze
    gs.agent_pos   = agent_pos
    gs.pursuer_pos = pursuer_pos
    gs.turn        = Turn.PURSUER
    gs.step        = 0
    gs.step_limit  = 9999
    return gs


def _print_partial_stats(stats):
    icon = {"agent_win": "✓ agent wins", "pursuer_win": "✗ caught", "timeout": "⏱ timeout"}
    print(f"\n{'═'*48}")
    print(f"  {icon.get(stats['outcome'], stats['outcome'])}")
    print(f"  agent type      : {stats['agent_type']}")
    print(f"  pursuer         : {stats['pursuer']}")
    print(f"  dynamic walls   : {stats['dynamic']}")
    print(f" wall shifts      : {stats['wall_shifts']}")
    print(f"  visibility r    : {stats['visibility_radius']}")
    print(f"  steps taken     : {stats['steps']}")
    print(f"  map explored    : {stats['cells_explored']*100:.1f}%")
    print(f"  total time      : {stats['total_time_ms']:.1f} ms")
    if stats.get("phase_switch_step"):
        print(f"  explore→navigate: step {stats['phase_switch_step']}")
    print(f"{'═'*48}\n")


def run_partial_comparison(maze, dynamic=False, visibility_radius=4):
    
    agents   = ["online_astar", "online_lrta", "exploration"]
    pursuers = ["random", "greedy", "beam", "astar"]

    print(f"\n{'═'*74}")
    print(f"  partial observability comparison (r={visibility_radius}) "
          f"{'[dynamic]' if dynamic else '[static]'}")
    print(f"{'═'*74}")
    print(f"{'agent':<16} {'pursuer':<10} {'outcome':<14} {'steps':>6} "
          f"{'explored':>10} {'time(ms)':>10}")
    print(f"{'─'*74}")

    for agent_type in agents:
        for pursuer in pursuers:
            stats = simulate_partial_game(
                maze, agent_type=agent_type, pursuer_strategy=pursuer,
                visibility_radius=visibility_radius, dynamic=dynamic,
                verbose=False, step_limit=200
            )
            outcome_str = stats["outcome"].replace("_", " ").upper()
            explored    = f"{stats['cells_explored']*100:.1f}%"
            print(f"{agent_type:<16} {pursuer:<10} {outcome_str:<14} "
                  f"{stats['steps']:>6} {explored:>10} "
                  f"{stats['total_time_ms']:>10.1f}")

    print(f"{'─'*74}\n")