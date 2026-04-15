
from __future__ import annotations

import argparse
import csv
import math
import os
import statistics
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Iterable, Tuple

from maze import Maze
from main import simulate_game
from partial_game import simulate_partial_game

# These are all the agents and pursuer types we want to test.
# Full agents can see the whole maze, partial agents work with fog of war.
FULL_AGENTS = ["minimax", "alpha_beta", "expectimax", "lrta", "hill_climb", "beam_search"]
PARTIAL_AGENTS = ["online_astar", "online_lrta", "exploration"]
PURSUERS = ["random", "greedy", "beam", "astar"]

DEFAULT_SIZES = [11, 13, 17, 21]
DEFAULT_DENSITIES = [0.15, 0.25, 0.35]
DEFAULT_VISIBILITY = [2, 4, 6]
DEFAULT_TRIALS = 5

def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)

# Win is 1, anything else (caught or timed out) is 0.
def outcome_to_score(outcome: str) -> int:
    if outcome == "agent_win":
        return 1
    if outcome == "pursuer_win":
        return 0
    return 0  


def safe_mean(values: List[float]) -> float:
    return statistics.mean(values) if values else 0.0


def safe_stdev(values: List[float]) -> float:
    return statistics.stdev(values) if len(values) >= 2 else 0.0


def write_csv(path: Path, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def normalize_result_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isfinite(value):
            return round(value, 4)
        return None
    return value


def normalize_row(row: Dict[str, Any]) -> Dict[str, Any]:
    return {k: normalize_result_value(v) for k, v in row.items()}

# Runs a single full-visibility game with the given settings and packages the result into one row.
# Each row captures everything about that game: who played, what maze, what happened, how long it took.
def run_full_once(
    size: int,
    density: float,
    seed: int,
    dynamic: bool,
    agent: str,
    pursuer: str,
    step_limit: int,
    depth_limit: int,
    verbose: bool = False,
) -> Dict[str, Any]:
   
    maze = Maze.generate_random(size, size, density, seed=seed)
    stats = simulate_game(
        maze=maze,
        agent_algo=agent,
        pursuer_strategy=pursuer,
        dynamic=dynamic,
        depth_limit=depth_limit,
        verbose=verbose,
        step_limit=step_limit,
    )

    row = {
        "mode": "full",
        "agent": agent,
        "pursuer": pursuer,
        "size": size,
        "density": density,
        "seed": seed,
        "dynamic": dynamic,
        "visibility_radius": None,
        "outcome": stats.get("outcome"),
        "win": outcome_to_score(stats.get("outcome")),
        "steps": stats.get("steps", 0),
        "nodes_expanded": stats.get("total_nodes", 0),
        "time_ms": stats.get("total_time_ms", 0.0),
        "cells_explored": None,
        "wall_shifts": None,
        "phase_switch_step": None,
        "error": "",
    }
    return normalize_row(row)

# Same idea as run_full_once but for fog of war games.
# We also track extra things here like how much of the map was explored
# and when the Exploration agent switched from mapping to navigating.
def run_partial_once(
    size: int,
    density: float,
    seed: int,
    dynamic: bool,
    agent: str,
    pursuer: str,
    visibility_radius: int,
    step_limit: int,
    verbose: bool = False,
) -> Dict[str, Any]:
   
    maze = Maze.generate_random(size, size, density, seed=seed)
    stats = simulate_partial_game(
        maze=maze,
        agent_type=agent,
        pursuer_strategy=pursuer,
        visibility_radius=visibility_radius,
        dynamic=dynamic,
        verbose=verbose,
        step_limit=step_limit,
    )

    row = {
        "mode": "partial",
        "agent": agent,
        "pursuer": pursuer,
        "size": size,
        "density": density,
        "seed": seed,
        "dynamic": dynamic,
        "visibility_radius": visibility_radius,
        "outcome": stats.get("outcome"),
        "win": outcome_to_score(stats.get("outcome")),
        "steps": stats.get("steps", 0),
        "nodes_expanded": None,
        "time_ms": stats.get("total_time_ms", 0.0),
        "cells_explored": stats.get("cells_explored"),
        "wall_shifts": stats.get("wall_shifts"),
        "phase_switch_step": stats.get("phase_switch_step"),
        "error": "",
    }
    return normalize_row(row)

# The main loop for full-visibility experiments.
# It runs every combination of agent, pursuer, maze size, density, and dynamic setting.
# If a single game crashes for any reason, it logs the error and keeps going
# so one bad run doesn't kill the whole batch.
def run_full_experiments(
    sizes: List[int],
    densities: List[float],
    trials: int,
    dynamic_values: List[bool],
    agents: List[str],
    pursuers: List[str],
    step_limit: int,
    depth_limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    total = len(sizes) * len(densities) * trials * len(dynamic_values) * len(agents) * len(pursuers)
    count = 0

    for size in sizes:
        for density in densities:
            for trial_idx in range(trials):
                seed = 1000 * size + int(density * 100) * 10 + trial_idx
                for dynamic in dynamic_values:
                    for agent in agents:
                        for pursuer in pursuers:
                            count += 1
                            print(
                                f"[FULL {count}/{total}] "
                                f"size={size} density={density} seed={seed} dynamic={dynamic} "
                                f"agent={agent} pursuer={pursuer}"
                            )
                            try:
                                row = run_full_once(
                                    size=size,
                                    density=density,
                                    seed=seed,
                                    dynamic=dynamic,
                                    agent=agent,
                                    pursuer=pursuer,
                                    step_limit=step_limit,
                                    depth_limit=depth_limit,
                                )
                            except Exception as e:
                                row = {
                                    "mode": "full",
                                    "agent": agent,
                                    "pursuer": pursuer,
                                    "size": size,
                                    "density": density,
                                    "seed": seed,
                                    "dynamic": dynamic,
                                    "visibility_radius": None,
                                    "outcome": "error",
                                    "win": 0,
                                    "steps": None,
                                    "nodes_expanded": None,
                                    "time_ms": None,
                                    "cells_explored": None,
                                    "wall_shifts": None,
                                    "phase_switch_step": None,
                                    "error": f"{type(e).__name__}: {e}",
                                }
                                print("  -> ERROR:", row["error"])
                            rows.append(normalize_row(row))

    return rows

# Has an extra loop for visibility radius since that only applies to partial mode.
def run_partial_experiments(
    sizes: List[int],
    densities: List[float],
    trials: int,
    dynamic_values: List[bool],
    visibility_values: List[int],
    agents: List[str],
    pursuers: List[str],
    step_limit: int,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []

    total = (
        len(sizes) * len(densities) * trials * len(dynamic_values)
        * len(visibility_values) * len(agents) * len(pursuers)
    )
    count = 0

    for size in sizes:
        for density in densities:
            for trial_idx in range(trials):
                seed = 5000 + 1000 * size + int(density * 100) * 10 + trial_idx
                for dynamic in dynamic_values:
                    for visibility_radius in visibility_values:
                        for agent in agents:
                            for pursuer in pursuers:
                                count += 1
                                print(
                                    f"[PARTIAL {count}/{total}] "
                                    f"size={size} density={density} seed={seed} dynamic={dynamic} "
                                    f"radius={visibility_radius} agent={agent} pursuer={pursuer}"
                                )
                                try:
                                    row = run_partial_once(
                                        size=size,
                                        density=density,
                                        seed=seed,
                                        dynamic=dynamic,
                                        agent=agent,
                                        pursuer=pursuer,
                                        visibility_radius=visibility_radius,
                                        step_limit=step_limit,
                                    )
                                except Exception as e:
                                    row = {
                                        "mode": "partial",
                                        "agent": agent,
                                        "pursuer": pursuer,
                                        "size": size,
                                        "density": density,
                                        "seed": seed,
                                        "dynamic": dynamic,
                                        "visibility_radius": visibility_radius,
                                        "outcome": "error",
                                        "win": 0,
                                        "steps": None,
                                        "nodes_expanded": None,
                                        "time_ms": None,
                                        "cells_explored": None,
                                        "wall_shifts": None,
                                        "phase_switch_step": None,
                                        "error": f"{type(e).__name__}: {e}",
                                    }
                                    print("  -> ERROR:", row["error"])
                                rows.append(normalize_row(row))

    return rows


SUMMARY_FIELDS = [
    "mode",
    "agent",
    "pursuer",
    "dynamic",
    "visibility_radius",
    "size",
    "density",
    "n_runs",
    "win_rate",
    "avg_steps",
    "std_steps",
    "avg_time_ms",
    "std_time_ms",
    "avg_nodes_expanded",
    "avg_cells_explored",
    "avg_wall_shifts",
    "avg_phase_switch_step",
]

# This turns hundreds of individual game results into a clean comparison table.
def summarize_rows(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("outcome") == "error":
            continue
        key = (
            row.get("mode"),
            row.get("agent"),
            row.get("pursuer"),
            row.get("dynamic"),
            row.get("visibility_radius"),
            row.get("size"),
            row.get("density"),
        )
        grouped[key].append(row)

    summary_rows: List[Dict[str, Any]] = []

    for key, group in grouped.items():
        mode, agent, pursuer, dynamic, visibility_radius, size, density = key

        wins = [g["win"] for g in group if g.get("win") is not None]
        steps = [g["steps"] for g in group if g.get("steps") is not None]
        times = [g["time_ms"] for g in group if g.get("time_ms") is not None]
        nodes = [g["nodes_expanded"] for g in group if g.get("nodes_expanded") is not None]
        explored = [g["cells_explored"] for g in group if g.get("cells_explored") is not None]
        shifts = [g["wall_shifts"] for g in group if g.get("wall_shifts") is not None]
        phase_switch = [g["phase_switch_step"] for g in group if g.get("phase_switch_step") is not None]

        summary_rows.append(
            normalize_row({
                "mode": mode,
                "agent": agent,
                "pursuer": pursuer,
                "dynamic": dynamic,
                "visibility_radius": visibility_radius,
                "size": size,
                "density": density,
                "n_runs": len(group),
                "win_rate": safe_mean(wins),
                "avg_steps": safe_mean(steps),
                "std_steps": safe_stdev(steps),
                "avg_time_ms": safe_mean(times),
                "std_time_ms": safe_stdev(times),
                "avg_nodes_expanded": safe_mean(nodes) if nodes else None,
                "avg_cells_explored": safe_mean(explored) if explored else None,
                "avg_wall_shifts": safe_mean(shifts) if shifts else None,
                "avg_phase_switch_step": safe_mean(phase_switch) if phase_switch else None,
            })
        )

    summary_rows.sort(key=lambda r: (
        str(r["mode"]),
        str(r["agent"]),
        str(r["pursuer"]),
        str(r["dynamic"]),
        str(r["visibility_radius"]),
        str(r["size"]),
        str(r["density"]),
    ))
    return summary_rows

AGENT_SUMMARY_FIELDS = [
    "mode",
    "agent",
    "n_runs",
    "win_rate",
    "avg_steps",
    "avg_time_ms",
    "avg_nodes_expanded",
    "avg_cells_explored",
]

AGENT_PURSUER_SUMMARY_FIELDS = [
    "mode",
    "agent",
    "pursuer",
    "n_runs",
    "win_rate",
    "avg_steps",
    "avg_time_ms",
    "avg_nodes_expanded",
    "avg_cells_explored",
]

PARTIAL_VISIBILITY_SUMMARY_FIELDS = [
    "agent",
    "visibility_radius",
    "dynamic",
    "n_runs",
    "win_rate",
    "avg_steps",
    "avg_time_ms",
    "avg_cells_explored",
    "avg_wall_shifts",
]

# Zooms all the way out and collapses everything down to one row per agent
# so you can see at a glance which agent performed best overall across all conditions.
def summarize_by_agent(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("outcome") == "error":
            continue
        key = (row.get("mode"), row.get("agent"))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (mode, agent), group in grouped.items():
        wins = [g["win"] for g in group if g.get("win") is not None]
        steps = [g["steps"] for g in group if g.get("steps") is not None]
        times = [g["time_ms"] for g in group if g.get("time_ms") is not None]
        nodes = [g["nodes_expanded"] for g in group if g.get("nodes_expanded") is not None]
        explored = [g["cells_explored"] for g in group if g.get("cells_explored") is not None]

        out.append(normalize_row({
            "mode": mode,
            "agent": agent,
            "n_runs": len(group),
            "win_rate": safe_mean(wins),
            "avg_steps": safe_mean(steps),
            "avg_time_ms": safe_mean(times),
            "avg_nodes_expanded": safe_mean(nodes) if nodes else None,
            "avg_cells_explored": safe_mean(explored) if explored else None,
        }))

    out.sort(key=lambda r: (str(r["mode"]), str(r["agent"])))
    return out


def summarize_by_agent_pursuer(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("outcome") == "error":
            continue
        key = (row.get("mode"), row.get("agent"), row.get("pursuer"))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (mode, agent, pursuer), group in grouped.items():
        wins = [g["win"] for g in group if g.get("win") is not None]
        steps = [g["steps"] for g in group if g.get("steps") is not None]
        times = [g["time_ms"] for g in group if g.get("time_ms") is not None]
        nodes = [g["nodes_expanded"] for g in group if g.get("nodes_expanded") is not None]
        explored = [g["cells_explored"] for g in group if g.get("cells_explored") is not None]

        out.append(normalize_row({
            "mode": mode,
            "agent": agent,
            "pursuer": pursuer,
            "n_runs": len(group),
            "win_rate": safe_mean(wins),
            "avg_steps": safe_mean(steps),
            "avg_time_ms": safe_mean(times),
            "avg_nodes_expanded": safe_mean(nodes) if nodes else None,
            "avg_cells_explored": safe_mean(explored) if explored else None,
        }))

    out.sort(key=lambda r: (str(r["mode"]), str(r["agent"]), str(r["pursuer"])))
    return out

# Specifically for fog of war results as it shows how each agent's performance
# changes as we give it a wider or narrower field of view.
def summarize_partial_visibility(rows: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    grouped: Dict[Tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)

    for row in rows:
        if row.get("outcome") == "error" or row.get("mode") != "partial":
            continue
        key = (row.get("agent"), row.get("visibility_radius"), row.get("dynamic"))
        grouped[key].append(row)

    out: List[Dict[str, Any]] = []
    for (agent, visibility_radius, dynamic), group in grouped.items():
        wins = [g["win"] for g in group if g.get("win") is not None]
        steps = [g["steps"] for g in group if g.get("steps") is not None]
        times = [g["time_ms"] for g in group if g.get("time_ms") is not None]
        explored = [g["cells_explored"] for g in group if g.get("cells_explored") is not None]
        shifts = [g["wall_shifts"] for g in group if g.get("wall_shifts") is not None]

        out.append(normalize_row({
            "agent": agent,
            "visibility_radius": visibility_radius,
            "dynamic": dynamic,
            "n_runs": len(group),
            "win_rate": safe_mean(wins),
            "avg_steps": safe_mean(steps),
            "avg_time_ms": safe_mean(times),
            "avg_cells_explored": safe_mean(explored) if explored else None,
            "avg_wall_shifts": safe_mean(shifts) if shifts else None,
        }))

    out.sort(key=lambda r: (str(r["agent"]), str(r["dynamic"]), str(r["visibility_radius"])))
    return out

# Draws a grid where each cell is colored by win rate: darker means worse, brighter means better.
def plot_heatmap(matrix, row_labels, col_labels, title, save_path):
    import matplotlib.pyplot as plt

    plt.figure(figsize=(max(6, len(col_labels) * 1.2), max(4, len(row_labels) * 0.8)))
    plt.imshow(matrix, aspect="auto")
    plt.colorbar()
    plt.xticks(range(len(col_labels)), col_labels, rotation=30, ha="right")
    plt.yticks(range(len(row_labels)), row_labels)
    plt.title(title)

    for i in range(len(row_labels)):
        for j in range(len(col_labels)):
            plt.text(j, i, f"{matrix[i][j]:.2f}", ha="center", va="center")

    plt.tight_layout()
    plt.savefig(save_path, dpi=160)
    plt.close()

# Generates all the charts and saves them to a plots folder.
# If matplotlib isn't installed it just skips this step quietly instead of crashing.
def try_make_plots(raw_rows: List[Dict[str, Any]], summary_rows: List[Dict[str, Any]], out_dir: Path) -> None:
    try:
        import matplotlib.pyplot as plt
    except Exception:
        print("matplotlib not available; skipping plots.")
        return

    plots_dir = out_dir / "plots"
    ensure_dir(plots_dir)

    
    for mode in ("full", "partial"):
        subset = [r for r in summary_rows if r["mode"] == mode]
        if not subset:
            continue

        by_agent: Dict[str, List[float]] = defaultdict(list)
        for row in subset:
            by_agent[row["agent"]].append(row["win_rate"])

        agents = list(by_agent.keys())
        values = [safe_mean(by_agent[a]) for a in agents]

        plt.figure(figsize=(10, 5))
        plt.bar(agents, values)
        plt.ylim(0, 1)
        plt.ylabel("Win Rate")
        plt.title(f"Average Win Rate by Agent ({mode})")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"win_rate_by_agent_{mode}.png", dpi=160)
        plt.close()

    
    for mode in ("full", "partial"):
        subset = [r for r in summary_rows if r["mode"] == mode]
        if not subset:
            continue

        by_agent: Dict[str, List[float]] = defaultdict(list)
        for row in subset:
            if row["avg_time_ms"] is not None:
                by_agent[row["agent"]].append(row["avg_time_ms"])

        agents = list(by_agent.keys())
        values = [safe_mean(by_agent[a]) for a in agents]

        plt.figure(figsize=(10, 5))
        plt.bar(agents, values)
        plt.ylabel("Average Runtime (ms)")
        plt.title(f"Average Runtime by Agent ({mode})")
        plt.xticks(rotation=25, ha="right")
        plt.tight_layout()
        plt.savefig(plots_dir / f"runtime_by_agent_{mode}.png", dpi=160)
        plt.close()


    full_subset = [r for r in summary_rows if r["mode"] == "full"]
    if full_subset:
        agents = sorted({r["agent"] for r in full_subset})
        sizes = sorted({r["size"] for r in full_subset})

        plt.figure(figsize=(10, 6))
        for agent in agents:
            vals = []
            for size in sizes:
                matching = [r["win_rate"] for r in full_subset if r["agent"] == agent and r["size"] == size]
                vals.append(safe_mean(matching) if matching else 0.0)
            plt.plot(sizes, vals, marker="o", label=agent)

        plt.ylim(0, 1)
        plt.xlabel("Maze Size")
        plt.ylabel("Win Rate")
        plt.title("Full Mode: Win Rate vs Maze Size")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "full_win_rate_vs_size.png", dpi=160)
        plt.close()


    partial_subset = [r for r in summary_rows if r["mode"] == "partial" and r["avg_cells_explored"] is not None]
    if partial_subset:
        agents = sorted({r["agent"] for r in partial_subset})
        radii = sorted({r["visibility_radius"] for r in partial_subset if r["visibility_radius"] is not None})

        plt.figure(figsize=(10, 6))
        for agent in agents:
            vals = []
            for radius in radii:
                matching = [
                    r["avg_cells_explored"]
                    for r in partial_subset
                    if r["agent"] == agent and r["visibility_radius"] == radius
                ]
                vals.append(safe_mean(matching) if matching else 0.0)
            plt.plot(radii, vals, marker="o", label=agent)

        plt.ylim(0, 1)
        plt.xlabel("Visibility Radius")
        plt.ylabel("Average Fraction Explored")
        plt.title("Partial Mode: Exploration vs Visibility Radius")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "partial_explored_vs_radius.png", dpi=160)
        plt.close()

    
    full_agent_pursuer = [r for r in summary_rows if r["mode"] == "full"]
    if full_agent_pursuer:
        agents = sorted({r["agent"] for r in full_agent_pursuer})
        pursuers = sorted({r["pursuer"] for r in full_agent_pursuer})

        matrix = []
        for agent in agents:
            row = []
            for pursuer in pursuers:
                vals = [
                    r["win_rate"]
                    for r in full_agent_pursuer
                    if r["agent"] == agent and r["pursuer"] == pursuer
                ]
                row.append(safe_mean(vals) if vals else 0.0)
            matrix.append(row)

        plot_heatmap(
            matrix,
            agents,
            pursuers,
            "Full Mode: Agent vs Pursuer Win Rate",
            plots_dir / "full_agent_pursuer_winrate_heatmap.png",
        )

    
    partial_agent_pursuer = [r for r in summary_rows if r["mode"] == "partial"]
    if partial_agent_pursuer:
        agents = sorted({r["agent"] for r in partial_agent_pursuer})
        pursuers = sorted({r["pursuer"] for r in partial_agent_pursuer})

        matrix = []
        for agent in agents:
            row = []
            for pursuer in pursuers:
                vals = [
                    r["win_rate"]
                    for r in partial_agent_pursuer
                    if r["agent"] == agent and r["pursuer"] == pursuer
                ]
                row.append(safe_mean(vals) if vals else 0.0)
            matrix.append(row)

        plot_heatmap(
            matrix,
            agents,
            pursuers,
            "Partial Mode: Agent vs Pursuer Win Rate",
            plots_dir / "partial_agent_pursuer_winrate_heatmap.png",
        )

   
    partial_subset = [r for r in summary_rows if r["mode"] == "partial"]
    if partial_subset:
        agents = sorted({r["agent"] for r in partial_subset})
        radii = sorted({r["visibility_radius"] for r in partial_subset if r["visibility_radius"] is not None})

        plt.figure(figsize=(10, 6))
        for agent in agents:
            vals = []
            for radius in radii:
                matching = [
                    r["win_rate"]
                    for r in partial_subset
                    if r["agent"] == agent and r["visibility_radius"] == radius
                ]
                vals.append(safe_mean(matching) if matching else 0.0)
            plt.plot(radii, vals, marker="o", label=agent)

        plt.ylim(0, 1)
        plt.xlabel("Visibility Radius")
        plt.ylabel("Win Rate")
        plt.title("Partial Mode: Win Rate vs Visibility Radius")
        plt.legend()
        plt.tight_layout()
        plt.savefig(plots_dir / "partial_win_rate_vs_radius.png", dpi=160)
        plt.close()

    print(f"Plots written to: {plots_dir}")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Batch experiments for adversarial maze navigation")

    parser.add_argument("--mode", choices=["full", "partial", "both"], default="both")
    parser.add_argument("--sizes", type=int, nargs="+", default=DEFAULT_SIZES)
    parser.add_argument("--densities", type=float, nargs="+", default=DEFAULT_DENSITIES)
    parser.add_argument("--visibility", type=int, nargs="+", default=DEFAULT_VISIBILITY)
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS)
    parser.add_argument("--step-limit-full", type=int, default=100)
    parser.add_argument("--step-limit-partial", type=int, default=200)
    parser.add_argument("--depth-limit", type=int, default=4)

    parser.add_argument("--static-only", action="store_true")
    parser.add_argument("--dynamic-only", action="store_true")
    parser.add_argument("--no-plots", action="store_true")

    parser.add_argument("--outdir", type=str, default="experiment_outputs")

    return parser.parse_args()

# Figures out whether to run static mazes, dynamic mazes, or both
# based on the flags passed in. Throws an error if someone accidentally sets both flags at once.
def resolve_dynamic_values(args: argparse.Namespace) -> List[bool]:
    if args.static_only and args.dynamic_only:
        raise ValueError("Cannot set both --static-only and --dynamic-only.")
    if args.static_only:
        return [False]
    if args.dynamic_only:
        return [True]
    return [False, True]

RAW_FIELDS = [
    "mode",
    "agent",
    "pursuer",
    "size",
    "density",
    "seed",
    "dynamic",
    "visibility_radius",
    "outcome",
    "win",
    "steps",
    "nodes_expanded",
    "time_ms",
    "cells_explored",
    "wall_shifts",
    "phase_switch_step",
    "error",
]


def main() -> None:
    args = parse_args()
    dynamic_values = resolve_dynamic_values(args)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_dir = Path(args.outdir) / f"run_{timestamp}"
    ensure_dir(out_dir)

    raw_rows: List[Dict[str, Any]] = []

    if args.mode in ("full", "both"):
        full_rows = run_full_experiments(
            sizes=args.sizes,
            densities=args.densities,
            trials=args.trials,
            dynamic_values=dynamic_values,
            agents=FULL_AGENTS,
            pursuers=PURSUERS,
            step_limit=args.step_limit_full,
            depth_limit=args.depth_limit,
        )
        raw_rows.extend(full_rows)

    if args.mode in ("partial", "both"):
        partial_rows = run_partial_experiments(
            sizes=args.sizes,
            densities=args.densities,
            trials=args.trials,
            dynamic_values=dynamic_values,
            visibility_values=args.visibility,
            agents=PARTIAL_AGENTS,
            pursuers=PURSUERS,
            step_limit=args.step_limit_partial,
        )
        raw_rows.extend(partial_rows)

    raw_csv = out_dir / "raw_results.csv"
    write_csv(raw_csv, raw_rows, RAW_FIELDS)
    print(f"Raw results written to: {raw_csv}")

    summary_rows = summarize_rows(raw_rows)
    summary_csv = out_dir / "summary_results.csv"
    write_csv(summary_csv, summary_rows, SUMMARY_FIELDS)
    print(f"Summary results written to: {summary_csv}")

    summary_by_agent_rows = summarize_by_agent(raw_rows)
    summary_by_agent_csv = out_dir / "summary_by_agent.csv"
    write_csv(summary_by_agent_csv, summary_by_agent_rows, AGENT_SUMMARY_FIELDS)
    print(f"Agent summary written to: {summary_by_agent_csv}")

    summary_agent_pursuer_rows = summarize_by_agent_pursuer(raw_rows)
    summary_agent_pursuer_csv = out_dir / "summary_agent_pursuer.csv"
    write_csv(summary_agent_pursuer_csv, summary_agent_pursuer_rows, AGENT_PURSUER_SUMMARY_FIELDS)
    print(f"Agent-pursuer summary written to: {summary_agent_pursuer_csv}")

    summary_partial_visibility_rows = summarize_partial_visibility(raw_rows)
    summary_partial_visibility_csv = out_dir / "summary_partial_visibility.csv"
    write_csv(summary_partial_visibility_csv, summary_partial_visibility_rows, PARTIAL_VISIBILITY_SUMMARY_FIELDS)
    print(f"Partial visibility summary written to: {summary_partial_visibility_csv}")

    errors = [r for r in raw_rows if r.get("outcome") == "error"]
    if errors:
        error_csv = out_dir / "errors.csv"
        write_csv(error_csv, errors, RAW_FIELDS)
        print(f"Errors written to: {error_csv}")

    if not args.no_plots:
        try_make_plots(raw_rows, summary_rows, out_dir)


if __name__ == "__main__":
    main()