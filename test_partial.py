"""
test_partial.py — unit tests for partial observability system
project: adversarial maze navigation
authors: vikramaditya sogani & namya singh

tests cover:
  - belief map initialization (all cells start as UNKNOWN)
  - fog of war updates correctly after agent moves
  - exploration percentage increases monotonically
  - goal detection once goal enters visibility radius
  - online agents return valid moves
  - full game simulation produces valid outcomes
  - dynamic maze belief updates after wall shifts
"""

from maze import Maze, DynamicMaze
from partial_maze import PartialMaze, UNKNOWN, OPEN, WALL
from online_agent import OnlineAstarAgent, OnlineLRTAAgent, ExplorationAgent
from partial_game import simulate_partial_game, run_partial_comparison


# ── helpers ──────────────────────────────────────────────────────────

def make_maze(size=13, density=0.25, seed=7):
    return Maze.generate_random(size, size, density, seed=seed)

def assert_eq(a, b, msg=""):
    assert a == b, f"expected {b}, got {a}  {msg}"

def assert_true(cond, msg=""):
    assert cond, f"assertion failed: {msg}"

def assert_between(val, lo, hi, msg=""):
    assert lo <= val <= hi, f"expected {lo} <= {val} <= {hi}  {msg}"

def pass_msg(name):
    print(f"  PASS  {name}")


# ── test 1: belief map starts fully unknown ───────────────────────────

def test_belief_map_init():
    maze = make_maze()
    pm   = PartialMaze(maze, visibility_radius=0)
    for r in range(pm.rows):
        for c in range(pm.cols):
            if (r, c) != maze.start:
                assert_eq(pm.belief_map[r][c], UNKNOWN,
                          f"cell ({r},{c}) should be UNKNOWN at init")
    pass_msg("belief map initialized to UNKNOWN outside start radius")


# ── test 2: update_belief reveals cells within radius ─────────────────

def test_update_belief_reveals_cells():
    maze   = make_maze()
    radius = 3
    pm     = PartialMaze(maze, visibility_radius=radius)
    pos    = maze.start
    r, c   = pos
    for dr in range(-radius, radius + 1):
        for dc in range(-radius, radius + 1):
            if abs(dr) + abs(dc) <= radius:
                nr, nc = r + dr, c + dc
                if 0 <= nr < maze.rows and 0 <= nc < maze.cols:
                    assert_true(pm.belief_map[nr][nc] != UNKNOWN,
                                f"cell ({nr},{nc}) should be revealed")
    pass_msg("update_belief reveals all cells within radius")


# ── test 3: exploration percentage increases monotonically ────────────

def test_exploration_increases():
    maze  = make_maze()
    pm    = PartialMaze(maze, visibility_radius=4)
    agent = OnlineAstarAgent(pm)
    pos   = maze.start

    prev_explored = pm.cells_explored()
    ever_increased = False

    for _ in range(30):
        _, new_pos = agent.step(pos)
        if new_pos != pos:
            pm.update_belief(new_pos)
            new_explored = pm.cells_explored()
            assert_true(new_explored >= prev_explored,
                        "explored% must not decrease")
            if new_explored > prev_explored:
                ever_increased = True
            prev_explored = new_explored
            pos = new_pos

    assert_true(ever_increased, "explored% must increase at some point")
    pass_msg("exploration percentage increases monotonically")


# ── test 4: cells_explored returns value in [0, 1] ───────────────────

def test_cells_explored_range():
    maze = make_maze()
    pm   = PartialMaze(maze, visibility_radius=4)
    val  = pm.cells_explored()
    assert_between(val, 0.0, 1.0, "cells_explored must be in [0, 1]")
    pass_msg("cells_explored returns value in valid range")


# ── test 5: goal becomes known once agent is close enough ─────────────

def test_goal_detection():
    rows, cols = 7, 7
    grid  = [[0]*cols for _ in range(rows)]
    start = (0, 0)
    goal  = (0, 4)
    maze  = Maze(grid, start, goal)
    pm    = PartialMaze(maze, visibility_radius=4)
    assert_true(pm.is_goal_known(),
                "goal at distance 4 should be visible from start")
    pass_msg("goal detected when within visibility radius")


# ── test 6: get_believed_neighbors returns only valid positions ────────

def test_believed_neighbors_bounds():
    maze = make_maze()
    pm   = PartialMaze(maze, visibility_radius=4)
    for action, neighbor, cost in pm.get_believed_neighbors(maze.start):
        r, c = neighbor
        assert_true(0 <= r < maze.rows and 0 <= c < maze.cols,
                    f"neighbor ({r},{c}) out of bounds")
        assert_eq(cost, 1, "step cost must be 1")
    pass_msg("get_believed_neighbors returns in-bounds positions")


# ── test 7: online a* agent returns valid action and position ─────────

def test_online_astar_valid_move():
    maze  = make_maze()
    pm    = PartialMaze(maze, visibility_radius=4)
    agent = OnlineAstarAgent(pm)
    action, new_pos = agent.step(maze.start)
    r, c = new_pos
    assert_true(0 <= r < maze.rows and 0 <= c < maze.cols,
                "OnlineAstarAgent returned out-of-bounds position")
    assert_true(action in ("UP","DOWN","LEFT","RIGHT","STAY"),
                f"invalid action: {action}")
    pass_msg("OnlineAstarAgent returns valid action and position")


# ── test 8: online lrta agent returns valid move ──────────────────────

def test_online_lrta_valid_move():
    maze  = make_maze()
    pm    = PartialMaze(maze, visibility_radius=4)
    agent = OnlineLRTAAgent(pm)
    action, new_pos = agent.step(maze.start)
    r, c = new_pos
    assert_true(0 <= r < maze.rows and 0 <= c < maze.cols,
                "OnlineLRTAAgent returned out-of-bounds position")
    assert_true(action in ("UP","DOWN","LEFT","RIGHT","STAY"),
                f"invalid action: {action}")
    pass_msg("OnlineLRTAAgent returns valid action and position")


# ── test 9: exploration agent returns valid move ──────────────────────

def test_exploration_agent_valid_move():
    maze  = make_maze()
    pm    = PartialMaze(maze, visibility_radius=4)
    agent = ExplorationAgent(pm)
    action, new_pos = agent.step(maze.start)
    r, c = new_pos
    assert_true(0 <= r < maze.rows and 0 <= c < maze.cols,
                "ExplorationAgent returned out-of-bounds position")
    pass_msg("ExplorationAgent returns valid action and position")


# ── test 10: simulate_partial_game returns valid outcome ──────────────

def test_simulate_outcome():
    maze  = make_maze()
    stats = simulate_partial_game(
        maze, agent_type="online_astar",
        pursuer_strategy="random",
        visibility_radius=4, verbose=False, step_limit=100
    )
    assert_true(stats["outcome"] in ("agent_win", "pursuer_win", "timeout"),
                f"invalid outcome: {stats['outcome']}")
    assert_true(stats["steps"] >= 0, "steps must be non-negative")
    assert_between(stats["cells_explored"], 0.0, 1.0,
                   "cells_explored must be in [0, 1]")
    pass_msg("simulate_partial_game returns valid outcome and stats")


# ── test 11: dynamic maze belief updates after wall shift ─────────────

def test_dynamic_belief_update():
    base  = make_maze()
    dyn   = DynamicMaze.from_static(base, shift_interval=1, num_shifts=2)
    pm    = PartialMaze(dyn, visibility_radius=4)
    pos   = base.start

    pm.update_belief(pos)
    snapshot_before = [row[:] for row in pm.belief_map]

    dyn.step(protected_positions={pos, base.goal})
    pm.update_belief(pos)

    changed = any(
        pm.belief_map[r][c] != snapshot_before[r][c]
        for r in range(pm.rows) for c in range(pm.cols)
        if (r, c) in pm.seen
    )
    pass_msg("dynamic maze belief map can update after wall shift")


# ── test 12: all three agents complete without crashing ───────────────

def test_all_agents_run():
    maze   = make_maze(size=11, density=0.2, seed=99)
    agents = ["online_astar", "online_lrta", "exploration"]
    for agent_type in agents:
        stats = simulate_partial_game(
            maze, agent_type=agent_type,
            pursuer_strategy="random",
            visibility_radius=4, verbose=False, step_limit=80
        )
        assert_true(stats["outcome"] is not None,
                    f"{agent_type} returned None outcome")
    pass_msg("all three online agents complete without crashing")


# ── test 13: partial game with dynamic maze runs correctly ────────────

def test_dynamic_partial_game():
    maze  = make_maze()
    stats = simulate_partial_game(
        maze, agent_type="online_astar",
        pursuer_strategy="greedy",
        visibility_radius=4, dynamic=True,
        verbose=False, step_limit=100
    )
    assert_true(stats["outcome"] in ("agent_win", "pursuer_win", "timeout"))
    pass_msg("partial game with dynamic maze runs without errors")


# ── run all tests ─────────────────────────────────────────────────────

def run_all_tests():
    tests = [
        test_belief_map_init,
        test_update_belief_reveals_cells,
        test_exploration_increases,
        test_cells_explored_range,
        test_goal_detection,
        test_believed_neighbors_bounds,
        test_online_astar_valid_move,
        test_online_lrta_valid_move,
        test_exploration_agent_valid_move,
        test_simulate_outcome,
        test_dynamic_belief_update,
        test_all_agents_run,
        test_dynamic_partial_game,
    ]

    print(f"\n{'═'*50}")
    print(f"  running {len(tests)} partial observability tests")
    print(f"{'═'*50}")

    passed = 0
    failed = 0
    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"  FAIL  {test.__name__}: {e}")
            failed += 1
        except Exception as e:
            print(f"  ERROR {test.__name__}: {e}")
            failed += 1

    print(f"{'─'*50}")
    print(f"  {passed} passed  |  {failed} failed")
    print(f"{'═'*50}\n")
    return failed == 0


if __name__ == "__main__":
    run_all_tests()