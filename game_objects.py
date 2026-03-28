"""
game_objects.py — game objects: traps, power-ups, weighted terrain
project: adversarial maze navigation
authors: vikramaditya sogani & namya singh

objects that make the maze more complex:
  Trap       — freezes agent for N turns when stepped on
  PowerUp    — gives agent a temporary ability
  Terrain    — weighted cells with different movement costs
"""

import random


# ─────────────────────────────────────────────
#  cell type constants (extend maze OPEN=0 WALL=1)
# ─────────────────────────────────────────────

OPEN    = 0
WALL    = 1
TRAP    = 2
POWERUP = 3
MUD     = 4   # cost 3 to traverse
WATER   = 5   # cost 2 to traverse
ROAD    = 6   # cost 0.5 to traverse (fast lane)

# step costs for each terrain type
TERRAIN_COST = {
    OPEN    : 1,
    TRAP    : 1,   # cost is paid, then trap triggers
    POWERUP : 1,
    MUD     : 3,
    WATER   : 2,
    ROAD    : 1,   # visually distinct but same cost (simplification)
}

# display colors (R,G,B) for pygame
CELL_COLORS = {
    OPEN    : (15,  17,  35),   # dark navy
    WALL    : (45,  48,  70),   # slate
    TRAP    : (180, 60,  40),   # red-orange
    POWERUP : (130, 60, 200),   # purple
    MUD     : (90,  65,  30),   # brown
    WATER   : (30,  80, 150),   # blue
    ROAD    : (40,  60,  40),   # dark green
}

CELL_LABELS = {
    TRAP    : "T",
    POWERUP : "P",
    MUD     : "~",
    WATER   : "W",
    ROAD    : "+",
}


# ─────────────────────────────────────────────
#  power-up types
# ─────────────────────────────────────────────

PU_REVEAL   = "reveal"    # reveals full map for 10 steps
PU_SPEED    = "speed"     # agent moves twice per turn for 8 steps
PU_SHIELD   = "shield"    # immunity to traps for 6 steps
PU_FREEZE   = "freeze"    # freezes pursuer for 5 steps

POWERUP_TYPES = [PU_REVEAL, PU_SPEED, PU_SHIELD, PU_FREEZE]

POWERUP_COLORS = {
    PU_REVEAL : (200, 180, 255),
    PU_SPEED  : (255, 220, 50),
    PU_SHIELD : (50,  220, 180),
    PU_FREEZE : (100, 180, 255),
}

POWERUP_LABELS = {
    PU_REVEAL : "R",
    PU_SPEED  : "S",
    PU_SHIELD : "X",
    PU_FREEZE : "F",
}


# ─────────────────────────────────────────────
#  agent status tracker
# ─────────────────────────────────────────────

class AgentStatus:
    """
    tracks all active effects on the agent.
    call tick() each game step to decrement counters.
    """

    def __init__(self):
        self.frozen_turns  = 0    # cant move
        self.speed_turns   = 0    # moves twice per turn
        self.shield_turns  = 0    # immune to traps
        self.reveal_turns  = 0    # full map revealed
        self.score         = 0
        self.powerups_collected = []
        self.traps_hit     = 0

    def tick(self):
        """decrement all active effect counters by 1."""
        self.frozen_turns  = max(0, self.frozen_turns  - 1)
        self.speed_turns   = max(0, self.speed_turns   - 1)
        self.shield_turns  = max(0, self.shield_turns  - 1)
        self.reveal_turns  = max(0, self.reveal_turns  - 1)

    def apply_trap(self):
        if self.shield_turns > 0:
            return False   # blocked by shield
        self.frozen_turns = 3
        self.traps_hit += 1
        return True

    def apply_powerup(self, pu_type):
        self.powerups_collected.append(pu_type)
        self.score += 50
        if pu_type == PU_REVEAL:
            self.reveal_turns = 10
        elif pu_type == PU_SPEED:
            self.speed_turns = 8
        elif pu_type == PU_SHIELD:
            self.shield_turns = 6
        elif pu_type == PU_FREEZE:
            return True   # caller handles pursuer freeze
        return False

    def is_frozen(self):
        return self.frozen_turns > 0

    def has_speed(self):
        return self.speed_turns > 0

    def full_reveal(self):
        return self.reveal_turns > 0

    def active_effects(self):
        effects = []
        if self.frozen_turns  > 0: effects.append(f"FROZEN({self.frozen_turns})")
        if self.speed_turns   > 0: effects.append(f"SPEED({self.speed_turns})")
        if self.shield_turns  > 0: effects.append(f"SHIELD({self.shield_turns})")
        if self.reveal_turns  > 0: effects.append(f"REVEAL({self.reveal_turns})")
        return effects


# ─────────────────────────────────────────────
#  pursuer status
# ─────────────────────────────────────────────

class PursuerStatus:
    """tracks effects on a single pursuer."""

    def __init__(self):
        self.frozen_turns = 0

    def tick(self):
        self.frozen_turns = max(0, self.frozen_turns - 1)

    def freeze(self, turns=5):
        self.frozen_turns = turns

    def is_frozen(self):
        return self.frozen_turns > 0


# ─────────────────────────────────────────────
#  obstacle placer
# ─────────────────────────────────────────────

def place_objects(grid, rows, cols, start, goal, pursuer_starts,
                  num_traps=10, num_powerups=6,
                  num_mud=8, num_water=6, num_road=5,
                  seed=None):
    """
    places traps, power-ups, and terrain objects on the grid.
    never places on start, goal, or pursuer start positions.
    returns dict mapping (row,col) -> powerup_type for POWERUP cells.
    """
    if seed is not None:
        random.seed(seed)

    protected = {start, goal} | set(pursuer_starts)

    # collect all open cells eligible for objects
    candidates = [
        (r, c) for r in range(rows) for c in range(cols)
        if grid[r][c] == OPEN and (r, c) not in protected
    ]
    random.shuffle(candidates)

    idx = 0
    powerup_map = {}   # (r,c) -> pu_type

    def place(cell_type, count):
        nonlocal idx
        placed = 0
        while placed < count and idx < len(candidates):
            r, c = candidates[idx]; idx += 1
            grid[r][c] = cell_type
            placed += 1

    def place_powerups(count):
        nonlocal idx
        placed = 0
        pu_cycle = POWERUP_TYPES * (count // len(POWERUP_TYPES) + 1)
        while placed < count and idx < len(candidates):
            r, c = candidates[idx]; idx += 1
            grid[r][c] = POWERUP
            powerup_map[(r, c)] = pu_cycle[placed]
            placed += 1

    place(TRAP,    num_traps)
    place_powerups(num_powerups)
    place(MUD,     num_mud)
    place(WATER,   num_water)
    place(ROAD,    num_road)

    return powerup_map