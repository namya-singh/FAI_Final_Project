"""
visual_game.py - visual simulation for adversial maze navigation
project: adversarial maze navigation
authors: vikramaditya sogani & namya singh
"""

import pygame
import pygame.gfxdraw
import sys
import math
import heapq
import random
from collections import deque

from search import hill_climb, beam_search
from adversarial_search import expectimax
from game_state import GameState, Turn
from maze         import Maze, DynamicMaze
from game_objects import (
    OPEN, WALL, TRAP, POWERUP, MUD, WATER, ROAD,
    TERRAIN_COST, AgentStatus, PursuerStatus, place_objects,
    PU_REVEAL, PU_SPEED, PU_SHIELD, PU_FREEZE
)

ROWS, COLS  = 17, 17
CELL        = 38
FOG_RADIUS  = 4
MAZE_W  = COLS * CELL
MAZE_H  = ROWS * CELL
PANEL_W = 270
WIN_W   = MAZE_W + PANEL_W
WIN_H   = MAZE_H + 70

BLACK       = (0,    0,    0)
DARK_BG     = (5,    5,   20)
WALL_DARK   = (10,  20,   80)
WALL_BRIGHT = (30,  60,  200)
WALL_GLOW   = (60, 100,  255)
PACMAN_Y    = (255, 220,   0)
GHOST_COLS  = [(255, 80, 80), (255, 180, 80), (180, 80, 255)]
GHOST_SCARED= (60,  80,  200)
GHOST_EYE_W = (255, 255, 255)
GHOST_EYE_P = (30,  30,  180)
PELLET_C    = (255, 220, 180)
POWER_C     = (255, 255, 100)
TRAP_C      = (220,  40,  40)
MUD_C       = (100,  70,  30)
WATER_C     = (30,   90, 200)
ROAD_C      = (30,   80,  40)
GOAL_C      = (60,  240, 100)
PANEL_BG    = (8,    8,   25)
TEXT_W      = (245, 250, 255)
TEXT_DIM    = (170, 185, 235)
ACCENT      = (60,  160, 255)
GOOD_C      = (60,  220, 100)
BAD_C       = (220,  60,  60)
WARN_C      = (255, 180,  40)

ALGO_NAMES = {
    "minimax": "Minimax",
    "alpha_beta": "Alpha-Beta",
    "lrta": "LRTA*",
    "expectimax": "Expectimax",
    "hill_climb": "Hill Climb",
    "beam_search": "Beam Search",
    "manual": "Manual",
}

PURSUER_NAMES = {
    "random": "Random",
    "greedy": "Greedy",
    "beam": "Beam Search",
    "astar": "A*",
}

AGENT_ORDER = [
    "lrta",
    "minimax",
    "alpha_beta",
    "expectimax",
    "hill_climb",
    "beam_search",
    "manual",
]

PURSUER_ORDER = [
    "random",
    "greedy",
    "beam",
    "astar",
]


SCREEN_START    = "start"
SCREEN_GAME     = "game"
SCREEN_GAMEOVER = "gameover"
HIGH_SCORES     = []


def draw_pacman(surf, cx, cy, radius, mouth_angle, facing):
    start_a = math.radians(facing + mouth_angle)
    end_a   = math.radians(facing + 360 - mouth_angle)
    points  = [(cx, cy)]
    steps   = 40
    for i in range(steps + 1):
        a = start_a + (end_a - start_a) * i / steps
        points.append((cx + radius * math.cos(a), cy + radius * math.sin(a)))
    if len(points) > 2:
        pygame.gfxdraw.filled_polygon(surf, [(int(x), int(y)) for x, y in points], PACMAN_Y)
        pygame.gfxdraw.aapolygon(surf, [(int(x), int(y)) for x, y in points], PACMAN_Y)
    ex = cx - int(radius * 0.15 * math.cos(math.radians(facing)))
    ey = cy - int(radius * 0.5)
    pygame.gfxdraw.filled_circle(surf, int(ex), int(ey), max(2, radius // 5), (30, 20, 0))


def draw_ghost(surf, cx, cy, radius, color, scared=False, frozen=False):
    col = (100, 180, 255) if frozen else (GHOST_SCARED if scared else color)
    rect = pygame.Rect(cx - radius, cy - radius, radius * 2, radius * 2)
    pygame.draw.ellipse(surf, col, rect)
    pygame.draw.rect(surf, col, pygame.Rect(cx - radius, cy, radius * 2, radius))
    bump_r = radius // 3
    for i in range(3):
        bx = cx - radius + bump_r + i * bump_r * 2
        pygame.draw.circle(surf, DARK_BG, (bx, cy + radius), bump_r)
    if not scared and not frozen:
        for ex_off in [-radius // 3, radius // 3]:
            pygame.draw.circle(surf, GHOST_EYE_W, (cx + ex_off, cy - radius // 4), radius // 4)
            pygame.draw.circle(surf, GHOST_EYE_P, (cx + ex_off + 1, cy - radius // 4 + 1), radius // 8)
    else:
        for ex_off in [-radius // 3, radius // 3]:
            ex, ey, r2 = cx + ex_off, cy - radius // 4, radius // 5
            pygame.draw.line(surf, (255, 200, 200), (ex - r2, ey - r2), (ex + r2, ey + r2), 2)
            pygame.draw.line(surf, (255, 200, 200), (ex + r2, ey - r2), (ex - r2, ey + r2), 2)


def draw_wall_cell(surf, x, y, size, t):
    pygame.draw.rect(surf, WALL_DARK, (x, y, size, size))
    pulse = int(10 * math.sin(t / 600 + (x + y) * 0.01))
    glow  = tuple(min(255, WALL_BRIGHT[i] + pulse) for i in range(3))
    pygame.draw.rect(surf, glow, (x + 1, y + 1, size - 2, size - 2), 2)


def draw_pellet(surf, cx, cy, radius):
    pygame.gfxdraw.filled_circle(surf, cx, cy, radius, PELLET_C)
    pygame.gfxdraw.aacircle(surf, cx, cy, radius, PELLET_C)


def draw_powerup_pellet(surf, cx, cy, radius, t):
    r = radius + int(3 * math.sin(t / 200))
    pygame.gfxdraw.filled_circle(surf, cx, cy, r, POWER_C)
    pygame.gfxdraw.aacircle(surf, cx, cy, r, (255, 255, 200))


class SmoothPos:
    def __init__(self, row, col):
        self.row = row; self.col = col
        self.px  = col * CELL + CELL // 2
        self.py  = row * CELL + CELL // 2
        self.tx  = self.px; self.ty = self.py
        self.spd = 0.25

    def set_target(self, row, col):
        self.row = row; self.col = col
        self.tx  = col * CELL + CELL // 2
        self.ty  = row * CELL + CELL // 2

    def update(self):
        self.px += (self.tx - self.px) * self.spd
        self.py += (self.ty - self.py) * self.spd

    def center(self):
        return int(self.px), int(self.py)


class Particle:
    def __init__(self, x, y, color):
        angle = random.uniform(0, 2 * math.pi)
        speed = random.uniform(1.5, 5)
        self.x = x; self.y = y
        self.vx = math.cos(angle) * speed
        self.vy = math.sin(angle) * speed
        self.life = random.randint(20, 45)
        self.max_life = self.life
        self.color = color
        self.size = random.randint(2, 6)

    def update(self):
        self.x += self.vx; self.y += self.vy
        self.vy += 0.12; self.vx *= 0.97; self.life -= 1

    def draw(self, surf):
        col = tuple(min(255, v) for v in self.color[:3])
        pygame.gfxdraw.filled_circle(surf, int(self.x), int(self.y), self.size, col)

    def alive(self): return self.life > 0


def spawn_particles(particles, x, y, color, count=15):
    for _ in range(count):
        particles.append(Particle(x, y, color))


class FlashMsg:
    def __init__(self, text, color, duration=90):
        self.text = text; self.color = color
        self.timer = duration; self.max_time = duration

    def tick(self): self.timer -= 1
    def alive(self): return self.timer > 0


class WallAnim:
    def __init__(self, pos, appearing):
        self.pos = pos; self.appearing = appearing; self.timer = 25

    def alive(self): return self.timer > 0
    def tick(self): self.timer -= 1

    def draw(self, surf, t):
        r, c = self.pos
        x, y = c * CELL, r * CELL
        progress = 1 - self.timer / 25
        if self.appearing:
            size = int(CELL * progress)
            off  = (CELL - size) // 2
            col  = tuple(int(v * progress) for v in WALL_BRIGHT)
            pygame.draw.rect(surf, col, (x + off, y + off, size, size), border_radius=3)
        else:
            size = int(CELL * (1 - progress))
            off  = (CELL - size) // 2
            col  = tuple(int(v * (1 - progress)) for v in WALL_BRIGHT)
            if size > 0:
                pygame.draw.rect(surf, col, (x + off, y + off, size, size), border_radius=3)
            for _ in range(2):
                dx = random.randint(-CELL // 2, CELL // 2)
                dy = random.randint(-CELL // 2, CELL // 2)
                pygame.draw.circle(surf, WALL_GLOW, (x + CELL // 2 + dx, y + CELL // 2 + dy), 2)


class StartScreen:
    def __init__(self, screen, fonts):
        self.screen = screen; self.fonts = fonts
        self.algo = "lrta"; self.pursuer = "greedy"
        self.pursuers = 2; self.dynamic = True; self.fog = True
        self.algo_opts = ["lrta", "minimax", "alpha_beta", "expectimax", "hill_climb", "beam_search", "manual"]
        self.pursuer_opts = ["random", "greedy", "beam", "astar"]
        self.algo_idx = 0; self.pursuer_idx = 1; self.t = 0
        self.dots = [{"x": random.randint(0, WIN_W), "y": random.randint(0, WIN_H),
                      "r": random.randint(2, 5), "s": random.uniform(0.3, 1.2)}
                     for _ in range(40)]

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key in (pygame.K_RETURN, pygame.K_SPACE): return "start"
            elif event.key == pygame.K_LEFT:
                self.algo_idx = (self.algo_idx - 1) % len(self.algo_opts)
                self.algo = self.algo_opts[self.algo_idx]
            elif event.key == pygame.K_RIGHT:
                self.algo_idx = (self.algo_idx + 1) % len(self.algo_opts)
                self.algo = self.algo_opts[self.algo_idx]
            elif event.key == pygame.K_UP:
                self.pursuer_idx = (self.pursuer_idx - 1) % len(self.pursuer_opts)
                self.pursuer = self.pursuer_opts[self.pursuer_idx]
            elif event.key == pygame.K_DOWN:
                self.pursuer_idx = (self.pursuer_idx + 1) % len(self.pursuer_opts)
                self.pursuer = self.pursuer_opts[self.pursuer_idx]
            elif event.key == pygame.K_1: self.pursuers = 1
            elif event.key == pygame.K_2: self.pursuers = 2
            elif event.key == pygame.K_3: self.pursuers = 3
            elif event.key == pygame.K_d: self.dynamic = not self.dynamic
            elif event.key == pygame.K_f: self.fog = not self.fog
        return None

    def draw(self):
        self.t += 1
        self.screen.fill(DARK_BG)
        for d in self.dots:
            d["y"] += d["s"]
            if d["y"] > WIN_H: d["y"] = 0
            pulse = int(180 + 60 * math.sin(self.t / 30 + d["x"]))
            pygame.gfxdraw.filled_circle(self.screen, int(d["x"]), int(d["y"]),
                                         d["r"], (pulse, pulse // 2, 0))
        cy = 45
        title = self.fonts["xl"].render("MAZE  NAVIGATOR", True, PACMAN_Y)
        self.screen.blit(title, (WIN_W // 2 - title.get_width() // 2, cy))
        mouth = int(30 * abs(math.sin(self.t / 15)))
        draw_pacman(self.screen, WIN_W // 2 - title.get_width() // 2 - 40, cy + 18, 18, mouth, 0)
        cy += 42
        sub = self.fonts["sm"].render(
            "adversarial search  |  partial observability  |  dynamic maze", True, TEXT_DIM)
        self.screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, cy))
        cy += 36
        bw, bh = 560, 260
        bx = WIN_W // 2 - bw // 2
        pygame.draw.rect(self.screen, (12, 12, 40), (bx, cy, bw, bh), border_radius=12)
        pygame.draw.rect(self.screen, WALL_BRIGHT, (bx, cy, bw, bh), 2, border_radius=12)

        def row(label, value, yy, highlight=False):
            ls = self.fonts["md"].render(label, True, TEXT_W)
            vs = self.fonts["md"].render(value, True, PACMAN_Y if highlight else ACCENT)
            self.screen.blit(ls, (bx + 34, cy + yy))
            self.screen.blit(vs, (bx + bw - vs.get_width() - 34, cy + yy))

        row("agent algorithm", f"< {ALGO_NAMES.get(self.algo, self.algo)} >", 18, True)
        row("pursuer strategy", f"< {PURSUER_NAMES.get(self.pursuer, self.pursuer)} >", 50, True)
        row("number of pursuers", f"{self.pursuers}  (press 1/2/3)", 82)
        row("dynamic walls", "ON  (D to toggle)" if self.dynamic else "OFF (D to toggle)", 114)
        row("fog of war", "ON  (F to toggle)" if self.fog else "OFF (F to toggle)", 146)
        row("split view", "toggle in game (B)", 178)
        hint = self.fonts["sm"].render(
            "< / > change agent     up/down change pursuer     ENTER to start", True, TEXT_DIM)
        self.screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, cy + bh + 10))
        cy2 = cy + bh + 28
        for i, line in enumerate([
            "SPACE pause   |   R reset maze   |   1-7 switch agent",
            "Z/X/C/V pursuer   |   +/- speed   |   F fog toggle",
            "D dynamic walls   |   B split view   |   ESC quit",
        ]):
            s = self.fonts["sm"].render(line, True, TEXT_W)
            self.screen.blit(s, (WIN_W // 2 - s.get_width() // 2, cy2 + i * 20))
        if HIGH_SCORES:
            cy3 = cy2 + 72
            hs = self.fonts["md"].render("HIGH SCORES", True, ACCENT)
            self.screen.blit(hs, (WIN_W // 2 - hs.get_width() // 2, cy3))
            for i, (sc, nm) in enumerate(HIGH_SCORES[:3]):
                s = self.fonts["sm"].render(f"#{i+1}  {nm:<18} {sc:>6}", True, TEXT_W)
                self.screen.blit(s, (WIN_W // 2 - s.get_width() // 2, cy3 + 22 + i * 18))


class GameOverScreen:
    def __init__(self, screen, fonts, outcome, stats):
        self.screen = screen; self.fonts = fonts
        self.outcome = outcome; self.stats = stats; self.t = 0

    def handle(self, event):
        if event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:      return "reset"
            if event.key == pygame.K_RETURN: return "start"
            if event.key in (pygame.K_ESCAPE, pygame.K_q): return "quit"
        return None

    def draw(self):
        self.t += 1
        overlay = pygame.Surface((WIN_W, WIN_H), pygame.SRCALPHA)
        overlay.fill((0, 0, 0, 180))
        self.screen.blit(overlay, (0, 0))
        won   = self.outcome == "GOAL!"
        col   = GOOD_C if won else BAD_C
        msg   = "YOU REACHED THE GOAL!" if won else "CAUGHT BY A PURSUER!"
        bounce = int(6 * math.sin(self.t / 15))
        title  = self.fonts["xl"].render(msg, True, col)
        self.screen.blit(title, (WIN_W // 2 - title.get_width() // 2, WIN_H // 2 - 110 + bounce))
        sub = self.fonts["lg"].render("WELL PLAYED" if won else "GAME OVER", True, PACMAN_Y)
        self.screen.blit(sub, (WIN_W // 2 - sub.get_width() // 2, WIN_H // 2 - 65))
        bw, bh = 580, 280
        bx = WIN_W // 2 - bw // 2
        by = WIN_H // 2 - 30
        pygame.draw.rect(self.screen, (10, 10, 35), (bx, by, bw, bh), border_radius=10)
        pygame.draw.rect(self.screen, col, (bx, by, bw, bh), 2, border_radius=10)
        items = [("steps taken", str(self.stats.get("steps", 0))),
                 ("map explored", f"{self.stats.get('explored', 0):.0f}%"),
                 ("wall shifts",  str(self.stats.get("shifts", 0))),
                 ("final score",  str(self.stats.get("score", 0)))]
        for i, (k, v) in enumerate(items):
            row_i = i // 2; col_i = i % 2
            ks = self.fonts["sm"].render(k, True, TEXT_DIM)
            vs = self.fonts["md"].render(v, True, PACMAN_Y if k == "final score" else TEXT_W)
            xx = bx + 20 + col_i * (bw // 2)
            yy = by + 20 + row_i * 55
            self.screen.blit(ks, (xx, yy))
            self.screen.blit(vs, (xx, yy + 18))
        hint = self.fonts["md"].render(
            "R — new maze     ENTER — menu     ESC — quit", True, TEXT_DIM)
        self.screen.blit(hint, (WIN_W // 2 - hint.get_width() // 2, WIN_H // 2 + 145))


class VisualGame:
    def __init__(self, agent_algo="lrta", pursuer_strategy="greedy",
                 num_pursuers=2, dynamic=True, fog_of_war=True):
        pygame.init()
        pygame.display.set_caption("Maze Navigator  —  Adversarial AI")
        self.sound_on = False
        try:
            pygame.mixer.init(frequency=44100, size=-16, channels=1, buffer=512)
            self.sound_on = True
            self._build_sounds()
        except Exception:
            pass
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()
        self.fonts  = {
            "xl": pygame.font.SysFont("monospace", 32, bold=True),
            "lg": pygame.font.SysFont("monospace", 24, bold=True),
            "md": pygame.font.SysFont("monospace", 20, bold=True),
            "sm": pygame.font.SysFont("monospace", 15, bold=True),
        }
        self.agent_algo = agent_algo; self.pursuer_strategy = pursuer_strategy
        self.num_pursuers = num_pursuers; self.dynamic = dynamic; self.fog_of_war = fog_of_war
        self.speed = 7; self.paused = False; self.tick_acc = 0; self.t = 0
        self.show_belief_split = False
        self.manual_dir = None; self.screen_mode = SCREEN_START
        self.start_screen = StartScreen(self.screen, self.fonts)
        self.go_screen = None; self.game_state_data = {}

    def _build_sounds(self):
        import numpy as np
        sr = 44100
        def make_beep(freq, dur, vol=0.3, shape="sine"):
            t = np.linspace(0, dur, int(sr * dur), endpoint=False)
            wave = np.sin(2 * np.pi * freq * t) if shape == "sine" else np.sign(np.sin(2 * np.pi * freq * t))
            wave = (wave * vol * 32767).astype(np.int16)
            return pygame.sndarray.make_sound(np.column_stack([wave, wave]))
        try:
            self.snd_pellet  = make_beep(880,  0.05)
            self.snd_powerup = make_beep(1200, 0.15)
            self.snd_trap    = make_beep(200,  0.2, shape="square")
            self.snd_win     = make_beep(660,  0.4)
            self.snd_lose    = make_beep(150,  0.5, shape="square")
        except Exception:
            self.sound_on = False

    def _play(self, snd_name):
        if not self.sound_on: return
        s = getattr(self, f"snd_{snd_name}", None)
        if s: s.play()

    def reset(self):
        base = Maze.generate_random(ROWS, COLS, obstacle_density=0.27, seed=random.randint(0, 99999))
        self.maze = DynamicMaze.from_static(base, shift_interval=4, num_shifts=5, seed=42) \
                    if self.dynamic else base
        self.grid = self.maze.grid
        raw_starts = [(ROWS-1, 0), (0, COLS-1), (ROWS-1, COLS-2)]
        self.pursuer_starts = [list(ps) if self.maze.is_walkable(ps) else [1, COLS-2]
                               for ps in raw_starts[:self.num_pursuers]]
        self.powerup_map = place_objects(
            self.grid, ROWS, COLS,
            start=self.maze.start, goal=self.maze.goal,
            pursuer_starts=[tuple(p) for p in self.pursuer_starts],
            num_traps=12, num_powerups=8, num_mud=10, num_water=8, num_road=6,
            seed=random.randint(0, 99999))
        self.agent_pos   = list(self.maze.start)
        self.goal_pos    = list(self.maze.goal)
        self.pursuer_pos = [list(p) for p in self.pursuer_starts]
        self.agent_anim   = SmoothPos(*self.agent_pos)
        self.pursuer_anim = [SmoothPos(*p) for p in self.pursuer_pos]
        self.mouth_angle  = 0; self.mouth_open = True; self.facing = 0
        self.agent_status   = AgentStatus()
        self.pursuer_status = [PursuerStatus() for _ in self.pursuer_starts]
        self.seen = set(); self._update_fog(self.agent_pos)
        self.lrta_h = {}; self.agent_path = []
        self.steps = 0; self.wall_shifts = 0; self.nodes_exp = 0; self.score = 0
        self.game_over = False; self.outcome = ""
        self.particles = []; self.flash_msgs = []; self.wall_anims = []; self.tick_acc = 0
        self.go_screen = None; self.screen_mode = SCREEN_GAME
        self.agent_history = deque(maxlen=10)
        self.pursuer_history = deque(maxlen=10)
        self.agent_stuck_count = 0
        self.pursuer_stuck_count = 0
        self.agent_history.append(tuple(self.agent_pos))
        self.pursuer_history.append(tuple(tuple(p) for p in self.pursuer_pos))

    def _update_fog(self, pos):
        r, c   = pos
        radius = ROWS if self.agent_status.full_reveal() else FOG_RADIUS
        for dr in range(-radius, radius + 1):
            for dc in range(-radius, radius + 1):
                if abs(dr) + abs(dc) <= radius:
                    nr, nc = r + dr, c + dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        self.seen.add((nr, nc))

    def _visible(self, pos):
        if not self.fog_of_war or self.agent_status.full_reveal(): return True
        ar, ac = self.agent_pos
        return abs(pos[0] - ar) + abs(pos[1] - ac) <= FOG_RADIUS

    def _walkable(self, pos):
        r, c = pos
        return 0 <= r < ROWS and 0 <= c < COLS and self.grid[r][c] != WALL

    def _neighbors(self, pos):
        r, c = pos
        return [[r+dr, c+dc] for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]
                if self._walkable([r+dr, c+dc])]

    def _belief_neighbors(self, pos):
        r, c = pos
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if 0 <= nr < ROWS and 0 <= nc < COLS:
                if (nr, nc) not in self.seen or self.grid[nr][nc] != WALL:
                    out.append([nr, nc])
        return out

    def _manhattan(self, a, b): return abs(a[0]-b[0]) + abs(a[1]-b[1])

    def _planning_maze(self, start=None, goal=None):
        """
        Builds a temporary Maze for search algorithms.
        All non-wall gameplay cells are treated as OPEN so that
        traps, power-ups, mud, water, and road remain traversable.
        """
        s = tuple(start or self.agent_pos)
        g = tuple(goal or self.goal_pos)

        grid_copy = []
        for row in self.grid:
            grid_copy.append([Maze.WALL if cell == WALL else Maze.OPEN for cell in row])

        grid_copy[s[0]][s[1]] = Maze.OPEN
        grid_copy[g[0]][g[1]] = Maze.OPEN
        return Maze(grid_copy, s, g)

    def _nearest_pursuer(self):
        return min(self.pursuer_pos, key=lambda pp: self._manhattan(self.agent_pos, pp))

    def _make_adv_state(self):
        """
        Builds a lightweight GameState for adversarial-search algorithms.
        Uses the nearest pursuer as the adversarial opponent in visual mode.
        """
        pursuer = tuple(self._nearest_pursuer())
        temp_maze = self._planning_maze(start=self.agent_pos, goal=self.goal_pos)
        return GameState(
            maze=temp_maze,
            agent_pos=tuple(self.agent_pos),
            pursuer_pos=pursuer,
            turn=Turn.AGENT,
            step=0,
            step_limit=50,
        )

    def _greedy_goal_fallback(self, pos):
        nb = self._neighbors(pos)
        if not nb:
            return pos
        return min(nb, key=lambda n: self._manhattan(n, self.goal_pos))

    def _next_agent_algo(self):
        if self.agent_algo == "manual":
            return "manual"
        if self.agent_algo not in AGENT_ORDER:
            return "lrta"
        idx = AGENT_ORDER.index(self.agent_algo)
        for step in range(1, len(AGENT_ORDER) + 1):
            cand = AGENT_ORDER[(idx + step) % len(AGENT_ORDER)]
            if cand != "manual":
                return cand
        return self.agent_algo

    def _next_pursuer_algo(self):
        if self.pursuer_strategy not in PURSUER_ORDER:
            return "greedy"
        idx = PURSUER_ORDER.index(self.pursuer_strategy)
        return PURSUER_ORDER[(idx + 1) % len(PURSUER_ORDER)]

    def _switch_agent_algo(self, new_algo, reason=""):
        if new_algo == self.agent_algo:
            return
        old = self.agent_algo
        self.agent_algo = new_algo
        self.agent_path = []
        self.lrta_h = {}
        msg = f"Agent switched: {ALGO_NAMES.get(old, old)} → {ALGO_NAMES.get(new_algo, new_algo)}"
        if reason:
            msg += f" ({reason})"
        self._flash(msg, WARN_C)

    def _switch_pursuer_algo(self, new_algo, reason=""):
        if new_algo == self.pursuer_strategy:
            return
        old = self.pursuer_strategy
        self.pursuer_strategy = new_algo
        msg = f"Pursuer switched: {PURSUER_NAMES.get(old, old)} → {PURSUER_NAMES.get(new_algo, new_algo)}"
        if reason:
            msg += f" ({reason})"
        self._flash(msg, ACCENT)

    def _agent_is_stuck(self):
        hist = list(self.agent_history)
        if len(hist) < 6:
            return False
        if len(set(hist[-4:])) == 1:
            return True
        tail = hist[-6:]
        if len(set(tail)) <= 2:
            return True
        return False

    def _pursuer_is_stuck(self):
        hist = list(self.pursuer_history)
        if len(hist) < 6:
            return False
        if len(set(hist[-4:])) == 1:
            return True
        tail = hist[-6:]
        if len(set(tail)) <= 2:
            return True
        return False

    def _astar(self, start, goal, belief=True):
        h = lambda p: self._manhattan(p, goal)
        ctr = 0; heap = [(h(start), 0, ctr, start, [start])]; vis = {}
        while heap:
            f, g, _, cur, path = heapq.heappop(heap)
            k = (cur[0], cur[1])
            if vis.get(k, 9999) <= g: continue
            vis[k] = g
            if cur[0] == goal[0] and cur[1] == goal[1]:
                self.nodes_exp += len(vis); return path
            for n in (self._belief_neighbors(cur) if belief else self._neighbors(cur)):
                ctr += 1
                heapq.heappush(heap, (g+1+h(n), g+1, ctr, n, path+[n]))
        return None

    def _lrta_step(self, pos):
        def hget(p):
            k = (p[0], p[1])
            if k not in self.lrta_h: self.lrta_h[k] = self._manhattan(p, self.goal_pos)
            return self.lrta_h[k]
        nb = self._belief_neighbors(pos)
        if not nb: return pos
        mc = min(1 + hget(n) for n in nb)
        self.lrta_h[(pos[0], pos[1])] = max(hget(pos), mc)
        return min(nb, key=lambda n: 1 + hget(n))

    def _minimax_step(self, pos, p_positions, depth=3, ab=False):
        def score(ap, pps, d, is_max, alpha, beta):
            if ap[0]==self.goal_pos[0] and ap[1]==self.goal_pos[1]: return 1000
            for pp in pps:
                if ap[0]==pp[0] and ap[1]==pp[1]: return -1000
            if d == 0:
                return (-self._manhattan(ap, self.goal_pos)*10 +
                        min(self._manhattan(ap, pp) for pp in pps)*5)
            if is_max:
                best = -math.inf
                for n in (self._neighbors(ap) or [ap]):
                    v = score(n, pps, d-1, False, alpha, beta); best = max(best, v)
                    if ab: alpha = max(alpha, best)
                    if ab and best >= beta: break
                return best
            else:
                best = math.inf
                for n in (self._neighbors(pps[0]) or [pps[0]]):
                    v = score(ap, [n]+pps[1:], d-1, True, alpha, beta); best = min(best, v)
                    if ab: beta = min(beta, best)
                    if ab and best <= alpha: break
                return best
        moves = self._neighbors(pos)
        if not moves: return pos
        self.nodes_exp += len(moves) * (4 ** depth)
        return max(moves, key=lambda n: score(n, p_positions, depth-1, False, -math.inf, math.inf))

    def _pursuer_step(self, pp, idx):
        nb = self._neighbors(pp)
        if not nb: return pp
        if self.pursuer_strategy == "random": return random.choice(nb)
        if self.pursuer_strategy == "greedy":
            return min(nb, key=lambda n: self._manhattan(n, self.agent_pos))
        if self.pursuer_strategy == "beam":
            path = self._astar(pp, self.agent_pos, belief=False)
            return path[1] if path and len(path) > 1 else nb[0]
        path = self._astar(pp, self.agent_pos, belief=False)
        return path[1] if path and len(path) > 1 else nb[0]

    def tick(self):
        if self.game_over: return
        if self.dynamic:
            protected = {tuple(self.agent_pos), tuple(self.goal_pos)}
            for pp in self.pursuer_pos: protected.add(tuple(pp))
            prev_grid = [row[:] for row in self.grid]
            shifted   = self.maze.step(protected_positions=protected)
            if shifted:
                self.wall_shifts += 1; self.lrta_h = {}; self.agent_path = []
                for r in range(ROWS):
                    for c in range(COLS):
                        if prev_grid[r][c] != self.grid[r][c]:
                            self.wall_anims.append(WallAnim((r, c), self.grid[r][c] == WALL))
        self.agent_status.tick()
        for ps in self.pursuer_status: ps.tick()
        if not self.agent_status.is_frozen():
            self._do_agent_move()
            if self.agent_status.has_speed() and not self.game_over:
                self._do_agent_move()
        else:
            spawn_particles(self.particles, *self.agent_anim.center(), BAD_C, 4)
        if self.game_over: return
        for i, pp in enumerate(self.pursuer_pos):
            if self.pursuer_status[i].is_frozen(): continue
            new_pp = self._pursuer_step(pp, i)
            self.pursuer_pos[i] = new_pp
            self.pursuer_anim[i].set_target(*new_pp)
            if new_pp[0]==self.agent_pos[0] and new_pp[1]==self.agent_pos[1]:
                self._end_game("CAUGHT"); return
        if self.mouth_open:
            self.mouth_angle = min(35, self.mouth_angle + 8)
            if self.mouth_angle >= 35:
                self.mouth_open = False
        else:
            self.mouth_angle = max(2, self.mouth_angle - 8)
            if self.mouth_angle <= 2:
                self.mouth_open = True
        self.steps += 1

        self.agent_history.append(tuple(self.agent_pos))
        self.pursuer_history.append(tuple(tuple(p) for p in self.pursuer_pos))
        if self.agent_algo != "manual" and self._agent_is_stuck():
            self.agent_stuck_count += 1
        else:
            self.agent_stuck_count = 0
        if self.agent_stuck_count >= 2:
            new_algo = self._next_agent_algo()
            self._switch_agent_algo(new_algo, reason="stuck loop")
            self.agent_stuck_count = 0
            self.agent_history.clear()
            self.agent_history.append(tuple(self.agent_pos))
        if self._pursuer_is_stuck():
            self.pursuer_stuck_count += 1
        else:
            self.pursuer_stuck_count = 0
        if self.pursuer_stuck_count >= 2:
            new_algo = self._next_pursuer_algo()
            self._switch_pursuer_algo(new_algo, reason="stuck loop")
            self.pursuer_stuck_count = 0
            self.pursuer_history.clear()
            self.pursuer_history.append(tuple(tuple(p) for p in self.pursuer_pos))

    def _do_agent_move(self):
        pos = self.agent_pos
        algo = self.agent_algo

        if algo == "manual":
            if self.manual_dir is None:
                return
            dr, dc = self.manual_dir
            self.manual_dir = None
            new_pos = [pos[0] + dr, pos[1] + dc]
            self.agent_path = []

        elif algo == "lrta":
            new_pos = self._lrta_step(pos)
            self.agent_path = []

        elif algo == "minimax":
            new_pos = self._minimax_step(pos, self.pursuer_pos, 3, False)
            self.agent_path = []

        elif algo == "alpha_beta":
            new_pos = self._minimax_step(pos, self.pursuer_pos, 4, True)
            self.agent_path = []

        elif algo == "expectimax":
            state = self._make_adv_state()
            result = expectimax(state, depth_limit=4)
            self.nodes_exp += result.nodes_expanded
            if result.best_move_pos is not None:
                new_pos = list(result.best_move_pos)
            else:
                new_pos = pos
            self.agent_path = []

        elif algo == "hill_climb":
            temp_maze = self._planning_maze(start=pos, goal=self.goal_pos)
            result = hill_climb(
                temp_maze,
                "manhattan",
                start=tuple(pos),
                allow_sideways=True,
                max_sideways=6,
            )
            self.nodes_exp += result.nodes_expanded
            if result.success and len(result.path) > 1:
                new_pos = list(result.path[1])
                self.agent_path = [list(p) for p in result.path]
            else:
                new_pos = self._greedy_goal_fallback(pos)
                self.agent_path = []

        elif algo == "beam_search":
            temp_maze = self._planning_maze(start=pos, goal=self.goal_pos)
            result = beam_search(
                temp_maze,
                beam_width=3,
                heuristic_name="manhattan",
                start=tuple(pos),
            )
            self.nodes_exp += result.nodes_expanded
            if result.success and len(result.path) > 1:
                new_pos = list(result.path[1])
                self.agent_path = [list(p) for p in result.path]
            else:
                new_pos = self._greedy_goal_fallback(pos)
                self.agent_path = []

        else:
            new_pos = pos
            self.agent_path = []

        if not self._walkable(new_pos):
            new_pos = pos

        if new_pos[1] > pos[1]:
            self.facing = 0
        elif new_pos[1] < pos[1]:
            self.facing = 180
        elif new_pos[0] > pos[0]:
            self.facing = 90
        elif new_pos[0] < pos[0]:
            self.facing = 270

        self.agent_pos = new_pos
        self.agent_anim.set_target(*new_pos)
        self._update_fog(new_pos)

        if new_pos[0] == self.goal_pos[0] and new_pos[1] == self.goal_pos[1]:
            self.score += 500
            self._end_game("GOAL!")
            return

        cell = self.grid[new_pos[0]][new_pos[1]]
        if cell == TRAP:
            if self.agent_status.apply_trap():
                self._flash("TRAP!  frozen 3 turns", BAD_C)
                spawn_particles(self.particles, *self.agent_anim.center(), TRAP_C, 14)
                self._play("trap")
            else:
                self._flash("Shield blocked the trap!", GOOD_C)

        elif cell == POWERUP:
            pu = self.powerup_map.get((new_pos[0], new_pos[1]), PU_SPEED)
            freeze = self.agent_status.apply_powerup(pu)
            self.grid[new_pos[0]][new_pos[1]] = OPEN
            self.powerup_map.pop((new_pos[0], new_pos[1]), None)
            self.score += 50
            if freeze:
                for ps in self.pursuer_status:
                    ps.freeze(5)
                self._flash("FREEZE!  pursuers frozen 5 turns", (100, 180, 255))
            else:
                self._flash(f"Power-up: {pu.upper()}!", POWER_C)
            spawn_particles(self.particles, *self.agent_anim.center(), POWER_C, 18)
            self._play("powerup")

        for pp in self.pursuer_pos:
            if new_pos[0] == pp[0] and new_pos[1] == pp[1]:
                self._end_game("CAUGHT")
                return

    def _end_game(self, outcome):
        self.game_over = True; self.outcome = outcome
        col = GOOD_C if outcome == "GOAL!" else BAD_C
        spawn_particles(self.particles, *self.agent_anim.center(), col, 30)
        self._play("win" if outcome == "GOAL!" else "lose")
        self.game_state_data = {
            "steps": self.steps,
            "explored": len(self.seen) * 100 / (ROWS * COLS),
            "shifts": self.wall_shifts, "score": self.score}
        HIGH_SCORES.append((self.score, self.agent_algo))
        HIGH_SCORES.sort(key=lambda x: -x[0])

    def _flash(self, text, color):
        self.flash_msgs.append(FlashMsg(text, color))

    def draw_game(self):
        self.t += 1
        self.screen.fill(DARK_BG)

        if self.show_belief_split:
            self._draw_split_view()
        else:
            self._draw_maze()
            self._draw_wall_anims()
            self._draw_particles()
            self._draw_agents()

        self._draw_panel()
        self._draw_bottom_bar()
        self._draw_flash_msgs()

        if self.game_over:
            if self.go_screen is None:
                self.go_screen = GameOverScreen(
                    self.screen,
                    self.fonts,
                    self.outcome,
                    self.game_state_data
                )
            self.go_screen.draw()

    def _draw_board_cell(self, surf, r, c, cell_size, show_truth, show_fog):
        x, y = c * cell_size, r * cell_size
        pos = (r, c)
        cell = self.grid[r][c]
        in_seen = pos in self.seen
        in_vis = self._visible(pos)
        if show_fog and not in_seen:
            pygame.draw.rect(surf, (2, 2, 8), (x, y, cell_size, cell_size))
            return
        if cell == WALL:
            pygame.draw.rect(surf, WALL_DARK, (x, y, cell_size, cell_size))
            pygame.draw.rect(surf, WALL_BRIGHT, (x + 1, y + 1, cell_size - 2, cell_size - 2), 2)
            if show_fog and not in_vis:
                dim = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
                dim.fill((0, 0, 0, 160))
                surf.blit(dim, (x, y))
            return
        floor_col = {MUD: MUD_C, WATER: WATER_C, ROAD: ROAD_C}.get(cell, (12, 12, 35))
        pygame.draw.rect(surf, floor_col, (x, y, cell_size, cell_size))
        if show_fog and not in_vis:
            dim = pygame.Surface((cell_size, cell_size), pygame.SRCALPHA)
            dim.fill((0, 0, 0, 140))
            surf.blit(dim, (x, y))
            return
        cx_, cy_ = x + cell_size // 2, y + cell_size // 2
        if cell == TRAP and show_truth:
            r2 = cell_size // 3
            pygame.draw.line(surf, TRAP_C, (cx_ - r2, cy_ - r2), (cx_ + r2, cy_ + r2), 2)
            pygame.draw.line(surf, TRAP_C, (cx_ + r2, cy_ - r2), (cx_ - r2, cy_ + r2), 2)
        elif cell == POWERUP and show_truth:
            rr = max(3, cell_size // 5)
            pygame.gfxdraw.filled_circle(surf, cx_, cy_, rr, POWER_C)
            pygame.gfxdraw.aacircle(surf, cx_, cy_, rr, (255, 255, 200))
        elif cell == MUD:
            s = self.fonts["sm"].render("~", True, (150, 110, 50))
            surf.blit(s, (cx_ - s.get_width() // 2, cy_ - s.get_height() // 2))
        elif cell == WATER:
            s = self.fonts["sm"].render("~", True, (80, 150, 255))
            surf.blit(s, (cx_ - s.get_width() // 2, cy_ - s.get_height() // 2))
        elif cell == ROAD:
            pygame.draw.line(surf, (60, 120, 60), (cx_, y + 3), (cx_, y + cell_size - 3), 2)
        else:
            rr = max(2, cell_size // 10)
            pygame.gfxdraw.filled_circle(surf, cx_, cy_, rr, PELLET_C)
            pygame.gfxdraw.aacircle(surf, cx_, cy_, rr, PELLET_C)

    def _draw_split_view(self):
        left_margin = 10
        top_margin = 24
        gap = 14
        usable_w = MAZE_W - 2 * left_margin
        each_w = (usable_w - gap) // 2
        each_h = MAZE_H - top_margin - 8
        left_x = left_margin
        right_x = left_margin + each_w + gap
        self._draw_board(
            self.screen,
            board_x=left_x,
            board_y=top_margin,
            board_w=each_w,
            board_h=each_h,
            show_truth=False,
            title="Agent Belief / Visible World",
        )
        self._draw_board(
            self.screen,
            board_x=right_x,
            board_y=top_margin,
            board_w=each_w,
            board_h=each_h,
            show_truth=True,
            title="True Maze State",
        )

    def _draw_board(self, surf, board_x, board_y, board_w, board_h, show_truth=False, title=""):
        cell_size = min(board_w // COLS, board_h // ROWS)

        board = pygame.Surface((COLS * cell_size, ROWS * cell_size))
        board.fill(DARK_BG)

        for r in range(ROWS):
            for c in range(COLS):
                self._draw_board_cell(
                    board,
                    r,
                    c,
                    cell_size,
                    show_truth=show_truth,
                    show_fog=(not show_truth)
                )

        gr, gc = self.goal_pos
        gx, gy = gc * cell_size, gr * cell_size
        pygame.draw.rect(board, GOAL_C, (gx + 2, gy + 2, cell_size - 4, cell_size - 4), border_radius=4)

        acx = self.agent_pos[1] * cell_size + cell_size // 2
        acy = self.agent_pos[0] * cell_size + cell_size // 2
        draw_pacman(board, acx, acy, max(6, cell_size // 2 - 2), self.mouth_angle, self.facing)

        for i, pp in enumerate(self.pursuer_pos):
            if show_truth or self._visible(pp) or self.agent_status.full_reveal():
                pcx = pp[1] * cell_size + cell_size // 2
                pcy = pp[0] * cell_size + cell_size // 2
                draw_ghost(
                    board,
                    pcx,
                    pcy,
                    max(6, cell_size // 2 - 2),
                    GHOST_COLS[i % len(GHOST_COLS)],
                    self.pursuer_status[i].is_frozen(),
                    self.pursuer_status[i].is_frozen()
                )

        self.screen.blit(board, (board_x, board_y))

        if title:
            s = self.fonts["md"].render(title, True, ACCENT)
            self.screen.blit(s, (board_x, board_y - 18))

    def _draw_maze(self):
        for r in range(ROWS):
            for c in range(COLS):
                x, y = c * CELL, r * CELL; pos = (r, c)
                cell = self.grid[r][c]; in_seen = pos in self.seen; in_vis = self._visible(pos)
                if self.fog_of_war and not in_seen:
                    pygame.draw.rect(self.screen, (2, 2, 8), (x, y, CELL, CELL)); continue
                if cell == WALL:
                    draw_wall_cell(self.screen, x, y, CELL, self.t)
                    if self.fog_of_war and not in_vis:
                        dim = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                        dim.fill((0, 0, 0, 160)); self.screen.blit(dim, (x, y))
                    continue
                floor_col = {MUD: MUD_C, WATER: WATER_C, ROAD: ROAD_C}.get(cell, (12, 12, 35))
                pygame.draw.rect(self.screen, floor_col, (x, y, CELL, CELL))
                if self.fog_of_war and not in_vis:
                    dim = pygame.Surface((CELL, CELL), pygame.SRCALPHA)
                    dim.fill((0, 0, 0, 140)); self.screen.blit(dim, (x, y)); continue
                cx_, cy_ = x + CELL // 2, y + CELL // 2
                if cell == TRAP:
                    r2 = CELL // 3
                    pygame.draw.line(self.screen, TRAP_C, (cx_-r2, cy_-r2), (cx_+r2, cy_+r2), 3)
                    pygame.draw.line(self.screen, TRAP_C, (cx_+r2, cy_-r2), (cx_-r2, cy_+r2), 3)
                    pygame.gfxdraw.aacircle(self.screen, cx_, cy_, r2+2, (*TRAP_C, 120))
                elif cell == POWERUP:
                    draw_powerup_pellet(self.screen, cx_, cy_, CELL // 5, self.t)
                    pu = self.powerup_map.get(pos, PU_SPEED)
                    lbl = {"reveal":"R","speed":"S","shield":"X","freeze":"F"}.get(pu, "P")
                    s = self.fonts["sm"].render(lbl, True, (20, 20, 20))
                    self.screen.blit(s, (cx_ - s.get_width()//2, cy_ - s.get_height()//2))
                elif cell == MUD:
                    s = self.fonts["sm"].render("~", True, (150, 110, 50))
                    self.screen.blit(s, (cx_ - s.get_width()//2, cy_ - s.get_height()//2))
                elif cell == WATER:
                    s = self.fonts["sm"].render("~", True, (80, 150, 255))
                    self.screen.blit(s, (cx_ - s.get_width()//2, cy_ - s.get_height()//2))
                elif cell == ROAD:
                    pygame.draw.line(self.screen, (60, 120, 60), (cx_, y+4), (cx_, y+CELL-4), 2)
                else:
                    draw_pellet(self.screen, cx_, cy_, 3)
        gr, gc = self.maze.goal; gx, gy = gc * CELL, gr * CELL
        pulse = int(3 * math.sin(self.t / 20))
        pygame.draw.rect(self.screen, GOAL_C,
                         (gx+4-pulse, gy+4-pulse, CELL-8+pulse*2, CELL-8+pulse*2), border_radius=6)
        s = self.fonts["md"].render("G", True, (10, 30, 10))
        self.screen.blit(s, (gx + CELL//2 - s.get_width()//2, gy + CELL//2 - s.get_height()//2))
        if self.agent_path and len(self.agent_path) > 1:
            for p in self.agent_path[1:min(len(self.agent_path), 12)]:
                if self._visible(p):
                    pygame.gfxdraw.filled_circle(self.screen,
                        p[1]*CELL+CELL//2, p[0]*CELL+CELL//2, 3, (60, 180, 255, 80))

    def _draw_wall_anims(self):
        alive = []
        for wa in self.wall_anims:
            wa.tick(); wa.draw(self.screen, self.t)
            if wa.alive(): alive.append(wa)
        self.wall_anims = alive

    def _draw_particles(self):
        alive = []
        for p in self.particles:
            p.update(); p.draw(self.screen)
            if p.alive(): alive.append(p)
        self.particles = alive

    def _draw_agents(self):
        self.agent_anim.update()
        for pa in self.pursuer_anim: pa.update()
        for i, pa in enumerate(self.pursuer_anim):
            if not self._visible(self.pursuer_pos[i]) and self.fog_of_war \
                    and not self.agent_status.full_reveal(): continue
            draw_ghost(self.screen, *pa.center(), CELL//2 - 2,
                       GHOST_COLS[i % len(GHOST_COLS)],
                       self.pursuer_status[i].is_frozen(),
                       self.pursuer_status[i].is_frozen())
        draw_pacman(self.screen, *self.agent_anim.center(),
                    CELL//2 - 3, self.mouth_angle, self.facing)

    def _draw_panel(self):
        px = MAZE_W
        pygame.draw.rect(self.screen, PANEL_BG, (px, 0, PANEL_W, MAZE_H))
        pygame.draw.line(self.screen, WALL_BRIGHT, (px, 0), (px, MAZE_H), 1)
        y = 14

        def txt(t, col=TEXT_W, f="md", indent=14, gap=7):
            nonlocal y
            s = self.fonts[f].render(t, True, col)
            self.screen.blit(s, (px + indent, y))
            y += s.get_height() + gap

        def div():
            nonlocal y
            pygame.draw.line(self.screen, (30, 35, 80),
                             (px+8, y+2), (px+PANEL_W-8, y+2), 1); y += 10

        mouth = int(25 * abs(math.sin(self.t / 15)))
        draw_pacman(self.screen, px + 20, y + 10, 10, mouth, 0)
        s = self.fonts["lg"].render("MAZE NAVIGATOR", True, PACMAN_Y)
        self.screen.blit(s, (px + 38, y)); y += 26; div()
        txt(f"algo    {ALGO_NAMES.get(self.agent_algo,'')}", TEXT_DIM)
        txt(f"pursuers  {PURSUER_NAMES.get(self.pursuer_strategy, self.pursuer_strategy)} x{self.num_pursuers}",
            TEXT_DIM)
        txt(f"dynamic {'ON' if self.dynamic else 'OFF'}", TEXT_DIM)
        txt(f"fog     {'ON' if self.fog_of_war else 'OFF'}", TEXT_W)
        txt(f"split   {'ON' if self.show_belief_split else 'OFF'}",
            GOOD_C if self.show_belief_split else WARN_C)
        div()
        txt("STATS", ACCENT, "lg", gap=9)
        txt(f"steps      {self.steps}")
        txt(f"explored   {len(self.seen)*100//(ROWS*COLS)}%")
        txt(f"shifts     {self.wall_shifts}")
        txt(f"nodes      {self.nodes_exp}")
        txt(f"score      {self.score}", PACMAN_Y); div()
        txt("EFFECTS", ACCENT, "lg")
        effects = self.agent_status.active_effects()
        for e in (effects or ["none"]):
            col = BAD_C if "FROZEN" in e else WARN_C if "SPEED" in e else (GOOD_C if e != "none" else TEXT_DIM)
            txt(f"  {e}", col)
        div()
        txt("KEYS", ACCENT, "lg")
        for line in ["SPACE  pause",  "R      new maze",   "1-7    algorithm",
                     "+/-    speed",  "F fog  D walls",    "B split view",
                     "Z..V pursuer",  "arrows manual",     "ESC    quit"]:
            txt(line, TEXT_DIM, "sm")
        self._draw_minimap()

    def _draw_minimap(self):
        """
        draws a scaled-down true maze overview in the side panel.
        shows walls, open cells, agent and pursuer positions,
        and the goal — ignoring fog of war entirely.
        this gives the viewer a god-eye view alongside the
        agent limited fog-of-war perspective.
        """
        mm_cols = min(PANEL_W - 30, 80)
        mm_rows = int(mm_cols * ROWS / COLS)
        cs      = mm_cols // COLS
        if cs < 2: cs = 2
        mm_w    = cs * COLS                                                                                     
        mm_h    = cs * ROWS
        ox      = MAZE_W + PANEL_W - mm_w - 15 
        oy = MAZE_H - mm_h - 2

        for r in range(ROWS):
            for c in range(COLS):
                x   = ox + c * cs
                y   = oy + r * cs
                cell = self.grid[r][c]
                if cell == 1:
                    col = (40, 50, 120)
                elif cell == 2:
                    col = (160, 40, 40)
                elif cell == 3:
                    col = (160, 140, 40)
                elif cell == 4:
                    col = (80, 55, 20)
                elif cell == 5:
                    col = (20, 60, 140)
                elif cell == 6:
                    col = (20, 60, 30)
                else:
                    col = (20, 22, 50)
                pygame.draw.rect(self.screen, col, (x, y, cs, cs))

        gr, gc = self.maze.goal
        pygame.draw.rect(self.screen, GOAL_C,
                         (ox + gc*cs, oy + gr*cs, cs, cs))

        for i, pp in enumerate(self.pursuer_pos):
            pcol = GHOST_COLS[i % len(GHOST_COLS)]
            pygame.draw.rect(self.screen, pcol,
                             (ox + pp[1]*cs, oy + pp[0]*cs, cs, cs))

        ar, ac = self.agent_pos
        pygame.draw.rect(self.screen, PACMAN_Y,
                         (ox + ac*cs, oy + ar*cs, cs, cs))

        label = self.fonts["sm"].render("true maze", True, TEXT_DIM)
        self.screen.blit(label, (ox, oy - 14))
        pygame.draw.rect(self.screen, (40, 45, 90), (ox-1, oy-1, mm_w+2, mm_h+2), 1)

    def _draw_bottom_bar(self):
        y = MAZE_H
        pygame.draw.rect(self.screen, PANEL_BG, (0, y, WIN_W, 70))
        pygame.draw.line(self.screen, WALL_BRIGHT, (0, y), (WIN_W, y), 1)
        status = "PAUSED" if self.paused else \
                 ("GAME OVER — " + self.outcome if self.game_over else "RUNNING")
        scol = WARN_C if self.paused else (BAD_C if self.game_over and self.outcome=="CAUGHT"
                else GOOD_C if self.game_over else ACCENT)
        s1 = self.fonts["md"].render(
            f"{ALGO_NAMES.get(self.agent_algo, self.agent_algo)}  vs  "
            f"{PURSUER_NAMES.get(self.pursuer_strategy, self.pursuer_strategy)} pursuers",
            True, TEXT_DIM
        )
        s2 = self.fonts["lg"].render(status, True, scol)
        s3 = self.fonts["sm"].render(f"speed {self.speed}x", True, TEXT_DIM)
        self.screen.blit(s1, (12, y + 8))
        self.screen.blit(s2, (12, y + 30))
        self.screen.blit(s3, (MAZE_W - 80, y + 8))

    def _draw_flash_msgs(self):
        alive = []; yy = MAZE_H // 2 - 60
        for fm in self.flash_msgs:
            fm.tick()
            if fm.alive():
                s = self.fonts["lg"].render(fm.text, True, fm.color)
                self.screen.blit(s, (MAZE_W//2 - s.get_width()//2, yy))
                yy += 26; alive.append(fm)
        self.flash_msgs = alive

    def run(self):
        FPS = 60
        while True:
            dt = self.clock.tick(FPS); self.t += 1
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()
                if self.screen_mode == SCREEN_START:
                    result = self.start_screen.handle(event)
                    if result == "start":
                        self.agent_algo = self.start_screen.algo
                        self.pursuer_strategy = self.start_screen.pursuer
                        self.num_pursuers = self.start_screen.pursuers
                        self.dynamic = self.start_screen.dynamic
                        self.fog_of_war = self.start_screen.fog
                        self.reset()
                elif self.screen_mode == SCREEN_GAME:
                    if event.type == pygame.KEYDOWN:
                        k = event.key
                        if k in (pygame.K_q, pygame.K_ESCAPE): pygame.quit(); sys.exit()
                        elif k == pygame.K_SPACE: self.paused = not self.paused
                        elif k == pygame.K_r: self.reset()
                        elif k == pygame.K_1:
                            self.agent_algo = "lrta"
                            self.lrta_h = {}
                            self.agent_path = []
                        elif k == pygame.K_2:
                            self.agent_algo = "minimax"
                            self.agent_path = []
                        elif k == pygame.K_3: self.agent_algo = "alpha_beta"; self.agent_path = []
                        elif k == pygame.K_4:
                            self.agent_algo = "expectimax"
                            self.agent_path = []
                        elif k == pygame.K_5:
                            self.agent_algo = "hill_climb"
                            self.agent_path = []
                        elif k == pygame.K_6:
                            self.agent_algo = "beam_search"
                            self.agent_path = []
                        elif k == pygame.K_7:
                            self.agent_algo = "manual"
                            self.agent_path = []
                        elif k == pygame.K_z: self.pursuer_strategy = "random"
                        elif k == pygame.K_x: self.pursuer_strategy = "greedy"
                        elif k == pygame.K_c: self.pursuer_strategy = "beam"
                        elif k == pygame.K_v: self.pursuer_strategy = "astar"
                        elif k == pygame.K_EQUALS: self.speed = min(20, self.speed + 1)
                        elif k == pygame.K_MINUS:  self.speed = max(1,  self.speed - 1)
                        elif k == pygame.K_f: self.fog_of_war = not self.fog_of_war
                        elif k == pygame.K_b: self.show_belief_split = not self.show_belief_split
                        elif k == pygame.K_d: self.dynamic = not self.dynamic; self.reset()
                        elif k == pygame.K_UP:    self.manual_dir = (-1,  0)
                        elif k == pygame.K_DOWN:  self.manual_dir = ( 1,  0)
                        elif k == pygame.K_LEFT:  self.manual_dir = ( 0, -1)
                        elif k == pygame.K_RIGHT: self.manual_dir = ( 0,  1)
                elif self.screen_mode == SCREEN_GAMEOVER and self.go_screen:
                    result = self.go_screen.handle(event)
                    if result == "reset": self.reset()
                    elif result == "start":
                        self.screen_mode = SCREEN_START
                        self.start_screen = StartScreen(self.screen, self.fonts)
                    elif result == "quit": pygame.quit(); sys.exit()
            if self.screen_mode == SCREEN_GAME and not self.paused and not self.game_over:
                self.tick_acc += dt
                ms = 1000 // max(1, self.speed)
                while self.tick_acc >= ms:
                    self.tick(); self.tick_acc -= ms
                    if self.game_over: self.screen_mode = SCREEN_GAMEOVER; break
            if self.screen_mode == SCREEN_START:
                self.start_screen.draw()
            elif self.screen_mode in (SCREEN_GAME, SCREEN_GAMEOVER):
                self.draw_game()
            pygame.display.flip()


def launch(agent_algo="lrta", pursuer_strategy="greedy",
           num_pursuers=2, dynamic=True, fog=True):
    game = VisualGame(agent_algo=agent_algo, pursuer_strategy=pursuer_strategy,
                      num_pursuers=num_pursuers, dynamic=dynamic, fog_of_war=fog)
    game.run()


if __name__ == "__main__":
    launch()