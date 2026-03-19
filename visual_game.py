"""
visual_game.py — pygame visual simulation
project: adversarial maze navigation
authors: vikramaditya sogani & namya singh

run with:  python main.py --visual
or directly: python visual_game.py

controls:
  SPACE      — pause / resume
  R          — reset / new maze
  1-4        — switch agent algorithm
  Q/ESC      — quit
  arrow keys — manual control (when manual mode selected)
  +/-        — speed up / slow down
"""

import pygame
import sys
import math
import heapq
import random
from collections import deque

from maze         import Maze, DynamicMaze
from game_objects import (
    OPEN, WALL, TRAP, POWERUP, MUD, WATER, ROAD,
    CELL_COLORS, CELL_LABELS, POWERUP_COLORS, POWERUP_LABELS,
    TERRAIN_COST, AgentStatus, PursuerStatus, place_objects,
    PU_REVEAL, PU_SPEED, PU_SHIELD, PU_FREEZE
)


# ─────────────────────────────────────────────
#  layout constants
# ─────────────────────────────────────────────

ROWS, COLS     = 17, 17
CELL_SIZE      = 36
FOG_RADIUS     = 4

MAZE_W = COLS * CELL_SIZE
MAZE_H = ROWS * CELL_SIZE
PANEL_W = 260
WIN_W   = MAZE_W + PANEL_W
WIN_H   = MAZE_H + 60   # bottom bar

# colors
C_BG         = (10,  12,  28)
C_WALL       = (50,  54,  80)
C_OPEN       = (18,  20,  42)
C_FOG        = (6,   7,   18)
C_AGENT      = (60, 180, 255)
C_GOAL       = (60, 220, 100)
C_START      = (100, 100, 160)
C_PURSUER    = [(220, 60, 60), (220, 130, 40), (200, 60, 160)]
C_PATH       = (60, 180, 255, 60)
C_TRAP       = (200, 60,  40)
C_MUD        = (110, 80,  30)
C_WATER      = (40, 100, 200)
C_ROAD       = (40,  80,  40)
C_PANEL_BG   = (14,  16,  34)
C_TEXT       = (200, 205, 230)
C_TEXT_DIM   = (100, 105, 130)
C_ACCENT     = (60, 180, 255)
C_WARN       = (220, 160, 40)
C_GOOD       = (60, 220, 100)
C_BAD        = (220, 60,  60)
C_POWERUP    = {
    PU_REVEAL : (200, 180, 255),
    PU_SPEED  : (255, 220,  50),
    PU_SHIELD : (50,  220, 180),
    PU_FREEZE : (100, 180, 255),
}

ALGO_NAMES = {
    "astar"    : "A* (belief map)",
    "lrta"     : "LRTA*",
    "minimax"  : "Minimax",
    "alphabeta": "Alpha-Beta",
    "manual"   : "Manual (arrows)",
}

PURSUER_NAMES = {
    "random" : "Random",
    "greedy" : "Greedy",
    "astar"  : "A* pursuer",
}


# ─────────────────────────────────────────────
#  game state
# ─────────────────────────────────────────────

class VisualGame:

    def __init__(self, agent_algo="astar", pursuer_strategy="greedy",
                 num_pursuers=2, dynamic=True, fog_of_war=True):

        pygame.init()
        pygame.display.set_caption("Adversarial Maze Navigator")
        self.screen = pygame.display.set_mode((WIN_W, WIN_H))
        self.clock  = pygame.time.Clock()

        # fonts
        self.font_lg  = pygame.font.SysFont("monospace", 15, bold=True)
        self.font_md  = pygame.font.SysFont("monospace", 13)
        self.font_sm  = pygame.font.SysFont("monospace", 11)
        self.font_xl  = pygame.font.SysFont("monospace", 22, bold=True)

        self.agent_algo       = agent_algo
        self.pursuer_strategy = pursuer_strategy
        self.num_pursuers     = num_pursuers
        self.dynamic          = dynamic
        self.fog_of_war       = fog_of_war

        self.speed      = 8    # frames per second for game ticks
        self.paused     = False
        self.game_over  = False
        self.outcome    = ""
        self.tick_timer = 0
        self.manual_dir = None

        # particle effects
        self.particles  = []
        self.flash_msg  = None
        self.flash_timer= 0

        self.reset()

    # ── maze setup ───────────────────────────

    def reset(self):
        """build a new maze and reset all game state."""
        base = Maze.generate_random(ROWS, COLS, obstacle_density=0.27,
                                    seed=random.randint(0, 9999))
        if self.dynamic:
            self.maze = DynamicMaze.from_static(base, shift_interval=7,
                                                num_shifts=3, seed=42)
        else:
            self.maze = base

        self.grid = self.maze.grid   # live reference

        # pursuer start positions (corners / edges)
        raw_starts = [
            (ROWS-1, 0), (0, COLS-1), (ROWS-1, COLS-2)
        ]
        self.pursuer_starts = []
        for ps in raw_starts[:self.num_pursuers]:
            if self.maze.is_walkable(ps):
                self.pursuer_starts.append(list(ps))
            else:
                self.pursuer_starts.append([1, COLS-2])

        # place game objects
        self.powerup_map = place_objects(
            self.grid, ROWS, COLS,
            start          = self.maze.start,
            goal           = self.maze.goal,
            pursuer_starts = [tuple(p) for p in self.pursuer_starts],
            num_traps      = 12,
            num_powerups   = 7,
            num_mud        = 10,
            num_water      = 8,
            num_road       = 6,
            seed           = random.randint(0, 9999),
        )

        self.agent_pos    = list(self.maze.start)
        self.goal_pos     = list(self.maze.goal)
        self.pursuer_pos  = [list(p) for p in self.pursuer_starts]

        self.agent_status   = AgentStatus()
        self.pursuer_status = [PursuerStatus() for _ in self.pursuer_starts]

        # fog of war
        self.seen = set()
        self._update_fog(self.agent_pos)

        # search state
        self.lrta_h    = {}
        self.agent_path= []    # current planned path (for display)

        # stats
        self.steps      = 0
        self.wall_shifts= 0
        self.nodes_exp  = 0
        self.game_over  = False
        self.outcome    = ""
        self.particles  = []
        self.flash_msg  = None
        self.tick_timer = 0
        self.step_double= False   # for speed power-up

    # ── fog of war ───────────────────────────

    def _update_fog(self, pos):
        r, c = pos
        radius = ROWS if self.agent_status.full_reveal() else FOG_RADIUS
        for dr in range(-radius, radius+1):
            for dc in range(-radius, radius+1):
                if abs(dr)+abs(dc) <= radius:
                    nr, nc = r+dr, c+dc
                    if 0 <= nr < ROWS and 0 <= nc < COLS:
                        self.seen.add((nr, nc))

    def _visible(self, pos):
        if not self.fog_of_war or self.agent_status.full_reveal():
            return True
        r, c = pos
        ar, ac = self.agent_pos
        return abs(r-ar)+abs(c-ac) <= FOG_RADIUS

    # ── helpers ──────────────────────────────

    def _walkable(self, pos):
        r, c = pos
        return (0 <= r < ROWS and 0 <= c < COLS and
                self.grid[r][c] != WALL)

    def _neighbors(self, pos):
        r, c = pos
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            np = [r+dr, c+dc]
            if self._walkable(np):
                out.append(np)
        return out

    def _belief_neighbors(self, pos):
        """neighbors using belief map — unknown = optimistically open."""
        r, c = pos
        out = []
        for dr, dc in [(-1,0),(1,0),(0,-1),(0,1)]:
            nr, nc = r+dr, c+dc
            if not (0 <= nr < ROWS and 0 <= nc < COLS):
                continue
            if (nr, nc) not in self.seen:
                out.append([nr, nc])   # unknown = assume open
            elif self.grid[nr][nc] != WALL:
                out.append([nr, nc])
        return out

    def _manhattan(self, a, b):
        return abs(a[0]-b[0]) + abs(a[1]-b[1])

    # ── search algorithms ────────────────────

    def _astar(self, start, goal, use_belief=True):
        h    = lambda p: self._manhattan(p, goal)
        ctr  = [0]
        heap = [(h(start), 0, start, [start])]
        vis  = {}
        while heap:
            f, g, cur, path = heapq.heappop(heap)
            k = (cur[0], cur[1])
            if vis.get(k, 999) <= g: continue
            vis[k] = g
            if cur[0]==goal[0] and cur[1]==goal[1]:
                self.nodes_exp += len(vis)
                return path
            nb = self._belief_neighbors(cur) if use_belief else self._neighbors(cur)
            for n in nb:
                ctr[0] += 1
                heapq.heappush(heap, (g+1+h(n), g+1, n, path+[n]))
        return None

    def _lrta_step(self, pos):
        def hget(p):
            k = (p[0], p[1])
            if k not in self.lrta_h:
                self.lrta_h[k] = self._manhattan(p, self.goal_pos)
            return self.lrta_h[k]

        nb = self._belief_neighbors(pos)
        if not nb: return pos
        min_cost = min(1 + hget(n) for n in nb)
        self.lrta_h[(pos[0], pos[1])] = max(hget(pos), min_cost)
        return min(nb, key=lambda n: 1 + hget(n))

    def _minimax_step(self, pos, p_positions, depth=3, alpha_beta=False):
        def score(ap, pps, d, is_max, alpha, beta):
            if ap[0]==self.goal_pos[0] and ap[1]==self.goal_pos[1]:
                return 1000
            for pp in pps:
                if ap[0]==pp[0] and ap[1]==pp[1]: return -1000
            if d == 0:
                dist_to_goal = self._manhattan(ap, self.goal_pos)
                min_p_dist   = min(self._manhattan(ap, pp) for pp in pps)
                return -dist_to_goal * 10 + min_p_dist * 5
            if is_max:
                best = -math.inf
                for n in self._neighbors(ap):
                    v = score(n, pps, d-1, False, alpha, beta)
                    best = max(best, v)
                    if alpha_beta:
                        alpha = max(alpha, best)
                        if best >= beta: break
                return best if best > -math.inf else score(ap, pps, d-1, False, alpha, beta)
            else:
                best = math.inf
                # only move first pursuer for efficiency
                for n in self._neighbors(pps[0]):
                    new_pps = [n] + pps[1:]
                    v = score(ap, new_pps, d-1, True, alpha, beta)
                    best = min(best, v)
                    if alpha_beta:
                        beta = min(beta, best)
                        if best <= alpha: break
                return best if best < math.inf else score(ap, pps, d-1, True, alpha, beta)

        moves = self._neighbors(pos)
        if not moves: return pos
        best_move = max(moves, key=lambda n: score(
            n, p_positions, depth-1, False, -math.inf, math.inf))
        self.nodes_exp += len(moves) * (4 ** depth)
        return best_move

    def _pursuer_move(self, pursuer_pos, idx):
        strat = self.pursuer_strategy
        nb = self._neighbors(pursuer_pos)
        if not nb: return pursuer_pos
        if strat == "random":
            return random.choice(nb)
        elif strat == "greedy":
            return min(nb, key=lambda n: self._manhattan(n, self.agent_pos))
        elif strat == "astar":
            path = self._astar(pursuer_pos, self.agent_pos, use_belief=False)
            return path[1] if path and len(path) > 1 else nb[0]
        return nb[0]

    # ── game tick ────────────────────────────

    def tick(self):
        if self.game_over: return

        # dynamic wall shift
        if self.dynamic:
            protected = {tuple(self.agent_pos), tuple(self.goal_pos)}
            for pp in self.pursuer_pos:
                protected.add(tuple(pp))
            shifted = self.maze.step(protected_positions=protected)
            if shifted:
                self.wall_shifts += 1
                self.lrta_h = {}   # invalidate learned h-values
                self.agent_path = []

        # tick status effects
        self.agent_status.tick()
        for ps in self.pursuer_status:
            ps.tick()

        # agent move
        if not self.agent_status.is_frozen():
            self._do_agent_move()
            # speed power-up: move twice
            if self.agent_status.has_speed() and not self.game_over:
                self._do_agent_move()
        else:
            self._spawn_particles(self.agent_pos, C_BAD, 6)

        if self.game_over: return

        # pursuer moves
        for i, pp in enumerate(self.pursuer_pos):
            if self.pursuer_status[i].is_frozen():
                self._spawn_particles(pp, C_WATER, 4)
                continue
            new_pp = self._pursuer_move(pp, i)
            self.pursuer_pos[i] = new_pp
            if new_pp[0]==self.agent_pos[0] and new_pp[1]==self.agent_pos[1]:
                self.game_over = True
                self.outcome   = "CAUGHT"
                self._spawn_particles(self.agent_pos, C_BAD, 20)
                return

        self.steps += 1

    def _do_agent_move(self):
        """compute and apply one agent move."""
        algo = self.agent_algo
        pos  = self.agent_pos

        if algo == "manual":
            if self.manual_dir is None: return
            dr, dc = self.manual_dir
            self.manual_dir = None
            new_pos = [pos[0]+dr, pos[1]+dc]
        elif algo == "astar":
            if not self.agent_path or len(self.agent_path) < 2:
                self.agent_path = self._astar(pos, self.goal_pos) or []
            new_pos = self.agent_path[1] if len(self.agent_path) > 1 else pos
            if len(self.agent_path) > 1: self.agent_path.pop(0)
        elif algo == "lrta":
            new_pos = self._lrta_step(pos)
        elif algo == "minimax":
            new_pos = self._minimax_step(pos, self.pursuer_pos, depth=3, alpha_beta=False)
            self.agent_path = []
        elif algo == "alphabeta":
            new_pos = self._minimax_step(pos, self.pursuer_pos, depth=4, alpha_beta=True)
            self.agent_path = []
        else:
            new_pos = pos

        if not self._walkable(new_pos):
            new_pos = pos

        self.agent_pos = new_pos
        self._update_fog(new_pos)

        # check goal
        if new_pos[0]==self.goal_pos[0] and new_pos[1]==self.goal_pos[1]:
            self.agent_status.score += 500
            self.game_over = True
            self.outcome   = "GOAL!"
            self._spawn_particles(new_pos, C_GOAL, 25)
            return

        # check cell effects
        cell = self.grid[new_pos[0]][new_pos[1]]
        if cell == TRAP:
            hit = self.agent_status.apply_trap()
            if hit:
                self._flash("TRAP! Frozen 3 turns", C_BAD)
                self._spawn_particles(new_pos, C_TRAP, 12)
            else:
                self._flash("Shield blocked trap!", C_GOOD)
        elif cell == POWERUP:
            pu_type = self.powerup_map.get((new_pos[0], new_pos[1]), PU_SPEED)
            freeze_pursuer = self.agent_status.apply_powerup(pu_type)
            # remove power-up from grid
            self.grid[new_pos[0]][new_pos[1]] = OPEN
            self.powerup_map.pop((new_pos[0], new_pos[1]), None)
            if freeze_pursuer:
                for ps in self.pursuer_status:
                    ps.freeze(5)
                self._flash("FREEZE! Pursuers frozen 5 turns", C_WATER)
            else:
                self._flash(f"Power-up: {pu_type.upper()}!", C_POWERUP.get(pu_type, C_ACCENT))
            self._spawn_particles(new_pos, C_POWERUP.get(pu_type, C_ACCENT), 15)

        # check pursuer collision after agent moves
        for pp in self.pursuer_pos:
            if new_pos[0]==pp[0] and new_pos[1]==pp[1]:
                self.game_over = True
                self.outcome   = "CAUGHT"
                self._spawn_particles(new_pos, C_BAD, 20)
                return

    # ── particle system ──────────────────────

    def _spawn_particles(self, pos, color, count):
        cx = pos[1]*CELL_SIZE + CELL_SIZE//2
        cy = pos[0]*CELL_SIZE + CELL_SIZE//2
        for _ in range(count):
            angle = random.uniform(0, 2*math.pi)
            speed = random.uniform(1, 4)
            self.particles.append({
                "x": cx, "y": cy,
                "vx": math.cos(angle)*speed,
                "vy": math.sin(angle)*speed,
                "life": random.randint(15, 35),
                "color": color,
                "size": random.randint(2, 5),
            })

    def _update_particles(self):
        alive = []
        for p in self.particles:
            p["x"]   += p["vx"]
            p["y"]   += p["vy"]
            p["vy"]  += 0.15   # gravity
            p["life"] -= 1
            if p["life"] > 0:
                alive.append(p)
        self.particles = alive

    def _flash(self, msg, color):
        self.flash_msg   = msg
        self.flash_color = color
        self.flash_timer = 90

    # ── drawing ──────────────────────────────

    def draw(self):
        self.screen.fill(C_BG)
        self._draw_maze()
        self._draw_particles()
        self._draw_agents()
        self._draw_panel()
        self._draw_bottom_bar()
        self._draw_overlay()
        pygame.display.flip()

    def _draw_maze(self):
        for r in range(ROWS):
            for c in range(COLS):
                x = c * CELL_SIZE
                y = r * CELL_SIZE
                pos = (r, c)
                cell = self.grid[r][c]

                in_seen    = pos in self.seen
                in_visible = self._visible(pos)

                # fog of war
                if self.fog_of_war and not in_seen:
                    pygame.draw.rect(self.screen, C_FOG, (x, y, CELL_SIZE, CELL_SIZE))
                    continue

                # base cell color
                if cell == WALL:
                    col = C_WALL
                elif cell == MUD:
                    col = C_MUD
                elif cell == WATER:
                    col = C_WATER
                elif cell == ROAD:
                    col = C_ROAD
                elif cell == TRAP:
                    col = C_TRAP
                elif cell == POWERUP:
                    pu = self.powerup_map.get(pos, PU_SPEED)
                    col = C_POWERUP.get(pu, (130, 60, 200))
                else:
                    col = C_OPEN

                # dim cells not in current visibility radius
                if self.fog_of_war and not in_visible:
                    col = tuple(int(v * 0.35) for v in col)

                pygame.draw.rect(self.screen, col, (x, y, CELL_SIZE, CELL_SIZE))

                # grid line
                pygame.draw.rect(self.screen, (col[0]//3, col[1]//3, col[2]//3),
                                 (x, y, CELL_SIZE, CELL_SIZE), 1)

                # cell label
                if cell in (TRAP, MUD, WATER, ROAD) and in_visible:
                    lbl = CELL_LABELS.get(cell, "")
                    s = self.font_sm.render(lbl, True, (200, 200, 200))
                    self.screen.blit(s, (x+CELL_SIZE//2-s.get_width()//2,
                                         y+CELL_SIZE//2-s.get_height()//2))
                elif cell == POWERUP and in_visible:
                    pu  = self.powerup_map.get(pos, PU_SPEED)
                    lbl = "PU"
                    s   = self.font_sm.render(lbl, True, (240, 240, 240))
                    self.screen.blit(s, (x+CELL_SIZE//2-s.get_width()//2,
                                         y+CELL_SIZE//2-s.get_height()//2))

        # draw start and goal
        self._draw_cell_icon(self.maze.start, "S", C_START)
        self._draw_cell_icon(self.maze.goal,  "G", C_GOAL)

        # planned path highlight
        if self.agent_path and len(self.agent_path) > 1:
            for p in self.agent_path[1:]:
                px = p[1]*CELL_SIZE + CELL_SIZE//2
                py = p[0]*CELL_SIZE + CELL_SIZE//2
                surf = pygame.Surface((8, 8), pygame.SRCALPHA)
                pygame.draw.circle(surf, (60, 180, 255, 80), (4, 4), 4)
                self.screen.blit(surf, (px-4, py-4))

    def _draw_cell_icon(self, pos, label, color):
        r, c = pos
        x = c*CELL_SIZE; y = r*CELL_SIZE
        pygame.draw.rect(self.screen, color,
                         (x+3, y+3, CELL_SIZE-6, CELL_SIZE-6), border_radius=4)
        s = self.font_md.render(label, True, (255, 255, 255))
        self.screen.blit(s, (x+CELL_SIZE//2-s.get_width()//2,
                              y+CELL_SIZE//2-s.get_height()//2))

    def _draw_agents(self):
        # draw pursuers
        for i, pp in enumerate(self.pursuer_pos):
            if not self._visible(pp) and self.fog_of_war and not self.agent_status.full_reveal():
                continue
            col = C_PURSUER[i % len(C_PURSUER)]
            if self.pursuer_status[i].is_frozen():
                col = C_WATER
            x = pp[1]*CELL_SIZE + CELL_SIZE//2
            y = pp[0]*CELL_SIZE + CELL_SIZE//2
            pygame.draw.circle(self.screen, col, (x, y), CELL_SIZE//2 - 3)
            pygame.draw.circle(self.screen, (255, 255, 255), (x, y), CELL_SIZE//2 - 3, 2)
            s = self.font_md.render("P", True, (255, 255, 255))
            self.screen.blit(s, (x-s.get_width()//2, y-s.get_height()//2))

        # draw agent
        ap = self.agent_pos
        x  = ap[1]*CELL_SIZE + CELL_SIZE//2
        y  = ap[0]*CELL_SIZE + CELL_SIZE//2
        col = C_AGENT
        if self.agent_status.is_frozen():
            col = (160, 160, 255)
        elif self.agent_status.has_speed():
            col = (255, 230, 50)
        elif self.agent_status.shield_turns > 0:
            col = (50, 230, 180)

        # pulsing glow ring
        t   = pygame.time.get_ticks()
        rad = CELL_SIZE//2 - 2 + int(2 * math.sin(t / 200))
        pygame.draw.circle(self.screen, tuple(int(v*0.4) for v in col), (x, y), rad+4)
        pygame.draw.circle(self.screen, col, (x, y), rad)
        pygame.draw.circle(self.screen, (255, 255, 255), (x, y), rad, 2)
        s = self.font_md.render("A", True, (10, 12, 28))
        self.screen.blit(s, (x-s.get_width()//2, y-s.get_height()//2))

    def _draw_particles(self):
        self._update_particles()
        for p in self.particles:
            alpha = int(255 * p["life"] / 35)
            col   = tuple(min(255, v) for v in p["color"][:3])
            pygame.draw.circle(self.screen, col,
                                (int(p["x"]), int(p["y"])), p["size"])

    def _draw_panel(self):
        px = MAZE_W
        pygame.draw.rect(self.screen, C_PANEL_BG, (px, 0, PANEL_W, MAZE_H))
        pygame.draw.line(self.screen, C_WALL, (px, 0), (px, MAZE_H), 1)

        y = 16
        def txt(text, color=C_TEXT, font=None, indent=12):
            nonlocal y
            f = font or self.font_md
            s = f.render(text, True, color)
            self.screen.blit(s, (px + indent, y))
            y += s.get_height() + 4

        def divider():
            nonlocal y
            pygame.draw.line(self.screen, C_WALL,
                             (px+8, y+2), (px+PANEL_W-8, y+2), 1)
            y += 10

        txt("MAZE NAVIGATOR", C_ACCENT, self.font_lg)
        divider()

        txt(f"algorithm : {ALGO_NAMES.get(self.agent_algo, self.agent_algo)}", C_TEXT_DIM)
        txt(f"pursuer   : {PURSUER_NAMES.get(self.pursuer_strategy, '')} x{self.num_pursuers}", C_TEXT_DIM)
        txt(f"dynamic   : {'ON' if self.dynamic else 'OFF'}", C_TEXT_DIM)
        txt(f"fog       : {'ON' if self.fog_of_war else 'OFF'}", C_TEXT_DIM)
        divider()

        txt("STATS", C_ACCENT, self.font_lg)
        txt(f"steps      : {self.steps}")
        txt(f"explored   : {len(self.seen)}/{ROWS*COLS} ({int(len(self.seen)*100/(ROWS*COLS))}%)")
        txt(f"wall shifts: {self.wall_shifts}")
        txt(f"nodes exp  : {self.nodes_exp}")
        txt(f"score      : {self.agent_status.score}", C_GOOD)
        divider()

        txt("ACTIVE EFFECTS", C_ACCENT, self.font_lg)
        effects = self.agent_status.active_effects()
        if effects:
            for e in effects:
                col = C_BAD if "FROZEN" in e else C_WARN if "SPEED" in e else C_GOOD
                txt(f"  {e}", col)
        else:
            txt("  none", C_TEXT_DIM)

        divider()

        txt("CONTROLS", C_ACCENT, self.font_lg)
        for line in [
            "SPACE  pause/resume",
            "R      new maze",
            "1      A* agent",
            "2      LRTA* agent",
            "3      Minimax",
            "4      Alpha-Beta",
            "5      Manual",
            "+/-    speed",
            "Q/ESC  quit",
        ]:
            txt(line, C_TEXT_DIM, self.font_sm)

    def _draw_bottom_bar(self):
        y = MAZE_H
        pygame.draw.rect(self.screen, C_PANEL_BG, (0, y, WIN_W, 60))
        pygame.draw.line(self.screen, C_WALL, (0, y), (WIN_W, y), 1)

        speed_txt = f"SPEED: {self.speed}x"
        algo_txt  = f"[{self.agent_algo.upper()}]  {ALGO_NAMES.get(self.agent_algo, '')} vs {self.pursuer_strategy} pursuer"
        status_txt= "PAUSED" if self.paused else ("GAME OVER — "+self.outcome if self.game_over else "RUNNING")
        status_col= C_WARN if self.paused else (C_BAD if self.game_over and self.outcome=="CAUGHT" else C_GOOD if self.game_over else C_ACCENT)

        s1 = self.font_md.render(algo_txt, True, C_TEXT_DIM)
        s2 = self.font_lg.render(status_txt, True, status_col)
        s3 = self.font_md.render(speed_txt, True, C_TEXT_DIM)

        self.screen.blit(s1, (12, y+8))
        self.screen.blit(s2, (12, y+28))
        self.screen.blit(s3, (WIN_W-PANEL_W-100, y+8))

        # flash message
        if self.flash_timer > 0:
            alpha = min(255, self.flash_timer * 5)
            fs = self.font_lg.render(self.flash_msg, True, self.flash_color)
            self.screen.blit(fs, (MAZE_W//2 - fs.get_width()//2, y+15))
            self.flash_timer -= 1

    def _draw_overlay(self):
        if not self.game_over: return
        surf = pygame.Surface((MAZE_W, MAZE_H), pygame.SRCALPHA)
        surf.fill((0, 0, 0, 150))
        self.screen.blit(surf, (0, 0))

        if self.outcome == "GOAL!":
            msg   = "YOU REACHED THE GOAL!"
            color = C_GOAL
        else:
            msg   = "AGENT WAS CAUGHT!"
            color = C_BAD

        s1 = self.font_xl.render(msg, True, color)
        s2 = self.font_md.render(f"steps: {self.steps}  |  score: {self.agent_status.score}  |  press R to reset", True, C_TEXT)
        self.screen.blit(s1, (MAZE_W//2 - s1.get_width()//2, MAZE_H//2 - 30))
        self.screen.blit(s2, (MAZE_W//2 - s2.get_width()//2, MAZE_H//2 + 10))

    # ── main loop ────────────────────────────

    def run(self):
        FPS      = 60
        tick_acc = 0

        while True:
            dt = self.clock.tick(FPS)

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    pygame.quit(); sys.exit()

                if event.type == pygame.KEYDOWN:
                    if event.key in (pygame.K_q, pygame.K_ESCAPE):
                        pygame.quit(); sys.exit()
                    elif event.key == pygame.K_SPACE:
                        self.paused = not self.paused
                    elif event.key == pygame.K_r:
                        self.reset()
                    elif event.key == pygame.K_1:
                        self.agent_algo = "astar";     self.agent_path=[]; self.lrta_h={}
                    elif event.key == pygame.K_2:
                        self.agent_algo = "lrta";      self.agent_path=[]; self.lrta_h={}
                    elif event.key == pygame.K_3:
                        self.agent_algo = "minimax";   self.agent_path=[]
                    elif event.key == pygame.K_4:
                        self.agent_algo = "alphabeta"; self.agent_path=[]
                    elif event.key == pygame.K_5:
                        self.agent_algo = "manual";    self.agent_path=[]
                    elif event.key == pygame.K_EQUALS or event.key == pygame.K_PLUS:
                        self.speed = min(20, self.speed + 1)
                    elif event.key == pygame.K_MINUS:
                        self.speed = max(1, self.speed - 1)
                    elif event.key == pygame.K_d:
                        self.dynamic = not self.dynamic; self.reset()
                    elif event.key == pygame.K_f:
                        self.fog_of_war = not self.fog_of_war
                    # manual controls
                    elif event.key == pygame.K_UP:    self.manual_dir = (-1,  0)
                    elif event.key == pygame.K_DOWN:  self.manual_dir = ( 1,  0)
                    elif event.key == pygame.K_LEFT:  self.manual_dir = ( 0, -1)
                    elif event.key == pygame.K_RIGHT: self.manual_dir = ( 0,  1)

            # game tick at controlled speed
            if not self.paused and not self.game_over:
                tick_acc += dt
                ms_per_tick = 1000 // max(1, self.speed)
                while tick_acc >= ms_per_tick:
                    self.tick()
                    tick_acc -= ms_per_tick
                    if self.game_over: break

            self.draw()


# ─────────────────────────────────────────────
#  entry point
# ─────────────────────────────────────────────

def launch(agent_algo="astar", pursuer_strategy="greedy",
           num_pursuers=2, dynamic=True, fog=True):
    game = VisualGame(
        agent_algo        = agent_algo,
        pursuer_strategy  = pursuer_strategy,
        num_pursuers      = num_pursuers,
        dynamic           = dynamic,
        fog_of_war        = fog,
    )
    game.run()


if __name__ == "__main__":
    launch()