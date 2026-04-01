"""
Celestial Billiard — A 2D billiard game with Newtonian gravity between balls.
Controls: Mouse drag to aim/shoot, Arrow keys for spin, F12 for screenshot, R to rerack.
"""

import os
import math
import time
import datetime
from enum import IntEnum, Enum
from dataclasses import dataclass, field

import numpy as np
import pygame

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.backends.backend_agg as agg


# ============================================================
# SECTION 1: IMPORTS AND CONFIG
# ============================================================

class CFG:
    # --- Window ---
    WINDOW_W       = 1400
    WINDOW_H       = 800
    FPS_CAP        = 60
    PHYSICS_HZ     = 120
    DT             = 1.0 / PHYSICS_HZ

    # --- Table (meters, 9-foot) ---
    TABLE_W        = 2.54
    TABLE_H        = 1.27
    POCKET_R       = 0.057
    RAIL_W         = 0.05

    # --- Play area layout (pixels) ---
    PLAY_AREA_W    = 980
    PLAY_AREA_H    = int(PLAY_AREA_W * (1.27 / 2.54))  # maintain 2:1
    PPM            = PLAY_AREA_W / 2.54  # pixels per meter

    # --- Ball ---
    BALL_R_M       = 0.028575
    BALL_R_PX      = int(BALL_R_M * PPM)
    BALL_MASS      = 0.17

    # --- Friction (slider-tunable) ---
    MU_SLIDE       = 0.20
    MU_ROLL        = 0.06
    MU_SPIN        = 0.04
    G_ACCEL        = 9.81

    # --- Restitution ---
    COR_BALL       = 0.95
    COR_RAIL       = 0.75

    # --- Gravity between balls ---
    G_GAME         = 5e-3
    SOFTENING_EPS  = BALL_R_M * 2.0

    # --- Speed threshold ---
    V_THRESHOLD    = 0.005

    # --- Colors ---
    COLOR_FELT     = (0, 100, 60)
    COLOR_RAIL     = (101, 67, 33)
    COLOR_BG       = (10, 5, 30)

    BALL_COLORS = [
        (240, 240, 240),  # 0 cue (white)
        (255, 220, 0),    # 1 yellow
        (0, 70, 200),     # 2 blue
        (200, 30, 30),    # 3 red
        (130, 0, 180),    # 4 purple
        (220, 100, 20),   # 5 orange
        (0, 150, 50),     # 6 green
        (130, 30, 10),    # 7 maroon
        (20, 20, 20),     # 8 black
        (255, 220, 0),    # 9 yellow stripe
        (0, 70, 200),     # 10 blue stripe
        (200, 30, 30),    # 11 red stripe
        (130, 0, 180),    # 12 purple stripe
        (220, 100, 20),   # 13 orange stripe
        (0, 150, 50),     # 14 green stripe
        (130, 30, 10),    # 15 maroon stripe
    ]

    # --- Head string (line behind which cue ball is placed) ---
    HEAD_STRING_X  = TABLE_W * 0.25  # 1/4 from head rail

    # --- Cue ---
    MAX_POWER_M    = 3.5      # realistic: soft=0.5, medium=1.5, fast=3.0, break=4+
    CUE_DRAG_PX    = 300

    # --- UI Layout ---
    UI_X           = PLAY_AREA_W + 20
    SLIDER_H       = 20
    INSET_SIZE     = 120


# ============================================================
# SECTION 2: VECTOR MATH UTILITIES
# ============================================================

def vec(x, y):
    return np.array([x, y], dtype=float)

def length(v):
    return np.linalg.norm(v)

def normalize(v):
    n = length(v)
    return v / n if n > 1e-12 else vec(0.0, 0.0)

def dot(a, b):
    return float(np.dot(a, b))

def cross2d(a, b):
    return float(a[0] * b[1] - a[1] * b[0])

def rotate90(v):
    return vec(-v[1], v[0])

def clamp(x, lo, hi):
    return max(lo, min(hi, x))


# ============================================================
# SECTION 3: BALL CLASS
# ============================================================

class CueBallGravity(IntEnum):
    FULL       = 0   # cue ball participates fully (attracts and is attracted)
    SOURCE     = 1   # cue ball attracts others but is not itself attracted
    NONE       = 2   # cue ball neither attracts nor is attracted

CUE_GRAVITY_LABELS = {
    CueBallGravity.FULL:   "Cue grav: FULL (mutual)",
    CueBallGravity.SOURCE: "Cue grav: SOURCE only",
    CueBallGravity.NONE:   "Cue grav: NONE",
}


class BallState(IntEnum):
    STATIONARY = 0
    SPINNING   = 1
    ROLLING    = 2
    SLIDING    = 3
    POCKETED   = 4


@dataclass
class Ball:
    id: int
    pos: np.ndarray
    vel: np.ndarray = field(default_factory=lambda: vec(0.0, 0.0))
    omega: np.ndarray = field(default_factory=lambda: np.zeros(3))
    mass: float = CFG.BALL_MASS
    radius: float = CFG.BALL_R_M
    state: int = BallState.STATIONARY

    @property
    def speed(self):
        return length(self.vel)

    @property
    def is_active(self):
        return self.state not in (BallState.STATIONARY, BallState.POCKETED)

    @property
    def color(self):
        return CFG.BALL_COLORS[self.id]

    @property
    def is_stripe(self):
        return 9 <= self.id <= 15

    def slip_velocity(self):
        """Slip velocity at contact point with cloth."""
        return self.vel - self.radius * vec(self.omega[1], -self.omega[0])

    def update_state(self):
        spd = length(self.vel)
        omega_mag = abs(self.omega[2])
        if self.state == BallState.POCKETED:
            return
        if spd < CFG.V_THRESHOLD and omega_mag < CFG.V_THRESHOLD:
            self.vel[:] = 0.0
            self.omega[:] = 0.0
            self.state = BallState.STATIONARY
        elif spd < CFG.V_THRESHOLD:
            self.vel[:] = 0.0
            self.state = BallState.SPINNING
        elif length(self.slip_velocity()) < CFG.V_THRESHOLD * 0.1:
            self.state = BallState.ROLLING
        else:
            self.state = BallState.SLIDING


# ============================================================
# SECTION 4: TABLE CLASS
# ============================================================

class Table:
    def __init__(self):
        self.w = CFG.TABLE_W
        self.h = CFG.TABLE_H
        self.pocket_r = CFG.POCKET_R
        self.pockets = [
            vec(0, 0),
            vec(self.w / 2, 0),
            vec(self.w, 0),
            vec(0, self.h),
            vec(self.w / 2, self.h),
            vec(self.w, self.h),
        ]

    def check_pockets(self, ball):
        if ball.state == BallState.POCKETED:
            return True
        for p in self.pockets:
            if length(ball.pos - p) < self.pocket_r:
                ball.state = BallState.POCKETED
                ball.vel[:] = 0.0
                ball.omega[:] = 0.0
                return True
        return False

    def clamp_to_play_surface(self, ball):
        r = ball.radius
        bounced = False
        if ball.pos[0] - r < 0:
            ball.pos[0] = r
            ball.vel[0] = abs(ball.vel[0]) * CFG.COR_RAIL
            ball.omega[2] *= 0.5
            bounced = True
        if ball.pos[0] + r > self.w:
            ball.pos[0] = self.w - r
            ball.vel[0] = -abs(ball.vel[0]) * CFG.COR_RAIL
            ball.omega[2] *= 0.5
            bounced = True
        if ball.pos[1] - r < 0:
            ball.pos[1] = r
            ball.vel[1] = abs(ball.vel[1]) * CFG.COR_RAIL
            ball.omega[2] *= 0.5
            bounced = True
        if ball.pos[1] + r > self.h:
            ball.pos[1] = self.h - r
            ball.vel[1] = -abs(ball.vel[1]) * CFG.COR_RAIL
            ball.omega[2] *= 0.5
            bounced = True
        return bounced


# ============================================================
# SECTION 5: PHYSICS ENGINE
# ============================================================

class PhysicsEngine:
    def __init__(self, balls, table, cue_gravity_mode=CueBallGravity.FULL):
        self.balls = balls
        self.table = table
        self.cue_gravity_mode = cue_gravity_mode
        self.energy_log = []
        self.sim_time = 0.0

    def _derivatives(self, pos_vel, active_balls):
        N = len(active_balls)
        derivs = np.zeros((N, 4))
        derivs[:, 0] = pos_vel[:, 2]
        derivs[:, 1] = pos_vel[:, 3]

        forces = np.zeros((N, 2))
        self._add_gravity_forces(pos_vel, active_balls, forces)

        for i, ball in enumerate(active_balls):
            vel_now = pos_vel[i, 2:4]
            f_fric = self._friction_force(ball, vel_now)
            forces[i] += f_fric

        for i, ball in enumerate(active_balls):
            derivs[i, 2] = forces[i, 0] / ball.mass
            derivs[i, 3] = forces[i, 1] / ball.mass

        return derivs

    def _add_gravity_forces(self, pos_vel, active_balls, forces):
        eps2 = CFG.SOFTENING_EPS ** 2
        G = CFG.G_GAME
        mode = self.cue_gravity_mode

        active_id_to_idx = {b.id: i for i, b in enumerate(active_balls)}

        for i, bi in enumerate(active_balls):
            # In SOURCE / NONE mode the cue ball is not attracted by anything
            if bi.id == 0 and mode != CueBallGravity.FULL:
                continue

            pi = pos_vel[i, :2]
            for bj in self.balls:
                if bj.id == bi.id or bj.state == BallState.POCKETED:
                    continue
                # In NONE mode the cue ball does not attract other balls either
                if bj.id == 0 and mode == CueBallGravity.NONE:
                    continue

                j = active_id_to_idx.get(bj.id)
                if j is not None:
                    pj = pos_vel[j, :2]
                else:
                    pj = bj.pos
                r_vec = pj - pi
                r2 = dot(r_vec, r_vec) + eps2
                r3 = r2 ** 1.5
                forces[i] += G * bi.mass * bj.mass / r3 * r_vec

    def _friction_force(self, ball, vel_now):
        v_mag = length(vel_now)
        if v_mag < 1e-9:
            return vec(0.0, 0.0)
        slip = ball.slip_velocity()
        slip_mag = length(slip)
        if slip_mag > CFG.V_THRESHOLD * 0.1:
            return -CFG.MU_SLIDE * ball.mass * CFG.G_ACCEL * normalize(slip)
        else:
            return -CFG.MU_ROLL * ball.mass * CFG.G_ACCEL * normalize(vel_now)

    def _rk4_step(self, active_balls, dt):
        N = len(active_balls)
        if N == 0:
            return
        y = np.array([[b.pos[0], b.pos[1], b.vel[0], b.vel[1]] for b in active_balls])

        k1 = self._derivatives(y, active_balls)
        k2 = self._derivatives(y + 0.5 * dt * k1, active_balls)
        k3 = self._derivatives(y + 0.5 * dt * k2, active_balls)
        k4 = self._derivatives(y + dt * k3, active_balls)

        y_new = y + (dt / 6.0) * (k1 + 2 * k2 + 2 * k3 + k4)

        for i, ball in enumerate(active_balls):
            ball.pos[:] = y_new[i, :2]
            ball.vel[:] = y_new[i, 2:4]

    def _evolve_spin(self, ball, dt):
        slip = ball.slip_velocity()
        slip_mag = length(slip)

        if slip_mag > CFG.V_THRESHOLD * 0.1:
            slip_hat = normalize(slip)
            fric_accel = -CFG.MU_SLIDE * CFG.G_ACCEL * slip_hat
            I_factor = 2.0 / 5.0 * ball.radius
            alpha_x = fric_accel[1] / I_factor
            alpha_y = -fric_accel[0] / I_factor
            ball.omega[0] += alpha_x * dt
            ball.omega[1] += alpha_y * dt
        else:
            ball.omega[0] = ball.vel[1] / ball.radius
            ball.omega[1] = -ball.vel[0] / ball.radius

        oz = ball.omega[2]
        if abs(oz) > 1e-9:
            decay = (5 * CFG.MU_SPIN * CFG.G_ACCEL) / (2 * ball.radius) * dt
            ball.omega[2] = oz - np.sign(oz) * min(decay, abs(oz))

    def _resolve_ball_collisions(self):
        balls = [b for b in self.balls if b.state != BallState.POCKETED]
        n = len(balls)
        for _ in range(2):  # 2 passes for multi-ball
            for i in range(n):
                for j in range(i + 1, n):
                    bi, bj = balls[i], balls[j]
                    delta = bj.pos - bi.pos
                    dist = length(delta)
                    min_dist = bi.radius + bj.radius
                    if dist < min_dist and dist > 1e-9:
                        normal = delta / dist
                        overlap = min_dist - dist
                        bi.pos -= normal * overlap * 0.5
                        bj.pos += normal * overlap * 0.5

                        rel_vel = bj.vel - bi.vel
                        v_normal = dot(rel_vel, normal)
                        if v_normal >= 0:
                            continue

                        e = CFG.COR_BALL
                        mi, mj = bi.mass, bj.mass
                        impulse_scalar = -(1 + e) * v_normal / (1.0 / mi + 1.0 / mj)
                        impulse = impulse_scalar * normal

                        bi.vel -= impulse / mi
                        bj.vel += impulse / mj

                        if bi.state == BallState.STATIONARY:
                            bi.state = BallState.SLIDING
                        if bj.state == BallState.STATIONARY:
                            bj.state = BallState.SLIDING

    def step(self, dt=None):
        if dt is None:
            dt = CFG.DT
        # Only moving balls participate in RK4 integration.
        # Stationary balls still exert gravity on moving ones (handled in _add_gravity_forces)
        # but are not themselves moved by gravity — they only start moving when hit.
        active = [b for b in self.balls
                  if b.state not in (BallState.STATIONARY, BallState.POCKETED)]

        self._rk4_step(active, dt)

        for b in active:
            self._evolve_spin(b, dt)
            self.table.check_pockets(b)
            self.table.clamp_to_play_surface(b)
            b.update_state()

        self._resolve_ball_collisions()
        self.sim_time += dt
        self._log_energy()

    def _log_energy(self):
        active = [b for b in self.balls if b.state != BallState.POCKETED]
        ke = sum(0.5 * b.mass * b.speed ** 2 for b in active)
        pe = 0.0
        mode = self.cue_gravity_mode
        for i in range(len(active)):
            for j in range(i + 1, len(active)):
                bi, bj = active[i], active[j]
                # Skip pairs that don't interact gravitationally
                if mode == CueBallGravity.NONE and (bi.id == 0 or bj.id == 0):
                    continue
                if mode == CueBallGravity.SOURCE and (bi.id == 0 or bj.id == 0):
                    # One-directional: PE is ill-defined; skip cue-ball pairs
                    continue
                r = length(bi.pos - bj.pos)
                r_soft = math.sqrt(r * r + CFG.SOFTENING_EPS ** 2)
                pe -= CFG.G_GAME * bi.mass * bj.mass / r_soft
        self.energy_log.append((self.sim_time, ke, pe, ke + pe))

    def cue_ball_moving(self):
        cue = next((b for b in self.balls if b.id == 0), None)
        if cue is None or cue.state == BallState.POCKETED:
            return False
        return cue.speed > CFG.V_THRESHOLD


# ============================================================
# SECTION 6: CUE CLASS
# ============================================================

class Cue:
    def __init__(self):
        self.active = False
        self.start_pos_px = None
        self.current_pos_px = None
        self.contact_a = 0.0  # horizontal offset (english)
        self.contact_b = 0.0  # vertical offset (topspin/backspin)

    def begin_drag(self, mouse_px):
        self.active = True
        self.start_pos_px = np.array(mouse_px, dtype=float)
        self.current_pos_px = np.array(mouse_px, dtype=float)

    def update_drag(self, mouse_px):
        self.current_pos_px = np.array(mouse_px, dtype=float)

    def adjust_contact(self, da, db):
        self.contact_a = clamp(self.contact_a + da, -0.95, 0.95)
        self.contact_b = clamp(self.contact_b + db, -0.95, 0.95)
        mag = math.sqrt(self.contact_a ** 2 + self.contact_b ** 2)
        if mag > 0.95:
            scale = 0.95 / mag
            self.contact_a *= scale
            self.contact_b *= scale

    def release(self, cue_ball):
        if not self.active:
            return False
        drag = self.current_pos_px - self.start_pos_px
        drag_len = length(drag)
        if drag_len < 5:
            self.active = False
            return False

        shot_dir = -normalize(drag)
        power_frac = min(drag_len / CFG.CUE_DRAG_PX, 1.0)
        V = power_frac * CFG.MAX_POWER_M

        cue_ball.vel = shot_dir * V

        a, b = self.contact_a, self.contact_b
        R = cue_ball.radius
        # Spin from contact offset
        # Topspin/backspin: b > 0 means hit above center → topspin
        cue_ball.omega[0] = (5.0 / 2.0) * (b / R) * (V / R) * shot_dir[1]
        cue_ball.omega[1] = -(5.0 / 2.0) * (b / R) * (V / R) * shot_dir[0]
        # Sidespin/english
        cue_ball.omega[2] = (5.0 / 2.0) * (a / R) * (V / R)

        cue_ball.state = BallState.SLIDING
        self.active = False
        return True

    def get_power_fraction(self):
        if not self.active or self.start_pos_px is None:
            return 0.0
        return min(length(self.current_pos_px - self.start_pos_px) / CFG.CUE_DRAG_PX, 1.0)

    def get_shot_direction(self):
        if not self.active or self.start_pos_px is None:
            return vec(1, 0)
        drag = self.current_pos_px - self.start_pos_px
        return -normalize(drag) if length(drag) > 1 else vec(1, 0)


# ============================================================
# SECTION 7: UI CLASS (sliders, inset, energy plot)
# ============================================================

class Slider:
    def __init__(self, x, y, w, label, min_val, max_val, initial, target_attr, tooltip=""):
        self.rect = pygame.Rect(x, y, w, CFG.SLIDER_H)
        self.label = label
        self.min_val = min_val
        self.max_val = max_val
        self.value = initial
        self.target = target_attr
        self.tooltip = tooltip
        self.dragging = False

    def handle_event(self, event):
        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.rect.collidepoint(event.pos):
                self.dragging = True
                self._update_value(event.pos[0])
        elif event.type == pygame.MOUSEBUTTONUP:
            self.dragging = False
        elif event.type == pygame.MOUSEMOTION and self.dragging:
            self._update_value(event.pos[0])

    def _update_value(self, mouse_x):
        rx = clamp(mouse_x - self.rect.x, 0, self.rect.w)
        self.value = self.min_val + (rx / self.rect.w) * (self.max_val - self.min_val)
        setattr(CFG, self.target, self.value)

    def draw(self, surface, font):
        pygame.draw.rect(surface, (60, 60, 60), self.rect, border_radius=4)
        fill_w = int((self.value - self.min_val) / max(self.max_val - self.min_val, 1e-12) * self.rect.w)
        fill_rect = pygame.Rect(self.rect.x, self.rect.y, fill_w, self.rect.h)
        pygame.draw.rect(surface, (0, 160, 100), fill_rect, border_radius=4)
        hx = self.rect.x + fill_w
        pygame.draw.circle(surface, (220, 220, 220), (hx, self.rect.y + CFG.SLIDER_H // 2), 7)
        txt = font.render(f"{self.label}: {self.value:.4f}", True, (200, 200, 200))
        surface.blit(txt, (self.rect.x, self.rect.y - 15))
        # Tooltip on hover over label or slider area
        if self.tooltip:
            hover_rect = pygame.Rect(self.rect.x, self.rect.y - 15, self.rect.w, self.rect.h + 15)
            mx, my = pygame.mouse.get_pos()
            if hover_rect.collidepoint(mx, my):
                tip_surf = font.render(self.tooltip, True, (255, 255, 200))
                tip_bg = pygame.Rect(mx + 10, my - 14, tip_surf.get_width() + 8, tip_surf.get_height() + 4)
                # Keep tooltip on screen
                if tip_bg.right > CFG.WINDOW_W:
                    tip_bg.right = CFG.WINDOW_W - 4
                pygame.draw.rect(surface, (40, 40, 50), tip_bg, border_radius=3)
                pygame.draw.rect(surface, (100, 100, 120), tip_bg, 1, border_radius=3)
                surface.blit(tip_surf, (tip_bg.x + 4, tip_bg.y + 2))


class InsetView:
    def draw(self, surface, cue, rect):
        pygame.draw.rect(surface, (25, 25, 35), rect)
        cx, cy = rect.centerx, rect.centery
        r = rect.width // 2 - 10
        # Ball
        pygame.draw.circle(surface, (230, 230, 230), (cx, cy), r)
        pygame.draw.circle(surface, (150, 150, 150), (cx, cy), r, 2)
        # Crosshair
        pygame.draw.line(surface, (100, 100, 100), (cx - r, cy), (cx + r, cy), 1)
        pygame.draw.line(surface, (100, 100, 100), (cx, cy - r), (cx, cy + r), 1)
        # Contact point (red dot)
        dot_x = int(cx + cue.contact_a * r * 0.85)
        dot_y = int(cy - cue.contact_b * r * 0.85)
        pygame.draw.circle(surface, (255, 40, 40), (dot_x, dot_y), 5)
        # Arrow hints
        c = (180, 180, 60)
        # Up
        pygame.draw.polygon(surface, c, [
            (rect.centerx, rect.top + 3),
            (rect.centerx - 5, rect.top + 11),
            (rect.centerx + 5, rect.top + 11)])
        # Down
        pygame.draw.polygon(surface, c, [
            (rect.centerx, rect.bottom - 3),
            (rect.centerx - 5, rect.bottom - 11),
            (rect.centerx + 5, rect.bottom - 11)])
        # Left
        pygame.draw.polygon(surface, c, [
            (rect.left + 3, rect.centery),
            (rect.left + 11, rect.centery - 5),
            (rect.left + 11, rect.centery + 5)])
        # Right
        pygame.draw.polygon(surface, c, [
            (rect.right - 3, rect.centery),
            (rect.right - 11, rect.centery - 5),
            (rect.right - 11, rect.centery + 5)])
        # Border
        pygame.draw.rect(surface, (100, 100, 100), rect, 2)


class EnergyPlot:
    def __init__(self, w=380, h=200):
        self.w, self.h = w, h
        self.fig, self.ax = plt.subplots(figsize=(w / 100, h / 100), dpi=100)
        self.fig.patch.set_facecolor('#0a0520')
        self.ax.set_facecolor('#0a0520')
        self.canvas = agg.FigureCanvasAgg(self.fig)
        self.surface = None

    def update(self, energy_log):
        if len(energy_log) < 2:
            return
        data = energy_log[-500:]
        times = [d[0] for d in data]
        KEs = [d[1] for d in data]
        PEs = [d[2] for d in data]
        TEs = [d[3] for d in data]
        self.ax.clear()
        self.ax.plot(times, KEs, 'y-', label='KE', linewidth=1)
        self.ax.plot(times, PEs, 'c-', label='PE', linewidth=1)
        self.ax.plot(times, TEs, 'w-', label='Total', linewidth=1.5)
        self.ax.legend(loc='upper right', fontsize=6, facecolor='#0a0520',
                       labelcolor='white', edgecolor='gray')
        self.ax.tick_params(colors='gray', labelsize=6)
        for spine in self.ax.spines.values():
            spine.set_color('gray')
        self.ax.set_xlabel('Time (s)', color='gray', fontsize=7)
        self.ax.set_ylabel('Energy (J)', color='gray', fontsize=7)
        self.fig.tight_layout(pad=0.5)
        self.canvas.draw()
        buf = self.canvas.buffer_rgba()
        self.surface = pygame.image.frombuffer(bytes(buf), (self.w, self.h), "RGBA")

    def draw(self, screen, pos):
        if self.surface:
            screen.blit(self.surface, pos)


# ============================================================
# SECTION 8: RENDERER
# ============================================================

class Renderer:
    def __init__(self, screen, table_origin):
        self.screen = screen
        self.origin = table_origin
        self.font_sm = pygame.font.SysFont("monospace", 11)
        self.font_med = pygame.font.SysFont("monospace", 14)

    def m_to_px(self, pos_m):
        px = self.origin + pos_m * CFG.PPM
        return int(px[0]), int(px[1])

    def draw_background(self):
        self.screen.fill(CFG.COLOR_BG)
        # Solar system placeholder
        # Sun
        sun_x, sun_y = 100, 400
        pygame.draw.circle(self.screen, (255, 200, 30), (sun_x, sun_y), 50)
        pygame.draw.circle(self.screen, (255, 240, 100), (sun_x, sun_y), 35)
        # Orbit rings
        orbits = [
            (120, 10, (40, 40, 50)),   # Mercury
            (190, 16, (40, 40, 50)),   # Venus
            (280, 22, (40, 40, 50)),   # Earth
            (380, 30, (40, 40, 50)),   # Mars
        ]
        for r, _, col in orbits:
            pygame.draw.ellipse(self.screen, col,
                                (sun_x - r, sun_y - r // 3, r * 2, r * 2 // 3), 1)
        # Planets
        planets = [
            (sun_x + 110, sun_y - 8,   6,  (180, 130, 90),  "Mer"),
            (sun_x + 170, sun_y - 18,  9,  (220, 190, 120), "Ven"),
            (sun_x + 260, sun_y - 25,  10, (60, 120, 220),  "Ear"),
            (sun_x + 360, sun_y - 35,  8,  (200, 80, 40),   "Mar"),
        ]
        for px, py, r, col, name in planets:
            pygame.draw.circle(self.screen, col, (px, py), r)
            lbl = self.font_sm.render(name, True, (120, 120, 140))
            self.screen.blit(lbl, (px - 10, py + r + 2))
        # Scale annotation
        # Table width in pixels ≈ 980, "1 AU ~ table width"
        note = self.font_sm.render("~ 1 AU = table width ~", True, (60, 60, 80))
        self.screen.blit(note, (20, CFG.WINDOW_H - 40))

    def draw_table(self, table):
        ox, oy = int(self.origin[0]), int(self.origin[1])
        w_px = int(table.w * CFG.PPM)
        h_px = int(table.h * CFG.PPM)
        rail = int(CFG.RAIL_W * CFG.PPM)

        # Rail border
        pygame.draw.rect(self.screen, CFG.COLOR_RAIL,
                         (ox - rail, oy - rail, w_px + 2 * rail, h_px + 2 * rail),
                         border_radius=8)
        # Felt
        pygame.draw.rect(self.screen, CFG.COLOR_FELT, (ox, oy, w_px, h_px))
        # Pockets
        for p in table.pockets:
            ppx, ppy = self.m_to_px(p)
            pygame.draw.circle(self.screen, (15, 15, 15), (ppx, ppy),
                               int(table.pocket_r * CFG.PPM))

    def draw_ball(self, ball):
        if ball.state == BallState.POCKETED:
            return
        cx, cy = self.m_to_px(ball.pos)
        r = max(3, int(ball.radius * CFG.PPM))
        color = ball.color

        if ball.is_stripe:
            pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy), r)
            band_h = max(2, r * 2 // 3)
            band_rect = pygame.Rect(cx - r, cy - band_h // 2, 2 * r, band_h)
            pygame.draw.rect(self.screen, color, band_rect)
            # Clip to circle by redrawing edge
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), r, 1)
        else:
            pygame.draw.circle(self.screen, color, (cx, cy), r)
            pygame.draw.circle(self.screen, (0, 0, 0), (cx, cy), r, 1)

        # Number
        if ball.id > 0:
            fg = (255, 255, 255) if ball.id == 8 else (0, 0, 0)
            # White circle behind number for stripes
            if ball.is_stripe:
                pygame.draw.circle(self.screen, (240, 240, 240), (cx, cy), r // 2)
            num_surf = self.font_sm.render(str(ball.id), True, fg)
            self.screen.blit(num_surf, (cx - num_surf.get_width() // 2,
                                        cy - num_surf.get_height() // 2))

    def draw_velocity_arrows(self, balls):
        arrow_scale = 30.0  # pixels per m/s
        for ball in balls:
            if ball.state == BallState.POCKETED or ball.speed < 0.001:
                continue
            cx, cy = self.m_to_px(ball.pos)
            tip_x = cx + ball.vel[0] * arrow_scale
            tip_y = cy + ball.vel[1] * arrow_scale
            pygame.draw.line(self.screen, (255, 255, 100),
                             (cx, cy), (int(tip_x), int(tip_y)), 2)
            # Arrowhead
            dir_v = vec(tip_x - cx, tip_y - cy)
            dlen = length(dir_v)
            if dlen > 4:
                d_hat = dir_v / dlen
                perp = rotate90(d_hat) * 4
                base = vec(tip_x, tip_y) - d_hat * 8
                pygame.draw.polygon(self.screen, (255, 255, 100), [
                    (int(tip_x), int(tip_y)),
                    (int(base[0] + perp[0]), int(base[1] + perp[1])),
                    (int(base[0] - perp[0]), int(base[1] - perp[1]))
                ])

    def draw_cue(self, cue, cue_ball):
        if not cue.active:
            return
        ball_px = np.array(self.m_to_px(cue_ball.pos), dtype=float)
        drag = cue.current_pos_px - cue.start_pos_px
        drag_len = length(drag)
        if drag_len < 3:
            return

        shot_dir = -normalize(drag)
        ball_r_px = cue_ball.radius * CFG.PPM

        # Cue stick behind ball
        cue_start = ball_px - shot_dir * (ball_r_px + 5 + drag_len * 0.3)
        cue_end = cue_start - shot_dir * 200
        pygame.draw.line(self.screen, (180, 140, 60),
                         (int(cue_start[0]), int(cue_start[1])),
                         (int(cue_end[0]), int(cue_end[1])), 5)
        # Cue tip
        pygame.draw.circle(self.screen, (140, 180, 200),
                           (int(cue_start[0]), int(cue_start[1])), 3)

        # Ghost aiming line
        aim_end = ball_px + shot_dir * 300
        pygame.draw.line(self.screen, (100, 100, 100),
                         (int(ball_px[0]), int(ball_px[1])),
                         (int(aim_end[0]), int(aim_end[1])), 1)

        # Power indicator
        power = cue.get_power_fraction()
        pwr_r = int(255 * power)
        pwr_g = int(255 * (1 - power))
        pygame.draw.circle(self.screen, (pwr_r, pwr_g, 0),
                           (int(ball_px[0]), int(ball_px[1])),
                           int(ball_r_px) + 4, 2)


# ============================================================
# SECTION 9: GAME CLASS (main loop + state machine)
# ============================================================

class GameState(Enum):
    AIMING = "aiming"
    SIMULATING = "simulating"
    STOPPED = "stopped"
    PLACING = "placing"  # cue ball pocketed, click to place on head string


class Game:
    def __init__(self):
        pygame.init()
        self.screen = pygame.display.set_mode((CFG.WINDOW_W, CFG.WINDOW_H))
        pygame.display.set_caption("Celestial Billiard")
        self.clock = pygame.time.Clock()

        table_w_px = int(CFG.TABLE_W * CFG.PPM)
        table_h_px = int(CFG.TABLE_H * CFG.PPM)
        self.table_origin = vec(
            (CFG.PLAY_AREA_W - table_w_px) // 2 + 10,
            (CFG.WINDOW_H - table_h_px) // 2
        )

        self.table = Table()
        self.balls = self._setup_balls()
        self.physics = PhysicsEngine(self.balls, self.table, self.cue_gravity_mode)
        self.cue = Cue()
        self.renderer = Renderer(self.screen, self.table_origin)
        self.inset = InsetView()
        self.energy_plot = EnergyPlot(
            w=min(380, CFG.WINDOW_W - CFG.UI_X - 10),
            h=200
        )
        self.sliders = self._build_sliders()

        self.game_state = GameState.AIMING
        self.cue_gravity_mode = CueBallGravity.FULL
        self.setup_mode = False  # S key toggles: 2-ball setup for tuning
        self.accumulator = 0.0
        self.last_time = time.perf_counter()
        self.plot_update_counter = 0

    def _setup_balls(self):
        balls = []
        R = CFG.BALL_R_M
        spacing = R * 2.05

        apex_x = CFG.TABLE_W * 0.75
        apex_y = CFG.TABLE_H * 0.5

        # 8-ball rack: 1 at apex, 8 at center of row 3
        # Place in row-major order, swap 8 to position 4 (center of row 3)
        ids = [1, 2, 9, 3, 8, 10, 4, 11, 12, 5, 6, 13, 14, 15, 7]

        idx = 0
        row_offset_x = spacing * math.cos(math.radians(30))
        for row in range(5):
            for col in range(row + 1):
                x = apex_x + row * row_offset_x
                y = apex_y + (col - row / 2.0) * spacing
                ball_id = ids[idx]
                balls.append(Ball(id=ball_id, pos=vec(x, y)))
                idx += 1

        # Cue ball
        balls.append(Ball(id=0, pos=vec(CFG.TABLE_W * 0.25, CFG.TABLE_H * 0.5)))
        return balls

    def _get_cue_ball(self):
        return next(b for b in self.balls if b.id == 0)

    def _build_sliders(self):
        x = CFG.UI_X
        y_start = CFG.INSET_SIZE + 50
        w = CFG.WINDOW_W - x - 20
        gap = 38
        return [
            Slider(x, y_start + 0 * gap, w, "Slide Friction",  0.01, 0.50, CFG.MU_SLIDE,   "MU_SLIDE",
                   "Kinetic friction while ball slides on cloth (higher = stops faster)"),
            Slider(x, y_start + 1 * gap, w, "Roll Friction",  0.005, 0.30, CFG.MU_ROLL,    "MU_ROLL",
                   "Rolling resistance on cloth (real: 0.01=fast cloth, 0.02=slow cloth)"),
            Slider(x, y_start + 2 * gap, w, "Spin Friction",  0.005, 0.20, CFG.MU_SPIN,    "MU_SPIN",
                   "Friction that decays vertical spin (english) over time"),
            Slider(x, y_start + 3 * gap, w, "Gravity",         0.0,  0.05, CFG.G_GAME,     "G_GAME",
                   "Gravitational constant between balls (0 = normal billiards)"),
            Slider(x, y_start + 4 * gap, w, "Softening",      0.01,  0.30, CFG.SOFTENING_EPS, "SOFTENING_EPS",
                   "Prevents infinite gravity at close range (larger = weaker close gravity)"),
            Slider(x, y_start + 5 * gap, w, "Ball Bounce",     0.0,  1.0,  CFG.COR_BALL,   "COR_BALL",
                   "Ball-ball elasticity: 1.0 = perfect bounce, 0.0 = absorb all energy"),
            Slider(x, y_start + 6 * gap, w, "Rail Bounce",     0.0,  1.0,  CFG.COR_RAIL,   "COR_RAIL",
                   "Rail elasticity: 1.0 = perfect cushion rebound, 0.0 = dead rail"),
            Slider(x, y_start + 7 * gap, w, "Stop Speed",    0.001, 0.05, CFG.V_THRESHOLD, "V_THRESHOLD",
                   "Speed below which the cue ball is considered stopped"),
        ]

    def _setup_balls_two(self):
        """Setup mode: just cue ball and one object ball for tuning physics."""
        balls = []
        balls.append(Ball(id=0, pos=vec(CFG.TABLE_W * 0.25, CFG.TABLE_H * 0.5)))
        balls.append(Ball(id=1, pos=vec(CFG.TABLE_W * 0.65, CFG.TABLE_H * 0.5)))
        return balls

    def _placing_position(self, mouse_pos):
        """Convert mouse click to a valid cue ball position on the head string.
        Returns position in meters, or None if spot is blocked by another ball."""
        # Head string is a vertical line at x = HEAD_STRING_X
        head_x = CFG.HEAD_STRING_X
        # Convert mouse y to table y in meters
        table_y_m = (mouse_pos[1] - self.table_origin[1]) / CFG.PPM
        R = CFG.BALL_R_M
        # Clamp to table bounds
        table_y_m = clamp(table_y_m, R, CFG.TABLE_H - R)
        candidate = vec(head_x, table_y_m)
        # Check no overlap with other balls
        for b in self.balls:
            if b.id == 0 or b.state == BallState.POCKETED:
                continue
            if length(b.pos - candidate) < R * 2.1:
                return None
        return candidate

    def _reset(self):
        self.balls = self._setup_balls_two() if self.setup_mode else self._setup_balls()
        self.physics = PhysicsEngine(self.balls, self.table, self.cue_gravity_mode)
        self.cue = Cue()
        self.game_state = GameState.AIMING
        self.energy_plot.surface = None
        self.physics.energy_log.clear()

    def run(self):
        running = True
        while running:
            now = time.perf_counter()
            frame_time = min(now - self.last_time, 0.05)
            self.last_time = now
            self.accumulator += frame_time

            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                self._handle_event(event)

            if self.game_state == GameState.SIMULATING:
                steps = 0
                while self.accumulator >= CFG.DT and steps < 10:
                    self.physics.step()
                    self.accumulator -= CFG.DT
                    steps += 1
                # Update plot periodically during simulation
                self.plot_update_counter += 1
                if self.plot_update_counter % 30 == 0:
                    self.energy_plot.update(self.physics.energy_log)
                if not self.physics.cue_ball_moving():
                    self.energy_plot.update(self.physics.energy_log)
                    cue_ball = self._get_cue_ball()
                    if cue_ball.state == BallState.POCKETED:
                        self.game_state = GameState.PLACING
                    else:
                        self.game_state = GameState.STOPPED
            else:
                self.accumulator = 0.0

            self._render()
            self.clock.tick(CFG.FPS_CAP)

        pygame.quit()

    def _handle_event(self, event):
        # Sliders get priority
        for s in self.sliders:
            s.handle_event(event)

        cue_ball = self._get_cue_ball()

        if event.type == pygame.MOUSEBUTTONDOWN and event.button == 1:
            if self.game_state == GameState.PLACING:
                # Place cue ball on head string at clicked y position
                place_pos = self._placing_position(event.pos)
                if place_pos is not None:
                    cue_ball.pos[:] = place_pos
                    cue_ball.vel[:] = 0.0
                    cue_ball.omega[:] = 0.0
                    cue_ball.state = BallState.STATIONARY
                    self.game_state = GameState.AIMING
            elif self.game_state in (GameState.AIMING, GameState.STOPPED):
                if cue_ball.state != BallState.POCKETED:
                    cb_px = np.array(self.renderer.m_to_px(cue_ball.pos), dtype=float)
                    mouse_px = np.array(event.pos, dtype=float)
                    if length(mouse_px - cb_px) < max(CFG.BALL_R_PX * 3, 30):
                        self.cue.begin_drag(event.pos)
                        self.game_state = GameState.AIMING

        elif event.type == pygame.MOUSEMOTION:
            if self.cue.active:
                self.cue.update_drag(event.pos)

        elif event.type == pygame.MOUSEBUTTONUP and event.button == 1:
            if self.cue.active:
                if self.cue.release(cue_ball):
                    self.game_state = GameState.SIMULATING
                    self.physics.energy_log.clear()
                    self.physics.sim_time = 0.0
                    self.plot_update_counter = 0

        elif event.type == pygame.KEYDOWN:
            STEP = 0.08
            if self.cue.active:
                if event.key == pygame.K_UP:
                    self.cue.adjust_contact(0, +STEP)
                elif event.key == pygame.K_DOWN:
                    self.cue.adjust_contact(0, -STEP)
                elif event.key == pygame.K_LEFT:
                    self.cue.adjust_contact(-STEP, 0)
                elif event.key == pygame.K_RIGHT:
                    self.cue.adjust_contact(+STEP, 0)

            if event.key == pygame.K_F12:
                self._take_screenshot()
            elif event.key == pygame.K_g:
                self.cue_gravity_mode = CueBallGravity((self.cue_gravity_mode + 1) % 3)
                self.physics.cue_gravity_mode = self.cue_gravity_mode
            elif event.key == pygame.K_r:
                self._reset()
            elif event.key == pygame.K_s:
                self.setup_mode = not self.setup_mode
                self._reset()

    def _render(self):
        self.renderer.draw_background()
        self.renderer.draw_table(self.table)

        for ball in self.balls:
            self.renderer.draw_ball(ball)

        cue_ball = self._get_cue_ball()
        if self.game_state == GameState.AIMING:
            self.renderer.draw_cue(self.cue, cue_ball)

        if self.game_state == GameState.STOPPED:
            self.renderer.draw_velocity_arrows(self.balls)

        if self.game_state == GameState.PLACING:
            # Draw head string line
            hx = int(self.table_origin[0] + CFG.HEAD_STRING_X * CFG.PPM)
            ty = int(self.table_origin[1])
            by = int(self.table_origin[1] + CFG.TABLE_H * CFG.PPM)
            pygame.draw.line(self.screen, (200, 200, 100), (hx, ty), (hx, by), 1)
            # Draw ghost ball at mouse y on head string
            mouse_y = pygame.mouse.get_pos()[1]
            candidate = self._placing_position(pygame.mouse.get_pos())
            ghost_color = (180, 180, 180, 128) if candidate is not None else (200, 60, 60, 128)
            gy_m = clamp((mouse_y - self.table_origin[1]) / CFG.PPM, CFG.BALL_R_M, CFG.TABLE_H - CFG.BALL_R_M)
            gy_px = int(self.table_origin[1] + gy_m * CFG.PPM)
            r_px = max(3, int(CFG.BALL_R_M * CFG.PPM))
            pygame.draw.circle(self.screen, ghost_color[:3], (hx, gy_px), r_px, 2)

        # Right panel
        inset_rect = pygame.Rect(CFG.UI_X, 10, CFG.INSET_SIZE, CFG.INSET_SIZE)
        self.inset.draw(self.screen, self.cue, inset_rect)

        for s in self.sliders:
            s.draw(self.screen, self.renderer.font_sm)

        # Energy plot below sliders
        plot_y = CFG.INSET_SIZE + 50 + 9 * 38 + 20
        self.energy_plot.draw(self.screen, (CFG.UI_X, min(plot_y, CFG.WINDOW_H - 210)))

        # Status bar
        state_txt = self.game_state.value.upper()
        if self.game_state == GameState.SIMULATING:
            moving = sum(1 for b in self.balls if b.is_active)
            state_txt += f"  ({moving} balls moving)"
        mode_txt = "SETUP" if self.setup_mode else "GAME"
        grav_txt = CUE_GRAVITY_LABELS[self.cue_gravity_mode]
        lbl = self.renderer.font_med.render(
            f"{mode_txt} | {state_txt} | {grav_txt}  |  G=grav mode  S=setup  R=reset  F12=screenshot  Arrows=spin",
            True, (180, 180, 180))
        self.screen.blit(lbl, (10, CFG.WINDOW_H - 22))

        pygame.display.flip()

    def _take_screenshot(self):
        os.makedirs("screenshots", exist_ok=True)
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = f"screenshots/shot_{ts}.png"
        pygame.image.save(self.screen, path)
        print(f"Screenshot saved: {path}")


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    Game().run()
