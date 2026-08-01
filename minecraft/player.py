import math

import numpy as np

from . import blocks as B
from . import config as C
from .mat4 import forward_from_angles

EPS = 1e-4
HALF_W = C.PLAYER_WIDTH / 2.0


class Player:
    __slots__ = ("pos", "vel", "yaw", "pitch", "on_ground", "flying",
                 "hotbar", "selected")

    def __init__(self, pos=(0.0, 80.0, 0.0), yaw=0.0, pitch=0.0):
        self.pos = np.array(pos, dtype=np.float64)
        self.vel = np.zeros(3, dtype=np.float64)
        self.yaw = float(yaw)
        self.pitch = float(pitch)
        self.on_ground = False
        self.flying = False
        self.hotbar = list(B.PLACEABLE[:9])
        self.selected = 0

    @property
    def eye(self):
        return self.pos + np.array([0.0, C.EYE_HEIGHT, 0.0])

    @property
    def forward(self):
        return forward_from_angles(self.yaw, self.pitch)

    def look(self, dx, dy):
        self.yaw += dx * C.MOUSE_SENSITIVITY
        self.pitch -= dy * C.MOUSE_SENSITIVITY
        lim = math.pi / 2.0 - 1e-4
        self.pitch = max(-lim, min(lim, self.pitch))

    def toggle_fly(self):
        self.flying = not self.flying
        self.vel[1] = 0.0

    def aabb(self, pos=None):
        p = self.pos if pos is None else pos
        return ((p[0] - HALF_W, p[1], p[2] - HALF_W),
                (p[0] + HALF_W, p[1] + C.PLAYER_HEIGHT, p[2] + HALF_W))

    def update(self, dt, world, move):
        fwd_amt, strafe_amt, jump, sneak, sprint = move
        sy, cy = math.sin(self.yaw), math.cos(self.yaw)
        wish = np.array([sy * fwd_amt + cy * strafe_amt,
                         0.0,
                         -cy * fwd_amt + sy * strafe_amt])
        n = math.hypot(wish[0], wish[2])
        if n > 1e-9:
            wish /= n
        feet = world.get_block(math.floor(self.pos[0]),
                               math.floor(self.pos[1]),
                               math.floor(self.pos[2]))
        in_water = feet == B.WATER
        if self.flying:
            speed = C.FLY_SPRINT_SPEED if sprint else C.FLY_SPEED
            self.vel[0] = wish[0] * speed
            self.vel[2] = wish[2] * speed
            self.vel[1] = (speed if jump else 0.0) + (-speed if sneak else 0.0)
        else:
            speed = C.SPRINT_SPEED if sprint else C.WALK_SPEED
            if in_water:
                speed *= 0.5
            self.vel[0] = wish[0] * speed
            self.vel[2] = wish[2] * speed
            if in_water:
                self.vel[1] = max(self.vel[1] - C.GRAVITY * 0.25 * dt, -4.0)
                if jump:
                    self.vel[1] = 4.0
            else:
                self.vel[1] = max(self.vel[1] - C.GRAVITY * dt,
                                  -C.TERMINAL_VELOCITY)
                if jump and self.on_ground:
                    self.vel[1] = C.JUMP_VELOCITY
                    self.on_ground = False
        speed_max = max(abs(self.vel[0]), abs(self.vel[1]), abs(self.vel[2]))
        steps = max(1, math.ceil(speed_max * dt / 0.4))
        sub = dt / steps
        self.on_ground = False
        for _ in range(steps):
            self._move_axis(world, 0, self.vel[0] * sub)
            self._move_axis(world, 1, self.vel[1] * sub)
            self._move_axis(world, 2, self.vel[2] * sub)

    def _move_axis(self, world, axis, delta):
        if delta == 0.0:
            return
        new_val = self.pos[axis] + delta
        p = (self.pos[0], new_val, self.pos[2]) if axis == 1 else (
            new_val if axis == 0 else self.pos[0],
            self.pos[1],
            new_val if axis == 2 else self.pos[2])
        (x0, y0, z0), (x1, y1, z1) = self.aabb(np.array(p))
        best = None
        for bx in range(math.floor(x0), math.floor(x1 - EPS) + 1):
            for by in range(math.floor(y0), math.floor(y1 - EPS) + 1):
                for bz in range(math.floor(z0), math.floor(z1 - EPS) + 1):
                    if not world.is_solid(bx, by, bz):
                        continue
                    if axis == 0:
                        cand = bx - HALF_W - EPS if delta > 0 else bx + 1 + HALF_W + EPS
                    elif axis == 1:
                        cand = by - C.PLAYER_HEIGHT - EPS if delta > 0 else by + 1 + EPS
                    else:
                        cand = bz - HALF_W - EPS if delta > 0 else bz + 1 + HALF_W + EPS
                    if best is None or abs(cand - self.pos[axis]) < abs(best - self.pos[axis]):
                        best = cand
        if best is None:
            self.pos[axis] = new_val
            return
        self.pos[axis] = best
        if axis == 1:
            if delta < 0:
                self.on_ground = True
            self.vel[1] = 0.0
