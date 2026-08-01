import math

import numpy as np


def perspective(fov_deg, aspect, near, far):
    f = 1.0 / math.tan(math.radians(fov_deg) / 2.0)
    m = np.zeros((4, 4), dtype=np.float64)
    m[0, 0] = f / aspect
    m[1, 1] = f
    m[2, 2] = (far + near) / (near - far)
    m[2, 3] = 2 * far * near / (near - far)
    m[3, 2] = -1.0
    return m


def ortho(l, r, b, t, n, f):
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = 2.0 / (r - l)
    m[1, 1] = 2.0 / (t - b)
    m[2, 2] = -2.0 / (f - n)
    m[0, 3] = -(r + l) / (r - l)
    m[1, 3] = -(t + b) / (t - b)
    m[2, 3] = -(f + n) / (f - n)
    return m


def look_at(eye, target, up):
    eye = np.asarray(eye, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    up = np.asarray(up, dtype=np.float64)
    f = target - eye
    f = f / np.linalg.norm(f)
    s = np.cross(f, up)
    s = s / np.linalg.norm(s)
    u = np.cross(s, f)
    m = np.identity(4, dtype=np.float64)
    m[0, :3] = s
    m[1, :3] = u
    m[2, :3] = -f
    m[0, 3] = -np.dot(s, eye)
    m[1, 3] = -np.dot(u, eye)
    m[2, 3] = np.dot(f, eye)
    return m


def translation(x, y, z):
    m = np.identity(4, dtype=np.float64)
    m[0, 3] = x
    m[1, 3] = y
    m[2, 3] = z
    return m


def rotation_x(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=np.float64)
    m[1, 1] = c
    m[1, 2] = -s
    m[2, 1] = s
    m[2, 2] = c
    return m


def rotation_y(a):
    c, s = math.cos(a), math.sin(a)
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = c
    m[0, 2] = s
    m[2, 0] = -s
    m[2, 2] = c
    return m


def scale_uniform(s):
    m = np.identity(4, dtype=np.float64)
    m[0, 0] = s
    m[1, 1] = s
    m[2, 2] = s
    return m


def to_gl(m):
    return np.ascontiguousarray(m.T.astype(np.float32))


def forward_from_angles(yaw, pitch):
    cp = math.cos(pitch)
    return np.array([math.sin(yaw) * cp, math.sin(pitch), -math.cos(yaw) * cp])


def frustum_planes(mvp):
    m = np.asarray(mvp, dtype=np.float64)
    planes = np.array([
        m[3] + m[0],
        m[3] - m[0],
        m[3] + m[1],
        m[3] - m[1],
        m[3] + m[2],
        m[3] - m[2],
    ])
    norms = np.linalg.norm(planes[:, :3], axis=1)
    return planes / norms[:, None]


def aabb_visible(planes, mins, maxs):
    for p in planes:
        x = maxs[0] if p[0] >= 0 else mins[0]
        y = maxs[1] if p[1] >= 0 else mins[1]
        z = maxs[2] if p[2] >= 0 else mins[2]
        if p[0] * x + p[1] * y + p[2] * z + p[3] < 0:
            return False
    return True
