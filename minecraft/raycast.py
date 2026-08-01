import math

from . import blocks as B


def raycast(world, origin, direction, max_dist):
    ox, oy, oz = float(origin[0]), float(origin[1]), float(origin[2])
    dx, dy, dz = float(direction[0]), float(direction[1]), float(direction[2])
    x, y, z = math.floor(ox), math.floor(oy), math.floor(oz)
    step_x = 1 if dx > 0 else -1
    step_y = 1 if dy > 0 else -1
    step_z = 1 if dz > 0 else -1
    t_max_x = ((x + (step_x > 0)) - ox) / dx if dx != 0.0 else math.inf
    t_max_y = ((y + (step_y > 0)) - oy) / dy if dy != 0.0 else math.inf
    t_max_z = ((z + (step_z > 0)) - oz) / dz if dz != 0.0 else math.inf
    t_delta_x = abs(1.0 / dx) if dx != 0.0 else math.inf
    t_delta_y = abs(1.0 / dy) if dy != 0.0 else math.inf
    t_delta_z = abs(1.0 / dz) if dz != 0.0 else math.inf
    t = 0.0
    px, py, pz = x, y, z
    first = True
    while t <= max_dist:
        if not first:
            bid = world.get_block(x, y, z)
            if bid != B.AIR and bid != B.WATER:
                return (x, y, z), (px, py, pz), bid
        first = False
        px, py, pz = x, y, z
        if t_max_x <= t_max_y and t_max_x <= t_max_z:
            x += step_x
            t = t_max_x
            t_max_x += t_delta_x
        elif t_max_y <= t_max_z:
            y += step_y
            t = t_max_y
            t_max_y += t_delta_y
        else:
            z += step_z
            t = t_max_z
            t_max_z += t_delta_z
    return None
