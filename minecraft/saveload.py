import os

import numpy as np

from . import config as C
from .chunk import Chunk


def save_path(seed):
    os.makedirs(C.SAVE_DIR, exist_ok=True)
    return os.path.join(C.SAVE_DIR, f"world_{seed}.npz")


def save_world(world, player, time_of_day, path):
    data = {
        "seed": np.int64(world.seed),
        "time_of_day": np.float64(time_of_day),
        "pos": np.asarray(player.pos, dtype=np.float64),
        "vel": np.asarray(player.vel, dtype=np.float64),
        "angles": np.array([player.yaw, player.pitch]),
        "flying": np.bool_(player.flying),
        "hotbar": np.asarray(player.hotbar, dtype=np.uint8),
        "selected": np.int64(player.selected),
    }
    for (cx, cz), chunk in world.chunks.items():
        data[f"c_{cx}_{cz}"] = chunk.blocks
    np.savez_compressed(path, **data)


def load_world(world, path):
    with np.load(path) as z:
        if int(z["seed"]) != world.seed:
            return None
        meta = {
            "time_of_day": float(z["time_of_day"]),
            "pos": z["pos"].astype(np.float64),
            "vel": z["vel"].astype(np.float64),
            "yaw": float(z["angles"][0]),
            "pitch": float(z["angles"][1]),
            "flying": bool(z["flying"]),
            "hotbar": [int(b) for b in z["hotbar"]],
            "selected": int(z["selected"]),
        }
        for key in z.files:
            if not key.startswith("c_"):
                continue
            _, cx, cz = key.split("_")
            chunk = Chunk(int(cx), int(cz), np.array(z[key], dtype=np.uint8))
            world.chunks[(chunk.cx, chunk.cz)] = chunk
    return meta
