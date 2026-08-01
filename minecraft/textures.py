import numpy as np

TILE = 16
GRID = 8
ATLAS_SIZE = TILE * GRID


def _rgb(base, spread, rng, size=TILE):
    base = np.asarray(base, dtype=np.int16).reshape(1, 1, 3)
    n = rng.randint(-spread, spread + 1, size=(size, size, 3))
    return np.clip(base + n, 0, 255).astype(np.uint8)


def _patches(base_a, base_b, rng, scale=4):
    small = rng.random((TILE // scale, TILE // scale)) > 0.5
    m = np.repeat(np.repeat(small, scale, 0), scale, 1)
    a = np.asarray(base_a, dtype=np.int16)
    b = np.asarray(base_b, dtype=np.int16)
    j = rng.randint(-8, 9, size=(TILE, TILE, 3))
    out = np.where(m[..., None], a, b) + j
    return np.clip(out, 0, 255).astype(np.uint8)


def _rgba(rgb, alpha=255):
    a = np.full(rgb.shape[:2] + (1,), alpha, dtype=np.uint8)
    return np.concatenate([rgb, a], axis=2)


def _ore(rng, blob_rgb, blobs=5):
    rgb = _patches((125, 125, 125), (115, 115, 115), rng)
    for _ in range(blobs):
        cx, cy = rng.randint(1, 14, size=2)
        w, h = rng.randint(1, 3, size=2)
        j = rng.randint(-25, 26, size=(h, w, 3))
        rgb[cy:cy + h, cx:cx + w] = np.clip(np.asarray(blob_rgb, dtype=np.int16) + j, 0, 255)
    return _rgba(rgb)


def _grass_top(rng):
    return _rgba(_patches((99, 158, 52), (111, 168, 62), rng))


def _grass_side(rng):
    rgb = _patches((134, 96, 67), (121, 85, 58), rng)
    depth = rng.randint(12, 16, size=(1, TILE))
    rows = np.arange(TILE).reshape(TILE, 1)
    green = _patches((99, 158, 52), (111, 168, 62), rng)
    rgb = np.where((rows >= depth)[..., None], green, rgb)
    return _rgba(rgb)


def _snow_side(rng):
    rgb = _patches((134, 96, 67), (121, 85, 58), rng)
    depth = rng.randint(12, 16, size=(1, TILE))
    rows = np.arange(TILE).reshape(TILE, 1)
    white = _rgb((240, 248, 248), 6, rng)
    rgb = np.where((rows >= depth)[..., None], white, rgb)
    return _rgba(rgb)


def _dirt(rng):
    return _rgba(_patches((134, 96, 67), (121, 85, 58), rng))


def _stone(rng):
    return _rgba(_patches((125, 125, 125), (114, 114, 114), rng))


def _sand(rng):
    return _rgba(_patches((219, 207, 163), (208, 195, 150), rng))


def _water(rng):
    rgb = _rgb((45, 90, 200), 18, rng)
    rows = np.arange(TILE).reshape(TILE, 1)
    band = ((rows + rng.randint(0, 3, size=(1, TILE))) % 8) < 2
    rgb = np.where(band[..., None], np.clip(rgb.astype(np.int16) + 35, 0, 255), rgb).astype(np.uint8)
    return _rgba(rgb, alpha=196)


def _log_side(rng):
    cols = np.arange(TILE).reshape(1, TILE)
    stripe = (cols + rng.randint(0, 2, size=(1, TILE))) % 4
    stripe = np.broadcast_to(stripe, (TILE, TILE))
    base = np.where((stripe < 2)[..., None], np.array([104, 82, 50]), np.array([86, 66, 40])).astype(np.int16)
    j = rng.randint(-10, 11, size=(TILE, TILE, 3))
    return _rgba(np.clip(base + j, 0, 255).astype(np.uint8))


def _log_top(rng):
    yy, xx = np.mgrid[0:TILE, 0:TILE]
    d = np.maximum(np.abs(xx - 7.5), np.abs(yy - 7.5))
    ring = (d.astype(np.int32) % 2) == 0
    a = np.asarray([177, 142, 90], dtype=np.int16)
    b = np.asarray([104, 82, 50], dtype=np.int16)
    j = rng.randint(-8, 9, size=(TILE, TILE, 3))
    return _rgba(np.clip(np.where(ring[..., None], a, b) + j, 0, 255).astype(np.uint8))


def _leaves(rng):
    rgb = _patches((43, 105, 31), (52, 122, 38), rng, scale=2)
    holes = rng.random((TILE, TILE)) < 0.18
    alpha = np.where(holes, 0, 255).astype(np.uint8)[..., None]
    return np.concatenate([rgb, alpha], axis=2)


def _planks(rng):
    rows = np.arange(TILE).reshape(TILE, 1)
    board = (rows // 4) % 2
    seam = (rows % 4) == 0
    base = np.where(board, np.array([172, 140, 88]), np.array([162, 130, 80])).astype(np.int16)
    j = rng.randint(-9, 10, size=(TILE, TILE, 3))
    rgb = np.clip(base + j, 0, 255)
    rgb[seam[:, 0]] = np.array([110, 88, 54])
    return _rgba(rgb.astype(np.uint8))


def _cobble(rng):
    small = rng.randint(90, 150, size=(4, 4, 3))
    big = np.repeat(np.repeat(small, 4, 0), 4, 1)
    j = rng.randint(-12, 13, size=(TILE, TILE, 3))
    rgb = np.clip(big + j, 0, 255)
    rows = np.arange(TILE).reshape(TILE, 1)
    cols = np.arange(TILE).reshape(1, TILE)
    mortar = ((rows % 4) == 0) | ((cols + rows // 4 % 2 * 2) % 4 == 0)
    rgb = np.where(mortar[..., None], np.array([70, 70, 70]), rgb)
    return _rgba(rgb.astype(np.uint8))


def _glass(rng):
    rgb = np.full((TILE, TILE, 3), 220, dtype=np.uint8)
    alpha = np.zeros((TILE, TILE), dtype=np.uint8)
    rows = np.arange(TILE).reshape(TILE, 1)
    cols = np.arange(TILE).reshape(1, TILE)
    edge = (rows == 0) | (rows == TILE - 1) | (cols == 0) | (cols == TILE - 1)
    alpha[edge] = 200
    streak = (rows - cols) % 11 == 0
    alpha |= np.where(streak, 60, 0).astype(np.uint8)
    return np.concatenate([rgb, alpha[..., None]], axis=2)


def _bedrock(rng):
    m = rng.random((TILE, TILE, 1)) > 0.5
    a = np.asarray([55, 55, 55], dtype=np.int16)
    b = np.asarray([110, 110, 110], dtype=np.int16)
    j = rng.randint(-12, 13, size=(TILE, TILE, 3))
    return _rgba(np.clip(np.where(m, a, b) + j, 0, 255).astype(np.uint8))


def _snow(rng):
    return _rgba(_rgb((240, 248, 248), 6, rng))


def _brick(rng):
    rows = np.arange(TILE).reshape(TILE, 1)
    cols = np.arange(TILE).reshape(1, TILE)
    mortar = np.broadcast_to((rows % 4) == 0, (TILE, TILE)).copy()
    offset = np.where((rows // 4) % 2 == 0, 0, 2)
    mortar |= ((cols + offset) % 4) == 0
    j = rng.randint(-10, 11, size=(TILE, TILE, 3))
    rgb = np.clip(np.asarray([150, 68, 55], dtype=np.int16) + j, 0, 255)
    rgb = np.where(mortar[..., None], np.array([188, 178, 172]), rgb)
    return _rgba(rgb.astype(np.uint8))


def _gravel(rng):
    palette = np.array([[128, 124, 120], [110, 104, 100], [143, 136, 128], [98, 92, 88], [134, 96, 67]])
    idx = rng.randint(0, len(palette), size=(TILE, TILE))
    rgb = palette[idx]
    j = rng.randint(-8, 9, size=(TILE, TILE, 3))
    return _rgba(np.clip(rgb.astype(np.int16) + j, 0, 255).astype(np.uint8))


def _sandstone_top(rng):
    return _rgba(_rgb((216, 202, 155), 7, rng))


def _sandstone_bottom(rng):
    return _rgba(_rgb((202, 186, 138), 7, rng))


def _sandstone_side(rng):
    rgb = _rgb((214, 200, 152), 7, rng)
    rows = np.arange(TILE).reshape(TILE, 1)
    band = (rows < 2) | (rows >= TILE - 2)
    rgb = np.where(band, np.clip(rgb.astype(np.int16) - 30, 0, 255), rgb).astype(np.uint8)
    return _rgba(rgb)


BUILDERS = [
    ("grass_top", lambda r: _grass_top(r)),
    ("grass_side", lambda r: _grass_side(r)),
    ("dirt", _dirt),
    ("stone", _stone),
    ("sand", _sand),
    ("water", _water),
    ("log_top", _log_top),
    ("log_side", _log_side),
    ("leaves", _leaves),
    ("planks", _planks),
    ("cobblestone", _cobble),
    ("glass", _glass),
    ("bedrock", _bedrock),
    ("coal_ore", lambda r: _ore(r, (35, 35, 35))),
    ("iron_ore", lambda r: _ore(r, (216, 175, 147))),
    ("gold_ore", lambda r: _ore(r, (250, 234, 77))),
    ("diamond_ore", lambda r: _ore(r, (93, 236, 231))),
    ("snow", _snow),
    ("snow_side", _snow_side),
    ("brick", _brick),
    ("gravel", _gravel),
    ("sandstone_top", _sandstone_top),
    ("sandstone_side", _sandstone_side),
    ("sandstone_bottom", _sandstone_bottom),
]

TILE_INDEX = {name: i for i, (name, _) in enumerate(BUILDERS)}


def build_atlas(seed=0):
    atlas = np.zeros((ATLAS_SIZE, ATLAS_SIZE, 4), dtype=np.uint8)
    for i, (name, fn) in enumerate(BUILDERS):
        rng = np.random.RandomState((seed + i * 7919) & 0x7FFFFFFF)
        tile = fn(rng)
        col = i % GRID
        row = i // GRID
        atlas[row * TILE:(row + 1) * TILE, col * TILE:(col + 1) * TILE] = tile
    return atlas


def build_palette(atlas):
    pal = np.zeros((len(BUILDERS), 4), dtype=np.uint8)
    for i, (name, _) in enumerate(BUILDERS):
        col = i % GRID
        row = i // GRID
        tile = atlas[row * TILE:(row + 1) * TILE,
                     col * TILE:(col + 1) * TILE].astype(np.float32)
        a = tile[..., 3:4] / 255.0
        wsum = a.sum()
        if wsum > 0:
            rgb = (tile[..., :3] * a).sum(axis=(0, 1)) / wsum
        else:
            rgb = tile[..., :3].mean(axis=(0, 1))
        pal[i, :3] = np.clip(rgb, 0, 255).astype(np.uint8)
        pal[i, 3] = int(tile[..., 3].mean())
    return pal


def tile_uv(name):
    i = TILE_INDEX[name]
    col = i % GRID
    row = i // GRID
    e = 0.5 / ATLAS_SIZE
    u0 = col * TILE / ATLAS_SIZE + e
    v0 = row * TILE / ATLAS_SIZE + e
    u1 = (col + 1) * TILE / ATLAS_SIZE - e
    v1 = (row + 1) * TILE / ATLAS_SIZE - e
    return u0, v0, u1, v1
