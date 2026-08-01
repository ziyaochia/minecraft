import sys
import time
sys.path.insert(0, r"F:\PycharmProjects\Minecraft")
import numpy as np
from PIL import Image
from minecraft.world import World
from minecraft import blocks as B
from minecraft.mesher import build_mesh
from minecraft.config import SEA_LEVEL

SEED = 20260731
w = World(SEED)

t0 = time.time()
for cx in range(-1, 2):
    for cz in range(-1, 2):
        w.ensure_chunk(cx, cz)
t1 = time.time()
print(f"generated 9 chunks in {(t1 - t0) * 1000:.0f} ms ({(t1 - t0) / 9 * 1000:.1f} ms/chunk)")

c = w.get_chunk(0, 0)
ids, counts = np.unique(c.blocks, return_counts=True)
print("block histogram chunk(0,0):", {B.NAMES[i]: int(n) for i, n in zip(ids, counts)})

COLORS = {
    B.GRASS: (90, 160, 60), B.DIRT: (134, 96, 67), B.STONE: (125, 125, 125),
    B.SAND: (219, 207, 163), B.WATER: (45, 90, 200), B.SNOW: (240, 248, 248),
    B.GRAVEL: (128, 124, 120), B.LOG: (104, 82, 50), B.LEAVES: (43, 105, 31),
    B.SANDSTONE: (214, 200, 152), B.BEDROCK: (60, 60, 60),
}
N = 48
img = np.zeros((N, N, 3), dtype=np.uint8)
hmap = np.zeros((N, N), dtype=np.int32)
for wx in range(-16, 32):
    for wz in range(-16, 32):
        col = None
        for y in range(255, -1, -1):
            b = w.get_block(wx, y, wz)
            if b != B.AIR:
                col = b
                hmap[wx + 16, wz + 16] = y
                break
        img[wx + 16, wz + 16] = COLORS.get(col, (255, 0, 255))
Image.fromarray(img).resize((384, 384), Image.NEAREST).save("world_topdown.png")

gray = ((hmap - hmap.min()) / max(1, hmap.max() - hmap.min()) * 255).astype(np.uint8)
Image.fromarray(gray).resize((384, 384), Image.NEAREST).save("world_height.png")

t0 = time.time()
meshes = {}
for cx in range(-1, 2):
    for cz in range(-1, 2):
        padded = w.get_padded(cx, cz)
        meshes[(cx, cz)] = build_mesh(padded, cx * 16, cz * 16)
t1 = time.time()
print(f"meshed 9 chunks in {(t1 - t0) * 1000:.0f} ms ({(t1 - t0) / 9 * 1000:.1f} ms/chunk)")
for k, (ov, oi, tv, ti) in meshes.items():
    print(f"  chunk {k}: opaque {len(ov)} verts {len(oi)} idx | trans {len(tv)} verts {len(ti)} idx")
    assert len(oi) % 6 == 0 and len(ov) == len(oi) // 6 * 4
    assert oi.max() < len(ov) if len(oi) else True

sw = World(1)
slab = np.zeros((16, 256, 16), dtype=np.uint8)
slab[:, 10, :] = B.STONE
from minecraft.chunk import Chunk
sw.chunks[(0, 0)] = Chunk(0, 0, slab)
ov, oi, tv, ti = build_mesh(sw.get_padded(0, 0))
print("slab test: faces =", len(oi) // 6, "(expect 576)")
assert len(oi) // 6 == 576, "slab face count wrong"
uniq = np.unique(ov[:, 5])
assert len(uniq) == 4 and np.allclose(sorted(uniq.tolist()), [0.5, 0.6, 0.8, 1.0])
print("slab shades:", sorted(set(np.unique(ov[:, 5]).tolist())))

sw.chunks[(0, 0)].blocks[0, 10, 0] = B.AIR
ov2, oi2, _, _ = build_mesh(sw.get_padded(0, 0))
print("slab with hole: faces =", len(oi2) // 6, "(expect 576 - 4 + 2 = 574)")
assert len(oi2) // 6 == 574

sw2 = World(1)
wb = np.zeros((16, 256, 16), dtype=np.uint8)
wb[:, 5, :] = B.WATER
sw2.chunks[(0, 0)] = Chunk(0, 0, wb)
ov, oi, tv, ti = build_mesh(sw2.get_padded(0, 0))
print("water slab: opaque faces =", len(oi) // 6, "translucent faces =", len(ti) // 6, "(expect 0 / 576)")
assert len(oi) == 0 and len(ti) // 6 == 576
print("OK")
