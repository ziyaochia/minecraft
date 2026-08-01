import math
import sys

sys.path.insert(0, r"F:\PycharmProjects\Minecraft")
from minecraft import blocks as B
from minecraft.interact import break_block, place_block
from minecraft.player import EPS, HALF_W, Player
from minecraft.raycast import raycast
from minecraft.world import World

DT = 1.0 / 60.0
w = World(12345)
for cx in range(-1, 2):
    for cz in range(-1, 2):
        w.ensure_chunk(cx, cz)
for chunk in w.chunks.values():
    chunk.blocks[:, 0:5, :] = B.STONE
    chunk.blocks[:, 5:41, :] = B.AIR
w.chunks[(1, 0)].blocks[0, 5:10, :] = B.STONE
w.chunks[(1, 1)].blocks[4:7, 5:7, 4:7] = B.WATER
w.chunks[(0, 0)].blocks[8, 0, 8] = B.BEDROCK

FLOOR = 5.0 + EPS

p = Player((8.0, 12.0, 8.0))
for _ in range(300):
    p.update(DT, w, (0, 0, False, False, False))
assert p.on_ground, "should land"
assert abs(p.pos[1] - FLOOR) < 1e-9, f"land height {p.pos[1]}"
assert p.vel[1] == 0.0
print(f"fall+land OK  y={p.pos[1]:.6f}")

pw = Player((12.0, FLOOR, 8.0), yaw=math.pi / 2)
for _ in range(600):
    pw.update(DT, w, (1, 0, False, False, False))
expected_x = 16.0 - HALF_W - EPS
assert abs(pw.pos[0] - expected_x) < 1e-9, f"wall stop {pw.pos[0]}"
assert abs(pw.pos[1] - FLOOR) < 1e-9
print(f"wall stop OK  x={pw.pos[0]:.6f}")

pj = Player((4.0, FLOOR, 4.0))
for _ in range(5):
    pj.update(DT, w, (0, 0, False, False, False))
max_y = 0.0
landed = False
for i in range(90):
    pj.update(DT, w, (0, 0, i < 20, False, False))
    max_y = max(max_y, pj.pos[1])
    if i > 30 and pj.on_ground:
        landed = True
v, y, ymax = 8.6, 0.0, 0.0
for k in range(60):
    if k > 0:
        v -= 32.0 * DT
    y += v * DT
    ymax = max(ymax, y)
apex = FLOOR + ymax
assert abs(max_y - apex) < 1e-6, f"apex {max_y} vs {apex}"
assert landed, "should land again"
print(f"jump OK  apex={max_y:.4f} (theory {apex:.4f})")

eye = p.eye
hit, prev, bid = raycast(w, eye, (0.0, -1.0, 0.0), 6.0)
assert (hit, prev, bid) == ((8, 4, 8), (8, 5, 8), B.STONE), (hit, prev, bid)
print("raycast down OK")

assert raycast(w, (8.0, 7.0, 8.0), (1.0, 0.0, 0.0), 6.0) is None
hit, prev, bid = raycast(w, (8.0, 7.0, 8.0), (1.0, 0.0, 0.0), 10.0)
assert hit == (16, 7, 8) and prev == (15, 7, 8), (hit, prev)
print("raycast reach OK")

hit, prev, bid = raycast(w, (21.0, 9.0, 21.0), (0.0, -1.0, 0.0), 10.0)
assert hit == (21, 4, 21) and bid == B.STONE, (hit, bid)
print("raycast through water OK")

aff = break_block(w, (8, 4, 8))
assert w.get_block(8, 4, 8) == B.AIR
assert aff == [(0, 0)], aff
aff = break_block(w, (15, 4, 8))
assert set(aff) == {(0, 0), (1, 0)}, aff
break_block(w, (8, 0, 8))
assert w.get_block(8, 0, 8) == B.BEDROCK, "bedrock unbreakable"
print("break OK (border remesh set, bedrock)")

assert place_block(w, (7, 5, 7), B.STONE, player=p) == []
assert w.get_block(7, 5, 7) == B.AIR, "overlap rejected"
aff = place_block(w, (10, 5, 10), B.COBBLESTONE, player=p)
assert aff and w.get_block(10, 5, 10) == B.COBBLESTONE
place_block(w, (21, 5, 21), B.STONE)
assert w.get_block(21, 5, 21) == B.STONE, "place into water"
print("place OK (overlap reject, normal, into water)")

pf = Player((-4.0, FLOOR, -4.0))
pf.toggle_fly()
for _ in range(30):
    pf.update(DT, w, (0, 0, True, False, False))
assert abs(pf.pos[1] - (FLOOR + 6.0)) < 0.01, pf.pos[1]
for _ in range(60):
    pf.update(DT, w, (0, 0, False, True, False))
assert abs(pf.pos[1] - FLOOR) < 1e-9, pf.pos[1]
pf.toggle_fly()
print("fly OK")

ps = Player((21.5, 9.0, 21.5))
for _ in range(240):
    ps.update(DT, w, (0, 0, False, False, False))
assert ps.on_ground and abs(ps.pos[1] - (6.0 + EPS)) < 1e-9, ps.pos[1]
print(f"water sink OK  rests on placed stone y={ps.pos[1]:.6f}")

print("ALL PLAYER TESTS PASSED")
