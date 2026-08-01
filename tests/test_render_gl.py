import sys
import math
sys.path.insert(0, r"F:\PycharmProjects\Minecraft")
import numpy as np
import moderngl
from PIL import Image
from minecraft.world import World
from minecraft import blocks as B
from minecraft.mesher import build_mesh
from minecraft.textures import build_atlas
from minecraft.render_gl import GLRenderer
from minecraft.mat4 import perspective, look_at, frustum_planes, aabb_visible, forward_from_angles
from minecraft.sky import SkyState

SEED = 20260731
R = 6
w = World(SEED)
for cx in range(-R, R):
    for cz in range(-R, R):
        w.ensure_chunk(cx, cz)

atlas = build_atlas(SEED)
ctx = moderngl.create_context(standalone=True, require=330)
rend = GLRenderer(ctx, atlas)

meshes = {}
for (cx, cz) in list(w.chunks.keys()):
    ov, oi, tv, ti = build_mesh(w.get_padded(cx, cz), cx * 16, cz * 16)
    meshes[(cx, cz)] = rend.create_chunk_mesh(
        ov, oi, tv, ti,
        (cx * 16, 0, cz * 16), (cx * 16 + 16, 256, cz * 16 + 16))

W, H = 960, 540
fbo = ctx.simple_framebuffer((W, H))
fbo.use()
ctx.enable(moderngl.DEPTH_TEST)

eye = np.array([30.0, 95.0, 30.0])
target = np.array([-20.0, 60.0, -40.0])
fwd = target - eye
fwd = fwd / np.linalg.norm(fwd)
yaw = math.atan2(fwd[0], -fwd[2])
pitch = math.asin(fwd[1])
forward = forward_from_angles(yaw, pitch)
right = np.cross(forward, [0, 1, 0])
right = right / np.linalg.norm(right)
up = np.cross(right, forward)

proj = perspective(70, W / H, 0.1, 1000)
view = look_at(eye, eye + forward, up)
mvp = proj @ view

sky = SkyState(0.32)
planes = frustum_planes(mvp)
visible = [m for (cx, cz), m in meshes.items()
           if aabb_visible(planes, m.aabb_min, m.aabb_max)]
print(f"visible chunks: {len(visible)} / {len(meshes)}")

fbo.clear(sky.horizon[0], sky.horizon[1], sky.horizon[2], 1.0)
rend.draw_sky((right, up, forward), math.tan(math.radians(70) / 2), W / H, sky, eye, 100.0)
rend.draw_world(visible, mvp, sky, 60, 96 * 1.0, translucent=False)
rend.draw_world(visible, mvp, sky, 60, 96 * 1.0, translucent=True)
rend.draw_outline(mvp, (4, 66, 4))
rend.draw_hud(W, H, B.PLACEABLE[:9], 2)

img = Image.frombytes("RGB", (W, H), fbo.read(components=3)).transpose(Image.FLIP_TOP_BOTTOM)
img.save("render_gl_1.png")
print("OK")
