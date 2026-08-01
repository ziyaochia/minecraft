import math

import moderngl
import numpy as np

from . import blocks as B
from .mat4 import ortho, rotation_x, rotation_y, scale_uniform, to_gl, translation

CHUNK_VS = """
#version 330 core
in vec3 in_pos;
in vec2 in_uv;
in float in_shade;
uniform mat4 mvp;
out vec2 v_uv;
out float v_shade;
out float v_dist;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
    v_shade = in_shade;
    v_dist = gl_Position.w;
}
"""

CHUNK_FS = """
#version 330 core
in vec2 v_uv;
in float v_shade;
in float v_dist;
uniform sampler2D tex;
uniform vec3 fog_color;
uniform float fog_near;
uniform float fog_far;
uniform float brightness;
out vec4 frag;
void main() {
    vec4 c = texture(tex, v_uv);
    if (c.a < 0.35) discard;
    vec3 col = c.rgb * v_shade * brightness;
    float f = clamp((v_dist - fog_near) / (fog_far - fog_near), 0.0, 1.0);
    col = mix(col, fog_color, f);
    frag = vec4(col, c.a);
}
"""

SKY_VS = """
#version 330 core
out vec2 v_ndc;
void main() {
    vec2 p = vec2((gl_VertexID == 1) ? 3.0 : -1.0, (gl_VertexID == 2) ? 3.0 : -1.0);
    v_ndc = p;
    gl_Position = vec4(p, 0.0, 1.0);
}
"""

SKY_COMMON = """
#version 330 core
uniform vec3 sun_dir;
uniform vec3 zenith;
uniform vec3 horizon;
uniform float day_factor;
uniform float time;
uniform vec3 cam_pos;
float hash(vec2 p) { return fract(sin(dot(p, vec2(127.1, 311.7))) * 43758.5453123); }
float vnoise(vec2 p) {
    vec2 i = floor(p);
    vec2 f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash(i), hash(i + vec2(1, 0)), f.x),
               mix(hash(i + vec2(0, 1)), hash(i + vec2(1, 1)), f.x), f.y);
}
vec3 sky_color(vec3 dir) {
    float h = clamp(dir.y, 0.0, 1.0);
    vec3 col = mix(horizon, zenith, pow(h, 0.6));
    if (dir.y < 0.0) col = mix(horizon, horizon * 0.55, clamp(-dir.y * 4.0, 0.0, 1.0));
    float sd = dot(dir, sun_dir);
    float sun_disc = smoothstep(0.99935, 0.99965, sd);
    float glow = pow(clamp(sd, 0.0, 1.0), 32.0) * 0.4 * day_factor;
    col += vec3(1.0, 0.95, 0.8) * sun_disc * max(day_factor, 0.12);
    col += vec3(1.0, 0.7, 0.4) * glow;
    float md = dot(dir, -sun_dir);
    float moon_disc = smoothstep(0.99955, 0.99985, md);
    col += vec3(0.85, 0.88, 0.95) * moon_disc * (1.0 - day_factor) * 0.9;
    if (day_factor < 0.7) {
        vec2 gp = vec2(atan(dir.x, dir.z) * 57.0, dir.y * 110.0);
        vec2 cell = floor(gp);
        float star = step(0.993, hash(cell));
        float tw = 0.6 + 0.4 * sin(time * 2.0 + hash(cell + 7.0) * 20.0);
        col += vec3(star) * tw * (1.0 - day_factor) * smoothstep(0.02, 0.25, dir.y);
    }
    if (dir.y > 0.005) {
        float dist = (130.0 - cam_pos.y) / dir.y;
        if (dist > 0.0 && dist < 2500.0) {
            vec2 cuv = (cam_pos.xz + dir.xz * dist) * 0.045 + vec2(time * 0.008, 0.0);
            float c = vnoise(cuv) * 0.65 + vnoise(cuv * 2.7 + 13.1) * 0.35;
            float cl = smoothstep(0.58, 0.74, c);
            float fade = 1.0 - clamp(dist / 1800.0, 0.0, 1.0);
            vec3 cloud_col = mix(vec3(0.07, 0.08, 0.11), vec3(1.02), day_factor);
            float ca = cl * 0.8 * fade * max(day_factor, 0.3);
            col = mix(col, cloud_col, ca);
        }
    }
    return col;
}
"""

SKY_FS = SKY_COMMON + """
in vec2 v_ndc;
out vec4 frag;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform vec3 cam_forward;
uniform float tan_half_fov;
uniform float aspect;
void main() {
    vec3 dir = normalize(cam_forward + v_ndc.x * tan_half_fov * aspect * cam_right
                         + v_ndc.y * tan_half_fov * cam_up);
    frag = vec4(sky_color(dir), 1.0);
}
"""

SOLID_VS = """
#version 330 core
in vec2 in_pos;
uniform mat4 mvp;
void main() { gl_Position = mvp * vec4(in_pos, 0.0, 1.0); }
"""

SOLID_FS = """
#version 330 core
uniform vec4 color;
out vec4 frag;
void main() { frag = color; }
"""

ICON_VS = """
#version 330 core
in vec3 in_pos;
in vec2 in_uv;
in float in_shade;
uniform mat4 mvp;
out vec2 v_uv;
out float v_shade;
void main() {
    gl_Position = mvp * vec4(in_pos, 1.0);
    v_uv = in_uv;
    v_shade = in_shade;
}
"""

ICON_FS = """
#version 330 core
in vec2 v_uv;
in float v_shade;
uniform sampler2D tex;
out vec4 frag;
void main() {
    vec4 c = texture(tex, v_uv);
    if (c.a < 0.35) discard;
    frag = vec4(c.rgb * v_shade, 1.0);
}
"""

LINE_CUBE = np.array([
    (0, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 1), (1, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 0),
    (0, 1, 0), (1, 1, 0), (1, 1, 0), (1, 1, 1), (1, 1, 1), (0, 1, 1), (0, 1, 1), (0, 1, 0),
    (0, 0, 0), (0, 1, 0), (1, 0, 0), (1, 1, 0), (1, 0, 1), (1, 1, 1), (0, 0, 1), (0, 1, 1),
], dtype=np.float32)

ICON_FACES = [
    ((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)], [(0, 0), (1, 0), (1, 1), (0, 1)], 1.0),
    ((0, 0, 1), [(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)], [(0, 0), (0, 1), (1, 1), (1, 0)], 0.8),
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [(0, 0), (0, 1), (1, 1), (1, 0)], 0.6),
]


class ChunkMesh:
    __slots__ = ("vao_o", "n_o", "vao_t", "n_t", "vbo_o", "ibo_o", "vbo_t", "ibo_t", "aabb_min", "aabb_max")

    def release(self):
        for obj in (self.vao_o, self.vbo_o, self.ibo_o, self.vao_t, self.vbo_t, self.ibo_t):
            if obj is not None:
                obj.release()


class GLRenderer:
    def __init__(self, ctx, atlas):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=CHUNK_VS, fragment_shader=CHUNK_FS)
        self.sky_prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=SKY_FS)
        self.solid_prog = ctx.program(vertex_shader=SOLID_VS, fragment_shader=SOLID_FS)
        self.icon_prog = ctx.program(vertex_shader=ICON_VS, fragment_shader=ICON_FS)
        self.tex = ctx.texture((atlas.shape[1], atlas.shape[0]), 4, atlas.tobytes(), dtype="f1")
        self.tex.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.tex.anisotropy = 1.0
        self.prog["tex"].value = 0
        self.icon_prog["tex"].value = 0
        self.sky_vao = ctx.vertex_array(self.sky_prog, [])
        self.outline_vbo = ctx.buffer(LINE_CUBE.tobytes())
        self.outline_vao = ctx.vertex_array(self.solid_prog, [(self.outline_vbo, "3f", "in_pos")])
        self.quad_vbo = ctx.buffer(reserve=6 * 2 * 4)
        self.quad_vao = ctx.vertex_array(self.solid_prog, [(self.quad_vbo, "2f", "in_pos")])
        self.icon_vaos = {}
        self._build_icons()

    def _build_icons(self):
        from .textures import tile_uv
        for bid in B.PLACEABLE:
            blk = B.BLOCKS[bid]
            verts = []
            idx = []
            for normal, corners, uvs, shade in ICON_FACES:
                if normal[1] > 0:
                    name = blk.tex_top
                elif normal[1] < 0:
                    name = blk.tex_bottom
                else:
                    name = blk.tex_side
                u0, v0, u1, v1 = tile_uv(name)
                base = len(verts)
                for (cx, cy, cz), (cu, cv) in zip(corners, uvs):
                    verts.append((cx - 0.5, cy - 0.5, cz - 0.5,
                                  u0 + cu * (u1 - u0), v0 + cv * (v1 - v0), shade))
                idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
            vbo = self.ctx.buffer(np.array(verts, dtype=np.float32).tobytes())
            ibo = self.ctx.buffer(np.array(idx, dtype=np.uint32).tobytes())
            vao = self.ctx.vertex_array(self.icon_prog, [(vbo, "3f 2f 1f", "in_pos", "in_uv", "in_shade")], ibo)
            self.icon_vaos[bid] = vao

    def create_chunk_mesh(self, ov, oi, tv, ti, aabb_min, aabb_max):
        m = ChunkMesh()
        m.aabb_min = aabb_min
        m.aabb_max = aabb_max
        m.vbo_o = m.ibo_o = m.vao_o = None
        m.vbo_t = m.ibo_t = m.vao_t = None
        m.n_o = len(oi)
        m.n_t = len(ti)
        if m.n_o:
            m.vbo_o = self.ctx.buffer(ov.tobytes())
            m.ibo_o = self.ctx.buffer(oi.tobytes())
            m.vao_o = self.ctx.vertex_array(self.prog, [(m.vbo_o, "3f 2f 1f", "in_pos", "in_uv", "in_shade")], m.ibo_o)
        if m.n_t:
            m.vbo_t = self.ctx.buffer(tv.tobytes())
            m.ibo_t = self.ctx.buffer(ti.tobytes())
            m.vao_t = self.ctx.vertex_array(self.prog, [(m.vbo_t, "3f 2f 1f", "in_pos", "in_uv", "in_shade")], m.ibo_t)
        return m

    def draw_sky(self, view_dirs, tan_half_fov, aspect, sky, cam_pos, time_s):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        right, up, forward = view_dirs
        p = self.sky_prog
        p["cam_right"].value = tuple(right)
        p["cam_up"].value = tuple(up)
        p["cam_forward"].value = tuple(forward)
        p["tan_half_fov"].value = tan_half_fov
        p["aspect"].value = aspect
        p["sun_dir"].value = tuple(sky.sun_dir)
        p["zenith"].value = tuple(sky.zenith)
        p["horizon"].value = tuple(sky.horizon)
        p["day_factor"].value = sky.day_factor
        p["time"].value = time_s
        p["cam_pos"].value = tuple(cam_pos)
        self.sky_vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)

    def draw_world(self, meshes, mvp, sky, fog_near, fog_far, translucent=False):
        p = self.prog
        p["mvp"].write(to_gl(mvp))
        p["fog_color"].value = tuple(sky.horizon)
        p["fog_near"].value = fog_near
        p["fog_far"].value = fog_far
        p["brightness"].value = sky.brightness
        self.tex.use(0)
        ctx = self.ctx
        if translucent:
            ctx.enable(moderngl.BLEND)
            ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        for m in meshes:
            if translucent:
                if m.vao_t is not None:
                    m.vao_t.render(moderngl.TRIANGLES)
            else:
                if m.vao_o is not None:
                    m.vao_o.render(moderngl.TRIANGLES)
        if translucent:
            ctx.disable(moderngl.BLEND)

    def draw_outline(self, mvp, block_pos):
        if block_pos is None:
            return
        m = mvp @ translation(block_pos[0] - 0.002, block_pos[1] - 0.002, block_pos[2] - 0.002) @ scale_uniform(1.004)
        self.solid_prog["mvp"].write(to_gl(m))
        self.solid_prog["color"].value = (0.0, 0.0, 0.0, 0.85)
        self.ctx.line_width = 2.0
        self.outline_vao.render(moderngl.LINES)
        self.ctx.line_width = 1.0

    def _draw_rect(self, ortho_m, x0, y0, x1, y1, color):
        verts = np.array([
            (x0, y0), (x1, y0), (x1, y1),
            (x0, y0), (x1, y1), (x0, y1),
        ], dtype=np.float32)
        self.quad_vbo.write(verts.tobytes())
        self.solid_prog["mvp"].write(to_gl(ortho_m))
        self.solid_prog["color"].value = color
        self.quad_vao.render(moderngl.TRIANGLES, vertices=6)

    def draw_hud(self, width, height, hotbar, selected):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        ctx.enable(moderngl.BLEND)
        ctx.blend_func = (moderngl.SRC_ALPHA, moderngl.ONE_MINUS_SRC_ALPHA)
        om = ortho(0, width, 0, height, -1, 1)
        cx, cy = width / 2.0, height / 2.0
        s = 9.0
        t = 1.2
        self._draw_rect(om, cx - s, cy - t, cx + s, cy + t, (1, 1, 1, 0.75))
        self._draw_rect(om, cx - t, cy - s, cx + t, cy + s, (1, 1, 1, 0.75))
        slot = 46.0
        pad = 4.0
        total = 9 * slot
        x0 = cx - total / 2.0
        y0 = 12.0
        for i in range(9):
            sx = x0 + i * slot
            self._draw_rect(om, sx, y0, sx + slot - pad, y0 + slot - pad, (0.08, 0.08, 0.09, 0.55))
        hx = x0 + selected * slot
        w = 3.0
        self._draw_rect(om, hx - w, y0 - w, hx + slot - pad + w, y0 + slot - pad + w, (1, 1, 1, 0.9))
        self._draw_rect(om, hx, y0, hx + slot - pad, y0 + slot - pad, (0.35, 0.35, 0.36, 0.6))
        ctx.disable(moderngl.BLEND)
        self.tex.use(0)
        size = 26.0
        for i, bid in enumerate(hotbar):
            if bid == B.AIR:
                continue
            ix = x0 + i * slot + (slot - pad) / 2.0
            iy = y0 + (slot - pad) / 2.0
            m = (ortho(0, width, 0, height, -100, 100)
                 @ translation(ix, iy, 0)
                 @ scale_uniform(size)
                 @ rotation_x(math.radians(-28))
                 @ rotation_y(math.radians(45)))
            self.icon_prog["mvp"].write(to_gl(m))
            self.icon_vaos[bid].render(moderngl.TRIANGLES)
