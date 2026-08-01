import moderngl
import numpy as np

from . import blocks as B
from . import config as C
from .render_gl import SKY_COMMON, SKY_VS
from .textures import TILE_INDEX, build_palette

VOL_R = 5

RT_FS = SKY_COMMON + """
in vec2 v_ndc;
out vec4 frag;
uniform vec3 cam_right;
uniform vec3 cam_up;
uniform vec3 cam_forward;
uniform float tan_half_fov;
uniform float aspect;
uniform usampler3D voxels;
uniform sampler2D palette;
uniform ivec3 vol_origin;
uniform ivec3 vol_size;
uniform float fog_near;
uniform float fog_far;
uniform float brightness;

const uint WATER = 5u;
const uint GLASS_B = 10u;

uint block_at(ivec3 p) {
    ivec3 v = p - vol_origin;
    if (any(lessThan(v, ivec3(0))) || any(greaterThanEqual(v, vol_size))) return 0u;
    return texelFetch(voxels, v, 0).x;
}

bool opaque(uint b) { return b != 0u && b != WATER && b != GLASS_B; }

float shadow_factor(vec3 hp) {
    vec3 ro = hp + sun_dir * 1e-3;
    ivec3 p = ivec3(floor(ro));
    ivec3 st = ivec3(sign(sun_dir));
    vec3 inv = 1.0 / sun_dir;
    vec3 t_delta = abs(inv);
    vec3 t_max3 = (vec3(p) + max(vec3(st), vec3(0.0)) - ro) * inv;
    float t = 0.0;
    for (int i = 0; i < 320; i++) {
        if (t_max3.x < t_max3.y && t_max3.x < t_max3.z) {
            p.x += st.x; t = t_max3.x; t_max3.x += t_delta.x;
        } else if (t_max3.y < t_max3.z) {
            p.y += st.y; t = t_max3.y; t_max3.y += t_delta.y;
        } else {
            p.z += st.z; t = t_max3.z; t_max3.z += t_delta.z;
        }
        if (t > 300.0) break;
        if (opaque(block_at(p))) return 0.35;
    }
    return 1.0;
}

bool march_hit(vec3 ro, vec3 rd, out ivec3 cell, out vec3 normal, out float t) {
    ivec3 p = ivec3(floor(ro));
    ivec3 st = ivec3(sign(rd));
    vec3 inv = 1.0 / rd;
    vec3 t_delta = abs(inv);
    vec3 t_max3 = (vec3(p) + max(vec3(st), vec3(0.0)) - ro) * inv;
    t = 0.0;
    normal = vec3(0.0, 1.0, 0.0);
    for (int i = 0; i < 384; i++) {
        if (t_max3.x < t_max3.y && t_max3.x < t_max3.z) {
            p.x += st.x; t = t_max3.x; t_max3.x += t_delta.x; normal = vec3(-st.x, 0.0, 0.0);
        } else if (t_max3.y < t_max3.z) {
            p.y += st.y; t = t_max3.y; t_max3.y += t_delta.y; normal = vec3(0.0, -st.y, 0.0);
        } else {
            p.z += st.z; t = t_max3.z; t_max3.z += t_delta.z; normal = vec3(0.0, 0.0, -st.z);
        }
        if (t > 320.0) return false;
        if (opaque(block_at(p))) { cell = p; return true; }
    }
    return false;
}

vec3 shade_hit(ivec3 cell, vec3 normal, uint bid, vec3 hp, float t) {
    int row = normal.y > 0.5 ? 0 : (normal.y < -0.5 ? 2 : 1);
    vec3 base = texelFetch(palette, ivec2(int(bid), row), 0).rgb;
    float fs = normal.y > 0.5 ? 1.0 : (normal.y < -0.5 ? 0.5 :
                                        (abs(normal.x) > 0.5 ? 0.6 : 0.8));
    float lit = day_factor > 0.02 ? shadow_factor(hp + normal * 2e-3) : 1.0;
    vec3 col = base * fs * brightness * lit;
    float f = clamp((t - fog_near) / (fog_far - fog_near), 0.0, 1.0);
    return mix(col, horizon, f);
}

void main() {
    vec3 rd = normalize(cam_forward + v_ndc.x * tan_half_fov * aspect * cam_right
                        + v_ndc.y * tan_half_fov * cam_up);
    vec3 ro = cam_pos;
    vec3 inv = 1.0 / rd;
    vec3 t0 = (vec3(vol_origin) - ro) * inv;
    vec3 t1 = (vec3(vol_origin + vol_size) - ro) * inv;
    vec3 tsm = min(t0, t1);
    vec3 tbg = max(t0, t1);
    float t_enter = max(max(tsm.x, tsm.y), tsm.z);
    float t_exit = min(min(tbg.x, tbg.y), tbg.z);
    if (t_exit <= max(t_enter, 0.0)) { frag = vec4(sky_color(rd), 1.0); return; }
    float t_start = max(t_enter, 0.0);
    vec3 p_start = ro + rd * t_start;
    ivec3 p = clamp(ivec3(floor(p_start)), vol_origin, vol_origin + vol_size - 1);
    ivec3 st = ivec3(sign(rd));
    vec3 t_delta = abs(inv);
    vec3 t_max3 = (vec3(p) + max(vec3(st), vec3(0.0)) - p_start) * inv + t_start;
    float t = t_start;
    vec3 tint = vec3(1.0);
    vec3 refl_col = vec3(0.0);
    float refl_f = 0.0;
    for (int i = 0; i < 700; i++) {
        vec3 normal;
        if (t_max3.x < t_max3.y && t_max3.x < t_max3.z) {
            p.x += st.x; t = t_max3.x; t_max3.x += t_delta.x; normal = vec3(-st.x, 0.0, 0.0);
        } else if (t_max3.y < t_max3.z) {
            p.y += st.y; t = t_max3.y; t_max3.y += t_delta.y; normal = vec3(0.0, -st.y, 0.0);
        } else {
            p.z += st.z; t = t_max3.z; t_max3.z += t_delta.z; normal = vec3(0.0, 0.0, -st.z);
        }
        if (t > t_exit) break;
        uint b = block_at(p);
        if (b == 0u) continue;
        if (b == WATER || b == GLASS_B) {
            vec3 wc = texelFetch(palette, ivec2(int(b), 1), 0).rgb;
            if (b == WATER && refl_f == 0.0 && normal.y > 0.5) {
                vec3 hp = ro + rd * t;
                vec3 rd2 = reflect(rd, normal);
                ivec3 c2; vec3 n2; float t2;
                if (march_hit(hp + normal * 1e-3, rd2, c2, n2, t2))
                    refl_col = shade_hit(c2, n2, block_at(c2),
                                         hp + normal * 1e-3 + rd2 * t2, t2);
                else
                    refl_col = sky_color(rd2);
                refl_f = 0.25 + 0.55 * pow(1.0 - max(dot(-rd, normal), 0.0), 2.0);
            }
            float k = b == WATER ? 0.35 : 0.10;
            tint *= mix(vec3(1.0), wc, k);
            continue;
        }
        vec3 hp = ro + rd * t;
        vec3 col = shade_hit(p, normal, b, hp, t) * tint;
        if (refl_f > 0.0) col = mix(col, refl_col, refl_f);
        frag = vec4(col, 1.0);
        return;
    }
    vec3 col = sky_color(rd);
    if (refl_f > 0.0) col = mix(col, refl_col, refl_f);
    frag = vec4(col, 1.0);
}
"""


class RTRenderer:
    def __init__(self, ctx, atlas):
        self.ctx = ctx
        self.prog = ctx.program(vertex_shader=SKY_VS, fragment_shader=RT_FS)
        self.side = VOL_R * 2 + 1
        self.size = self.side * C.CHUNK_SIZE
        self.voxels = ctx.texture3d((self.size, C.WORLD_HEIGHT, self.size),
                                    1, dtype="u1")
        self.voxels.filter = (moderngl.NEAREST, moderngl.NEAREST)
        pal = build_palette(atlas)
        data = np.zeros((3, 256, 4), dtype=np.uint8)
        for bid, block in B.BLOCKS.items():
            data[0, bid] = pal[TILE_INDEX[block.tex_top]]
            data[1, bid] = pal[TILE_INDEX[block.tex_side]]
            data[2, bid] = pal[TILE_INDEX[block.tex_bottom]]
        self.palette = ctx.texture((256, 3), 4, data.tobytes(), dtype="f1")
        self.palette.filter = (moderngl.NEAREST, moderngl.NEAREST)
        self.prog["voxels"].value = 0
        self.prog["palette"].value = 1
        self.vao = ctx.vertex_array(self.prog, [])
        self.vol_origin = None

    def rebuild_volume(self, world, pcx, pcz):
        S = C.CHUNK_SIZE
        ox, oz = pcx - VOL_R, pcz - VOL_R
        self.vol_origin = (ox * S, 0, oz * S)
        arr = np.zeros((self.size, C.WORLD_HEIGHT, self.size), dtype=np.uint8)
        for (cx, cz), chunk in world.chunks.items():
            lx, lz = cx - ox, cz - oz
            if 0 <= lx < self.side and 0 <= lz < self.side:
                arr[lz * S:(lz + 1) * S, :, lx * S:(lx + 1) * S] = \
                    chunk.blocks.transpose(2, 1, 0)
        self.voxels.write(arr.tobytes())

    def upload_chunk(self, world, cx, cz):
        if self.vol_origin is None:
            return
        S = C.CHUNK_SIZE
        lx = cx - self.vol_origin[0] // S
        lz = cz - self.vol_origin[2] // S
        chunk = world.chunks.get((cx, cz))
        if chunk is None or not (0 <= lx < self.side and 0 <= lz < self.side):
            return
        slab = np.ascontiguousarray(chunk.blocks.transpose(2, 1, 0))
        self.voxels.write(slab.tobytes(),
                          viewport=(lx * S, 0, lz * S, S, C.WORLD_HEIGHT, S))

    def draw(self, view_dirs, tan_half_fov, aspect, sky, cam_pos,
             fog_near, fog_far, time_s):
        ctx = self.ctx
        ctx.disable(moderngl.DEPTH_TEST)
        right, up, forward = view_dirs
        p = self.prog
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
        p["fog_near"].value = fog_near
        p["fog_far"].value = fog_far
        p["brightness"].value = sky.brightness
        p["vol_origin"].value = self.vol_origin
        p["vol_size"].value = (self.size, C.WORLD_HEIGHT, self.size)
        self.voxels.use(0)
        self.palette.use(1)
        self.vao.render(moderngl.TRIANGLES, vertices=3)
        ctx.enable(moderngl.DEPTH_TEST)
