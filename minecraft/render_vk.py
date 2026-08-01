import math

import numpy as np
import wgpu
from rendercanvas.glfw import get_glfw_present_info

from . import blocks as B
from .mat4 import (ortho, rotation_x, rotation_y, scale_uniform, to_gl,
                   translation)
from .render_gl import ICON_FACES, LINE_CUBE

VK_FIX = np.array([
    [1.0, 0.0, 0.0, 0.0],
    [0.0, 1.0, 0.0, 0.0],
    [0.0, 0.0, 0.5, 0.5],
    [0.0, 0.0, 0.0, 1.0],
])

CHUNK_WGSL = """
struct Params {
    mvp: mat4x4f,
    fog_color: vec4f,
    scalars: vec4f,
};
@group(0) @binding(0) var<uniform> params: Params;
@group(1) @binding(0) var tex: texture_2d<f32>;
@group(1) @binding(1) var tex_sampler: sampler;

struct Vin {
    @location(0) pos: vec3f,
    @location(1) uv: vec2f,
    @location(2) shade: f32,
};
struct Vout {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) shade: f32,
    @location(2) dist: f32,
};
@vertex
fn vs_main(v: Vin) -> Vout {
    var o: Vout;
    o.clip = params.mvp * vec4f(v.pos, 1.0);
    o.uv = v.uv;
    o.shade = v.shade;
    o.dist = o.clip.w;
    return o;
}
@fragment
fn fs_main(i: Vout) -> @location(0) vec4f {
    let c = textureSample(tex, tex_sampler, i.uv);
    if (c.a < 0.35) { discard; }
    var col = c.rgb * i.shade * params.scalars.z;
    let f = clamp((i.dist - params.scalars.x) /
                  (params.scalars.y - params.scalars.x), 0.0, 1.0);
    col = mix(col, params.fog_color.rgb, f);
    return vec4f(col, c.a);
}
"""

SKY_WGSL = """
struct SkyU {
    cam_right: vec4f,
    cam_up: vec4f,
    cam_forward: vec4f,
    sun_dir: vec4f,
    zenith: vec4f,
    horizon: vec4f,
    cam_pos: vec4f,
    scalars: vec4f,
};
@group(0) @binding(0) var<uniform> sky: SkyU;

struct Vout {
    @builtin(position) pos: vec4f,
    @location(0) ndc: vec2f,
};

fn hash2(p: vec2f) -> f32 {
    return fract(sin(dot(p, vec2f(127.1, 311.7))) * 43758.5453123);
}

fn vnoise(p: vec2f) -> f32 {
    let i = floor(p);
    var f = fract(p);
    f = f * f * (3.0 - 2.0 * f);
    return mix(mix(hash2(i), hash2(i + vec2f(1.0, 0.0)), f.x),
               mix(hash2(i + vec2f(0.0, 1.0)), hash2(i + vec2f(1.0, 1.0)), f.x),
               f.y);
}

fn sky_color(dir: vec3f) -> vec3f {
    let sun_dir = sky.sun_dir.xyz;
    let day_factor = sky.scalars.z;
    let time = sky.scalars.w;
    let cam_pos = sky.cam_pos.xyz;
    let h = clamp(dir.y, 0.0, 1.0);
    var col = mix(sky.horizon.rgb, sky.zenith.rgb, pow(h, 0.6));
    if (dir.y < 0.0) {
        col = mix(sky.horizon.rgb, sky.horizon.rgb * 0.55,
                  clamp(-dir.y * 4.0, 0.0, 1.0));
    }
    let sd = dot(dir, sun_dir);
    let sun_disc = smoothstep(0.99935, 0.99965, sd);
    let glow = pow(clamp(sd, 0.0, 1.0), 32.0) * 0.4 * day_factor;
    col += vec3f(1.0, 0.95, 0.8) * sun_disc * max(day_factor, 0.12);
    col += vec3f(1.0, 0.7, 0.4) * glow;
    let md = dot(dir, -sun_dir);
    let moon_disc = smoothstep(0.99955, 0.99985, md);
    col += vec3f(0.85, 0.88, 0.95) * moon_disc * (1.0 - day_factor) * 0.9;
    if (day_factor < 0.7) {
        let gp = vec2f(atan2(dir.x, dir.z) * 57.0, dir.y * 110.0);
        let cell = floor(gp);
        let star = step(0.993, hash2(cell));
        let tw = 0.6 + 0.4 * sin(time * 2.0 + hash2(cell + vec2f(7.0)) * 20.0);
        col += vec3f(star) * tw * (1.0 - day_factor) *
               smoothstep(0.02, 0.25, dir.y);
    }
    if (dir.y > 0.005) {
        let dist = (130.0 - cam_pos.y) / dir.y;
        if (dist > 0.0 && dist < 2500.0) {
            let cuv = (cam_pos.xz + dir.xz * dist) * 0.045 +
                      vec2f(time * 0.008, 0.0);
            let c = vnoise(cuv) * 0.65 + vnoise(cuv * 2.7 + vec2f(13.1)) * 0.35;
            let cl = smoothstep(0.58, 0.74, c);
            let fade = 1.0 - clamp(dist / 1800.0, 0.0, 1.0);
            let cloud_col = mix(vec3f(0.07, 0.08, 0.11), vec3f(1.02), day_factor);
            let ca = cl * 0.8 * fade * max(day_factor, 0.3);
            col = mix(col, cloud_col, ca);
        }
    }
    return col;
}

@vertex
fn vs_main(@builtin(vertex_index) vi: u32) -> Vout {
    var o: Vout;
    let p = vec2f(f32((vi << 1u) & 2u), f32(vi & 2u)) * 2.0 - 1.0;
    o.pos = vec4f(p, 0.0, 1.0);
    o.ndc = p;
    return o;
}

@fragment
fn fs_main(o: Vout) -> @location(0) vec4f {
    let dir = normalize(sky.cam_forward.xyz +
        o.ndc.x * sky.scalars.x * sky.scalars.y * sky.cam_right.xyz +
        o.ndc.y * sky.scalars.x * sky.cam_up.xyz);
    return vec4f(sky_color(dir), 1.0);
}
"""

SOLID_WGSL = """
struct Solid {
    mvp: mat4x4f,
    color: vec4f,
};
@group(0) @binding(0) var<uniform> s: Solid;
@vertex
fn vs_main(@location(0) pos: vec3f) -> @builtin(position) vec4f {
    return s.mvp * vec4f(pos, 1.0);
}
@fragment
fn fs_main() -> @location(0) vec4f {
    return s.color;
}
"""

RECT_WGSL = """
struct Vin {
    @location(0) pos: vec2f,
    @location(1) color: vec4f,
};
struct Vout {
    @builtin(position) clip: vec4f,
    @location(0) color: vec4f,
};
@vertex
fn vs_main(v: Vin) -> Vout {
    var o: Vout;
    o.clip = vec4f(v.pos, 0.5, 1.0);
    o.color = v.color;
    return o;
}
@fragment
fn fs_main(o: Vout) -> @location(0) vec4f {
    return o.color;
}
"""

ICON_WGSL = """
@group(0) @binding(0) var tex: texture_2d<f32>;
@group(0) @binding(1) var tex_sampler: sampler;
struct Vin {
    @location(0) pos: vec4f,
    @location(1) uv: vec2f,
    @location(2) shade: f32,
};
struct Vout {
    @builtin(position) clip: vec4f,
    @location(0) uv: vec2f,
    @location(1) shade: f32,
};
@vertex
fn vs_main(v: Vin) -> Vout {
    var o: Vout;
    o.clip = v.pos;
    o.uv = v.uv;
    o.shade = v.shade;
    return o;
}
@fragment
fn fs_main(o: Vout) -> @location(0) vec4f {
    let c = textureSample(tex, tex_sampler, o.uv);
    if (c.a < 0.35) { discard; }
    return vec4f(c.rgb * o.shade, 1.0);
}
"""

BLEND_ALPHA = {
    "color": {"src_factor": "src-alpha", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
    "alpha": {"src_factor": "one", "dst_factor": "one-minus-src-alpha",
              "operation": "add"},
}


class VKMesh:
    __slots__ = ("vbo_o", "ibo_o", "n_o", "vbo_t", "ibo_t", "n_t",
                 "aabb_min", "aabb_max")

    def release(self):
        for b in (self.vbo_o, self.ibo_o, self.vbo_t, self.ibo_t):
            if b is not None:
                b.destroy()


class VKRenderer:
    def __init__(self, win, atlas):
        import glfw
        self.adapter = wgpu.gpu.request_adapter_sync(
            power_preference="high-performance")
        self.device = self.adapter.request_device_sync()
        self.context = wgpu.gpu.get_canvas_context(get_glfw_present_info(win))
        w, h = glfw.get_framebuffer_size(win)
        self.context.set_physical_size(w, h)
        usage = wgpu.TextureUsage.RENDER_ATTACHMENT | wgpu.TextureUsage.COPY_SRC
        fmt = self._pick_format()
        self.context.configure(device=self.device, format=fmt, usage=usage)
        self.surface_format = fmt
        self.queue = self.device.queue
        self.shot_path = None
        self._depth_size = None
        self._depth_view = None
        self._build_tex(atlas)
        self._build_pipelines(fmt)
        self._build_buffers()

    def _pick_format(self):
        preferred = self.context.get_preferred_format(self.adapter)
        try:
            formats = self.context._get_capabilities(self.adapter)["formats"]
            for f in formats:
                if "srgb" not in f:
                    return f
        except Exception:
            pass
        return preferred

    def _build_tex(self, atlas):
        h, w = atlas.shape[:2]
        self.tex = self.device.create_texture(
            size=(w, h, 1), format="rgba8unorm",
            usage=wgpu.TextureUsage.TEXTURE_BINDING | wgpu.TextureUsage.COPY_DST)
        self.queue.write_texture({"texture": self.tex}, atlas.tobytes(),
                                 {"bytes_per_row": w * 4, "rows_per_image": h},
                                 (w, h, 1))
        self.tex_view = self.tex.create_view()
        self.sampler = self.device.create_sampler()

    def _build_pipelines(self, fmt):
        d = self.device
        chunk_mod = d.create_shader_module(code=CHUNK_WGSL)
        sky_mod = d.create_shader_module(code=SKY_WGSL)
        solid_mod = d.create_shader_module(code=SOLID_WGSL)
        rect_mod = d.create_shader_module(code=RECT_WGSL)
        icon_mod = d.create_shader_module(code=ICON_WGSL)
        chunk_layout = [{"array_stride": 24, "step_mode": "vertex",
                         "attributes": [
                             {"format": "float32x3", "offset": 0,
                              "shader_location": 0},
                             {"format": "float32x2", "offset": 12,
                              "shader_location": 1},
                             {"format": "float32", "offset": 20,
                              "shader_location": 2}]}]
        self.pipe_chunk = d.create_render_pipeline(
            layout="auto",
            vertex={"module": chunk_mod, "entry_point": "vs_main",
                    "buffers": chunk_layout},
            fragment={"module": chunk_mod, "entry_point": "fs_main",
                      "targets": [{"format": fmt, "blend": BLEND_ALPHA}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus",
                           "depth_write_enabled": True,
                           "depth_compare": "less"})
        self.pipe_sky = d.create_render_pipeline(
            layout="auto",
            vertex={"module": sky_mod, "entry_point": "vs_main", "buffers": []},
            fragment={"module": sky_mod, "entry_point": "fs_main",
                      "targets": [{"format": fmt}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus",
                           "depth_write_enabled": False,
                           "depth_compare": "always"})
        self.pipe_solid = d.create_render_pipeline(
            layout="auto",
            vertex={"module": solid_mod, "entry_point": "vs_main",
                    "buffers": [{"array_stride": 12, "step_mode": "vertex",
                                 "attributes": [
                                     {"format": "float32x3", "offset": 0,
                                      "shader_location": 0}]}]},
            fragment={"module": solid_mod, "entry_point": "fs_main",
                      "targets": [{"format": fmt}]},
            primitive={"topology": "line-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus",
                           "depth_write_enabled": False,
                           "depth_compare": "less"})
        self.pipe_rect = d.create_render_pipeline(
            layout="auto",
            vertex={"module": rect_mod, "entry_point": "vs_main",
                    "buffers": [{"array_stride": 24, "step_mode": "vertex",
                                 "attributes": [
                                     {"format": "float32x2", "offset": 0,
                                      "shader_location": 0},
                                     {"format": "float32x4", "offset": 8,
                                      "shader_location": 1}]}]},
            fragment={"module": rect_mod, "entry_point": "fs_main",
                      "targets": [{"format": fmt, "blend": BLEND_ALPHA}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus",
                           "depth_write_enabled": False,
                           "depth_compare": "always"})
        self.pipe_icon = d.create_render_pipeline(
            layout="auto",
            vertex={"module": icon_mod, "entry_point": "vs_main",
                    "buffers": [{"array_stride": 28, "step_mode": "vertex",
                                 "attributes": [
                                     {"format": "float32x4", "offset": 0,
                                      "shader_location": 0},
                                     {"format": "float32x2", "offset": 16,
                                      "shader_location": 1},
                                     {"format": "float32", "offset": 24,
                                      "shader_location": 2}]}]},
            fragment={"module": icon_mod, "entry_point": "fs_main",
                      "targets": [{"format": fmt}]},
            primitive={"topology": "triangle-list", "cull_mode": "none"},
            depth_stencil={"format": "depth24plus",
                           "depth_write_enabled": False,
                           "depth_compare": "always"})

    def _build_buffers(self):
        d = self.device
        U = wgpu.BufferUsage.UNIFORM | wgpu.BufferUsage.COPY_DST
        self.chunk_ubo = d.create_buffer(size=96, usage=U)
        self.sky_ubo = d.create_buffer(size=128, usage=U)
        self.solid_ubo = d.create_buffer(size=80, usage=U)
        V = wgpu.BufferUsage.VERTEX
        VD = V | wgpu.BufferUsage.COPY_DST
        self.outline_vbo = d.create_buffer_with_data(
            data=LINE_CUBE.tobytes(), usage=V)
        self.rect_vbo = d.create_buffer(size=2048, usage=VD)
        self.icon_vbo = d.create_buffer(size=4608, usage=VD)
        self.bg_chunk_params = d.create_bind_group(
            layout=self.pipe_chunk.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {
                "buffer": self.chunk_ubo, "offset": 0, "size": 96}}])
        self.bg_chunk_tex = d.create_bind_group(
            layout=self.pipe_chunk.get_bind_group_layout(1),
            entries=[{"binding": 0, "resource": self.tex_view},
                     {"binding": 1, "resource": self.sampler}])
        self.bg_sky = d.create_bind_group(
            layout=self.pipe_sky.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {
                "buffer": self.sky_ubo, "offset": 0, "size": 128}}])
        self.bg_solid = d.create_bind_group(
            layout=self.pipe_solid.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": {
                "buffer": self.solid_ubo, "offset": 0, "size": 80}}])
        self.bg_icon_tex = d.create_bind_group(
            layout=self.pipe_icon.get_bind_group_layout(0),
            entries=[{"binding": 0, "resource": self.tex_view},
                     {"binding": 1, "resource": self.sampler}])
        self._build_icon_geometry()

    def _build_icon_geometry(self):
        from .textures import tile_uv
        self.icon_geo = {}
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
                                  u0 + cu * (u1 - u0), v0 + cv * (v1 - v0),
                                  shade))
                idx.extend([base, base + 1, base + 2, base, base + 2, base + 3])
            v = np.array(verts, dtype=np.float32)
            i = np.array(idx, dtype=np.int64)
            self.icon_geo[bid] = (v[i][:, :3], v[i][:, 3:5], v[i][:, 5])

    def create_chunk_mesh(self, ov, oi, tv, ti, aabb_min, aabb_max):
        m = VKMesh()
        m.aabb_min = aabb_min
        m.aabb_max = aabb_max
        m.vbo_o = m.ibo_o = m.vbo_t = m.ibo_t = None
        m.n_o = len(oi)
        m.n_t = len(ti)
        if m.n_o:
            m.vbo_o = self.device.create_buffer_with_data(
                data=ov.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            m.ibo_o = self.device.create_buffer_with_data(
                data=oi.tobytes(), usage=wgpu.BufferUsage.INDEX)
        if m.n_t:
            m.vbo_t = self.device.create_buffer_with_data(
                data=tv.tobytes(), usage=wgpu.BufferUsage.VERTEX)
            m.ibo_t = self.device.create_buffer_with_data(
                data=ti.tobytes(), usage=wgpu.BufferUsage.INDEX)
        return m

    def begin_frame(self, clear_rgb, w, h):
        self.context.set_physical_size(w, h)
        if self._depth_size != (w, h):
            if hasattr(self, "_depth_tex"):
                self._depth_tex.destroy()
            self._depth_tex = self.device.create_texture(
                size=(w, h, 1), format="depth24plus",
                usage=wgpu.TextureUsage.RENDER_ATTACHMENT)
            self._depth_view = self._depth_tex.create_view()
            self._depth_size = (w, h)
        self.cur_tex = self.context.get_current_texture()
        self._fb = (w, h)
        self._enc = self.device.create_command_encoder()
        self._pass = self._enc.begin_render_pass(
            color_attachments=[{
                "view": self.cur_tex.create_view(),
                "load_op": "clear", "store_op": "store",
                "clear_value": (clear_rgb[0], clear_rgb[1], clear_rgb[2], 1.0)}],
            depth_stencil_attachment={
                "view": self._depth_view,
                "depth_clear_value": 1.0,
                "depth_load_op": "clear",
                "depth_store_op": "store"})

    def end_frame(self):
        self._pass.end()
        self.queue.submit([self._enc.finish()])
        if self.shot_path:
            self._readback(self.shot_path)
            self.shot_path = None
        self.context.present()

    def _readback(self, path):
        from PIL import Image
        w, h = self._fb
        data = self.queue.read_texture(
            {"texture": self.cur_tex, "origin": (0, 0, 0)},
            {"bytes_per_row": w * 4, "rows_per_image": h}, (w, h, 1))
        arr = np.frombuffer(data, np.uint8).reshape(h, w, 4)
        if self.surface_format.startswith("bgra"):
            arr = arr[:, :, [2, 1, 0]]
        else:
            arr = arr[:, :, :3]
        Image.fromarray(np.ascontiguousarray(arr)).save(path)
        print(f"screenshot saved: {path} ({w}x{h})")

    def draw_sky(self, view_dirs, tan_half_fov, aspect, sky, cam_pos, time_s):
        right, up, forward = view_dirs
        buf = np.zeros(32, dtype=np.float32)
        buf[0:3] = right
        buf[4:7] = up
        buf[8:11] = forward
        buf[12:15] = sky.sun_dir
        buf[16:19] = sky.zenith
        buf[20:23] = sky.horizon
        buf[24:27] = cam_pos
        buf[28:32] = (tan_half_fov, aspect, sky.day_factor, time_s)
        self.queue.write_buffer(self.sky_ubo, 0, buf.tobytes())
        rp = self._pass
        rp.set_pipeline(self.pipe_sky)
        rp.set_bind_group(0, self.bg_sky)
        rp.draw(3)

    def draw_world(self, meshes, mvp, sky, fog_near, fog_far,
                   translucent=False):
        buf = np.zeros(24, dtype=np.float32)
        buf[:16] = to_gl(mvp).reshape(-1)
        buf[16:19] = sky.horizon
        buf[20:24] = (fog_near, fog_far, sky.brightness, 0.0)
        self.queue.write_buffer(self.chunk_ubo, 0, buf.tobytes())
        rp = self._pass
        rp.set_pipeline(self.pipe_chunk)
        rp.set_bind_group(0, self.bg_chunk_params)
        rp.set_bind_group(1, self.bg_chunk_tex)
        for m in meshes:
            if translucent:
                if m.n_t:
                    rp.set_vertex_buffer(0, m.vbo_t)
                    rp.set_index_buffer(m.ibo_t, "uint32")
                    rp.draw_indexed(m.n_t)
            elif m.n_o:
                rp.set_vertex_buffer(0, m.vbo_o)
                rp.set_index_buffer(m.ibo_o, "uint32")
                rp.draw_indexed(m.n_o)

    def draw_outline(self, mvp, block_pos):
        if block_pos is None:
            return
        m = (mvp @ translation(block_pos[0] - 0.002, block_pos[1] - 0.002,
                               block_pos[2] - 0.002) @ scale_uniform(1.004))
        buf = np.zeros(20, dtype=np.float32)
        buf[:16] = to_gl(m).reshape(-1)
        buf[16:20] = (0.0, 0.0, 0.0, 0.85)
        self.queue.write_buffer(self.solid_ubo, 0, buf.tobytes())
        rp = self._pass
        rp.set_pipeline(self.pipe_solid)
        rp.set_bind_group(0, self.bg_solid)
        rp.set_vertex_buffer(0, self.outline_vbo)
        rp.draw(len(LINE_CUBE))

    def draw_hud(self, width, height, hotbar, selected):
        rects = []
        cx, cy = width / 2.0, height / 2.0
        s, t = 9.0, 1.2
        rects.append((cx - s, cy - t, cx + s, cy + t, (1, 1, 1, 0.75)))
        rects.append((cx - t, cy - s, cx + t, cy + s, (1, 1, 1, 0.75)))
        slot, pad = 46.0, 4.0
        total = 9 * slot
        x0, y0 = cx - total / 2.0, 12.0
        for i in range(9):
            sx = x0 + i * slot
            rects.append((sx, y0, sx + slot - pad, y0 + slot - pad,
                          (0.08, 0.08, 0.09, 0.55)))
        hx = x0 + selected * slot
        w3 = 3.0
        rects.append((hx - w3, y0 - w3, hx + slot - pad + w3,
                      y0 + slot - pad + w3, (1, 1, 1, 0.9)))
        rects.append((hx, y0, hx + slot - pad, y0 + slot - pad,
                      (0.35, 0.35, 0.36, 0.6)))
        data = np.zeros((len(rects) * 6, 6), dtype=np.float32)
        for i, (rx0, ry0, rx1, ry1, col) in enumerate(rects):
            xa = rx0 / width * 2.0 - 1.0
            xb = rx1 / width * 2.0 - 1.0
            ya = ry0 / height * 2.0 - 1.0
            yb = ry1 / height * 2.0 - 1.0
            quad = ((xa, ya), (xb, ya), (xb, yb),
                    (xa, ya), (xb, yb), (xa, yb))
            for j, (px, py) in enumerate(quad):
                data[i * 6 + j] = (px, py, *col)
        self.queue.write_buffer(self.rect_vbo, 0, data.tobytes())
        rp = self._pass
        rp.set_pipeline(self.pipe_rect)
        rp.set_vertex_buffer(0, self.rect_vbo)
        rp.draw(len(rects) * 6)
        per = 18
        verts = np.zeros((9 * per, 7), dtype=np.float32)
        n = 0
        size = 26.0
        for i, bid in enumerate(hotbar):
            if bid == B.AIR:
                continue
            ix = x0 + i * slot + (slot - pad) / 2.0
            iy = y0 + (slot - pad) / 2.0
            m = (ortho(0, width, 0, height, -100, 100)
                 @ translation(ix, iy, 0) @ scale_uniform(size)
                 @ rotation_x(math.radians(-28))
                 @ rotation_y(math.radians(45)))
            pos3, uv, shade = self.icon_geo[bid]
            pos = np.concatenate(
                [pos3, np.ones((len(pos3), 1), dtype=np.float32)], axis=1) @ m.T
            pos[:, 2] = 0.5 * pos[:, 2] + 0.5 * pos[:, 3]
            row = n * per
            verts[row:row + per, :4] = pos
            verts[row:row + per, 4:6] = uv
            verts[row:row + per, 6] = shade
            n += 1
        if n:
            self.queue.write_buffer(self.icon_vbo, 0,
                                    verts[:n * per].tobytes())
            rp.set_pipeline(self.pipe_icon)
            rp.set_bind_group(0, self.bg_icon_tex)
            rp.set_vertex_buffer(0, self.icon_vbo)
            for i in range(n):
                rp.draw(per, 1, i * per, 0)
