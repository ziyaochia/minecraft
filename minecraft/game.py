import math
import os
import time
from concurrent.futures import ThreadPoolExecutor

import glfw
import moderngl
import numpy as np

from . import blocks as B
from . import config as C
from .interact import break_block, place_block
from .mat4 import (aabb_visible, forward_from_angles, frustum_planes, look_at,
                   perspective, to_gl)
from .mesher import build_mesh
from .player import Player
from .raycast import raycast
from .render_gl import GLRenderer
from .saveload import load_world, save_path, save_world
from .sky import SkyState
from .textures import build_atlas
from .world import World

S = C.CHUNK_SIZE


def _gen_worker(world, cx, cz):
    return cx, cz, world.generator.generate(cx, cz)


def _mesh_worker(world, cx, cz):
    ov, oi, tv, ti = build_mesh(world.get_padded(cx, cz), cx * S, cz * S)
    return cx, cz, ov, oi, tv, ti


class GameGL:
    def __init__(self, args):
        self.args = args
        if not glfw.init():
            raise RuntimeError("glfw init failed")
        self.win = self._create_window()
        self.world = World(args.seed)
        self.atlas = build_atlas(args.seed)
        self._init_graphics()
        self.persist = bool(args.save) or not args.screenshot
        self.save_file = save_path(args.seed)
        self._loaded = None
        if self.persist and not args.fresh and os.path.exists(self.save_file):
            self._loaded = load_world(self.world, self.save_file)
        self.meshes = {}
        self.pending = {}
        self.need_mesh = set()
        self.redo = set()
        self.uploads = []
        self.pool = ThreadPoolExecutor(max_workers=2)
        self.mesh_pool = ThreadPoolExecutor(max_workers=1)
        self._pregen()
        self.player = Player(pos=self._spawn(), yaw=args.yaw, pitch=args.pitch)
        if self._loaded and not args.pos:
            m = self._loaded
            self.player.pos = m["pos"]
            self.player.vel = m["vel"]
            self.player.yaw = m["yaw"]
            self.player.pitch = m["pitch"]
            self.player.flying = m["flying"]
            self.player.hotbar = m["hotbar"]
            self.player.selected = m["selected"]
        if args.time is not None:
            self.time_of_day = args.time % 1.0
        elif self._loaded:
            self.time_of_day = self._loaded["time_of_day"] % 1.0
        else:
            self.time_of_day = 0.32
        self.keys = set()
        self.mouse_dx = 0.0
        self.mouse_dy = 0.0
        self.captured = False
        self.break_cool = 0.0
        self.place_cool = 0.0
        self.fps_time = 0.0
        self.fps_count = 0
        self.frame_index = 0
        self.rd = args.rd or C.RENDER_DISTANCE
        self.headless = bool(args.screenshot)
        self.demo_steps = {}
        if args.demo:
            for part in args.demo.split(","):
                f, act, *rest = part.split(":")
                self.demo_steps.setdefault(int(f), []).append(
                    (act, rest[0] if rest else None))
        self._install_callbacks()
        if args.screenshot:
            self._set_capture(True)

    def _create_window(self):
        glfw.window_hint(glfw.CONTEXT_VERSION_MAJOR, 3)
        glfw.window_hint(glfw.CONTEXT_VERSION_MINOR, 3)
        glfw.window_hint(glfw.OPENGL_PROFILE, glfw.OPENGL_CORE_PROFILE)
        glfw.window_hint(glfw.OPENGL_FORWARD_COMPAT, True)
        win = glfw.create_window(C.WINDOW_WIDTH, C.WINDOW_HEIGHT,
                                 C.WINDOW_TITLE, None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("window creation failed")
        glfw.make_context_current(win)
        glfw.swap_interval(1 if C.VSYNC else 0)
        return win

    def _init_graphics(self):
        self.ctx = moderngl.create_context()
        self.renderer = GLRenderer(self.ctx, self.atlas)

    def _pregen(self):
        cx = math.floor(self._spawn_xz()[0] / S)
        cz = math.floor(self._spawn_xz()[1] / S)
        for dx in range(-2, 3):
            for dz in range(-2, 3):
                self.world.ensure_chunk(cx + dx, cz + dz)
        self._premesh(cx, cz)

    def _premesh(self, cx, cz):
        for dx in range(-1, 2):
            for dz in range(-1, 2):
                c = (cx + dx, cz + dz)
                ov, oi, tv, ti = _mesh_worker(self.world, *c)[2:]
                self.meshes[c] = self._upload(c, ov, oi, tv, ti)

    def _spawn_xz(self):
        if self.args.pos:
            parts = [float(v) for v in self.args.pos.split(",")]
            return parts[0], parts[2]
        return 0.5, 0.5

    def _spawn(self):
        if self.args.pos:
            return tuple(float(v) for v in self.args.pos.split(","))
        x, z = self._spawn_xz()
        bx, bz = math.floor(x), math.floor(z)
        top = 100
        for y in range(C.WORLD_HEIGHT - 2, 0, -1):
            if self.world.is_solid(bx, y, bz):
                top = y + 1
                break
        return (x, top + 0.05, z)

    def _install_callbacks(self):
        glfw.set_key_callback(self.win, self._on_key)
        glfw.set_cursor_pos_callback(self.win, self._on_cursor)
        glfw.set_mouse_button_callback(self.win, self._on_button)
        glfw.set_scroll_callback(self.win, self._on_scroll)

    def _set_capture(self, on):
        self.captured = on
        glfw.set_input_mode(self.win, glfw.CURSOR,
                            glfw.CURSOR_DISABLED if on else glfw.CURSOR_NORMAL)
        if on and glfw.raw_mouse_motion_supported():
            glfw.set_input_mode(self.win, glfw.RAW_MOUSE_MOTION, True)

    def _on_key(self, win, key, scancode, action, mods):
        if action == glfw.PRESS:
            self.keys.add(key)
            if key == glfw.KEY_ESCAPE:
                if self.captured:
                    self._set_capture(False)
                else:
                    glfw.set_window_should_close(self.win, True)
            elif key == glfw.KEY_F:
                self.player.toggle_fly()
            elif glfw.KEY_1 <= key <= glfw.KEY_9:
                self.player.selected = key - glfw.KEY_1
        elif action == glfw.RELEASE:
            self.keys.discard(key)

    def _on_cursor(self, win, x, y):
        if not hasattr(self, "_last_x"):
            self._last_x, self._last_y = x, y
            return
        if self.captured:
            self.mouse_dx += x - self._last_x
            self.mouse_dy += y - self._last_y
        self._last_x, self._last_y = x, y

    def _on_button(self, win, button, action, mods):
        if action != glfw.PRESS:
            return
        if not self.captured:
            self._set_capture(True)
            return
        if button == glfw.MOUSE_BUTTON_LEFT:
            self._do_break()
            self.break_cool = 0.0
        elif button == glfw.MOUSE_BUTTON_RIGHT:
            self._do_place()
            self.place_cool = 0.0
        elif button == glfw.MOUSE_BUTTON_MIDDLE:
            self._do_pick()

    def _on_scroll(self, win, dx, dy):
        if dy:
            self.player.selected = (self.player.selected - int(math.copysign(1, dy))) % 9

    def _target(self):
        return raycast(self.world, self.player.eye, self.player.forward, C.REACH)

    def _do_break(self):
        hit = self._target()
        if hit is None:
            return
        for c in break_block(self.world, hit[0]):
            self._mark_dirty(c)

    def _do_place(self):
        hit = self._target()
        if hit is None:
            return
        block_id = self.player.hotbar[self.player.selected]
        for c in place_block(self.world, hit[1], block_id, player=self.player):
            self._mark_dirty(c)

    def _do_pick(self):
        hit = self._target()
        if hit is None:
            return
        bid = hit[2]
        if bid in self.player.hotbar:
            self.player.selected = self.player.hotbar.index(bid)
        else:
            self.player.hotbar[self.player.selected] = bid

    def _mark_dirty(self, c):
        if c in self.pending:
            self.redo.add(c)
        else:
            self.need_mesh.add(c)

    def _upload(self, c, ov, oi, tv, ti):
        cx, cz = c
        return self.renderer.create_chunk_mesh(
            ov, oi, tv, ti, (cx * S, 0, cz * S), (cx * S + S, C.WORLD_HEIGHT, cz * S + S))

    def _stream(self):
        pcx = math.floor(self.player.pos[0] / S)
        pcz = math.floor(self.player.pos[2] / S)
        R = self.rd
        desired = []
        for dx in range(-R, R + 1):
            for dz in range(-R, R + 1):
                d = max(abs(dx), abs(dz))
                desired.append((d, pcx + dx, pcz + dz))
        desired.sort()
        gen_budget = C.CHUNK_GEN_PER_FRAME
        for d, cx, cz in desired:
            c = (cx, cz)
            if c in self.world.chunks:
                if c not in self.meshes and c not in self.pending and \
                        self._neighbors_ready(cx, cz):
                    self.need_mesh.add(c)
            elif c not in self.pending and gen_budget > 0 and len(self.pending) < 32:
                self.pending[c] = self.pool.submit(_gen_worker, self.world, cx, cz)
                gen_budget -= 1
        by_dist = sorted(self.need_mesh,
                         key=lambda c: max(abs(c[0] - pcx), abs(c[1] - pcz)))
        mesh_budget = C.CHUNK_GEN_PER_FRAME
        for c in by_dist:
            if mesh_budget <= 0:
                break
            if c in self.pending or c not in self.world.chunks or \
                    not self._neighbors_ready(*c):
                self.need_mesh.discard(c)
                continue
            self.need_mesh.discard(c)
            self.pending[c] = self.mesh_pool.submit(_mesh_worker, self.world, *c)
            mesh_budget -= 1
        for c, fut in list(self.pending.items()):
            if not fut.done():
                continue
            del self.pending[c]
            result = fut.result()
            if len(result) == 3:
                cx, cz, chunk = result
                if (cx, cz) not in self.world.chunks:
                    self.world.add_chunk(chunk)
                for n in ((cx-1,cz),(cx+1,cz),(cx,cz-1),(cx,cz+1),
                          (cx-1,cz-1),(cx+1,cz-1),(cx-1,cz+1),(cx+1,cz+1)):
                    if n in self.meshes:
                        self._mark_dirty(n)
            else:
                cx, cz, ov, oi, tv, ti = result
                if c in self.redo:
                    self.redo.discard(c)
                    self.need_mesh.add(c)
                else:
                    self.uploads.append(result)
        budget = C.MESH_UPLOADS_PER_FRAME
        while self.uploads and budget > 0:
            cx, cz, ov, oi, tv, ti = self.uploads.pop(0)
            old = self.meshes.get((cx, cz))
            if old is not None:
                old.release()
            self.meshes[(cx, cz)] = self._upload((cx, cz), ov, oi, tv, ti)
            budget -= 1
        for c in list(self.meshes):
            if max(abs(c[0] - pcx), abs(c[1] - pcz)) > R + 2:
                self.meshes.pop(c).release()

    def _neighbors_ready(self, cx, cz):
        ch = self.world.chunks
        return all((cx + dx, cz + dz) in ch
                   for dx in (-1, 0, 1) for dz in (-1, 0, 1)
                   if (dx, dz) != (0, 0))

    def _input_move(self):
        if self.headless:
            return 0, 0, False, False, False
        g = glfw
        k = self.keys
        fwd = (1 if g.KEY_W in k else 0) - (1 if g.KEY_S in k else 0)
        strafe = (1 if g.KEY_D in k else 0) - (1 if g.KEY_A in k else 0)
        jump = g.KEY_SPACE in k
        sneak = g.KEY_LEFT_SHIFT in k or g.KEY_RIGHT_SHIFT in k
        sprint = g.KEY_LEFT_CONTROL in k
        return fwd, strafe, jump, sneak, sprint

    def _repeat_actions(self, dt):
        if not self.captured:
            return
        self.break_cool += dt
        self.place_cool += dt
        g = glfw
        if g.get_mouse_button(self.win, g.MOUSE_BUTTON_LEFT) == g.PRESS and \
                self.break_cool > 0.25:
            self._do_break()
            self.break_cool = 0.0
        if g.get_mouse_button(self.win, g.MOUSE_BUTTON_RIGHT) == g.PRESS and \
                self.place_cool > 0.25:
            self._do_place()
            self.place_cool = 0.0

    def _frame(self, dt):
        if not self.headless:
            self.player.look(self.mouse_dx, self.mouse_dy)
        self.mouse_dx = self.mouse_dy = 0.0
        self.player.update(dt, self.world, self._input_move())
        self._repeat_actions(dt)
        self._stream()
        self.time_of_day = (self.time_of_day + dt / C.DAY_LENGTH) % 1.0
        sky = SkyState(self.time_of_day)
        w, h = glfw.get_framebuffer_size(self.win)
        eye = self.player.eye
        forward = self.player.forward
        right = np.cross(forward, np.array([0.0, 1.0, 0.0]))
        right /= np.linalg.norm(right)
        up = np.cross(right, forward)
        proj = perspective(C.FOV, w / h, C.NEAR, C.FAR)
        mvp = proj @ look_at(eye, eye + forward, up)
        self._draw(sky, mvp, eye, (right, up, forward), w, h,
                   self.frame_index * dt)
        self.frame_index += 1

    def _draw(self, sky, mvp, eye, view_dirs, w, h, time_s):
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.screen.use()
        self.ctx.clear(sky.horizon[0], sky.horizon[1], sky.horizon[2], 1.0, depth=1.0)
        planes = frustum_planes(mvp)
        visible = [m for m in self.meshes.values()
                   if aabb_visible(planes, m.aabb_min, m.aabb_max)]
        fog_far = self.rd * S * 0.95
        self.renderer.draw_sky(view_dirs,
                               math.tan(math.radians(C.FOV) / 2.0), w / h,
                               sky, eye, time_s)
        self.renderer.draw_world(visible, mvp, sky, fog_far * 0.6, fog_far)
        self.renderer.draw_world(visible, mvp, sky, fog_far * 0.6, fog_far,
                                 translucent=True)
        hit = self._target()
        self.renderer.draw_outline(mvp, hit[0] if hit else None)
        self.renderer.draw_hud(w, h, self.player.hotbar, self.player.selected)

    def run(self):
        shot = self.args.screenshot
        fixed_dt = 1.0 / 60.0
        last = time.perf_counter()
        while not glfw.window_should_close(self.win):
            if shot:
                dt = fixed_dt
            else:
                now = time.perf_counter()
                dt = min(now - last, 0.05)
                last = now
            self._frame(dt)
            steps = self.demo_steps.get(self.frame_index)
            if steps:
                for act, path in steps:
                    if act == "place":
                        self._do_place()
                    elif act == "break":
                        self._do_break()
                    elif act == "shot":
                        self._save_screenshot(path)
            if shot and self.frame_index >= self.args.frames:
                self._save_screenshot(shot)
                break
            self._swap_and_poll()
            self._tick_fps()
        self.pool.shutdown(wait=False, cancel_futures=True)
        self.mesh_pool.shutdown(wait=False, cancel_futures=True)
        if self.persist:
            save_world(self.world, self.player, self.time_of_day,
                       self.save_file)
        glfw.terminate()

    def _swap_and_poll(self):
        glfw.swap_buffers(self.win)
        glfw.poll_events()

    def _tick_fps(self):
        self.fps_count += 1
        now = time.perf_counter()
        if now - self.fps_time >= 0.5:
            fps = self.fps_count / (now - self.fps_time)
            self.fps_time, self.fps_count = now, 0
            p = self.player.pos
            glfw.set_window_title(
                self.win,
                f"Minecraft - {fps:.0f} fps | {p[0]:.1f} {p[1]:.1f} {p[2]:.1f} | "
                f"{len(self.meshes)} chunks")

    def _save_screenshot(self, path):
        from PIL import Image
        w, h = glfw.get_framebuffer_size(self.win)
        data = self.ctx.screen.read(components=3)
        Image.frombytes("RGB", (w, h), data).transpose(
            Image.FLIP_TOP_BOTTOM).save(path)
        print(f"screenshot saved: {path} ({w}x{h})")
