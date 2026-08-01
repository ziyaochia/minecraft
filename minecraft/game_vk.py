import math

import glfw

from . import config as C
from .game import S, GameGL
from .mat4 import aabb_visible, frustum_planes
from .render_vk import VK_FIX, VKRenderer


class GameVK(GameGL):
    def _create_window(self):
        glfw.window_hint(glfw.CLIENT_API, glfw.NO_API)
        glfw.window_hint(glfw.RESIZABLE, True)
        win = glfw.create_window(C.WINDOW_WIDTH, C.WINDOW_HEIGHT,
                                 C.WINDOW_TITLE, None, None)
        if not win:
            glfw.terminate()
            raise RuntimeError("window creation failed")
        return win

    def _init_graphics(self):
        self.ctx = None
        self.vk = VKRenderer(self.win, self.atlas)

    def _upload(self, c, ov, oi, tv, ti):
        cx, cz = c
        return self.vk.create_chunk_mesh(
            ov, oi, tv, ti, (cx * S, 0, cz * S),
            (cx * S + S, C.WORLD_HEIGHT, cz * S + S))

    def _draw(self, sky, mvp, eye, view_dirs, w, h, time_s):
        planes = frustum_planes(mvp)
        visible = [m for m in self.meshes.values()
                   if aabb_visible(planes, m.aabb_min, m.aabb_max)]
        fog_far = self.rd * S * 0.95
        mvp_vk = VK_FIX @ mvp
        vk = self.vk
        vk.begin_frame(sky.horizon, w, h)
        vk.draw_sky(view_dirs, math.tan(math.radians(C.FOV) / 2.0), w / h,
                    sky, eye, time_s)
        vk.draw_world(visible, mvp_vk, sky, fog_far * 0.6, fog_far)
        vk.draw_world(visible, mvp_vk, sky, fog_far * 0.6, fog_far,
                      translucent=True)
        hit = self._target()
        vk.draw_outline(mvp_vk, hit[0] if hit else None)
        vk.draw_hud(w, h, self.player.hotbar, self.player.selected)
        vk.end_frame()

    def _swap_and_poll(self):
        glfw.poll_events()

    def _save_screenshot(self, path):
        self.vk.shot_path = path
        self._frame(1.0 / 60.0)
        self.vk.shot_path = None
