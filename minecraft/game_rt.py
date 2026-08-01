import math

from . import config as C
from .game import S, GameGL, _gen_worker
from .render_rt import VOL_R, RTRenderer


class GameRT(GameGL):
    def __init__(self, args):
        super().__init__(args)
        self.rt = RTRenderer(self.ctx, self.atlas)
        self._vol_center = None

    def _premesh(self, cx, cz):
        pass

    def _mark_dirty(self, c):
        self.rt.upload_chunk(self.world, *c)

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
            if c not in self.world.chunks and c not in self.pending and \
                    gen_budget > 0 and len(self.pending) < 32:
                self.pending[c] = self.pool.submit(_gen_worker, self.world, cx, cz)
                gen_budget -= 1
        for c, fut in list(self.pending.items()):
            if not fut.done():
                continue
            del self.pending[c]
            cx, cz, chunk = fut.result()
            if (cx, cz) not in self.world.chunks:
                self.world.add_chunk(chunk)
            self.rt.upload_chunk(self.world, cx, cz)
        if self._vol_center != (pcx, pcz):
            self._vol_center = (pcx, pcz)
            self.rt.rebuild_volume(self.world, pcx, pcz)

    def _draw(self, sky, mvp, eye, view_dirs, w, h, time_s):
        self.ctx.viewport = (0, 0, w, h)
        self.ctx.screen.use()
        self.ctx.clear(sky.horizon[0], sky.horizon[1], sky.horizon[2], 1.0, depth=1.0)
        fog_far = VOL_R * S * 0.95
        self.rt.draw(view_dirs, math.tan(math.radians(C.FOV) / 2.0), w / h,
                     sky, eye, fog_far * 0.6, fog_far, time_s)
        hit = self._target()
        self.renderer.draw_outline(mvp, hit[0] if hit else None)
        self.renderer.draw_hud(w, h, self.player.hotbar, self.player.selected)
