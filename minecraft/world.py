import numpy as np

from . import blocks as B
from .chunk import Chunk
from .config import CHUNK_SIZE, WORLD_HEIGHT
from .worldgen import WorldGenerator

S = CHUNK_SIZE
H = WORLD_HEIGHT


class World:
    def __init__(self, seed):
        self.seed = seed
        self.chunks = {}
        self.generator = WorldGenerator(seed)

    def get_chunk(self, cx, cz):
        return self.chunks.get((cx, cz))

    def add_chunk(self, chunk):
        self.chunks[(chunk.cx, chunk.cz)] = chunk

    def ensure_chunk(self, cx, cz):
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            chunk = self.generator.generate(cx, cz)
            self.chunks[(cx, cz)] = chunk
        return chunk

    def get_block(self, x, y, z):
        if y < 0 or y >= H:
            return B.AIR
        cx, lx = divmod(x, S)
        cz, lz = divmod(z, S)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return B.AIR
        return int(chunk.blocks[lx, y, lz])

    def set_block(self, x, y, z, block_id):
        if y < 0 or y >= H:
            return []
        cx, lx = divmod(x, S)
        cz, lz = divmod(z, S)
        chunk = self.chunks.get((cx, cz))
        if chunk is None:
            return []
        chunk.blocks[lx, y, lz] = block_id
        affected = {(cx, cz)}
        if lx == 0:
            affected.add((cx - 1, cz))
        elif lx == S - 1:
            affected.add((cx + 1, cz))
        if lz == 0:
            affected.add((cx, cz - 1))
        elif lz == S - 1:
            affected.add((cx, cz + 1))
        return [c for c in affected if c in self.chunks]

    def is_solid(self, x, y, z):
        if y < 0:
            return True
        if y >= H:
            return False
        return B.IS_SOLID[self.get_block(x, y, z)]

    def get_padded(self, cx, cz):
        out = np.zeros((S + 2, H + 2, S + 2), dtype=np.uint8)
        center = self.chunks.get((cx, cz))
        if center is None:
            return out
        out[1:S + 1, 1:H + 1, 1:S + 1] = center.blocks
        west = self.chunks.get((cx - 1, cz))
        east = self.chunks.get((cx + 1, cz))
        north = self.chunks.get((cx, cz - 1))
        south = self.chunks.get((cx, cz + 1))
        if west is not None:
            out[0, 1:H + 1, 1:S + 1] = west.blocks[S - 1, :, :]
        if east is not None:
            out[S + 1, 1:H + 1, 1:S + 1] = east.blocks[0, :, :]
        if north is not None:
            out[1:S + 1, 1:H + 1, 0] = north.blocks[:, :, S - 1]
        if south is not None:
            out[1:S + 1, 1:H + 1, S + 1] = south.blocks[:, :, 0]
        nw = self.chunks.get((cx - 1, cz - 1))
        ne = self.chunks.get((cx + 1, cz - 1))
        sw = self.chunks.get((cx - 1, cz + 1))
        se = self.chunks.get((cx + 1, cz + 1))
        if nw is not None:
            out[0, 1:H + 1, 0] = nw.blocks[S - 1, :, S - 1]
        if ne is not None:
            out[S + 1, 1:H + 1, 0] = ne.blocks[0, :, S - 1]
        if sw is not None:
            out[0, 1:H + 1, S + 1] = sw.blocks[S - 1, :, 0]
        if se is not None:
            out[S + 1, 1:H + 1, S + 1] = se.blocks[0, :, 0]
        return out
