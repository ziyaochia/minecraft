import numpy as np

from .config import CHUNK_SIZE, WORLD_HEIGHT


class Chunk:
    __slots__ = ("cx", "cz", "blocks")

    def __init__(self, cx, cz, blocks=None):
        self.cx = cx
        self.cz = cz
        if blocks is None:
            blocks = np.zeros((CHUNK_SIZE, WORLD_HEIGHT, CHUNK_SIZE), dtype=np.uint8)
        self.blocks = blocks
