import numpy as np

from . import blocks as B
from .chunk import Chunk
from .config import CHUNK_SIZE, WORLD_HEIGHT, SEA_LEVEL
from .noise import SimplexNoise

S = CHUNK_SIZE
H = WORLD_HEIGHT

ORE_SPECS = [
    (B.COAL_ORE, 10, 4, 100, 5, 8),
    (B.IRON_ORE, 8, 4, 56, 4, 7),
    (B.GOLD_ORE, 2, 4, 28, 3, 5),
    (B.DIAMOND_ORE, 1, 4, 15, 3, 5),
]


class WorldGenerator:
    def __init__(self, seed):
        self.seed = seed
        self.n_cont = SimplexNoise(seed ^ 0x9E3779B1)
        self.n_detail = SimplexNoise(seed ^ 0x3C6EF372)
        self.n_ridge = SimplexNoise(seed ^ 0x1B873593)
        self.n_mount = SimplexNoise(seed ^ 0x51EB851F)
        self.n_temp = SimplexNoise(seed ^ 0x2545F491)
        self.n_hum = SimplexNoise(seed ^ 0x6A09E667)
        self.n_cave_a = SimplexNoise(seed ^ 0x452821E6)
        self.n_cave_b = SimplexNoise(seed ^ 0xBB67AE85)

    def _chunk_rng(self, cx, cz, salt):
        h = (self.seed & 0x7FFFFFFF) ^ (cx * 3418731287) ^ (cz * 1328979875) ^ salt
        return np.random.RandomState(h & 0x7FFFFFFF)

    def _column_params(self, cx, cz):
        xs = cx * S + np.arange(S)
        zs = cz * S + np.arange(S)
        X, Z = np.meshgrid(xs, zs, indexing="ij")
        X = X.astype(np.float64)
        Z = Z.astype(np.float64)
        cont = self.n_cont.fbm2(X * 0.0035, Z * 0.0035, 4)
        detail = self.n_detail.fbm2(X * 0.02, Z * 0.02, 3)
        ridge = 1.0 - np.abs(self.n_ridge.fbm2(X * 0.006, Z * 0.006, 3))
        mount = np.clip((self.n_mount.fbm2(X * 0.002 + 53.7, Z * 0.002 - 17.3, 2) - 0.12) * 2.8, 0.0, 1.0)
        h = 66.0 + cont * 14.0 + detail * 4.0 + mount * ridge ** 2 * 95.0
        height = np.clip(h, 4, 200).astype(np.int32)
        temp = self.n_temp.fbm2(X * 0.0016 + 31.7, Z * 0.0016 - 11.9, 2)
        hum = self.n_hum.fbm2(X * 0.0016 - 71.3, Z * 0.0016 + 47.1, 2)
        return X, Z, height, temp, hum

    def generate(self, cx, cz):
        X, Z, height, temp, hum = self._column_params(cx, cz)
        blocks = np.zeros((S, H, S), dtype=np.uint8)
        Y = np.arange(H, dtype=np.int32).reshape(1, H, 1)
        hgt = height[:, None, :]
        solid = Y < hgt
        depth = hgt - Y
        blocks[solid] = B.STONE

        desert = temp > 0.25
        snowy = temp < -0.30
        plains = ~(desert | snowy)
        beach = (height <= SEA_LEVEL + 2) & (height >= SEA_LEVEL - 2)
        ocean_floor = height < SEA_LEVEL - 2
        mountain = height > 110
        high_snow = height > 140

        surf = depth == 1
        sub = (depth > 1) & (depth <= 4)

        d3 = desert[:, None, :] & surf
        blocks[d3] = B.SAND
        s3 = snowy[:, None, :] & surf
        blocks[s3] = B.SNOW
        p3 = plains[:, None, :] & surf
        blocks[p3] = B.GRASS

        sub_d = desert[:, None, :] & sub
        blocks[sub_d] = B.SANDSTONE
        sub_o = (~desert)[:, None, :] & sub
        blocks[sub_o] = B.DIRT

        b3 = beach[:, None, :] & surf
        blocks[b3] = B.SAND
        sub_b = beach[:, None, :] & sub
        blocks[sub_b] = B.SAND

        gravel_sel = (self.n_detail.noise2(X * 0.09, Z * 0.09) > 0.15)
        of = ocean_floor[:, None, :] & surf
        blocks[of & gravel_sel[:, None, :]] = B.GRAVEL
        of_sub = ocean_floor[:, None, :] & sub
        blocks[of_sub] = B.DIRT

        m3 = mountain[:, None, :] & surf
        blocks[m3] = B.STONE
        hs3 = high_snow[:, None, :] & surf
        blocks[hs3] = B.SNOW
        hs_sub = high_snow[:, None, :] & sub
        blocks[hs_sub] = B.STONE

        water_mask = (~solid) & (Y <= SEA_LEVEL)
        blocks[water_mask] = B.WATER

        self._carve_caves(blocks, X, Z, height)
        self._place_ores(blocks, cx, cz)
        self._place_trees(blocks, height, temp, hum, cx, cz)
        self._place_bedrock(blocks, cx, cz)
        return Chunk(cx, cz, blocks)

    def _carve_caves(self, blocks, X, Z, height):
        Y = np.arange(H, dtype=np.float64).reshape(1, H, 1)
        X3 = np.broadcast_to(X[:, None, :], (S, H, S))
        Z3 = np.broadcast_to(Z[:, None, :], (S, H, S))
        c1 = self.n_cave_a.noise3(X3 * 0.045, Y * 0.075, Z3 * 0.045)
        c2 = self.n_cave_b.noise3(X3 * 0.045 + 400.0, Y * 0.075, Z3 * 0.045)
        carve = (np.abs(c1) < 0.055) & (np.abs(c2) < 0.055)
        depth_ok = Y < (height[:, None, :] - 5)
        carve &= depth_ok & (Y > 5) & (blocks != B.WATER)
        blocks[carve] = B.AIR

    def _place_ores(self, blocks, cx, cz):
        for ore_id, tries, y_min, y_max, v_min, v_max in ORE_SPECS:
            rng = self._chunk_rng(cx, cz, ore_id * 2654435761)
            for _ in range(tries):
                x = rng.randint(0, S)
                y = rng.randint(y_min, y_max + 1)
                z = rng.randint(0, S)
                size = rng.randint(v_min, v_max + 1)
                for _ in range(size):
                    if 0 <= x < S and 0 <= z < S and 0 <= y < H:
                        if blocks[x, y, z] == B.STONE:
                            blocks[x, y, z] = ore_id
                    x += rng.randint(-1, 2)
                    y += rng.randint(-1, 2)
                    z += rng.randint(-1, 2)

    def _place_trees(self, blocks, height, temp, hum, cx, cz):
        rng = self._chunk_rng(cx, cz, 0x5EED5)
        desert = temp > 0.25
        density = np.clip(hum * 1.5 + 0.6, 0.0, 2.0)
        n_tries = int(2 + density.mean() * 3)
        for _ in range(n_tries):
            x = rng.randint(2, S - 2)
            z = rng.randint(2, S - 2)
            if desert[x, z]:
                continue
            if rng.random() > 0.25 + density[x, z] * 0.5:
                continue
            y = height[x, z]
            if y <= SEA_LEVEL + 1 or y + 8 >= H:
                continue
            surface = blocks[x, y - 1, z]
            if surface not in (B.GRASS, B.SNOW):
                continue
            trunk = rng.randint(4, 7)
            for dy in range(trunk):
                blocks[x, y + dy, z] = B.LOG
            for dy in range(trunk - 2, trunk + 1):
                r = 2 if dy < trunk else 1
                for dx in range(-r, r + 1):
                    for dz in range(-r, r + 1):
                        if dx == 0 and dz == 0 and dy < trunk:
                            continue
                        if abs(dx) == r and abs(dz) == r and rng.random() < 0.6:
                            continue
                        lx, ly, lz = x + dx, y + dy, z + dz
                        if 0 <= lx < S and 0 <= lz < S and ly < H:
                            if blocks[lx, ly, lz] == B.AIR:
                                blocks[lx, ly, lz] = B.LEAVES

    def _place_bedrock(self, blocks, cx, cz):
        rng = self._chunk_rng(cx, cz, 0xBED00C)
        blocks[:, 0, :] = B.BEDROCK
        for y in range(1, 5):
            p = (5 - y) / 5.0 * 0.9
            m = rng.random((S, S)) < p
            blocks[:, y, :][m] = B.BEDROCK
