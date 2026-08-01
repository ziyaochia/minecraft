import numpy as np

GRAD3 = np.array([
    [1, 1, 0], [-1, 1, 0], [1, -1, 0], [-1, -1, 0],
    [1, 0, 1], [-1, 0, 1], [1, 0, -1], [-1, 0, -1],
    [0, 1, 1], [0, -1, 1], [0, 1, -1], [0, -1, -1],
], dtype=np.float64)

F2 = 0.5 * (np.sqrt(3.0) - 1.0)
G2 = (3.0 - np.sqrt(3.0)) / 6.0
F3 = 1.0 / 3.0
G3 = 1.0 / 6.0


class SimplexNoise:
    def __init__(self, seed):
        rng = np.random.RandomState(seed & 0x7FFFFFFF)
        p = rng.permutation(256).astype(np.int64)
        self.perm = np.concatenate([p, p])

    def noise2(self, x, y):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        s = (x + y) * F2
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        t = (i + j) * G2
        x0 = x - (i - t)
        y0 = y - (j - t)
        i1 = (x0 > y0).astype(np.int64)
        j1 = 1 - i1
        x1 = x0 - i1 + G2
        y1 = y0 - j1 + G2
        x2 = x0 - 1.0 + 2.0 * G2
        y2 = y0 - 1.0 + 2.0 * G2
        ii = (i & 255)
        jj = (j & 255)
        perm = self.perm
        gi0 = perm[ii + perm[jj]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1]] % 12
        gi2 = perm[ii + 1 + perm[jj + 1]] % 12
        g0 = GRAD3[gi0]
        g1 = GRAD3[gi1]
        g2 = GRAD3[gi2]
        n = np.zeros_like(x0)
        t0 = 0.5 - x0 * x0 - y0 * y0
        m = t0 > 0
        t0c = np.where(m, t0, 0.0)
        n += np.where(m, t0c ** 4 * (g0[..., 0] * x0 + g0[..., 1] * y0), 0.0)
        t1 = 0.5 - x1 * x1 - y1 * y1
        m = t1 > 0
        t1c = np.where(m, t1, 0.0)
        n += np.where(m, t1c ** 4 * (g1[..., 0] * x1 + g1[..., 1] * y1), 0.0)
        t2 = 0.5 - x2 * x2 - y2 * y2
        m = t2 > 0
        t2c = np.where(m, t2, 0.0)
        n += np.where(m, t2c ** 4 * (g2[..., 0] * x2 + g2[..., 1] * y2), 0.0)
        return 70.0 * n

    def noise3(self, x, y, z):
        x = np.asarray(x, dtype=np.float64)
        y = np.asarray(y, dtype=np.float64)
        z = np.asarray(z, dtype=np.float64)
        s = (x + y + z) * F3
        i = np.floor(x + s).astype(np.int64)
        j = np.floor(y + s).astype(np.int64)
        k = np.floor(z + s).astype(np.int64)
        t = (i + j + k) * G3
        x0 = x - (i - t)
        y0 = y - (j - t)
        z0 = z - (k - t)
        x_ge_y = x0 >= y0
        y_ge_z = y0 >= z0
        x_ge_z = x0 >= z0
        i1 = (x_ge_y & x_ge_z).astype(np.int64)
        j1 = ((~x_ge_y.astype(bool)) & y_ge_z).astype(np.int64)
        k1 = ((~x_ge_z.astype(bool)) & (~y_ge_z.astype(bool))).astype(np.int64)
        i2 = (x_ge_y | x_ge_z).astype(np.int64)
        j2 = ((y0 > x0) | y_ge_z).astype(np.int64)
        k2 = 1 - (x_ge_z & y_ge_z).astype(np.int64)
        x1 = x0 - i1 + G3
        y1 = y0 - j1 + G3
        z1 = z0 - k1 + G3
        x2 = x0 - i2 + 2.0 * G3
        y2 = y0 - j2 + 2.0 * G3
        z2 = z0 - k2 + 2.0 * G3
        x3 = x0 - 1.0 + 3.0 * G3
        y3 = y0 - 1.0 + 3.0 * G3
        z3 = z0 - 1.0 + 3.0 * G3
        ii = i & 255
        jj = j & 255
        kk = k & 255
        perm = self.perm
        gi0 = perm[ii + perm[jj + perm[kk]]] % 12
        gi1 = perm[ii + i1 + perm[jj + j1 + perm[kk + k1]]] % 12
        gi2 = perm[ii + i2 + perm[jj + j2 + perm[kk + k2]]] % 12
        gi3 = perm[ii + 1 + perm[jj + 1 + perm[kk + 1]]] % 12
        g0 = GRAD3[gi0]
        g1 = GRAD3[gi1]
        g2 = GRAD3[gi2]
        g3 = GRAD3[gi3]
        n = np.zeros_like(x0)
        for (gi, xx, yy, zz) in ((g0, x0, y0, z0), (g1, x1, y1, z1),
                                 (g2, x2, y2, z2), (g3, x3, y3, z3)):
            tt = 0.6 - xx * xx - yy * yy - zz * zz
            m = tt > 0
            tc = np.where(m, tt, 0.0)
            n += np.where(m, tc ** 4 * (gi[..., 0] * xx + gi[..., 1] * yy + gi[..., 2] * zz), 0.0)
        return 32.0 * n

    def fbm2(self, x, y, octaves, persistence=0.5, lacunarity=2.0):
        total = np.zeros_like(np.asarray(x, dtype=np.float64))
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            total += amp * self.noise2(x * freq, y * freq)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm

    def fbm3(self, x, y, z, octaves, persistence=0.5, lacunarity=2.0):
        total = np.zeros_like(np.asarray(x, dtype=np.float64))
        amp = 1.0
        freq = 1.0
        norm = 0.0
        for _ in range(octaves):
            total += amp * self.noise3(x * freq, y * freq, z * freq)
            norm += amp
            amp *= persistence
            freq *= lacunarity
        return total / norm
