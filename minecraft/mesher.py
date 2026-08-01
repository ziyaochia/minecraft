import numpy as np

from . import blocks as B
from .config import CHUNK_SIZE, WORLD_HEIGHT
from .textures import BUILDERS, TILE_INDEX

S = CHUNK_SIZE
H = WORLD_HEIGHT

OPAQUE_LUT = np.array(B.IS_OPAQUE, dtype=bool)
TRANS_LUT = np.array(B.IS_TRANSLUCENT, dtype=bool)

TEX_TOP = np.zeros(256, dtype=np.int32)
TEX_SIDE = np.zeros(256, dtype=np.int32)
TEX_BOTTOM = np.zeros(256, dtype=np.int32)
for _bid, _blk in B.BLOCKS.items():
    TEX_TOP[_bid] = TILE_INDEX[_blk.tex_top]
    TEX_SIDE[_bid] = TILE_INDEX[_blk.tex_side]
    TEX_BOTTOM[_bid] = TILE_INDEX[_blk.tex_bottom]

TILE_UV = np.zeros((len(BUILDERS), 4), dtype=np.float32)
from .textures import tile_uv
for _i, (_name, _) in enumerate(BUILDERS):
    TILE_UV[_i] = tile_uv(_name)

AO_FACTOR = np.array([0.55, 0.7, 0.85, 1.0], dtype=np.float32)

_QUADS = [
    ((1, 0, 0), [(1, 0, 0), (1, 1, 0), (1, 1, 1), (1, 0, 1)], [(0, 0), (0, 1), (1, 1), (1, 0)]),
    ((-1, 0, 0), [(0, 0, 1), (0, 1, 1), (0, 1, 0), (0, 0, 0)], [(0, 0), (0, 1), (1, 1), (1, 0)]),
    ((0, 1, 0), [(0, 1, 1), (1, 1, 1), (1, 1, 0), (0, 1, 0)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
    ((0, -1, 0), [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)], [(0, 0), (1, 0), (1, 1), (0, 1)]),
    ((0, 0, 1), [(1, 0, 1), (1, 1, 1), (0, 1, 1), (0, 0, 1)], [(0, 0), (0, 1), (1, 1), (1, 0)]),
    ((0, 0, -1), [(0, 0, 0), (0, 1, 0), (1, 1, 0), (1, 0, 0)], [(0, 0), (0, 1), (1, 1), (1, 0)]),
]


class _Face:
    __slots__ = ("normal", "corners", "uvs", "ao_offs")

    def __init__(self, normal, corners, uvs):
        self.normal = normal
        self.corners = corners
        self.uvs = uvs
        na = next(i for i in range(3) if normal[i] != 0)
        varying = [i for i in range(3) if i != na]
        offs = []
        for c in corners:
            s1 = [0, 0, 0]
            s2 = [0, 0, 0]
            co = [0, 0, 0]
            d1 = 2 * c[varying[0]] - 1
            d2 = 2 * c[varying[1]] - 1
            s1[varying[0]] = d1
            s2[varying[1]] = d2
            co[varying[0]] = d1
            co[varying[1]] = d2
            offs.append((tuple(s1), tuple(s2), tuple(co)))
        self.ao_offs = offs


FACES = [_Face(n, c, u) for n, c, u in _QUADS]


def _shift(arr, ox, oy, oz):
    return arr[1 + ox:1 + ox + S, 1 + oy:1 + oy + H, 1 + oz:1 + oz + S]


def build_mesh(padded, world_ox=0.0, world_oz=0.0):
    T = padded
    OP = OPAQUE_LUT[T]
    cur = _shift(T, 0, 0, 0)
    opaque_v = []
    opaque_i = []
    trans_v = []
    trans_i = []
    for face in FACES:
        nx, ny, nz = face.normal
        nxt = _shift(T, nx, ny, nz)
        draw = B.CULL[cur, nxt]
        if not draw.any():
            continue
        xs, ys, zs = np.nonzero(draw)
        n_faces = len(xs)
        ids = cur[xs, ys, zs]
        if ny > 0:
            tex_idx = TEX_TOP[ids]
        elif ny < 0:
            tex_idx = TEX_BOTTOM[ids]
        else:
            tex_idx = TEX_SIDE[ids]
        tuv = TILE_UV[tex_idx]
        shade_base = B.FACE_SHADE[face.normal]
        verts = np.empty((n_faces * 4, 6), dtype=np.float32)
        for j in range(4):
            cx, cy, cz = face.corners[j]
            s1o, s2o, coo = face.ao_offs[j]
            s1 = _shift(OP, nx + s1o[0], ny + s1o[1], nz + s1o[2])
            s2 = _shift(OP, nx + s2o[0], ny + s2o[1], nz + s2o[2])
            co = _shift(OP, nx + coo[0], ny + coo[1], nz + coo[2])
            ao = np.where(s1 & s2, 0, 3 - (s1.astype(np.uint8) + s2.astype(np.uint8) + co.astype(np.uint8)))
            aov = ao[xs, ys, zs]
            shade = shade_base * AO_FACTOR[aov]
            u = tuv[:, 0] + face.uvs[j][0] * (tuv[:, 2] - tuv[:, 0])
            v = tuv[:, 1] + face.uvs[j][1] * (tuv[:, 3] - tuv[:, 1])
            verts[j::4, 0] = xs + cx + world_ox
            verts[j::4, 1] = ys + cy
            verts[j::4, 2] = zs + cz + world_oz
            verts[j::4, 3] = u
            verts[j::4, 4] = v
            verts[j::4, 5] = shade
        idx = (np.arange(n_faces, dtype=np.uint32)[:, None] * 4 +
               np.array([[0, 1, 2, 0, 2, 3]], dtype=np.uint32)).ravel()
        transl = TRANS_LUT[ids]
        if (~transl).any():
            m = ~transl
            vmask = np.repeat(m, 4)
            new_index = np.cumsum(vmask).astype(np.uint32) - 1
            opaque_v.append(verts[vmask])
            opaque_i.append(new_index[idx[np.repeat(m, 6)]])
        if transl.any():
            vmask = np.repeat(transl, 4)
            new_index = np.cumsum(vmask).astype(np.uint32) - 1
            trans_v.append(verts[vmask])
            trans_i.append(new_index[idx[np.repeat(transl, 6)]])

    def _merge(vlist, ilist):
        if not vlist:
            return np.zeros((0, 6), dtype=np.float32), np.zeros(0, dtype=np.uint32)
        verts = np.concatenate(vlist)
        offset = 0
        parts = []
        for v, i in zip(vlist, ilist):
            parts.append(i + offset)
            offset += len(v)
        return verts, np.concatenate(parts)

    ov, oi = _merge(opaque_v, opaque_i)
    tv, ti = _merge(trans_v, trans_i)
    return ov, oi, tv, ti
