AIR = 0
GRASS = 1
DIRT = 2
STONE = 3
SAND = 4
WATER = 5
LOG = 6
LEAVES = 7
PLANKS = 8
COBBLESTONE = 9
GLASS = 10
BEDROCK = 11
COAL_ORE = 12
IRON_ORE = 13
GOLD_ORE = 14
DIAMOND_ORE = 15
SNOW = 16
BRICK = 17
GRAVEL = 18
SANDSTONE = 19


class Block:
    __slots__ = ("id", "name", "opaque", "solid", "translucent", "tex_top", "tex_side", "tex_bottom", "emissive")

    def __init__(self, id, name, tex, opaque=True, solid=True, translucent=False, emissive=False):
        self.id = id
        self.name = name
        self.opaque = opaque
        self.solid = solid
        self.translucent = translucent
        self.emissive = emissive
        if isinstance(tex, str):
            self.tex_top = self.tex_side = self.tex_bottom = tex
        else:
            self.tex_top, self.tex_side, self.tex_bottom = tex


BLOCKS = {}


def _reg(block):
    BLOCKS[block.id] = block
    return block


_reg(Block(AIR, "air", "glass", opaque=False, solid=False))
_reg(Block(GRASS, "grass", ("grass_top", "grass_side", "dirt")))
_reg(Block(DIRT, "dirt", "dirt"))
_reg(Block(STONE, "stone", "stone"))
_reg(Block(SAND, "sand", "sand"))
_reg(Block(WATER, "water", "water", opaque=False, solid=False, translucent=True))
_reg(Block(LOG, "log", ("log_top", "log_side", "log_top")))
_reg(Block(LEAVES, "leaves", "leaves", opaque=False))
_reg(Block(PLANKS, "planks", "planks"))
_reg(Block(COBBLESTONE, "cobblestone", "cobblestone"))
_reg(Block(GLASS, "glass", "glass", opaque=False, translucent=True))
_reg(Block(BEDROCK, "bedrock", "bedrock"))
_reg(Block(COAL_ORE, "coal_ore", "coal_ore"))
_reg(Block(IRON_ORE, "iron_ore", "iron_ore"))
_reg(Block(GOLD_ORE, "gold_ore", "gold_ore"))
_reg(Block(DIAMOND_ORE, "diamond_ore", "diamond_ore"))
_reg(Block(SNOW, "snow", ("snow", "snow_side", "dirt")))
_reg(Block(BRICK, "brick", "brick"))
_reg(Block(GRAVEL, "gravel", "gravel"))
_reg(Block(SANDSTONE, "sandstone", ("sandstone_top", "sandstone_side", "sandstone_bottom")))

PLACEABLE = [GRASS, DIRT, STONE, COBBLESTONE, PLANKS, LOG, LEAVES, SAND, SANDSTONE,
             GLASS, BRICK, GRAVEL, SNOW, COAL_ORE, IRON_ORE, GOLD_ORE, DIAMOND_ORE, BEDROCK]

_is_opaque = [False] * 256
_is_solid = [False] * 256
_is_translucent = [False] * 256
_names = ["air"] * 256
for _b in BLOCKS.values():
    _is_opaque[_b.id] = _b.opaque
    _is_solid[_b.id] = _b.solid
    _is_translucent[_b.id] = _b.translucent
    _names[_b.id] = _b.name

IS_OPAQUE = _is_opaque
IS_SOLID = _is_solid
IS_TRANSLUCENT = _is_translucent
NAMES = _names


def build_cull_table():
    import numpy as np
    table = np.zeros((256, 256), dtype=bool)
    for a in range(256):
        if a == AIR:
            continue
        for b in range(256):
            if _is_opaque[b]:
                continue
            if a == b and (a == WATER or a == GLASS or a == LEAVES):
                continue
            table[a, b] = True
    return table


CULL = build_cull_table()

FACE_SHADE = {
    (0, 1, 0): 1.0,
    (0, -1, 0): 0.5,
    (1, 0, 0): 0.6,
    (-1, 0, 0): 0.6,
    (0, 0, 1): 0.8,
    (0, 0, -1): 0.8,
}


def face_texture(block_id, normal):
    b = BLOCKS[block_id]
    if normal[1] > 0:
        return b.tex_top
    if normal[1] < 0:
        return b.tex_bottom
    return b.tex_side
