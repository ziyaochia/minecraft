from . import blocks as B
from . import config as C
from .player import HALF_W


def block_intersects_player(player, bx, by, bz):
    return (bx + 1.0 > player.pos[0] - HALF_W and bx < player.pos[0] + HALF_W and
            by + 1.0 > player.pos[1] and by < player.pos[1] + C.PLAYER_HEIGHT and
            bz + 1.0 > player.pos[2] - HALF_W and bz < player.pos[2] + HALF_W)


def break_block(world, pos):
    x, y, z = pos
    if world.get_block(x, y, z) in (B.AIR, B.WATER, B.BEDROCK):
        return []
    return world.set_block(x, y, z, B.AIR)


def place_block(world, pos, block_id, player=None):
    x, y, z = pos
    if world.get_block(x, y, z) not in (B.AIR, B.WATER):
        return []
    if player is not None and B.IS_SOLID[block_id] and \
            block_intersects_player(player, x, y, z):
        return []
    return world.set_block(x, y, z, block_id)
