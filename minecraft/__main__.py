import argparse

from . import config as C


def main():
    ap = argparse.ArgumentParser(prog="minecraft")
    ap.add_argument("--seed", type=int, default=C.DEFAULT_SEED)
    ap.add_argument("--renderer", default="gl", choices=["gl", "rt", "vk"])
    ap.add_argument("--time", type=float, default=None)
    ap.add_argument("--fresh", action="store_true", help="ignore saved world")
    ap.add_argument("--save", action="store_true",
                    help="persist even in screenshot mode")
    ap.add_argument("--pos", default=None, help="x,y,z spawn override")
    ap.add_argument("--yaw", type=float, default=0.0)
    ap.add_argument("--pitch", type=float, default=-0.15)
    ap.add_argument("--screenshot", default=None)
    ap.add_argument("--frames", type=int, default=60)
    ap.add_argument("--demo", default=None,
                    help="comma list of frame:place|break|shot:path")
    ap.add_argument("--rd", type=int, default=None, help="render distance")
    args = ap.parse_args()
    if args.renderer == "gl":
        from .game import GameGL
        GameGL(args).run()
    elif args.renderer == "rt":
        from .game_rt import GameRT
        GameRT(args).run()
    else:
        from .game_vk import GameVK
        GameVK(args).run()


if __name__ == "__main__":
    main()
