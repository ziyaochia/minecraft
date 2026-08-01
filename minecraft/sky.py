import math

import numpy as np

DAY_ZENITH = np.array([0.30, 0.55, 0.92])
DAY_HORIZON = np.array([0.72, 0.83, 0.94])
NIGHT_ZENITH = np.array([0.008, 0.012, 0.03])
NIGHT_HORIZON = np.array([0.03, 0.04, 0.08])
SUNSET_TINT = np.array([0.95, 0.45, 0.18])


class SkyState:
    __slots__ = ("sun_dir", "day_factor", "zenith", "horizon", "brightness", "angle")

    def __init__(self, time_of_day):
        t = (time_of_day % 1.0)
        self.angle = t * 2.0 * math.pi
        self.sun_dir = np.array([math.cos(self.angle), math.sin(self.angle), 0.18])
        self.sun_dir = self.sun_dir / np.linalg.norm(self.sun_dir)
        elev = self.sun_dir[1]
        self.day_factor = float(np.clip(elev * 6.0 + 0.5, 0.0, 1.0))
        dusk = float(np.clip(1.0 - abs(elev) * 5.0, 0.0, 1.0))
        self.zenith = NIGHT_ZENITH + (DAY_ZENITH - NIGHT_ZENITH) * self.day_factor
        self.horizon = NIGHT_HORIZON + (DAY_HORIZON - NIGHT_HORIZON) * self.day_factor
        self.horizon = self.horizon + SUNSET_TINT * dusk * 0.45
        self.brightness = 0.25 + 0.75 * self.day_factor
