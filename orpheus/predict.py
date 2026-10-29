"""Aim ahead of the head's motion.

Between a camera frame and the speaker actually pointing somewhere there is
detection, smoothing (the One Euro filter trades jitter for lag), the serial
link, and the servos slewing. Aiming at where the ears are *now* means the beam
always trails a moving head. The predictor estimates each ear's velocity and
aims at where it will be `lead_ms` from now.

Velocity is estimated per point (both ears and the interaural centre) rather
than as head translation + rotation: over ~100 ms the ears' paths are close to
straight, and this handles turning and sliding with one rule.
"""

from __future__ import annotations

import math

import numpy as np

from .head import Listener


class Predictor:
    def __init__(self, lead_ms: float = 100.0, vel_tau_s: float = 0.08,
                 max_shift_mm: float = 120.0) -> None:
        self.lead = lead_ms / 1000.0
        self.tau = vel_tau_s
        self.max_shift = max_shift_mm
        self.reset()

    def reset(self) -> None:
        self._prev: np.ndarray | None = None
        self._t: float | None = None
        self._vel: np.ndarray | None = None

    @property
    def velocity_mm_s(self) -> np.ndarray | None:
        """Velocities of (ear_l, ear_r, centre), mm/s."""
        return None if self._vel is None else self._vel.copy()

    def __call__(self, L: Listener, t: float) -> Listener:
        pts = np.stack([L.ear_l, L.ear_r, L.t]).astype(float)
        if self._prev is None or self._t is None or not (0 < t - self._t < 0.5):
            # first frame, or a gap long enough that the old velocity means nothing
            self._prev, self._t, self._vel = pts, t, np.zeros_like(pts)
            return L
        dt = t - self._t
        raw = (pts - self._prev) / dt
        a = dt / (self.tau + dt)
        self._vel = a * raw + (1 - a) * self._vel
        self._prev, self._t = pts, t
        if self.lead <= 0:
            return L

        shift = self._vel * self.lead
        n = np.linalg.norm(shift, axis=1, keepdims=True)
        shift = np.where(n > self.max_shift, shift * self.max_shift / np.maximum(n, 1e-9), shift)
        p = pts + shift
        return Listener(t=p[2], R=L.R, ear_l=p[0], ear_r=p[1], radius_mm=L.radius_mm,
                        method=L.method, landmarks=L.landmarks)


def servo_lag_s(slew_deg_s: float, typical_move_deg: float, link_ms: float = 15.0) -> float:
    """Rough lag budget for choosing lead_ms: serial + half the slew time of a typical move."""
    return link_ms / 1000.0 + 0.5 * typical_move_deg / max(slew_deg_s, 1e-6)


def angular_error_deg(miss_mm: float, distance_mm: float) -> float:
    return math.degrees(math.atan2(miss_mm, max(distance_mm, 1e-6)))
