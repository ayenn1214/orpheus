"""Image -> 3D listener position -> per-stand absolute pan/tilt servo angles.

The 2017 build nudged servos 1-2 deg per frame until the face was centred in
the image. That only works when the camera rides on the stand. Here the
camera is fixed and each stand's pivot position is known, so the angles are
solved directly: no hunting, both stands converge on the same point, and
stands can sit anywhere (including either side of a desk).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .config import CameraConfig, ServoConfig, StandConfig

# A Haar face box sits on the face surface; the head's acoustic centre
# (between the ears) is roughly this far behind it.
FACE_TO_HEAD_CENTER_MM = 80.0


@dataclass(frozen=True)
class Intrinsics:
    fx: float
    fy: float
    cx: float
    cy: float

    @classmethod
    def from_camera(cls, cam: CameraConfig) -> "Intrinsics":
        """Checkerboard calibration when present (scaled to the current resolution),
        otherwise a pinhole guess from the horizontal field of view."""
        if cam.matrix:
            M = cam.matrix
            cw, ch = cam.calib_size or (cam.width, cam.height)
            sx, sy = cam.width / cw, cam.height / ch
            return cls(fx=M[0][0] * sx, fy=M[1][1] * sy, cx=M[0][2] * sx, cy=M[1][2] * sy)
        f = (cam.width / 2.0) / math.tan(math.radians(cam.hfov_deg) / 2.0)
        return cls(fx=f, fy=f, cx=cam.width / 2.0, cy=cam.height / 2.0)

    def matrix(self) -> np.ndarray:
        return np.array([[self.fx, 0, self.cx], [0, self.fy, self.cy], [0, 0, 1]], np.float64)


def face_to_head_xyz(
    bbox: tuple[float, float, float, float],
    K: Intrinsics,
    face_width_mm: float,
    image_width: int | None = None,
    mirror: bool = False,
) -> np.ndarray:
    """Pinhole back-projection of a face box (x, y, w, h) to a head-centre point (mm)."""
    x, y, w, h = bbox
    if w <= 0:
        raise ValueError("face box width must be positive")
    u = x + w / 2.0
    v = y + h / 2.0
    if mirror:
        if image_width is None:
            raise ValueError("image_width required when mirror=True")
        u = image_width - u
    z_face = K.fx * face_width_mm / w
    X = (u - K.cx) * z_face / K.fx
    Y = (v - K.cy) * z_face / K.fy
    return np.array([X, Y, z_face + FACE_TO_HEAD_CENTER_MM])


def calibrate_face_width(bbox_w_px: float, distance_mm: float, K: Intrinsics) -> float:
    """Face box width in mm, given the box width at a measured camera-to-face distance."""
    return bbox_w_px * distance_mm / K.fx


def stand_ear(stand: StandConfig) -> str:
    """The ear a stand serves, fixed for the session. A listener facing the camera has
    their LEFT on camera +X, so 'auto' takes it from which side of the camera the stand is.
    Fixed rather than 'nearest ear': when the head turns, the nearer ear can swap and
    both stands would chase the same (visible) ear."""
    if stand.ear in ("left", "right", "center"):
        return stand.ear
    if getattr(stand, "channel", "") == "FC":
        return "center"
    return "left" if stand.position_mm[0] >= 0 else "right"


def aim_point(listener, stand: StandConfig) -> np.ndarray:
    """Interaural centre, or this stand's own ear - seen or virtual (eyes + offset)."""
    if stand.aim == "head":
        return listener.t
    ear = stand_ear(stand)
    if ear == "center":
        return listener.t
    return listener.ear_l if ear == "left" else listener.ear_r


def driver_position(stand: StandConfig, target: np.ndarray) -> np.ndarray:
    """Tweeter acoustic centre once the stand is aimed at `target`."""
    p = np.asarray(stand.position_mm, float)
    d = np.asarray(target, float) - p
    n = np.linalg.norm(d)
    return p if n < 1e-6 else p + stand.driver_offset_mm * d / n


def stand_angles(target_xyz: np.ndarray, stand: StandConfig) -> tuple[float, float]:
    """Pan/tilt in degrees from the stand's zero direction (+Z, toed-in by yaw_offset).

    pan  > 0 : turn toward +X (camera's right)
    tilt > 0 : raise (toward -Y)
    """
    d = target_xyz - np.asarray(stand.position_mm, dtype=float)
    dx, dy, dz = d
    # wrap to (-180, 180]: a surround stand behind the listener has yaw_offset ~180
    pan = (math.degrees(math.atan2(dx, dz)) - stand.yaw_offset_deg + 180.0) % 360.0 - 180.0
    tilt = math.degrees(math.atan2(-dy, math.hypot(dx, dz)))
    return pan, tilt


def to_servo(angle_deg: float, servo: ServoConfig) -> float:
    cmd = servo.center_deg + servo.sign * angle_deg
    return min(max(cmd, servo.min_deg), servo.max_deg)


def solve(listener, stands: list[StandConfig]) -> tuple[list[tuple[float, float]], list[np.ndarray]]:
    """Servo commands (pan_deg, tilt_deg) and resulting driver positions, per stand."""
    cmds, drivers = [], []
    for s in stands:
        tgt = aim_point(listener, s)
        pan, tilt = stand_angles(tgt, s)
        cmds.append((to_servo(pan, s.pan), to_servo(tilt, s.tilt)))
        drivers.append(driver_position(s, tgt))
    return cmds, drivers
