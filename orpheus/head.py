"""Head model, 6-DoF head pose, and ear (tragion) localisation.

Ears are almost never visible to a front-facing camera, and Viola-Jones cannot
find them (the 2017 report's conclusion). So the ears are not detected - they
are *solved*: facial landmarks + a generic anthropometric head model give the
head pose by PnP, and the ears ride along at fixed head-frame positions.

Head frame H (mm), chosen so a listener facing the camera has R = I:
    origin  interaural centre (midpoint of the two tragions)
    +X      toward the listener's LEFT ear   (image right when facing camera)
    +Y      down
    +Z      toward the back of the head      (nose points along -Z)
Camera frame C: OpenCV convention, +X right, +Y down, +Z out of the lens.
    X_C = R @ X_H + t      (t = interaural centre in camera coordinates)
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from .filters import OneEuro
from .geometry import Intrinsics

# Generic adult model, head frame, mm. Built from mean adult anthropometry
# (pupils ~10 mm above the Frankfort plane, tragion ~82 mm behind the pupils).
# Absolute size is set per listener by HeadModel.scale (see --calibrate).
_BASE_POINTS = {
    "eye_l":   (31.5, -10.0, -82.0),
    "eye_r":   (-31.5, -10.0, -82.0),
    "nose":    (0.0, 35.0, -105.0),
    "mouth_l": (25.0, 62.0, -88.0),
    "mouth_r": (-25.0, 62.0, -88.0),
    "chin":    (0.0, 100.0, -85.0),
}


@dataclass
class HeadModel:
    """Anthropometry. Defaults are adult means; `scale` personalises them."""
    bitragion_mm: float = 145.0    # ear-to-ear (tragion) breadth
    width_mm: float = 152.0        # maximum head breadth (X1 in Algazi et al.)
    height_mm: float = 230.0       # vertex to menton (X2)
    depth_mm: float = 195.0        # glabella to back of head (X3)
    scale: float = 1.0             # from IPD/distance calibration
    # Measured tragion positions in the (scaled) head frame, mm - set by ear
    # enrollment (enroll.py). None = generic +-bitragion/2 on the X axis.
    ear_l_mm: tuple[float, float, float] | None = None
    ear_r_mm: tuple[float, float, float] | None = None

    def points(self) -> dict[str, np.ndarray]:
        return {k: self.scale * np.array(v) for k, v in _BASE_POINTS.items()}

    @property
    def ears_measured(self) -> bool:
        return self.ear_l_mm is not None and self.ear_r_mm is not None

    def ears(self) -> tuple[np.ndarray, np.ndarray]:
        if self.ears_measured:
            return np.array(self.ear_l_mm, float), np.array(self.ear_r_mm, float)
        half = self.scale * self.bitragion_mm / 2.0
        return np.array([half, 0.0, 0.0]), np.array([-half, 0.0, 0.0])

    def ear_distance_mm(self) -> float:
        el, er = self.ears()
        return float(np.linalg.norm(el - er))

    def sphere_radius_mm(self) -> float:
        """Best-fit spherical-head radius for ITD (Algazi, Avendano & Duda 2001):
        a = 0.51*X1 + 0.019*X2 + 0.18*X3 + 32 mm, X = half-dimensions.
        This is the anthropometric 'ITD scaling factor' of the 2017 report, Sec 2.1.3,
        reduced to the regression the HRTF literature settled on. With enrolled
        ears, head breadth X1 is taken from the measured ear distance (mean
        breadth/bitragion ratio 152/145) instead of the population mean."""
        s = self.scale
        if self.ears_measured:
            x1 = self.ear_distance_mm() * self.width_mm / self.bitragion_mm
        else:
            x1 = s * self.width_mm
        return 0.51 * x1 / 2 + 0.019 * s * self.height_mm / 2 \
            + 0.18 * s * self.depth_mm / 2 + 32.0


# ---------------------------------------------------------------- rotations
def _rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])


def _ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def _rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


def rotation_from_ypr(yaw_deg: float, pitch_deg: float = 0.0, roll_deg: float = 0.0) -> np.ndarray:
    """yaw > 0: facing toward camera +X; pitch > 0: looking up; roll > 0: left ear down."""
    return _ry(-math.radians(yaw_deg)) @ _rx(-math.radians(pitch_deg)) @ _rz(math.radians(roll_deg))


def ypr_from_rotation(R: np.ndarray) -> tuple[float, float, float]:
    """Inverse of rotation_from_ypr (Y-X-Z Euler decomposition)."""
    a = math.atan2(R[0, 2], R[2, 2])
    b = math.asin(max(-1.0, min(1.0, -R[1, 2])))
    c = math.atan2(R[1, 0], R[1, 1])
    return -math.degrees(a), -math.degrees(b), math.degrees(c)


# ------------------------------------------------------------------- pose
@dataclass
class Listener:
    t: np.ndarray                  # interaural centre, camera frame, mm
    R: np.ndarray                  # head -> camera rotation
    ear_l: np.ndarray              # camera frame, mm
    ear_r: np.ndarray
    radius_mm: float               # spherical-head radius for the HRTF
    method: str                    # "pnp" (landmarks) or "box" (face box only)
    landmarks: dict[str, tuple[float, float]] = field(default_factory=dict)

    @property
    def ypr(self) -> tuple[float, float, float]:
        return ypr_from_rotation(self.R)

    def to_head(self, p_cam: np.ndarray) -> np.ndarray:
        """Camera-frame point -> head-frame point."""
        return self.R.T @ (np.asarray(p_cam, float) - self.t)


def make_listener(R, t, model: HeadModel, method: str, landmarks=None) -> Listener:
    """t is the head-model origin from PnP; Listener.t is the interaural centre
    (they differ only when enrolled ears are not symmetric about the origin)."""
    R, t = np.asarray(R, float), np.asarray(t, float)
    el, er = model.ears()
    ear_l, ear_r = R @ el + t, R @ er + t
    return Listener(t=(ear_l + ear_r) / 2, R=R, ear_l=ear_l, ear_r=ear_r,
                    radius_mm=model.sphere_radius_mm(), method=method,
                    landmarks=dict(landmarks or {}))


class PoseSolver:
    """Landmarks -> head pose by PnP, with temporal warm start."""

    MIN_POINTS = 4

    def __init__(self, K: Intrinsics, model: HeadModel) -> None:
        self.K = K
        self.model = model
        self._cam = np.array([[K.fx, 0, K.cx], [0, K.fy, K.cy], [0, 0, 1]], dtype=np.float64)
        self._rvec: np.ndarray | None = None
        self._tvec: np.ndarray | None = None

    def reset(self) -> None:
        self._rvec = self._tvec = None

    def solve(self, pts2d: dict[str, tuple[float, float]]) -> tuple[np.ndarray, np.ndarray] | None:
        model = self.model.points()
        names = [n for n in pts2d if n in model]
        if len(names) < self.MIN_POINTS:
            return None
        obj = np.array([model[n] for n in names], dtype=np.float64)
        img = np.array([pts2d[n] for n in names], dtype=np.float64)

        ok = False
        if self._rvec is not None:
            ok, rvec, tvec = cv2.solvePnP(obj, img, self._cam, None, self._rvec.copy(),
                                          self._tvec.copy(), True, cv2.SOLVEPNP_ITERATIVE)
            ok = ok and self._plausible(rvec, tvec)
        if not ok:
            flag = cv2.SOLVEPNP_SQPNP if len(names) >= 4 else cv2.SOLVEPNP_EPNP
            ok, rvec, tvec = cv2.solvePnP(obj, img, self._cam, None, flags=flag)
            if ok:
                ok, rvec, tvec = cv2.solvePnP(obj, img, self._cam, None, rvec, tvec, True,
                                              cv2.SOLVEPNP_ITERATIVE)
            ok = ok and self._plausible(rvec, tvec)
        if not ok:
            self.reset()
            return None
        self._rvec, self._tvec = rvec, tvec
        R, _ = cv2.Rodrigues(rvec)
        return R, tvec.ravel()

    @staticmethod
    def _plausible(rvec, tvec) -> bool:
        if tvec[2] < 150 or tvec[2] > 6000:
            return False
        R, _ = cv2.Rodrigues(rvec)
        yaw, pitch, _ = ypr_from_rotation(R)
        return abs(yaw) < 85 and abs(pitch) < 70


def pose_from_box(box, K: Intrinsics, face_width_mm: float, model: HeadModel,
                  eyes: tuple | None = None, profile_sign: int = 0,
                  profile_yaw_deg: float = 70.0) -> tuple[np.ndarray, np.ndarray]:
    """Fallback when fewer than 4 landmarks: position from box width, roll from
    the eye line, yaw 0 (frontal) or +-profile_yaw_deg (profile detection)."""
    from .geometry import face_to_head_xyz

    t = face_to_head_xyz(box, K, face_width_mm)
    roll = 0.0
    if eyes is not None:
        (ur, vr), (ul, vl) = eyes          # subject's right eye (image left), left eye
        roll = math.degrees(math.atan2(vl - vr, ul - ur))
    yaw = profile_sign * profile_yaw_deg
    return rotation_from_ypr(yaw, 0.0, roll), t


def pose_from_eyes(eye_r_px, eye_l_px, K: Intrinsics, ipd_mm: float, model: HeadModel,
                   yaw_deg: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Eye-anchored pose for when only the two eyes are found (haar backend, or PnP
    failed). Depth from the enrolled eye spacing, roll from the eye line; the head
    origin is placed so the model's eye midpoint lands on the observed one. The ears
    then sit at  eye midpoint + R * (stored eye-to-ear offset).

    Yaw is not observable from two points; it is taken as `yaw_deg` (0 = facing the
    camera). Apparent eye spacing shrinks by cos(yaw), so an unmodelled 30 deg turn
    reads ~15 % too far away."""
    (ur, vr), (ul, vl) = eye_r_px, eye_l_px
    ipd_px = math.hypot(ul - ur, vl - vr) / max(math.cos(math.radians(yaw_deg)), 0.2)
    z = K.fx * ipd_mm / max(ipd_px, 1e-6)
    um, vm = (ul + ur) / 2, (vl + vr) / 2
    eye_mid_cam = np.array([(um - K.cx) * z / K.fx, (vm - K.cy) * z / K.fy, z])
    roll = math.degrees(math.atan2(vl - vr, ul - ur))
    R = rotation_from_ypr(yaw_deg, 0.0, roll)
    pts = model.points()
    eye_mid_h = (pts["eye_l"] + pts["eye_r"]) / 2
    return R, eye_mid_cam - R @ eye_mid_h


# -------------------------------------------------------------- smoothing
class PoseFilter:
    """One Euro on translation and on the rotation quaternion."""

    def __init__(self, min_cutoff_hz: float, beta: float) -> None:
        self.ft = OneEuro(min_cutoff_hz, beta)
        self.fq = OneEuro(min_cutoff_hz * 1.5, beta * 400)   # quaternion units ~ rad
        self._q_prev: np.ndarray | None = None

    def reset(self) -> None:
        self.ft.reset()
        self.fq.reset()
        self._q_prev = None

    def __call__(self, R: np.ndarray, t: np.ndarray, time_s: float) -> tuple[np.ndarray, np.ndarray]:
        q = _quat_from_R(R)
        if self._q_prev is not None and q @ self._q_prev < 0:
            q = -q                                 # stay on one hemisphere
        qf = self.fq(q, time_s)
        qf /= np.linalg.norm(qf)
        self._q_prev = qf
        return _R_from_quat(qf), self.ft(t, time_s)


def _quat_from_R(R: np.ndarray) -> np.ndarray:
    rvec, _ = cv2.Rodrigues(np.asarray(R, np.float64))
    ang = float(np.linalg.norm(rvec))
    if ang < 1e-12:
        return np.array([1.0, 0.0, 0.0, 0.0])
    ax = rvec.ravel() / ang
    return np.concatenate([[math.cos(ang / 2)], math.sin(ang / 2) * ax])


def _R_from_quat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = q
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ])
