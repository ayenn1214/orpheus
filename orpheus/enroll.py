"""Ear enrollment: find each ear once, measure it, store it, reuse it.

A front-facing camera cannot see the ears during normal listening, so they are
measured in a one-time guided session and stored in a listener profile:

  1. SCALE   Face the camera at a measured distance. Sets the head-model scale
             (monocular PnP knows shape, not size) and records the listening seat.
  2. LEFT    Turn until the left ear faces the camera. Keyframes are captured
             automatically across a spread of angles, each with its head pose.
  3. RIGHT   Same for the right ear.
  4. MARK    In each keyframe the tragion (the small flap in front of the ear
             canal) is located: an ear detector if one is configured, otherwise
             a click, with the model's prediction pre-placed.
  5. SOLVE   Each marked pixel is a ray; in the head frame the rays from the
             different poses cross at the ear. Least-squares intersection gives
             the ear position, and from the two ears the distances that matter:
             ear-to-ear, each ear relative to the eyes, sphere radius for the HRTF,
             and speaker-to-ear distances at the reference seat.

At run time the profile replaces the generic ears: the tracked pose carries the
measured ears, the stands aim at them, and the audio is time-aligned to them.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from .earfind import make_templates
from .geometry import Intrinsics
from .head import HeadModel, ypr_from_rotation

TRAGION_HINT = "the small flap in front of the ear canal (tragion)"


# ================================================================== geometry
@dataclass
class EarObservation:
    R: np.ndarray            # head -> camera rotation at capture
    t: np.ndarray            # head-model origin in camera frame, mm
    pixel: tuple[float, float]
    source: str = "click"    # "click", "detector", "prediction"


def ray_in_head(obs: EarObservation, K: Intrinsics) -> tuple[np.ndarray, np.ndarray]:
    """Camera ray through the pixel, expressed in the head frame: (origin, unit dir)."""
    u, v = obs.pixel
    d_cam = np.array([(u - K.cx) / K.fx, (v - K.cy) / K.fy, 1.0])
    d_cam /= np.linalg.norm(d_cam)
    origin = -obs.R.T @ obs.t                  # camera centre in head frame
    return origin, obs.R.T @ d_cam


def triangulate(origins: np.ndarray, dirs: np.ndarray, prior: np.ndarray | None = None,
                prior_sigma_mm: float = 30.0, ray_sigma_mm: float = 3.0):
    """Least-squares point closest to all rays, with an optional Gaussian prior.

    Returns (point, rms distance from the rays, max angle between rays in deg).
    The prior only matters when the rays are nearly parallel (poor baseline)."""
    A = np.zeros((3, 3))
    b = np.zeros(3)
    for o, d in zip(origins, dirs):
        P = np.eye(3) - np.outer(d, d)
        A += P
        b += P @ o
    if prior is not None:
        w = (ray_sigma_mm / prior_sigma_mm) ** 2
        A += w * np.eye(3)
        b += w * prior
    p = np.linalg.solve(A, b)
    dist = [np.linalg.norm((np.eye(3) - np.outer(d, d)) @ (p - o)) for o, d in zip(origins, dirs)]
    ang = max((math.degrees(math.acos(np.clip(d1 @ d2, -1, 1)))
               for i, d1 in enumerate(dirs) for d2 in dirs[i + 1:]), default=0.0)
    return p, float(np.sqrt(np.mean(np.square(dist)))), ang


def ear_view_angle(R: np.ndarray, t: np.ndarray, ear_head: np.ndarray) -> float:
    """Angle (deg) between the ear's outward direction and the line to the camera.
    0 = ear pointing straight at the lens, 90 = edge-on (facing the camera)."""
    out_h = np.array([np.sign(ear_head[0]) or 1.0, 0.0, 0.0])
    out_c = R @ out_h
    ear_c = R @ ear_head + t
    to_cam = -ear_c / np.linalg.norm(ear_c)
    return math.degrees(math.acos(np.clip(out_c @ to_cam, -1, 1)))


def project(p_cam: np.ndarray, K: Intrinsics) -> tuple[float, float]:
    return K.fx * p_cam[0] / p_cam[2] + K.cx, K.fy * p_cam[1] / p_cam[2] + K.cy


def reprojection_rms(points2d: dict, model: HeadModel, R, t, K: Intrinsics) -> float:
    m = model.points()
    e = [np.subtract(project(R @ m[k] + t, K), points2d[k]) for k in points2d if k in m]
    return float(np.sqrt(np.mean(np.sum(np.square(e), axis=1)))) if e else float("inf")


# =================================================================== profile
@dataclass
class ListenerProfile:
    name: str
    created: str
    head_scale: float
    ear_l_mm: tuple[float, float, float]          # head frame
    ear_r_mm: tuple[float, float, float]
    ear_distance_mm: float
    sphere_radius_mm: float
    ears_vs_eyes_mm: dict = field(default_factory=dict)    # readable offsets
    # Eye-anchored ear compensation: measured eye spacing and each ear's vector from the
    # eye midpoint (head frame, mm). With these, ears follow the eyes even when no
    # head-pose backend is running: ear = eye midpoint + R * offset.
    ipd_mm: float | None = None
    eye_mid_mm: tuple[float, float, float] | None = None
    ear_offsets_mm: dict = field(default_factory=dict)     # {"left": [x,y,z], "right": [...]}
    # Image patches of each ear from enrollment, for finding the ear that faces the camera
    ear_templates: dict = field(default_factory=dict)
    # SFace embedding of the listener's face, for automatic profile loading (faceid.py)
    face_embedding: list | None = None
    quality: dict = field(default_factory=dict)
    reference: dict = field(default_factory=dict)          # listening seat at enrollment
    camera: dict = field(default_factory=dict)

    def apply(self, head: HeadModel) -> None:
        head.scale = self.head_scale
        head.ear_l_mm = tuple(self.ear_l_mm)
        head.ear_r_mm = tuple(self.ear_r_mm)

    def save(self, path: str | Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps(asdict(self), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ListenerProfile":
        d = json.loads(Path(path).read_text())
        d["ear_l_mm"] = tuple(d["ear_l_mm"])
        d["ear_r_mm"] = tuple(d["ear_r_mm"])
        if d.get("eye_mid_mm") is not None:
            d["eye_mid_mm"] = tuple(d["eye_mid_mm"])
        return cls(**d)

    def reference_ears(self) -> tuple[np.ndarray, np.ndarray] | None:
        r = self.reference
        if not r:
            return None
        return np.array(r["ear_l_cam_mm"]), np.array(r["ear_r_cam_mm"])


def speaker_ear_distances(drivers: dict[str, np.ndarray], ear_l, ear_r) -> dict:
    return {name: {"ear_l": round(float(np.linalg.norm(p - ear_l)), 1),
                   "ear_r": round(float(np.linalg.norm(p - ear_r)), 1)}
            for name, p in drivers.items()}


# =================================================================== session
@dataclass
class Keyframe:
    side: str                 # "left" / "right" (listener's ear)
    frame: np.ndarray
    R: np.ndarray
    t: np.ndarray
    view_deg: float
    predicted: tuple[float, float]
    pixel: tuple[float, float] | None = None
    source: str = ""


class EnrollmentSession:
    """Hardware-free state machine. Feed it (frame, pose, landmarks) each frame;
    it decides what to capture and tells the UI what to ask the listener."""

    SCALE_FRAMES = 30
    # keyframe bins on the ear view angle (deg). Wider spread = better depth along the
    # ear axis; 12 views per ear halved the ear-distance scatter vs 6 views in simulation.
    BINS = ((20, 30), (30, 40), (40, 50), (50, 65))
    PER_BIN = 3
    MIN_VIEWS = 6                # 'next' is allowed once this many are captured
    MAX_REPROJ_PX = 4.0
    CENTER_TOL_PX = 25.0         # eyes must sit this close to the crosshair in step 1
    CROSS_VIEWS = 5              # crosshair captures per ear
    CROSS_AVG_FRAMES = 10        # pose averaged over the frames held still before SPACE
    CROSS_MIN_STEP_DEG = 6.0     # each capture turned at least this much further
    CROSS_GUARD_PX = 90.0        # reject a capture if the model's ear guess is this far off

    def __init__(self, K: Intrinsics, model: HeadModel, distance_mm: float, name: str,
                 mode: str = "crosshair") -> None:
        if mode not in ("crosshair", "sweep"):
            raise ValueError("mode must be 'crosshair' or 'sweep'")
        self.K = K
        self.model = model
        self.distance_mm = distance_mm
        self.name = name
        self.mode = mode
        self.step = "scale"
        self._ipd_px: list[float] = []
        self._scales: list[float] = []
        self._seat: list[tuple[np.ndarray, np.ndarray]] = []
        self.keyframes: list[Keyframe] = []
        self._bins = {s: {b: 0 for b in self.BINS} for s in ("left", "right")}
        self.message = ""

    # --------------------------------------------------------------- capture
    def progress(self) -> float:
        if self.step == "scale":
            return len(self._scales) / self.SCALE_FRAMES
        if self.step in ("left", "right") and self.mode == "crosshair":
            return self.views(self.step) / self.CROSS_VIEWS
        if self.step in ("left", "right"):
            need = len(self.BINS) * self.PER_BIN
            return sum(self._bins[self.step].values()) / need
        return 1.0

    def feed(self, frame: np.ndarray, R, t, points2d: dict) -> str:
        """R, t from the pose solver at the CURRENT model scale. Returns guidance text."""
        if R is None:
            self.message = "Face not found - face the camera"
            return self.message
        yaw, _, _ = ypr_from_rotation(R)
        reproj = reprojection_rms(points2d, self.model, R, t, self.K)

        if self.step == "scale":
            off = self._eye_offset_px(points2d)
            if abs(yaw) > 8:
                self.message = f"Face the camera straight on (yaw {yaw:+.0f})"
            elif off is None or np.hypot(*off) > self.CENTER_TOL_PX:
                self.message = "Centre your eyes on the cross" if off is None else \
                    f"Centre your eyes on the cross ({_nudge(off)})"
            elif reproj > self.MAX_REPROJ_PX:
                self.message = "Hold still"
            else:
                pts = self.model.points()
                eye_mid = R @ ((pts["eye_l"] + pts["eye_r"]) / 2) + t
                self._scales.append(self.distance_mm / eye_mid[2])
                self._ipd_px.append(float(np.hypot(*np.subtract(points2d["eye_l"], points2d["eye_r"]))))
                self._seat.append((R, t))
                self.message = f"Hold still, eyes {self.distance_mm:.0f} mm from the lens"
                if len(self._scales) >= self.SCALE_FRAMES:
                    self._finish_scale()
            return self.message

        if self.step in ("left", "right") and self.mode == "crosshair":
            hist = getattr(self, "_hist", [])
            hist = (hist + [(R, t)])[-self.CROSS_AVG_FRAMES:]
            self._hist = hist
            # average the held-still pose: single-frame PnP jitter is the main error source
            Rm = _mean_rotation([r for r, _ in hist])
            tm = np.mean([tt for _, tt in hist], axis=0)
            spread = max(np.linalg.norm(tt - tm) for _, tt in hist)
            self._live = (frame, Rm, tm, reproj if spread < 40 else float("inf"))   # 40 mm: PnP depth jitter
            side = self.step
            turn = "RIGHT" if side == "left" else "LEFT"
            n = self.views(side)
            self.message = (f"Turn to your {turn}, put your {side.upper()} ear on the cross, "
                            f"press SPACE ({n}/{self.CROSS_VIEWS})")
            if n:
                self.message += " - turn a little further each time"
            return self.message

        if self.step in ("left", "right"):
            side = self.step
            ear_h = self.model.ears()[0 if side == "left" else 1]
            view = ear_view_angle(R, t, ear_h)
            # the listener's left ear faces the camera when they turn to their right
            turn = "RIGHT" if side == "left" else "LEFT"
            if reproj > self.MAX_REPROJ_PX:
                self.message = f"Turn slowly to your {turn} - face points unsteady"
                return self.message
            for b in self.BINS:
                if b[0] <= view < b[1] and self._bins[side][b] < self.PER_BIN:
                    pred = project(R @ ear_h + t, self.K)
                    self.keyframes.append(Keyframe(side, frame.copy(), R.copy(), t.copy(),
                                                   view, pred))
                    self._bins[side][b] += 1
                    break
            missing = [b for b, n in self._bins[side].items() if n < self.PER_BIN]
            if not missing:
                self.advance()
            elif view >= 65 and min(b[0] for b in missing) >= 50:
                self.message = f"Turn slowly to your {turn} to show your {side} ear"
            elif view >= 60:
                self.message = f"Turn slowly to your {turn} to show your {side} ear"
            elif view < 25:
                self.message = "Too far - turn back toward the camera a little"
            else:
                self.message = f"Slowly sweep your head - {side} ear {view:.0f} deg"
            return self.message
        return self.message

    def _eye_offset_px(self, points2d) -> tuple[float, float] | None:
        if "eye_l" not in points2d or "eye_r" not in points2d:
            return None
        u, v = np.add(points2d["eye_l"], points2d["eye_r"]) / 2
        return u - self.K.cx, v - self.K.cy

    def capture_center(self) -> str:
        """Crosshair mode: the listener has put the current ear on the crosshair
        (principal point). Record that pixel with the head pose of this frame."""
        if self.mode != "crosshair" or self.step not in ("left", "right") or \
                getattr(self, "_live", None) is None:
            return "Nothing to capture"
        frame, R, t, reproj = self._live
        side = self.step
        if reproj > self.MAX_REPROJ_PX:
            return "Hold still for a moment, then press SPACE again"
        ear_h = self.model.ears()[0 if side == "left" else 1]
        view = ear_view_angle(R, t, ear_h)
        if view > 75:
            return f"Your {side} ear is not turned toward the camera yet"
        pred = project(R @ ear_h + t, self.K)
        if np.hypot(pred[0] - self.K.cx, pred[1] - self.K.cy) > self.CROSS_GUARD_PX:
            return f"That does not look like your {side} ear on the cross - check the side and try again"
        prev = [k.view_deg for k in self.keyframes if k.side == side]
        if prev and min(abs(view - v) for v in prev) < self.CROSS_MIN_STEP_DEG:
            return "Turn a little further than last time, then put the ear back on the cross"
        kf = Keyframe(side, frame.copy(), R.copy(), t.copy(), view, pred,
                      pixel=(self.K.cx, self.K.cy), source="crosshair")
        self.keyframes.append(kf)
        self._hist = []
        self._bins[side][self.BINS[0]] += 1          # counts views in crosshair mode
        if self.views(side) >= self.CROSS_VIEWS:
            self.advance()
            return "Captured. Now the other side" if self.step == "right" else "Captured both ears"
        return f"Captured {self.views(side)}/{self.CROSS_VIEWS}"

    def views(self, side: str) -> int:
        return sum(self._bins[side].values())

    def can_advance(self) -> bool:
        need = 3 if self.mode == "crosshair" else self.MIN_VIEWS
        return self.step in ("left", "right") and self.views(self.step) >= need

    def advance(self) -> None:
        """Finish the current ear (automatically when all bins are full, or on request
        once MIN_VIEWS are in - e.g. the detector loses the face at extreme turns)."""
        if self.step == "left":
            self.step, self.message = "right", "Now the other side"
        elif self.step == "right":
            self.step = "solve" if self.mode == "crosshair" else "mark"
            self.message = "Got it"

    def _finish_scale(self) -> None:
        s = float(np.median(self._scales))
        self.model.scale *= s
        # eyes were centred on the optical axis at a measured distance: ipd_px = f*IPD/D
        self.ipd_mm = float(np.median(self._ipd_px)) * self.distance_mm / self.K.fx
        # seat pose at the corrected scale: translation scales with the model
        self._seat = [(R, t * s) for R, t in self._seat]
        self.step = "left"

    # ----------------------------------------------------------------- solve
    def set_pixel(self, kf: Keyframe, pixel, source: str) -> None:
        kf.pixel = (float(pixel[0]), float(pixel[1]))
        kf.source = source

    def solve(self, drivers: dict[str, np.ndarray] | None = None) -> ListenerProfile:
        ears, quality = {}, {}
        generic = dict(zip(("left", "right"), self.model.ears()))
        for side in ("left", "right"):
            kfs = [k for k in self.keyframes if k.side == side and k.pixel is not None]
            if len(kfs) < 2:
                raise ValueError(f"{side} ear: need at least 2 marked views, have {len(kfs)}")
            rays = [ray_in_head(EarObservation(k.R, k.t, k.pixel), self.K) for k in kfs]
            o, d = map(np.array, zip(*rays))
            p, rms, base = triangulate(o, d, prior=generic[side])
            # one careless click should not move the ear: drop outlier rays, re-solve
            dist = np.array([np.linalg.norm((np.eye(3) - np.outer(di, di)) @ (p - oi))
                             for oi, di in zip(o, d)])
            keep = dist <= max(3 * np.median(dist), 8.0)
            if keep.sum() >= 2 and not keep.all():
                p, rms, base = triangulate(o[keep], d[keep], prior=generic[side])
            ears[side] = p
            quality[side] = {"views": int(keep.sum()), "rejected": int((~keep).sum()),
                             "ray_rms_mm": round(rms, 2),
                             "baseline_deg": round(base, 1),
                             "sources": sorted({k.source for k in kfs}),
                             "moved_from_generic_mm": round(float(np.linalg.norm(p - generic[side])), 1)}

        el, er = ears["left"], ears["right"]
        if not 100 <= np.linalg.norm(el - er) <= 200:
            raise ValueError(f"ear distance {np.linalg.norm(el - er):.0f} mm is not plausible "
                             "- re-mark the tragion points")

        head = HeadModel(**{**asdict(self.model), "ear_l_mm": tuple(el), "ear_r_mm": tuple(er)})
        pts = head.points()
        eye_mid = (pts["eye_l"] + pts["eye_r"]) / 2
        offsets, vectors = {}, {}
        for side, e in (("left", el), ("right", er)):
            d = e - eye_mid
            vectors[side] = np.round(d, 2).tolist()
            offsets[side] = {"behind_mm": round(float(d[2]), 1), "below_mm": round(float(d[1]), 1),
                             "out_mm": round(float(abs(d[0])), 1)}

        reference = {}
        if self._seat:
            R = _mean_rotation([r for r, _ in self._seat])
            t = np.median([tt for _, tt in self._seat], axis=0)
            ear_l, ear_r = R @ el + t, R @ er + t
            reference = {"head_center_mm": np.round((ear_l + ear_r) / 2, 1).tolist(),
                         "ypr_deg": [round(v, 1) for v in ypr_from_rotation(R)],
                         "R": np.round(R, 6).tolist(), "t_mm": np.round(t, 2).tolist(),
                         "ear_l_cam_mm": np.round(ear_l, 1).tolist(),
                         "ear_r_cam_mm": np.round(ear_r, 1).tolist()}
            if drivers:
                reference["speaker_to_ear_mm"] = speaker_ear_distances(drivers, ear_l, ear_r)

        return ListenerProfile(
            name=self.name, created=time.strftime("%Y-%m-%d %H:%M:%S"),
            head_scale=round(self.model.scale, 4),
            ear_l_mm=tuple(np.round(el, 2).tolist()), ear_r_mm=tuple(np.round(er, 2).tolist()),
            ear_distance_mm=round(float(np.linalg.norm(el - er)), 1),
            sphere_radius_mm=round(head.sphere_radius_mm(), 1),
            ears_vs_eyes_mm=offsets, quality=quality, reference=reference,
            ipd_mm=round(getattr(self, "ipd_mm", 0.0), 2) or None,
            ear_templates=make_templates(self.keyframes, {"left": el, "right": er}, self.K),
            eye_mid_mm=tuple(np.round(eye_mid, 2).tolist()), ear_offsets_mm=vectors,
            camera={"fx": self.K.fx, "cx": self.K.cx, "cy": self.K.cy})


def _nudge(off) -> str:
    du, dv = off
    parts = []
    if abs(du) > 8:   # du > 0: eyes on the camera's right = the listener's left
        parts.append(f"to your {'right' if du > 0 else 'left'} {abs(du):.0f} px")
    if abs(dv) > 8:
        parts.append(f"{'up' if dv > 0 else 'down'} {abs(dv):.0f} px")
    return "move " + ", ".join(parts) if parts else "almost"


def _mean_rotation(Rs: list[np.ndarray]) -> np.ndarray:
    U, _, Vt = np.linalg.svd(np.sum(Rs, axis=0))
    R = U @ Vt
    if np.linalg.det(R) < 0:
        U[:, -1] *= -1
        R = U @ Vt
    return R


# ============================================================ ear detector
class EarDetector:
    """Optional automatic tragion finder from a Haar ear cascade (e.g. the
    haarcascade_mcs_leftear.xml / _rightear.xml files from older OpenCV
    releases). Searches only around the model's prediction. The tragion is
    taken on the face-side edge of the detected ear box, at mid height."""

    def __init__(self, left_path: str, right_path: str) -> None:
        import cv2

        self.cv2 = cv2
        self.casc = {}
        for side, path in (("left", left_path), ("right", right_path)):
            c = cv2.CascadeClassifier(path)
            if c.empty():
                raise RuntimeError(f"could not load ear cascade {path}")
            self.casc[side] = c

    def __call__(self, kf: Keyframe, K: Intrinsics, ear_size_px: float) -> tuple[float, float] | None:
        cv2 = self.cv2
        u, v = kf.predicted
        r = int(1.6 * ear_size_px)
        H, W = kf.frame.shape[:2]
        x0, y0 = max(int(u) - r, 0), max(int(v) - r, 0)
        x1, y1 = min(int(u) + r, W), min(int(v) + r, H)
        if x1 - x0 < 20 or y1 - y0 < 20:
            return None
        g = cv2.equalizeHist(cv2.cvtColor(kf.frame[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
        hits = []
        for c in self.casc.values():   # left/right naming differs between cascade sets; try both
            hits += list(c.detectMultiScale(g, 1.05, 3, minSize=(max(int(ear_size_px * 0.4), 12),) * 2))
        if not hits:
            return None
        x, y, w, h = min(hits, key=lambda b: (x0 + b[0] + b[2] / 2 - u) ** 2
                         + (y0 + b[1] + b[3] / 2 - v) ** 2)
        # tragion: on the box edge nearer the face (toward the projected head origin)
        head_u = project(kf.t, K)[0]
        frac = 0.2 if head_u < x0 + x + w / 2 else 0.8
        return x0 + x + frac * w, y0 + y + h / 2
