"""Hidden-ear-only compensation.

At any moment at most one ear can face the camera. That ear is found directly in
the image; only the ear turned away (the missing one) is placed from the eyes with
the enrolled offset. Facing the camera head-on, both ears are edge-on and both
fall back to the offset.

The detector is personal: enrollment already captured each ear at a known pixel
(the crosshair, or your click), so those patches are stored in the profile as
templates and matched at run time around where the offset predicts the ear,
scaled for distance. No ear cascade or model download is needed.
"""

from __future__ import annotations

import base64
import math
from dataclasses import dataclass

import cv2
import numpy as np

from .head import Listener, make_listener, ypr_from_rotation

EAR_HEIGHT_MM = 65.0          # tragion-centred patch spans ~1.4x a typical ear height
PATCH_SCALE = 1.4
TEMPLATE_PX = 64
_CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(4, 4))


@dataclass
class EarFindConfig:
    enabled: bool = True
    visible_max_deg: float = 65.0     # ear counts as facing the camera below this view angle
    min_score: float = 0.55           # normalised cross-correlation acceptance
    search_px: int = 110              # search radius around the offset prediction
    max_correction_mm: float = 80.0   # with a full pose, ignore matches implying a larger jump


def _gray(img: np.ndarray) -> np.ndarray:
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    return _CLAHE.apply(g)


def _patch_px(fx: float, z_mm: float) -> float:
    return fx * EAR_HEIGHT_MM * PATCH_SCALE / max(z_mm, 1.0)


def view_angle(listener: Listener, side: str) -> float:
    """Angle between the ear's outward axis and the line to the camera (deg)."""
    ear = listener.ear_l if side == "left" else listener.ear_r
    out = listener.R @ np.array([1.0 if side == "left" else -1.0, 0.0, 0.0])
    to_cam = -ear / np.linalg.norm(ear)
    return math.degrees(math.acos(float(np.clip(out @ to_cam, -1, 1))))


# --------------------------------------------------------------- templates
def make_templates(keyframes, ears_head: dict, K) -> dict:
    """Crop each enrollment view around its marked tragion pixel.
    ears_head: {"left": xyz, "right": xyz} solved ears in the head frame."""
    out: dict[str, list] = {"left": [], "right": []}
    for kf in keyframes:
        if kf.pixel is None or kf.frame is None or kf.frame.ndim != 3:
            continue
        ear_cam = kf.R @ np.asarray(ears_head[kf.side]) + kf.t
        size = _patch_px(K.fx, ear_cam[2])
        u, v = kf.pixel
        h = int(round(size / 2))
        H, W = kf.frame.shape[:2]
        if h < 8 or u - h < 0 or v - h < 0 or u + h > W or v + h > H:
            continue
        crop = _gray(kf.frame)[int(v) - h:int(v) + h, int(u) - h:int(u) + h]
        crop = cv2.resize(crop, (TEMPLATE_PX, TEMPLATE_PX), interpolation=cv2.INTER_AREA)
        ok, png = cv2.imencode(".png", crop)
        if ok:
            out[kf.side].append({"view_deg": round(float(kf.view_deg), 1),
                                 "png_b64": base64.b64encode(png.tobytes()).decode()})
    return out


@dataclass
class _Template:
    view_deg: float
    img: np.ndarray


def _decode(entries) -> list[_Template]:
    out = []
    for e in entries or []:
        buf = np.frombuffer(base64.b64decode(e["png_b64"]), np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
        if img is not None:
            out.append(_Template(float(e["view_deg"]), img))
    return out


# ----------------------------------------------------------------- refiner
class EarRefiner:
    """Find the ear that faces the camera and use it as one more pose point.

    The seen ear's pixel is added to the face landmarks as a correspondence to the
    enrolled ear position, and the head pose is re-solved. The hidden ear is then
    placed with its stored offset under that corrected pose. With eyes only (two
    points, yaw unobservable) the seen ear is the third point that fixes yaw and
    depth; with a full landmark set it tightens the pose on the visible side.
    """

    def __init__(self, templates: dict, cfg: EarFindConfig, K) -> None:
        self.cfg = cfg
        self.K = K
        self.tpl = {s: _decode(templates.get(s)) for s in ("left", "right")}
        self.last_score = {"left": 0.0, "right": 0.0}

    @property
    def usable(self) -> bool:
        return self.cfg.enabled and any(self.tpl.values())

    def refine(self, frame_bgr: np.ndarray, R: np.ndarray, t: np.ndarray, points2d: dict,
               model, yaw_known: bool):
        """Returns (R, t, sources) with sources[side] in {"seen", "offset"}."""
        sources = {"left": "offset", "right": "offset"}
        if not self.usable:
            return R, t, sources
        ears_h = dict(zip(("left", "right"), model.ears()))
        L = make_listener(R, t, model, "tmp")
        if yaw_known:
            cands = [s for s in ("left", "right") if view_angle(L, s) <= self.cfg.visible_max_deg]
        else:
            cands = ["left", "right"]          # eyes only: let the image decide which side shows

        gray = _gray(frame_bgr)
        hits = {}
        for side in cands:
            if yaw_known:
                m = self._match(gray, L, side)
            else:
                m = self._match_band(gray, points2d, side, model)
            self.last_score[side] = m[2] if m else 0.0
            if m and m[2] >= self.cfg.min_score:
                hits[side] = m
        if not hits:
            return R, t, sources
        side = max(hits, key=lambda s: hits[s][2])     # only one ear can face the camera
        u, v, _ = hits[side]

        pose = self._resolve(points2d, model, ears_h[side], (u, v), R, t, side)
        if pose is None:
            return R, t, sources
        R2, t2 = pose
        # With a full pose a big jump means a false match. With eyes only the estimate can
        # legitimately be 100+ mm off (yaw unseen), so there the P3P checks decide instead.
        moved = np.linalg.norm((R2 @ ears_h[side] + t2) - (R @ ears_h[side] + t))
        if yaw_known and moved > self.cfg.max_correction_mm:
            return R, t, sources
        sources[side] = "seen"
        return R2, t2, sources

    def _resolve(self, points2d, model, ear_h, ear_px, R0, t0, side):
        pts = model.points()
        names = [n for n in points2d if n in pts]
        obj = np.array([pts[n] for n in names] + [ear_h], np.float64)
        img = np.array([points2d[n] for n in names] + [ear_px], np.float64)
        cam = np.array([[self.K.fx, 0, self.K.cx], [0, self.K.fy, self.K.cy], [0, 0, 1]], np.float64)
        rv0, _ = cv2.Rodrigues(R0)
        if len(obj) >= 4:
            ok, rv, tv = cv2.solvePnP(obj, img, cam, None, rv0.copy(), t0.reshape(3, 1).copy(),
                                      True, cv2.SOLVEPNP_ITERATIVE)
            return (cv2.Rodrigues(rv)[0], tv.ravel()) if ok else None
        # eyes + one ear: P3P gives up to 4 poses; keep the one with that ear facing the
        # camera, a plausible head angle, and nearest the eyes-only estimate
        try:
            n, rvs, tvs = cv2.solveP3P(obj, img, cam, None, cv2.SOLVEPNP_AP3P)
        except cv2.error:
            return None
        best, best_cost = None, float("inf")
        for rv, tv in zip(rvs, tvs):
            R, tv = cv2.Rodrigues(rv)[0], tv.ravel()
            if tv[2] < 150:
                continue
            Ltmp = make_listener(R, tv, model, "tmp")
            if view_angle(Ltmp, side) > 80:
                continue
            yaw, pitch, _ = ypr_from_rotation(R)
            if abs(yaw) > 85 or abs(pitch) > 60:
                continue
            cost = abs(tv[2] - t0[2]) + 2.0 * abs(np.linalg.norm(tv[:2] - t0[:2]))
            if cost < best_cost:
                best, best_cost = (R, tv), cost
        return best

    def _match_band(self, gray, points2d, side: str, model):
        """Eyes only: the 3D guess is unreliable, so search where an ear can appear
        relative to the eyes in the image - a band at eye level on that side, 0.6 to 4
        eye-spacings out - with the patch size taken from the eye spacing. A turned head
        shrinks the eye spacing by cos(yaw) but not the ear, hence the larger scales."""
        if "eye_l" not in points2d or "eye_r" not in points2d:
            return None
        el, er = np.asarray(points2d["eye_l"]), np.asarray(points2d["eye_r"])
        ipd_px = float(np.linalg.norm(el - er))
        if ipd_px < 8:
            return None
        mid = (el + er) / 2
        ipd_mm = float(np.linalg.norm(model.points()["eye_l"] - model.points()["eye_r"]))
        size = ipd_px * EAR_HEIGHT_MM * PATCH_SCALE / ipd_mm
        sgn = 1 if side == "left" else -1          # listener's left ear shows on image right
        H, W = gray.shape
        xa, xb = mid[0] + sgn * 0.6 * ipd_px, mid[0] + sgn * 4.0 * ipd_px
        x0, x1 = int(max(min(xa, xb) - size / 2, 0)), int(min(max(xa, xb) + size / 2, W))
        y0, y1 = int(max(mid[1] - 0.4 * ipd_px - size / 2, 0)), int(min(mid[1] + 1.0 * ipd_px + size / 2, H))
        roi = gray[y0:y1, x0:x1]
        best = None
        for tp in self.tpl[side]:
            for k in (0.9, 1.1, 1.35, 1.6):
                sz = max(int(round(size * k)), 8)
                if not roi.size or sz >= min(roi.shape):
                    continue
                res = cv2.matchTemplate(roi, cv2.resize(tp.img, (sz, sz)), cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if best is None or score > best[2]:
                    best = (x0 + loc[0] + sz / 2, y0 + loc[1] + sz / 2, float(score))
        return best

    def _match(self, gray, L: Listener, side: str):
        tpls = self.tpl[side]
        if not tpls:
            return None
        ear = L.ear_l if side == "left" else L.ear_r
        if ear[2] <= 1:
            return None
        u0 = self.K.fx * ear[0] / ear[2] + self.K.cx
        v0 = self.K.fy * ear[1] / ear[2] + self.K.cy
        size = int(round(_patch_px(self.K.fx, ear[2])))
        if size < 12:
            return None
        r = self.cfg.search_px + size // 2
        H, W = gray.shape
        x0, y0 = max(int(u0) - r, 0), max(int(v0) - r, 0)
        x1, y1 = min(int(u0) + r, W), min(int(v0) + r, H)
        roi = gray[y0:y1, x0:x1]
        if roi.shape[0] <= size or roi.shape[1] <= size:
            return None
        best = None
        va = view_angle(L, side)                     # the two enrollment views nearest now
        use = sorted(tpls, key=lambda tp: abs(tp.view_deg - va))[:2]
        scales = (0.9, 1.0, 1.1)
        for tp in use:
            for k in scales:
                s = max(int(round(size * k)), 8)
                if s >= min(roi.shape):
                    continue
                res = cv2.matchTemplate(roi, cv2.resize(tp.img, (s, s)), cv2.TM_CCOEFF_NORMED)
                _, score, _, loc = cv2.minMaxLoc(res)
                if best is None or score > best[2]:
                    best = (x0 + loc[0] + s / 2, y0 + loc[1] + s / 2, float(score))
        return best
