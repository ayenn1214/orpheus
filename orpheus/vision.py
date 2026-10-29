"""Viola-Jones face detection with CamShift (hue histogram) bridging.

Same two stages as the 2017 MATLAB build - vision.CascadeObjectDetector and
vision.HistogramBasedTracker - but run together every frame instead of
handing off permanently: detection is the primary source, CamShift only
carries the target through short dropouts (head turned, hand in front).
"""

from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np

from .config import DetectorConfig

Box = tuple[int, int, int, int]  # x, y, w, h (full-resolution pixels)


@dataclass
class Observation:
    box: Box
    source: str        # "frontal", "profile", "camshift"
    t: float


def _cascade(name: str) -> cv2.CascadeClassifier:
    c = cv2.CascadeClassifier(cv2.data.haarcascades + name)
    if c.empty():
        raise RuntimeError(f"could not load OpenCV cascade {name}")
    return c


class FaceTracker:
    def __init__(self, cfg: DetectorConfig) -> None:
        self.cfg = cfg
        self.frontal = _cascade("haarcascade_frontalface_default.xml")
        self.profile = _cascade("haarcascade_profileface.xml") if cfg.use_profile else None
        self.last: Observation | None = None
        self._last_detect_t: float = -1e9
        self._hist: np.ndarray | None = None
        self._window: Box | None = None
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self._term = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 1)

    # ------------------------------------------------------------ public
    def update(self, frame_bgr: np.ndarray, t: float) -> Observation | None:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        # Local (CLAHE) rather than global equalisation: a bright wall or window
        # stretches a global histogram and washes out the face.
        gray = self._clahe.apply(gray)

        found = self._detect(gray)
        if found is not None:
            box, source = found
            self._last_detect_t = t
            self._seed_histogram(frame_bgr, box)
            self.last = Observation(box, source, t)
            return self.last

        if self._hist is not None and t - self._last_detect_t <= self.cfg.lost_hold_s:
            box = self._camshift(frame_bgr)
            if box is not None:
                self.last = Observation(box, "camshift", t)
                return self.last

        self.last = None
        self._hist = None
        return None

    # ------------------------------------------------------- detection
    def _detect(self, gray: np.ndarray) -> tuple[Box, str] | None:
        s = self.cfg.detect_scale
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else gray
        min_px = max(int(self.cfg.min_face_px * s), 20)
        kw = dict(scaleFactor=self.cfg.scale_factor, minNeighbors=self.cfg.min_neighbors,
                  minSize=(min_px, min_px))

        candidates: list[tuple[Box, str]] = []
        for b in self.frontal.detectMultiScale(small, **kw):
            candidates.append((self._unscale(b, s), "frontal"))

        if not candidates and self.profile is not None:
            for b in self.profile.detectMultiScale(small, **kw):
                candidates.append((self._unscale(b, s), "profile"))
            flipped = cv2.flip(small, 1)
            W = small.shape[1]
            for x, y, w, h in self.profile.detectMultiScale(flipped, **kw):
                candidates.append((self._unscale((W - x - w, y, w, h), s), "profile"))

        if not candidates:
            return None
        return self._pick(candidates)

    @staticmethod
    def _unscale(b, s: float) -> Box:
        x, y, w, h = (int(round(v / s)) for v in b)
        return x, y, w, h

    def _pick(self, candidates: list[tuple[Box, str]]) -> tuple[Box, str]:
        """Stick with the current listener; otherwise take the largest (closest) face."""
        if self.last is not None:
            lx, ly, lw, lh = self.last.box
            lc = np.array([lx + lw / 2, ly + lh / 2])

            def dist(c):
                x, y, w, h = c[0]
                return np.linalg.norm(np.array([x + w / 2, y + h / 2]) - lc)

            nearest = min(candidates, key=dist)
            if dist(nearest) < 1.5 * lw:
                return nearest
        return max(candidates, key=lambda c: c[0][2] * c[0][3])

    # ---------------------------------------------------------- camshift
    @staticmethod
    def _hsv_mask(hsv: np.ndarray) -> np.ndarray:
        return cv2.inRange(hsv, (0, 40, 40), (180, 255, 255))

    def _seed_histogram(self, frame_bgr: np.ndarray, box: Box) -> None:
        x, y, w, h = box
        # inner region only: skip hair/background at the box edges
        ix, iy, iw, ih = x + w // 5, y + h // 5, max(3 * w // 5, 1), max(3 * h // 5, 1)
        roi = cv2.cvtColor(frame_bgr[iy:iy + ih, ix:ix + iw], cv2.COLOR_BGR2HSV)
        if roi.size == 0:
            return
        hist = cv2.calcHist([roi], [0], self._hsv_mask(roi), [32], [0, 180])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)
        self._hist = hist
        self._window = box

    def _camshift(self, frame_bgr: np.ndarray) -> Box | None:
        assert self._hist is not None and self._window is not None and self.last is not None
        hsv = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2HSV)
        back = cv2.calcBackProject([hsv], [0], self._hist, [0, 180], 1)
        back &= self._hsv_mask(hsv)
        _, window = cv2.CamShift(back, self._window, self._term)
        x, y, w, h = window
        if w < 10 or h < 10:
            return None
        self._window = (x, y, w, h)
        # CamShift's box size drifts with skin/background colour, so keep the last
        # detected size (which sets depth) and only take the centre from CamShift.
        _, _, lw, lh = self.last.box
        cx, cy = x + w / 2, y + h / 2
        return int(cx - lw / 2), int(cy - lh / 2), lw, lh
