"""Facial landmark backends. All return `Face`: a box plus named 2D points in
the head model's vocabulary (eye_l/eye_r/nose/mouth_l/mouth_r/chin, where
l/r are the LISTENER's left/right).

    haar       Viola-Jones face + eye cascades. Ships with OpenCV, no download.
               2 points -> roll only; yaw from profile detections.
    yunet      cv2.FaceDetectorYN, 5 points -> full PnP pose.
               model: face_detection_yunet_2023mar.onnx (opencv_zoo)
    mediapipe  MediaPipe Face Landmarker, 478 points (6 used) -> full PnP pose.
               model: face_landmarker.task
"""

from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from .config import DetectorConfig
from .vision import FaceTracker

Box = tuple[int, int, int, int]

MODEL_URLS = {
    "yunet": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/"
             "face_detection_yunet_2023mar.onnx",
    "mediapipe": "https://storage.googleapis.com/mediapipe-models/face_landmarker/"
                 "face_landmarker/float16/latest/face_landmarker.task",
}


@dataclass
class Face:
    box: Box
    points: dict[str, tuple[float, float]] = field(default_factory=dict)
    source: str = ""
    profile_sign: int = 0        # +1 facing camera +X, -1 facing -X, 0 frontal/unknown

    def yunet_row(self) -> np.ndarray | None:
        """The 15-value YuNet layout (box, subject-right eye, subject-left eye, nose,
        mouth right, mouth left, score) that cv2.FaceRecognizerSF.alignCrop expects.
        Built from named points, so any backend with 5 points can be identified."""
        need = ("eye_r", "eye_l", "nose", "mouth_r", "mouth_l")
        if not all(k in self.points for k in need):
            return None
        row = list(self.box) + [v for k in need for v in self.points[k]] + [1.0]
        return np.array(row, np.float32).reshape(1, 15)


def _iou(a: Box, b: Box) -> float:
    ax, ay, aw, ah = a
    bx, by, bw, bh = b
    ix = max(0, min(ax + aw, bx + bw) - max(ax, bx))
    iy = max(0, min(ay + ah, by + bh) - max(ay, by))
    inter = ix * iy
    return inter / max(aw * ah + bw * bh - inter, 1)


def _order_lr(a, b):
    """Return (listener-left, listener-right) for a pair of image points.
    Facing the camera (|yaw| < 90), the listener's left feature is image-right."""
    return (a, b) if a[0] >= b[0] else (b, a)


# ------------------------------------------------------------------- haar
class HaarBackend:
    name = "haar"

    def __init__(self, cfg: DetectorConfig) -> None:
        self.tracker = FaceTracker(cfg)
        self.eye = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_eye.xml")
        if self.eye.empty():
            raise RuntimeError("could not load haarcascade_eye.xml")
        self._clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))

    def __call__(self, frame_bgr: np.ndarray, t: float) -> Face | None:
        obs = self.tracker.update(frame_bgr, t)
        if obs is None:
            return None
        face = Face(box=obs.box, source=f"haar-{obs.source}")
        if obs.source == "camshift":
            return face
        eyes = self._eyes(frame_bgr, obs.box)
        x, _, w, _ = obs.box
        if len(eyes) == 2:
            face.points["eye_l"], face.points["eye_r"] = _order_lr(*eyes)
        elif len(eyes) == 1 and obs.source == "profile":
            # In profile the visible eye sits toward the side the nose points.
            face.profile_sign = 1 if eyes[0][0] > x + w / 2 else -1
        return face

    def detect_all(self, frame_bgr: np.ndarray, t: float, max_faces: int = 2) -> list[Face]:
        """The tracked face first, then other frontal faces (no CamShift for those)."""
        main = self(frame_bgr, t)
        out = [main] if main is not None else []
        if max_faces <= 1:
            return out
        cfg = self.tracker.cfg
        gray = self._clahe.apply(cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY))
        s = cfg.detect_scale
        small = cv2.resize(gray, None, fx=s, fy=s, interpolation=cv2.INTER_AREA) if s != 1 else gray
        m = max(int(cfg.min_face_px * s), 20)
        hits = self.tracker.frontal.detectMultiScale(small, cfg.scale_factor, cfg.min_neighbors,
                                                     minSize=(m, m))
        boxes = sorted((tuple(int(round(v / s)) for v in b) for b in hits),
                       key=lambda b: -b[2] * b[3])
        for box in boxes:
            if len(out) >= max_faces:
                break
            if any(_iou(box, f.box) > 0.2 for f in out):
                continue
            face = Face(box=box, source="haar-frontal")
            eyes = self._eyes(frame_bgr, box)
            if len(eyes) == 2:
                face.points["eye_l"], face.points["eye_r"] = _order_lr(*eyes)
            out.append(face)
        return out

    def _eyes(self, frame_bgr, box) -> list[tuple[float, float]]:
        x, y, w, h = box
        H, W = frame_bgr.shape[:2]
        x0, y0 = max(x, 0), max(y + h // 6, 0)
        x1, y1 = min(x + w, W), min(y + h * 6 // 10, H)
        if x1 - x0 < 20 or y1 - y0 < 10:
            return []
        roi = self._clahe.apply(cv2.cvtColor(frame_bgr[y0:y1, x0:x1], cv2.COLOR_BGR2GRAY))
        hits = self.eye.detectMultiScale(roi, 1.05, 3, minSize=(max(w // 10, 8),) * 2,
                                         maxSize=(w // 3, w // 3))
        cands = [(x0 + ex + ew / 2, y0 + ey + eh / 2, ew) for ex, ey, ew, eh in hits]
        # one per half of the face box: the candidate nearest the expected eye spot
        out = []
        for side in (0.30, 0.70):
            ex_u, ex_v = x + side * w, y + 0.40 * h
            half = [c for c in cands if (c[0] < x + w / 2) == (side < 0.5)]
            if half:
                c = min(half, key=lambda c: (c[0] - ex_u) ** 2 + (c[1] - ex_v) ** 2)
                out.append((c[0], c[1]))
        return out


# ------------------------------------------------------------------ yunet
class YuNetBackend:
    name = "yunet"

    def __init__(self, model_path: str, score: float = 0.75) -> None:
        if not hasattr(cv2, "FaceDetectorYN"):
            raise RuntimeError("this OpenCV build has no FaceDetectorYN (need >= 4.5.4)")
        self.det = cv2.FaceDetectorYN.create(model_path, "", (320, 320), score, 0.3, 50)
        self._size = None
        self._last: np.ndarray | None = None

    def __call__(self, frame_bgr: np.ndarray, t: float) -> Face | None:
        faces = self.detect_all(frame_bgr, t, 1)
        return faces[0] if faces else None

    def detect_all(self, frame_bgr: np.ndarray, t: float, max_faces: int = 2) -> list[Face]:
        H, W = frame_bgr.shape[:2]
        if self._size != (W, H):
            self.det.setInputSize((W, H))
            self._size = (W, H)
        _, faces = self.det.detect(frame_bgr)
        if faces is None or len(faces) == 0:
            self._last = None
            return []
        first = self._pick(faces)
        self._last = first
        rest = sorted((f for f in faces if f is not first), key=lambda f: -f[2] * f[3])
        return [self._face(f) for f in [first] + rest][:max_faces]

    @staticmethod
    def _face(f: np.ndarray) -> Face:
        x, y, w, h = (int(v) for v in f[:4])
        p = f[4:14].reshape(5, 2)
        eye_l, eye_r = _order_lr(tuple(p[0]), tuple(p[1]))
        mouth_l, mouth_r = _order_lr(tuple(p[3]), tuple(p[4]))
        return Face(box=(x, y, w, h), source="yunet",
                    points={"eye_l": eye_l, "eye_r": eye_r, "nose": tuple(p[2]),
                            "mouth_l": mouth_l, "mouth_r": mouth_r})

    def _pick(self, faces: np.ndarray) -> np.ndarray:
        if self._last is not None:
            c0 = self._last[:2] + self._last[2:4] / 2
            d = [np.linalg.norm(f[:2] + f[2:4] / 2 - c0) for f in faces]
            i = int(np.argmin(d))
            if d[i] < 1.5 * self._last[2]:
                return faces[i]
        return faces[int(np.argmax(faces[:, 2] * faces[:, 3]))]


# -------------------------------------------------------------- mediapipe
class MediaPipeBackend:
    name = "mediapipe"
    # Face Landmarker topology; "left"/"right" are the subject's.
    IDX = {"eye_l": (362, 263), "eye_r": (33, 133), "nose": (1,),
           "mouth_l": (291,), "mouth_r": (61,), "chin": (152,)}

    def __init__(self, model_path: str, num_faces: int = 2) -> None:
        import mediapipe as mp
        from mediapipe.tasks.python import BaseOptions
        from mediapipe.tasks.python.vision import (FaceLandmarker, FaceLandmarkerOptions,
                                                   RunningMode)

        self.mp = mp
        opts = FaceLandmarkerOptions(base_options=BaseOptions(model_asset_path=model_path),
                                     running_mode=RunningMode.VIDEO, num_faces=num_faces)
        self.lm = FaceLandmarker.create_from_options(opts)
        self._last_ms = -1

    def __call__(self, frame_bgr: np.ndarray, t: float) -> Face | None:
        faces = self.detect_all(frame_bgr, t, 1)
        return faces[0] if faces else None

    def detect_all(self, frame_bgr: np.ndarray, t: float, max_faces: int = 2) -> list[Face]:
        H, W = frame_bgr.shape[:2]
        ms = max(int(t * 1000), self._last_ms + 1)   # VIDEO mode needs increasing timestamps
        self._last_ms = ms
        img = self.mp.Image(image_format=self.mp.ImageFormat.SRGB,
                            data=cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB))
        res = self.lm.detect_for_video(img, ms)
        out = []
        for L in (res.face_landmarks or [])[:max_faces]:
            xy = np.array([(p.x * W, p.y * H) for p in L])
            pts = {k: tuple(xy[list(ix)].mean(axis=0)) for k, ix in self.IDX.items()}
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            out.append(Face(box=(int(x0), int(y0), int(x1 - x0), int(y1 - y0)), points=pts,
                            source="mediapipe"))
        return out


def detect_faces(backend, frame_bgr: np.ndarray, t: float, max_faces: int) -> list[Face]:
    """All faces from a backend; single-face callables (tests, custom) are wrapped."""
    if max_faces > 1 and hasattr(backend, "detect_all"):
        return backend.detect_all(frame_bgr, t, max_faces)
    face = backend(frame_bgr, t)
    if face is None:
        return []
    return list(face) if isinstance(face, (list, tuple)) else [face]


def make_backend(name: str, cfg: DetectorConfig, model_path: str = ""):
    if name == "haar":
        return HaarBackend(cfg)
    if not model_path:
        raise SystemExit(f"backend '{name}' needs a model file; download:\n  {MODEL_URLS[name]}\n"
                         f"and set detector.model_path (or --model)")
    if name == "yunet":
        return YuNetBackend(model_path)
    if name == "mediapipe":
        return MediaPipeBackend(model_path, num_faces=max(cfg.max_faces, 1))
    raise ValueError(f"unknown backend {name}")
