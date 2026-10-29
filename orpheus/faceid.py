"""Recognise who is sitting there and load their ear profile.

OpenCV's SFace recogniser turns an aligned face into a 128-number embedding;
two embeddings of the same person have cosine similarity above ~0.36. Each
listener profile stores one embedding (captured during enrollment, or later
with --add-face), so at run time the profile is found by nearest embedding.

Alignment needs five points (eyes, nose, mouth corners), so this works with the
yunet and mediapipe backends; the haar backend only finds eyes.

model: face_recognition_sface_2021dec.onnx from opencv_zoo
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .enroll import ListenerProfile
from .landmarks import Face

log = logging.getLogger(__name__)

MODEL_URL = ("https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/"
             "face_recognition_sface_2021dec.onnx")


def unit(v) -> np.ndarray:
    v = np.asarray(v, float).ravel()
    n = np.linalg.norm(v)
    return v / n if n > 0 else v


class SFaceEmbedder:
    """cv2.FaceRecognizerSF wrapper: (frame, Face) -> unit embedding."""

    def __init__(self, model_path: str) -> None:
        if not hasattr(cv2, "FaceRecognizerSF"):
            raise RuntimeError("this OpenCV build has no FaceRecognizerSF (need >= 4.5.4)")
        if not model_path or not Path(model_path).exists():
            raise SystemExit(f"face ID needs the SFace model; download:\n  {MODEL_URL}\n"
                             f"and set faceid.model_path (or --faceid-model)")
        self.rec = cv2.FaceRecognizerSF.create(model_path, "")

    def __call__(self, frame_bgr: np.ndarray, face: Face) -> np.ndarray | None:
        row = face.yunet_row()
        if row is None:
            return None
        crop = self.rec.alignCrop(frame_bgr, row)
        return unit(self.rec.feature(crop))


class ProfileLibrary:
    """All profiles in a folder that carry a face embedding."""

    def __init__(self, directory: str | Path) -> None:
        self.dir = Path(directory)
        self.profiles: list[ListenerProfile] = []
        self._emb: np.ndarray = np.zeros((0, 1))
        self.reload()

    def reload(self) -> None:
        self.profiles = []
        if self.dir.is_dir():
            for p in sorted(self.dir.glob("*.json")):
                try:
                    prof = ListenerProfile.load(p)
                except Exception as e:                       # noqa: BLE001 - skip bad files
                    log.warning("skipping %s: %s", p, e)
                    continue
                if prof.face_embedding:
                    self.profiles.append(prof)
        self._emb = np.array([unit(p.face_embedding) for p in self.profiles]) \
            if self.profiles else np.zeros((0, 1))
        log.info("face ID: %d profile(s) with a face in %s", len(self.profiles), self.dir)

    def match(self, emb: np.ndarray) -> tuple[ListenerProfile | None, float]:
        if not self.profiles:
            return None, 0.0
        sims = self._emb @ unit(emb)
        i = int(np.argmax(sims))
        return self.profiles[i], float(sims[i])


@dataclass
class IdState:
    """Per-listener identity bookkeeping (lives on a pipeline track)."""
    name: str = ""                  # confirmed profile name, "" = unknown
    candidate: str = ""
    streak: int = 0
    frames: int = 0
    score: float = 0.0


class FaceIdentifier:
    """Decides, per track, which profile it is, with hysteresis so one bad frame
    never swaps profiles mid-listen."""

    def __init__(self, embedder, library: ProfileLibrary, threshold: float = 0.363,
                 confirm_frames: int = 5, interval_frames: int = 15) -> None:
        self.embed = embedder
        self.lib = library
        self.threshold = threshold
        self.confirm = confirm_frames
        self.interval = interval_frames

    def update(self, ids: IdState, frame_bgr: np.ndarray, face: Face) -> ListenerProfile | None:
        """Returns a profile when the identity has just been confirmed or changed."""
        ids.frames += 1
        if ids.name and ids.frames % self.interval:
            return None                                      # identified: only re-check now and then
        emb = self.embed(frame_bgr, face)
        if emb is None:
            return None
        prof, score = self.lib.match(emb)
        name = prof.name if prof is not None and score >= self.threshold else ""
        ids.score = score
        if name == ids.candidate:
            ids.streak += 1
        else:
            ids.candidate, ids.streak = name, 1
        if ids.streak >= self.confirm and name != ids.name:
            ids.name = name
            if name:
                log.info("recognised %s (similarity %.2f)", name, score)
                return prof
        return None


def capture_embedding(embedder, faces_and_frames, min_samples: int = 10) -> list[float] | None:
    """Average embedding over frames where the listener faces the camera."""
    embs = [e for e in (embedder(fr, fc) for fr, fc in faces_and_frames) if e is not None]
    if len(embs) < min_samples:
        return None
    return unit(np.mean(embs, axis=0)).round(6).tolist()
