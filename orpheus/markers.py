"""Find the stands with the camera instead of a tape measure.

Each stand carries a printed ArUco marker (OpenCV's square fiducial tag) on its
pan/tilt head. From the marker's four corners and its printed size, the camera
solves the marker's full position; the stand's pivot is a fixed offset from it.

    python -m orpheus --config stands.json --make-markers markers/
    python -m orpheus --config stands.json --locate-stands

Accuracy comes from the camera calibration (run --calibrate-camera first), the
printed size (measure the black square with calipers, set markers.size_mm) and
how many pixels the marker covers. On a 640 px webcam, in simulation, a 100 mm
marker locates to about 4 mm per metre of distance (60 mm: about 8 mm/m).
Bigger markers or a 1280 px stream help more than more frames. Keep the marker
centre close to the pivot: orientation is the noisy part, and a long offset
turns orientation error into position error.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
import numpy as np

from .geometry import Intrinsics

log = logging.getLogger(__name__)


def dictionary(name: str):
    if not hasattr(cv2.aruco, name):
        raise SystemExit(f"unknown ArUco dictionary {name}")
    return cv2.aruco.getPredefinedDictionary(getattr(cv2.aruco, name))


def marker_image(dict_name: str, marker_id: int, size_mm: float, dpi: int = 300,
                 label: str = "") -> np.ndarray:
    """Printable marker at exact scale with a white quiet zone and a caption."""
    px = int(round(size_mm / 25.4 * dpi))
    m = cv2.aruco.generateImageMarker(dictionary(dict_name), marker_id, px)
    quiet = px // 4
    img = cv2.copyMakeBorder(m, quiet, quiet + px // 5, quiet, quiet, cv2.BORDER_CONSTANT, value=255)
    text = f"{label}  id {marker_id}  {dict_name}  {size_mm:g} mm @ {dpi} dpi (print at 100%)"
    scale = max(px / 900, 0.4)
    cv2.putText(img, text, (quiet, img.shape[0] - px // 12), cv2.FONT_HERSHEY_SIMPLEX, scale, 0,
                max(1, int(scale * 2)), cv2.LINE_AA)
    return img


def make_marker_sheets(cfg, out_dir: str | Path, dpi: int = 300) -> list[Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    # give stands without a marker the lowest free ids
    used = {s.marker_id for s in cfg.stands if s.marker_id >= 0}
    free = (i for i in range(1000) if i not in used)
    for s in cfg.stands:
        if s.marker_id < 0:
            s.marker_id = next(free)
    paths = []
    for s in cfg.stands:
        p = out / f"marker_{s.name}_id{s.marker_id}.png"
        cv2.imwrite(str(p), marker_image(cfg.markers.dictionary, s.marker_id, cfg.markers.size_mm,
                                         dpi, label=f"stand '{s.name}'"))
        paths.append(p)
    return paths


def detect(gray: np.ndarray, dict_name: str) -> dict[int, np.ndarray]:
    params = cv2.aruco.DetectorParameters()
    # sub-pixel corners: a 60 mm marker is only ~40 px wide at 1 m on a 640 px webcam,
    # so each tenth of a pixel is ~0.25 % of distance
    params.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    params.cornerRefinementWinSize = 3
    det = cv2.aruco.ArucoDetector(dictionary(dict_name), params)
    corners, ids, _ = det.detectMarkers(gray)
    if ids is None:
        return {}
    return {int(i): c.reshape(4, 2) for i, c in zip(ids.ravel(), corners)}


def marker_object_points(size_mm: float) -> np.ndarray:
    h = size_mm / 2
    # ArUco corner order (TL, TR, BR, BL); marker frame x right, y up, z toward the viewer
    return np.array([[-h, h, 0], [h, h, 0], [h, -h, 0], [-h, -h, 0]], np.float64)


def marker_pose(corners: np.ndarray, size_mm: float, K: Intrinsics, dist=None):
    ok, rvec, tvec = cv2.solvePnP(marker_object_points(size_mm), corners.astype(np.float64),
                                  K.matrix(), None if dist is None else np.asarray(dist, float),
                                  flags=cv2.SOLVEPNP_IPPE_SQUARE)
    if not ok:
        return None
    return cv2.Rodrigues(rvec)[0], tvec.ravel()


def pivot_from_marker(R: np.ndarray, t: np.ndarray, offset_mm) -> np.ndarray:
    return R @ np.asarray(offset_mm, float) + t


class StandLocator:
    """Accumulates marker sightings over frames; robust median per stand."""

    def __init__(self, cfg, K: Intrinsics, dist=None) -> None:
        self.cfg, self.K, self.dist = cfg, K, dist
        self.by_id = {s.marker_id: s for s in cfg.stands if s.marker_id >= 0}
        self.obs: dict[str, list[np.ndarray]] = {s.name: [] for s in self.by_id.values()}
        self.normals: dict[str, list[np.ndarray]] = {s.name: [] for s in self.by_id.values()}

    def add(self, frame_bgr: np.ndarray) -> dict[int, np.ndarray]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY) if frame_bgr.ndim == 3 else frame_bgr
        found = detect(gray, self.cfg.markers.dictionary)
        for mid, corners in found.items():
            s = self.by_id.get(mid)
            if s is None:
                continue
            pose = marker_pose(corners, self.cfg.markers.size_mm, self.K, self.dist)
            if pose is None:
                continue
            R, t = pose
            self.obs[s.name].append(pivot_from_marker(R, t, s.marker_offset_mm))
            self.normals[s.name].append(R @ np.array([0, 0, 1.0]))
        return found

    def result(self) -> dict[str, dict]:
        out = {}
        for name, pts in self.obs.items():
            if not pts:
                continue
            P = np.array(pts)
            med = np.median(P, axis=0)
            n = np.mean(self.normals[name], axis=0)
            out[name] = {"position_mm": np.round(med, 1).tolist(),
                         "spread_mm": round(float(np.median(np.linalg.norm(P - med, axis=1))), 2),
                         "sightings": len(pts),
                         # direction the marker faces, in degrees about +Y (0 = at the camera)
                         "facing_deg": round(float(np.degrees(np.arctan2(n[0], -n[2]))), 1)}
        return out

    def apply(self) -> dict[str, dict]:
        res = self.result()
        for s in self.cfg.stands:
            if s.name in res:
                s.position_mm = tuple(res[s.name]["position_mm"])
        return res


def run_locate_stands(cap, cfg, K: Intrinsics, dist=None, frames: int = 60,
                      config_path: str | None = None) -> int:
    loc = StandLocator(cfg, K, dist)
    if not loc.by_id:
        raise SystemExit("no stand has a marker_id - run --make-markers first (it assigns ids)")
    print(f"Looking for markers {sorted(loc.by_id)} ({cfg.markers.dictionary}, "
          f"{cfg.markers.size_mm:g} mm). Keep the stands still. q stops early.")
    n = 0
    while n < frames:
        ok, frame = cap.read()
        if not ok:
            break
        found = loc.add(frame)
        view = frame.copy()
        if found:
            cv2.aruco.drawDetectedMarkers(view, [c.reshape(1, 4, 2).astype(np.float32) for c in found.values()],
                                          np.array(list(found.keys())).reshape(-1, 1))
            n += 1
        cv2.putText(view, f"frames with markers {n}/{frames}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX,
                    0.65, (0, 230, 255), 2, cv2.LINE_AA)
        cv2.imshow("locate stands", view)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    res = loc.apply()
    if not res:
        print("no stand markers seen")
        return 1
    for name, r in res.items():
        x, y, z = r["position_mm"]
        print(f"  {name:<6} pivot x {x:+7.1f}  y {y:+7.1f}  z {z:+7.1f} mm   "
              f"(+-{r['spread_mm']:.1f} mm over {r['sightings']} sightings, marker faces {r['facing_deg']:+.0f} deg)")
    missing = [s.name for s in cfg.stands if s.marker_id >= 0 and s.name not in res]
    if missing:
        print(f"  not seen: {', '.join(missing)} - positions unchanged")
    if config_path:
        cfg.save(config_path)
        print(f"saved to {config_path}")
    return 0
