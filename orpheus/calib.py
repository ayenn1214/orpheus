"""Camera calibration from a printed checkerboard.

Replaces the field-of-view guess with a measured camera matrix and lens
distortion. Every frame is then undistorted before tracking, so face points,
ear matches and markers all live in the same ideal pinhole camera.

    python -m orpheus --config stands.json --calibrate-camera 9x6 --square 25
    python -m orpheus --config stands.json --calibrate-stereo 9x6 --square 25

COLSxROWS counts INNER corners (a board of 10x7 squares has 9x6). Measure the
printed square with calipers; print scaling is rarely exactly 100 %.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import cv2
import numpy as np

log = logging.getLogger(__name__)


def parse_board(spec: str) -> tuple[int, int]:
    try:
        c, r = (int(v) for v in spec.lower().split("x"))
    except ValueError as e:
        raise SystemExit(f"board size must look like 9x6 (inner corners), got {spec!r}") from e
    if c < 3 or r < 3:
        raise SystemExit("board needs at least 3x3 inner corners")
    return c, r


def board_points(cols: int, rows: int, square_mm: float) -> np.ndarray:
    g = np.zeros((rows * cols, 3), np.float32)
    g[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_mm
    return g


def find_board(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    """Sub-pixel inner corners (N,1,2) or None. The sector-based detector copes with
    blur and uneven light better than the classic one; the classic one is the fallback."""
    if hasattr(cv2, "findChessboardCornersSB"):
        ok, c = cv2.findChessboardCornersSB(gray, (cols, rows), flags=cv2.CALIB_CB_EXHAUSTIVE |
                                            cv2.CALIB_CB_ACCURACY)
        if ok:
            return c.astype(np.float32)
    ok, c = cv2.findChessboardCorners(gray, (cols, rows), flags=cv2.CALIB_CB_ADAPTIVE_THRESH |
                                      cv2.CALIB_CB_NORMALIZE_IMAGE)
    if not ok:
        return None
    return cv2.cornerSubPix(gray, c, (5, 5), (-1, -1),
                            (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.01))


def calibrate(objp: np.ndarray, corners: list, size) -> tuple[np.ndarray, np.ndarray, float]:
    rms, K, dist, _, _ = cv2.calibrateCamera([objp] * len(corners), corners, size, None, None)
    return K, dist.ravel(), float(rms)


def is_new_view(corners: np.ndarray, taken: list, min_move_px: float = 35.0,
                min_scale_change: float = 0.12) -> bool:
    """Accept a view only if the board moved or changed size/tilt enough since every
    earlier capture: twenty copies of one pose calibrate nothing."""
    c = corners.reshape(-1, 2)
    centre, spread = c.mean(axis=0), c.std(axis=0)
    for prev in taken:
        p = prev.reshape(-1, 2)
        moved = np.linalg.norm(centre - p.mean(axis=0))
        rescaled = np.abs(spread / np.maximum(p.std(axis=0), 1e-6) - 1).max()
        if moved < min_move_px and rescaled < min_scale_change:
            return False
    return True


class Undistorter:
    """Undistort whole frames, keeping the same camera matrix (pixels stay comparable)."""

    def __init__(self, K: np.ndarray, dist, size) -> None:
        self.maps = cv2.initUndistortRectifyMap(np.asarray(K, np.float64), np.asarray(dist, np.float64),
                                                None, np.asarray(K, np.float64), tuple(size),
                                                cv2.CV_16SC2)
        self.size = tuple(size)

    def __call__(self, frame: np.ndarray) -> np.ndarray:
        if (frame.shape[1], frame.shape[0]) != self.size:
            return frame
        return cv2.remap(frame, *self.maps, cv2.INTER_LINEAR)


def undistorter_for(cam) -> Undistorter | None:
    if not cam.matrix or not cam.dist or not any(abs(v) > 1e-9 for v in cam.dist):
        return None
    cw, ch = cam.calib_size or (cam.width, cam.height)
    K = np.array(cam.matrix, float)
    if (cw, ch) != (cam.width, cam.height):
        K[0] *= cam.width / cw
        K[1] *= cam.height / ch
    return Undistorter(K, cam.dist, (cam.width, cam.height))


# ---------------------------------------------------------------- interactive
def _overlay(img, text, line=0, color=(0, 230, 255)):
    org = (12, 28 + 26 * line)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 0, 0), 4, cv2.LINE_AA)
    cv2.putText(img, text, org, cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2, cv2.LINE_AA)


def run_camera_calibration(cap, cfg, cols: int, rows: int, square_mm: float, views: int = 20,
                           config_path: str | None = None) -> int:
    objp = board_points(cols, rows, square_mm)
    taken: list[np.ndarray] = []
    size = None
    print(f"Hold the {cols}x{rows} board in front of the camera. Views are taken automatically "
          "when the board is found in a new place: move it around the whole frame, near and far, "
          "and tilt it 20-40 deg in different directions. q quits.")
    while len(taken) < views:
        ok, frame = cap.read()
        if not ok:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        size = (gray.shape[1], gray.shape[0])
        c = find_board(gray, cols, rows)
        view = frame.copy()
        if c is not None:
            cv2.drawChessboardCorners(view, (cols, rows), c, True)
            if is_new_view(c, taken):
                taken.append(c)
                view[:] = 255 - view                         # flash
        for p in taken:
            cv2.polylines(view, [p.reshape(-1, 2)[[0, cols - 1, -1, -cols]].astype(np.int32)],
                          True, (80, 220, 80), 1)
        _overlay(view, f"views {len(taken)}/{views}")
        _overlay(view, "board found" if c is not None else "looking for the board", 1,
                 (80, 220, 80) if c is not None else (0, 0, 255))
        cv2.imshow("camera calibration", view)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    cv2.destroyAllWindows()
    if len(taken) < 8:
        print(f"only {len(taken)} views - need at least 8")
        return 1
    K, dist, rms = calibrate(objp, taken, size)
    hfov = 2 * np.degrees(np.arctan(size[0] / (2 * K[0, 0])))
    print(f"fx {K[0, 0]:.1f}  fy {K[1, 1]:.1f}  cx {K[0, 2]:.1f}  cy {K[1, 2]:.1f}  "
          f"HFOV {hfov:.1f} deg  reprojection {rms:.2f} px")
    print(f"distortion {np.round(dist, 4).tolist()}")
    if rms > 1.0:
        print("! reprojection above 1 px: check the square size and that the board is flat")
    cfg.camera.matrix = np.round(K, 4).tolist()
    cfg.camera.dist = np.round(dist, 6).tolist()
    cfg.camera.calib_size = list(size)
    cfg.camera.calib_rms_px = round(rms, 3)
    cfg.camera.hfov_deg = round(float(hfov), 2)
    if config_path:
        cfg.save(config_path)
        print(f"saved to {config_path}")
    return 0


def run_stereo_calibration(cfg, cols: int, rows: int, square_mm: float, out_path: str,
                           views: int = 20, config_path: str | None = None) -> int:
    from .depth import stereo_calibrate

    caps = [cv2.VideoCapture(i) for i in (cfg.camera.index, cfg.depth.right_index)]
    for c in caps:
        c.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
        c.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    if not all(c.isOpened() for c in caps):
        raise SystemExit("could not open both cameras (camera.index and depth.right_index)")
    objp = board_points(cols, rows, square_mm)
    left, right = [], []
    size = None
    print("Mount the two webcams side by side, 60-120 mm apart, pointing the same way. "
          "Hold the board so BOTH cameras see it; views are taken automatically. q quits.")
    while len(left) < views:
        if not all(c.grab() for c in caps):
            break
        fl, fr = caps[0].retrieve()[1], caps[1].retrieve()[1]
        gl, gr = cv2.cvtColor(fl, cv2.COLOR_BGR2GRAY), cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
        size = (gl.shape[1], gl.shape[0])
        cl, cr = find_board(gl, cols, rows), find_board(gr, cols, rows)
        if cl is not None and cr is not None and is_new_view(cl, left):
            left.append(cl)
            right.append(cr)
        view = np.hstack([fl, fr])
        _overlay(view, f"views {len(left)}/{views}   left {'ok' if cl is not None else '--'}   "
                       f"right {'ok' if cr is not None else '--'}")
        cv2.imshow("stereo calibration", view)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            break
    for c in caps:
        c.release()
    cv2.destroyAllWindows()
    if len(left) < 8:
        print(f"only {len(left)} paired views - need at least 8")
        return 1
    calib = stereo_calibrate(objp, left, right, size)
    Path(out_path).write_text(json.dumps(calib, indent=2))
    print(f"baseline {calib['baseline_mm']:.1f} mm, reprojection {calib['rms_px']:.2f} px -> {out_path}")
    cfg.depth.source = "stereo"
    cfg.depth.stereo_calib = str(out_path)
    if config_path:
        cfg.save(config_path)
        print(f"{config_path}: depth.source = stereo")
    return 0
