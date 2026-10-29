"""Depth cameras: measure distance instead of inferring it from face size.

A single webcam knows distance only through an assumed size (face width, eye
spacing, head model scale); a 10 % size error is a 10 % distance error, and it
moves every ear by tens of millimetres. A depth camera reads the distance of
each face landmark directly. Three sources, one interface:

    realsense  Intel RealSense D4xx (pyrealsense2), depth aligned to colour.
    oakd       Luxonis OAK-D (depthai v3), depth aligned to the colour camera.
    stereo     Two ordinary webcams, calibrated with --calibrate-stereo; images
               are rectified and depth comes from OpenCV semi-global matching.

Every source returns (ok, colour_bgr, depth_mm) with depth_mm the same size as
the colour image (0 = no reading), and exposes the colour camera's intrinsics,
which replace the configured/guessed ones.

How the depth is used (refine_pose_with_depth):
    pnp   rotation from the landmarks, distance from depth: the translation is
          rescaled along the camera ray so the landmarks sit at their measured
          depth (removes the head-size/distance ambiguity of monocular PnP).
    eyes  two 3D eye points give distance, head turn and roll - with a webcam
          alone, eyes-only tracking cannot see the turn at all.
    box   distance of the face from the depth at the box centre.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path

import cv2
import numpy as np

from .geometry import FACE_TO_HEAD_CENTER_MM, Intrinsics
from .head import HeadModel, rotation_from_ypr

log = logging.getLogger(__name__)


# ------------------------------------------------------------------ helpers
def sample_depth(depth_mm: np.ndarray, u: float, v: float, win: int = 2,
                 lo: float = 250.0, hi: float = 4000.0) -> float | None:
    """Median of valid depth readings in a (2*win+1)^2 window, or None."""
    H, W = depth_mm.shape[:2]
    x, y = int(round(u)), int(round(v))
    if not (0 <= x < W and 0 <= y < H):
        return None
    patch = depth_mm[max(y - win, 0):y + win + 1, max(x - win, 0):x + win + 1].ravel()
    patch = patch[(patch >= lo) & (patch <= hi)]
    if patch.size < max(1, (2 * win + 1) ** 2 // 4):
        return None
    return float(np.median(patch))


def backproject(u: float, v: float, z: float, K: Intrinsics) -> np.ndarray:
    return np.array([(u - K.cx) * z / K.fx, (v - K.cy) * z / K.fy, z])


def refine_pose_with_depth(R: np.ndarray, t: np.ndarray, method: str, points2d: dict, box,
                           depth_mm: np.ndarray, K: Intrinsics, model: HeadModel,
                           win: int = 2, lo: float = 250.0, hi: float = 4000.0):
    """Returns (R, t, used) - used is False when depth could not help this frame."""
    pts = model.points()
    meas = {}
    for name, (u, v) in points2d.items():
        if name in pts:
            z = sample_depth(depth_mm, u, v, win, lo, hi)
            if z is not None:
                meas[name] = backproject(u, v, z, K)

    if method == "eyes" and "eye_l" in meas and "eye_r" in meas:
        el, er = meas["eye_l"], meas["eye_r"]
        d = el - er
        ipd_model = float(np.linalg.norm(pts["eye_l"] - pts["eye_r"]))
        if 0.6 < np.linalg.norm(d) / ipd_model < 1.6:
            d = d / np.linalg.norm(d)
            # head +X (towards the left ear) seen in the camera frame
            yaw = math.degrees(math.atan2(d[2], d[0]))
            roll = math.degrees(math.asin(float(np.clip(d[1], -1, 1))))
            R2 = rotation_from_ypr(yaw, 0.0, roll)
            # eye centres are on the eyeball surface: compare like with like
            eye_mid_h = (pts["eye_l"] + pts["eye_r"]) / 2
            t2 = (el + er) / 2 - R2 @ eye_mid_h
            return R2, t2, True

    ratios = []
    for name, p in meas.items():
        pred = R @ pts[name] + t
        if pred[2] > 1:
            ratios.append(p[2] / pred[2])
    if ratios:
        s = float(np.median(ratios))
        if 0.6 < s < 1.6:
            return R, t * s, True
    if method == "box" and box is not None:
        x, y, w, h = box
        z = sample_depth(depth_mm, x + w / 2, y + h / 2, max(win, int(w * 0.1)), lo, hi)
        if z is not None and t[2] > 1:
            s = (z + FACE_TO_HEAD_CENTER_MM) / t[2]
            if 0.5 < s < 2.0:
                return R, t * s, True
    return R, t, False


# ------------------------------------------------------------------ sources
class DepthSource:
    intrinsics: Intrinsics

    def read(self) -> tuple[bool, np.ndarray | None, np.ndarray | None]:
        raise NotImplementedError

    def release(self) -> None:
        pass


class RealSenseSource(DepthSource):
    def __init__(self, width: int = 640, height: int = 480, fps: int = 30) -> None:
        try:
            import pyrealsense2 as rs
        except ImportError as e:
            raise SystemExit("RealSense needs pyrealsense2:  pip install pyrealsense2") from e
        self.rs = rs
        self.pipe = rs.pipeline()
        conf = rs.config()
        conf.enable_stream(rs.stream.color, width, height, rs.format.bgr8, fps)
        conf.enable_stream(rs.stream.depth, width, height, rs.format.z16, fps)
        try:
            prof = self.pipe.start(conf)
        except RuntimeError as e:
            raise SystemExit(f"RealSense: could not start {width}x{height}@{fps} ({e})") from e
        self.scale_mm = prof.get_device().first_depth_sensor().get_depth_scale() * 1000.0
        self.align = rs.align(rs.stream.color)
        i = prof.get_stream(rs.stream.color).as_video_stream_profile().get_intrinsics()
        self.intrinsics = Intrinsics(fx=i.fx, fy=i.fy, cx=i.ppx, cy=i.ppy)

    def read(self):
        frames = self.align.process(self.pipe.wait_for_frames(1000))
        c, d = frames.get_color_frame(), frames.get_depth_frame()
        if not c or not d:
            return False, None, None
        color = np.asanyarray(c.get_data()).copy()
        depth = np.asanyarray(d.get_data()).astype(np.float32) * self.scale_mm
        return True, color, depth

    def release(self) -> None:
        self.pipe.stop()


class OakDSource(DepthSource):
    """depthai v3 API (Pipeline.create(...).build(), Output.createOutputQueue())."""

    def __init__(self, width: int = 640, height: int = 480) -> None:
        try:
            import depthai as dai
        except ImportError as e:
            raise SystemExit("OAK-D needs depthai v3:  pip install depthai") from e
        if not hasattr(dai.node, "Camera") or not hasattr(dai.Node.Output, "createOutputQueue"):
            raise SystemExit("OAK-D support is written for depthai v3; found an older depthai")
        try:
            self.p = dai.Pipeline()
        except RuntimeError as e:
            raise SystemExit(f"OAK-D: {e}") from e
        cam = self.p.create(dai.node.Camera).build(dai.CameraBoardSocket.CAM_A)
        color = cam.requestOutput((width, height), dai.ImgFrame.Type.BGR888i)
        stereo = self.p.create(dai.node.StereoDepth).build(autoCreateCameras=True)
        stereo.setDepthAlign(dai.CameraBoardSocket.CAM_A)
        stereo.setOutputSize(width, height)
        stereo.setLeftRightCheck(True)
        stereo.setSubpixel(True)
        self.qc = color.createOutputQueue(maxSize=4, blocking=False)
        self.qd = stereo.depth.createOutputQueue(maxSize=4, blocking=False)
        self.p.start()
        M = self.p.getDefaultDevice().readCalibration().getCameraIntrinsics(
            dai.CameraBoardSocket.CAM_A, width, height)
        self.intrinsics = Intrinsics(fx=M[0][0], fy=M[1][1], cx=M[0][2], cy=M[1][2])

    def read(self):
        c, d = self.qc.get(), self.qd.get()
        if c is None or d is None:
            return False, None, None
        return True, c.getCvFrame(), d.getFrame().astype(np.float32)   # uint16 mm

    def release(self) -> None:
        self.p.stop()


class StereoMatcher:
    """Rectified stereo -> depth in the rectified left camera, via semi-global matching
    at half resolution (about 4x faster, enough for faces at 0.4-2 m)."""

    def __init__(self, calib: dict, num_disp: int = 96, block: int = 5) -> None:
        K1, D1 = np.array(calib["K1"]), np.array(calib["D1"])
        K2, D2 = np.array(calib["K2"]), np.array(calib["D2"])
        R, T = np.array(calib["R"]), np.array(calib["T"]).reshape(3)
        size = tuple(calib["size"])
        R1, R2, P1, P2, _, _, _ = cv2.stereoRectify(K1, D1, K2, D2, size, R, T, alpha=0)
        self.map_l = cv2.initUndistortRectifyMap(K1, D1, R1, P1, size, cv2.CV_16SC2)
        self.map_r = cv2.initUndistortRectifyMap(K2, D2, R2, P2, size, cv2.CV_16SC2)
        self.f = float(P1[0, 0])
        self.baseline = float(np.linalg.norm(T))
        self.intrinsics = Intrinsics(fx=P1[0, 0], fy=P1[1, 1], cx=P1[0, 2], cy=P1[1, 2])
        self.size = size
        self.sgbm = cv2.StereoSGBM_create(
            minDisparity=0, numDisparities=num_disp, blockSize=block,
            P1=8 * block * block, P2=32 * block * block, uniquenessRatio=10,
            speckleWindowSize=60, speckleRange=2, disp12MaxDiff=1,
            mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    def rectify(self, left: np.ndarray, right: np.ndarray):
        return (cv2.remap(left, *self.map_l, cv2.INTER_LINEAR),
                cv2.remap(right, *self.map_r, cv2.INTER_LINEAR))

    def depth(self, left_rect: np.ndarray, right_rect: np.ndarray) -> np.ndarray:
        gl = cv2.cvtColor(left_rect, cv2.COLOR_BGR2GRAY) if left_rect.ndim == 3 else left_rect
        gr = cv2.cvtColor(right_rect, cv2.COLOR_BGR2GRAY) if right_rect.ndim == 3 else right_rect
        hl, hr = cv2.pyrDown(gl), cv2.pyrDown(gr)
        disp = self.sgbm.compute(hl, hr).astype(np.float32) / 16.0
        with np.errstate(divide="ignore"):
            z = np.where(disp > 0.5, (self.f / 2) * self.baseline / disp, 0.0).astype(np.float32)
        return cv2.resize(z, (gl.shape[1], gl.shape[0]), interpolation=cv2.INTER_NEAREST)


class StereoPairSource(DepthSource):
    def __init__(self, left_index: int, right_index: int, calib_path: str) -> None:
        calib = json.loads(Path(calib_path).read_text())
        self.match = StereoMatcher(calib)
        w, h = self.match.size
        self.caps = []
        for idx in (left_index, right_index):
            cap = cv2.VideoCapture(idx)
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, w)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, h)
            if not cap.isOpened():
                raise SystemExit(f"stereo: could not open camera {idx}")
            self.caps.append(cap)
        self.intrinsics = self.match.intrinsics

    def read(self):
        # grab both first, then decode, so the two exposures are as close as possible
        if not all(c.grab() for c in self.caps):
            return False, None, None
        ok_l, left = self.caps[0].retrieve()
        ok_r, right = self.caps[1].retrieve()
        if not (ok_l and ok_r):
            return False, None, None
        lr, rr = self.match.rectify(left, right)
        return True, lr, self.match.depth(lr, rr)

    def release(self) -> None:
        for c in self.caps:
            c.release()


def stereo_calibrate(objp: np.ndarray, corners_l: list, corners_r: list, size) -> dict:
    """Intrinsics per camera, then the left->right transform with intrinsics fixed."""
    obj = [objp] * len(corners_l)
    _, K1, D1, _, _ = cv2.calibrateCamera(obj, corners_l, size, None, None)
    _, K2, D2, _, _ = cv2.calibrateCamera(obj, corners_r, size, None, None)
    rms, K1, D1, K2, D2, R, T, _, _ = cv2.stereoCalibrate(
        obj, corners_l, corners_r, K1, D1, K2, D2, size, flags=cv2.CALIB_FIX_INTRINSIC,
        criteria=(cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 1e-6))
    return {"K1": K1.tolist(), "D1": D1.ravel().tolist(), "K2": K2.tolist(),
            "D2": D2.ravel().tolist(), "R": R.tolist(), "T": T.ravel().tolist(),
            "size": list(size), "rms_px": float(rms),
            "baseline_mm": float(np.linalg.norm(T))}


def open_depth_source(cfg) -> DepthSource | None:
    src = cfg.depth.source
    w, h = cfg.camera.width, cfg.camera.height
    if src == "none":
        return None
    if src == "realsense":
        return RealSenseSource(w, h)
    if src == "oakd":
        return OakDSource(w, h)
    if src == "stereo":
        return StereoPairSource(cfg.camera.index, cfg.depth.right_index, cfg.depth.stereo_calib)
    raise ValueError(src)
