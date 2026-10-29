"""Interactive ear enrollment (OpenCV windows). Logic lives in enroll.py."""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np

from . import geometry
from .config import Config
from .enroll import TRAGION_HINT, EarDetector, EnrollmentSession, Keyframe, speaker_ear_distances
from .head import HeadModel, PoseSolver, make_listener, ypr_from_rotation
from .calib import undistorter_for
from .landmarks import make_backend

STEP_TITLE = {
    "scale": "1/3  Eyes on the cross, facing the camera",
    "left": "2/3  Left ear: turn to your RIGHT",
    "right": "3/3  Right ear: turn to your LEFT",
}
WIN = "ear enrollment"


def _text(img, s, org, scale=0.6, color=(255, 255, 255), thick=1):
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
    cv2.putText(img, s, org, cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _crosshair(img, K, color=(0, 230, 255)) -> None:
    c = (int(K.cx), int(K.cy))
    cv2.drawMarker(img, c, (0, 0, 0), cv2.MARKER_CROSS, 64, 4)
    cv2.drawMarker(img, c, color, cv2.MARKER_CROSS, 64, 2)
    cv2.circle(img, c, 14, color, 1)


def run(cfg: Config, cap, name: str, distance_mm: float, out_path: str | Path,
        ear_cascades: tuple[str, str] | None = None, config_path: str | None = None,
        mode: str = "crosshair") -> int:
    if cfg.detector.backend == "haar":
        raise SystemExit("Ear enrollment needs head yaw, which the haar backend cannot measure.\n"
                         "Use --backend yunet --model face_detection_yunet_2023mar.onnx "
                         "(or mediapipe).")
    cfg.head = HeadModel(**{**cfg.head.__dict__, "scale": 1.0, "ear_l_mm": None, "ear_r_mm": None})
    K = geometry.Intrinsics.from_camera(cfg.camera)
    backend = make_backend(cfg.detector.backend, cfg.detector, cfg.detector.model_path)
    solver = PoseSolver(K, cfg.head)
    sess = EnrollmentSession(K, cfg.head, distance_mm, name, mode=mode)
    status = ""
    und = undistorter_for(cfg.camera)
    embedder = None
    if cfg.faceid.enabled:
        from .faceid import SFaceEmbedder
        embedder = SFaceEmbedder(cfg.faceid.model_path)
    face_samples = []                       # (frame, face) while facing the camera, for face ID

    # ------------------------------------------------------------ capture
    flash = 0
    while sess.step in ("scale", "left", "right"):
        ok, frame = cap.read()
        if not ok:
            print("video ended before enrollment finished")
            return 1
        if cfg.camera.mirror:
            frame = cv2.flip(frame, 1)
        if und is not None:
            frame = und(frame)
        face = backend(frame, time.perf_counter())
        if embedder is not None and face is not None and sess.step == "scale" and len(face_samples) < 30:
            face_samples.append((frame.copy(), face))
        pose = solver.solve(face.points) if face is not None else None
        n_before = len(sess.keyframes)
        msg = sess.feed(frame, *(pose if pose else (None, None)), face.points if face else {})
        if len(sess.keyframes) > n_before:
            flash = 4

        view = frame.copy()
        if face is not None:
            for u, v in face.points.values():
                cv2.circle(view, (int(u), int(v)), 3, (255, 220, 0), -1)
            if "eye_l" in face.points and "eye_r" in face.points and sess.step == "scale":
                m = np.add(face.points["eye_l"], face.points["eye_r"]) / 2
                cv2.circle(view, (int(m[0]), int(m[1])), 6, (80, 220, 80), 2)
        if mode == "crosshair" or sess.step == "scale":
            _crosshair(view, K)
        if pose is not None and sess.step in ("left", "right") and mode == "sweep":
            ear = cfg.head.ears()[0 if sess.step == "left" else 1]
            p = pose[0] @ ear + pose[1]
            u, v = K.fx * p[0] / p[2] + K.cx, K.fy * p[1] / p[2] + K.cy
            cv2.circle(view, (int(u), int(v)), 10, (220, 80, 255), 2)
        # Mirror the preview so moving toward the cross feels natural (markers first, then
        # flip, then text so the text stays readable). The cross sits at the image centre.
        if not cfg.camera.mirror:
            view = cv2.flip(view, 1)
        H, W = view.shape[:2]
        _text(view, STEP_TITLE.get(sess.step, ""), (12, 28), 0.7, (0, 230, 255), 2)
        _text(view, msg, (12, 56))
        if pose is not None:
            yaw, pitch, roll = ypr_from_rotation(pose[0])
            _text(view, f"yaw {yaw:+.0f}  pitch {pitch:+.0f}", (12, 82), 0.5)
        cv2.rectangle(view, (12, H - 30), (W - 12, H - 14), (60, 60, 60), -1)
        cv2.rectangle(view, (12, H - 30), (12 + int((W - 24) * min(sess.progress(), 1)), H - 14),
                      (80, 220, 80), -1)
        if flash:
            cv2.rectangle(view, (0, 0), (W - 1, H - 1), (255, 255, 255), 6)
            flash -= 1
        if sess.can_advance():
            _text(view, "n = enough views, next", (12, H - 40), 0.5, (80, 220, 80))
        if status:
            _text(view, status, (12, 108), 0.55, (80, 220, 80))
        cv2.imshow(WIN, view)
        key = cv2.waitKey(1) & 0xFF
        if key == 32 and mode == "crosshair":
            status = sess.capture_center()
            if status.startswith("Captured"):
                flash = 4
        if key in (27, ord("q")):
            print("enrollment aborted")
            return 1
        if key == ord("n") and sess.can_advance():
            sess.advance()
    cv2.destroyWindow(WIN)
    print(f"head scale {cfg.head.scale:.3f}; {len(sess.keyframes)} ear views captured")

    # ------------------------------------------------------------ mark
    detector = EarDetector(*ear_cascades) if ear_cascades else None
    if mode == "sweep" and not _mark_all(sess, K, detector):
        print("enrollment aborted")
        return 1

    # ------------------------------------------------------------ solve
    try:
        prof = sess.solve()
    except ValueError as e:
        print(f"could not solve ears: {e}")
        return 1
    if embedder is not None:
        from .faceid import capture_embedding
        prof.face_embedding = capture_embedding(embedder, face_samples)
        print("  face ID: " + ("stored" if prof.face_embedding else "not enough clear frontal frames"))
    measured = HeadModel(**cfg.head.__dict__)
    prof.apply(measured)
    if prof.reference:
        seat = make_listener(np.array(prof.reference["R"]), np.array(prof.reference["t_mm"]),
                             measured, "seat")
        _, drivers = geometry.solve(seat, cfg.stands)
        prof.reference["speaker_to_ear_mm"] = speaker_ear_distances(
            {s.name: d for s, d in zip(cfg.stands, drivers)}, seat.ear_l, seat.ear_r)
    prof.save(out_path)
    print_summary(prof, out_path)

    if config_path:
        cfg.profile = str(out_path)
        cfg.head.ear_l_mm = cfg.head.ear_r_mm = None      # the profile carries them
        cfg.head.scale = 1.0
        cfg.save(config_path)
        print(f"{config_path}: profile -> {out_path}")
    return 0


def print_summary(prof, path) -> None:
    q = prof.quality
    print(f"\nListener profile '{prof.name}' -> {path}")
    print(f"  ear-to-ear          {prof.ear_distance_mm:.1f} mm")
    print(f"  HRTF sphere radius  {prof.sphere_radius_mm:.1f} mm")
    if prof.ipd_mm:
        print(f"  eye spacing (IPD)   {prof.ipd_mm:.1f} mm  (used to place the ears from the eyes alone)")
    for side in ("left", "right"):
        o = prof.ears_vs_eyes_mm[side]
        print(f"  {side:<5} ear          {o['behind_mm']:.0f} behind / {o['below_mm']:.0f} below / "
              f"{o['out_mm']:.0f} out from the eyes   "
              f"({q[side]['views']} views, {q[side]['rejected']} rejected, "
              f"{q[side]['baseline_deg']:.0f} deg spread, "
              f"ray fit {q[side]['ray_rms_mm']:.1f} mm)")
    for spk, d in prof.reference.get("speaker_to_ear_mm", {}).items():
        print(f"  seat: {spk:<6} speaker -> L ear {d['ear_l']:.0f} mm, R ear {d['ear_r']:.0f} mm")
    worst = max(q[s]["ray_rms_mm"] for s in q)
    if worst > 6:
        print("  ! ray fit above 6 mm - consider re-enrolling and clicking more carefully")


# ------------------------------------------------------------------ marking
def _mark_all(sess: EnrollmentSession, K, detector) -> bool:
    kfs = sess.keyframes
    i = 0
    while i < len(kfs):
        kf = kfs[i]
        ear_px = 65.0 * K.fx / max(float(np.linalg.norm(kf.t)), 1.0)   # ~65 mm ear height
        if detector is not None and kf.pixel is None:
            hit = detector(kf, K, ear_px)
            if hit is not None:
                sess.set_pixel(kf, hit, "detector")
        res = _mark_one(kf, i, len(kfs), ear_px)
        if res == "quit":
            return False
        if res == "back":
            i = max(i - 1, 0)
            continue
        if isinstance(res, tuple):
            sess.set_pixel(kf, res[0], res[1])
        i += 1
    return True


def _mark_one(kf: Keyframe, idx: int, total: int, ear_px: float):
    """Zoomed view around the predicted ear. Returns ((u,v), source), 'skip', 'back' or 'quit'."""
    u0, v0 = kf.predicted
    half = int(max(2.2 * ear_px, 40))
    H, W = kf.frame.shape[:2]
    x0, y0 = int(np.clip(u0 - half, 0, max(W - 2 * half, 0))), int(np.clip(v0 - half, 0, max(H - 2 * half, 0)))
    crop = kf.frame[y0:y0 + 2 * half, x0:x0 + 2 * half]
    zoom = 480.0 / max(crop.shape[:2])
    big = cv2.resize(crop, None, fx=zoom, fy=zoom, interpolation=cv2.INTER_CUBIC)
    sel = {"p": kf.pixel, "src": kf.source if kf.pixel else ""}

    def on_mouse(ev, x, y, *_):
        if ev == cv2.EVENT_LBUTTONDOWN:
            sel["p"] = (x0 + x / zoom, y0 + y / zoom)
            sel["src"] = "click"

    win = f"mark {kf.side} ear"
    cv2.namedWindow(win)
    cv2.setMouseCallback(win, on_mouse)
    while True:
        img = big.copy()
        pu, pv = (u0 - x0) * zoom, (v0 - y0) * zoom
        cv2.drawMarker(img, (int(pu), int(pv)), (0, 200, 255), cv2.MARKER_TILTED_CROSS, 18, 1)
        if sel["p"] is not None:
            q = (int((sel["p"][0] - x0) * zoom), int((sel["p"][1] - y0) * zoom))
            cv2.drawMarker(img, q, (80, 255, 80), cv2.MARKER_CROSS, 26, 2)
        _text(img, f"{kf.side} ear  {idx + 1}/{total}   view {kf.view_deg:.0f} deg", (10, 22), 0.55)
        _text(img, f"Click {TRAGION_HINT}", (10, 44), 0.45)
        _text(img, "Enter accept   s skip   b back   q quit   (orange x = model guess)",
              (10, img.shape[0] - 12), 0.42)
        cv2.imshow(win, img)
        k = cv2.waitKey(20) & 0xFF
        if k in (13, 10, 32) and sel["p"] is not None:
            cv2.destroyWindow(win)
            return sel["p"], sel["src"] or "click"
        if k == ord("s"):
            cv2.destroyWindow(win)
            return "skip"
        if k == ord("b"):
            cv2.destroyWindow(win)
            return "back"
        if k in (27, ord("q")):
            cv2.destroyWindow(win)
            return "quit"
