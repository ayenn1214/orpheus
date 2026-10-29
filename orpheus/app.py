"""Command-line entry point.

    python -m orpheus                                  # dry run, webcam 0, preview
    python -m orpheus --write-config stands.json
    python -m orpheus --config stands.json --backend yunet --model yunet.onnx \
        --enroll ankur --distance 600                               # find + measure ears, store profile
    python -m orpheus --config stands.json --calibrate 600   # (no-enroll fallback) scale only
    python -m orpheus --config stands.json --port COM6       # drive the stands
    python -m orpheus --config stands.json --port COM6 \\
        --audio virtual --audio-file song.wav                       # + tracked HRTF rendering
    python -m orpheus --backend yunet --model face_detection_yunet_2023mar.onnx

  setup tools
    --calibrate-camera 9x6 --square 25      checkerboard -> camera matrix + lens distortion
    --make-markers markers/                 printable ArUco markers, one per stand
    --locate-stands                         stand pivot positions from their markers
    --calibrate-stereo 9x6 --square 25      two webcams -> stereo depth calibration
    --add-face NAME                         store NAME's face for automatic profile loading

  run options
    --listeners single|main|midpoint        one listener, aim at the main one, or between two
    --faceid-model sface.onnx               recognise listeners, load their profiles
    --depth realsense|oakd|stereo           measured distance instead of size-based
    --lead-ms 100                           aim ahead of head motion (0 = off)
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

from . import geometry, hrtf
from .audio import MODES as AUDIO_MODES
from .config import Config
from .link import DryRunLink, SerialLink
from .pipeline import Pipeline, State

log = logging.getLogger("orpheus")

GREEN, ORANGE, BLUE, CYAN, MAGENTA = (80, 220, 80), (0, 200, 255), (255, 120, 0), (255, 220, 0), (220, 80, 255)
IPD_MM = 63.0


# ------------------------------------------------------------------ capture
def open_capture(cfg: Config, source: str | None) -> cv2.VideoCapture:
    if source is not None:
        cap = cv2.VideoCapture(int(source) if source.isdigit() else source)
    else:
        backend = cv2.CAP_DSHOW if sys.platform == "win32" else cv2.CAP_ANY
        cap = cv2.VideoCapture(cfg.camera.index, backend)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, cfg.camera.width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, cfg.camera.height)
    if not cap.isOpened():
        raise SystemExit(f"could not open video source {source or cfg.camera.index}")
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    if (w, h) != (cfg.camera.width, cfg.camera.height) and w > 0:
        log.warning("camera delivered %dx%d; using that for the optics model", w, h)
        cfg.camera.width, cfg.camera.height = w, h
    return cap


class CvSource:
    """Plain webcam / video file; no depth, intrinsics from the config."""

    intrinsics = None

    def __init__(self, cap: cv2.VideoCapture) -> None:
        self.cap = cap

    def read(self):
        ok, frame = self.cap.read()
        return ok, frame, None

    def release(self) -> None:
        self.cap.release()


def open_source(cfg: Config, source: str | None):
    if cfg.depth.source != "none" and source is None:
        from .depth import open_depth_source
        src = open_depth_source(cfg)
        log.info("depth: %s (camera intrinsics from the device)", cfg.depth.source)
        return src
    return CvSource(open_capture(cfg, source))


# ------------------------------------------------------------------ overlay
def _project(p, K) -> tuple[int, int] | None:
    if p[2] <= 1:
        return None
    return int(K.fx * p[0] / p[2] + K.cx), int(K.fy * p[1] / p[2] + K.cy)


def draw(frame: np.ndarray, st: State, cfg: Config, K, fps: float, audio=None) -> None:
    """Draw onto `frame`, which should be st.image (the frame as tracked)."""
    H, W = frame.shape[:2]
    for i, other in enumerate(st.listeners):
        if i == st.main_index:
            continue
        q = _project(other.t, K)
        if q:
            name = st.names[i] if i < len(st.names) and st.names[i] else "guest"
            cv2.circle(frame, q, 10, BLUE, 2)
            cv2.putText(frame, name, (q[0] + 12, q[1] - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.5, BLUE, 2)
    if st.aim is not None and cfg.predict.lead_ms > 0:
        for ear in (st.aim.ear_l, st.aim.ear_r):         # where the stands are aiming (ahead)
            q = _project(ear, K)
            if q:
                cv2.drawMarker(frame, q, ORANGE, cv2.MARKER_CROSS, 12, 2)
    if st.face is not None:
        x, y, w, h = st.face.box
        c = BLUE if "camshift" in st.face.source else GREEN
        cv2.rectangle(frame, (x, y), (x + w, y + h), c, 1)
        for name, (u, v) in st.face.points.items():
            cv2.circle(frame, (int(u), int(v)), 3, CYAN, -1)

    L = st.listener
    if L is not None:
        for ear, label, side in ((L.ear_l, "L", "left"), (L.ear_r, "R", "right")):
            q = _project(ear, K)
            if q:
                seen = st.ear_sources.get(side) == "seen"
                cv2.circle(frame, q, 7, GREEN if seen else MAGENTA, -1 if seen else 2)
                tag = f"{label} seen" if seen else f"{label} offset"
                cv2.putText(frame, tag, (q[0] + 9, q[1] + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                            GREEN if seen else MAGENTA, 2)
        if "eye_l" in L.landmarks and "eye_r" in L.landmarks:
            # ears not visible: show the stored eye-to-ear offsets being applied
            m = np.add(L.landmarks["eye_l"], L.landmarks["eye_r"]) / 2
            for ear, side in ((L.ear_l, "left"), (L.ear_r, "right")):
                q = _project(ear, K)
                if q and st.ear_sources.get(side) != "seen":
                    cv2.line(frame, (int(m[0]), int(m[1])), q, MAGENTA, 1, cv2.LINE_AA)
        nose_tip = L.R @ np.array([0, 0, -150.0]) + L.t
        a, b = _project(L.t, K), _project(nose_tip, K)
        if a and b:
            cv2.arrowedLine(frame, a, b, ORANGE, 2, tipLength=0.2)

    lines = [f"{fps:4.1f} fps   {cfg.detector.backend}"
             + (f"   depth {'on' if st.depth_used else 'no reading'}" if cfg.depth.source != "none" else "")
             + (f"   lead {cfg.predict.lead_ms:.0f} ms" if cfg.predict.lead_ms > 0 else "")]
    if st.listeners and cfg.listeners.mode != "single":
        main_name = st.names[st.main_index] if 0 <= st.main_index < len(st.names) else ""
        lines.append(f"{len(st.listeners)} listener(s), {cfg.listeners.mode}"
                     + (f", main {main_name}" if main_name else "")
                     + ("  - xtc off (2 listeners)" if st.xtc_suspended else ""))
    elif st.names and st.names[0]:
        lines.append(f"listener {st.names[0]}")
    if L is not None:
        yaw, pitch, roll = L.ypr
        X, Y, Z = L.t
        lines += [
            f"{L.method:<3} {st.face.source if st.face else ''}",
            f"head x{X:+5.0f} y{Y:+5.0f} z{Z:5.0f} mm",
            f"yaw {yaw:+5.1f} pitch {pitch:+5.1f} roll {roll:+5.1f}",
            f"ears L {st.ear_sources.get('left', 'offset')} / R {st.ear_sources.get('right', 'offset')}"
            f"  r {L.radius_mm:.1f} mm",
        ]
        for s, (p, t) in zip(cfg.stands, st.angles):
            lines.append(f"{s.name:<6} pan {p:6.1f} tilt {t:6.1f}  ({s.aim})")
    elif st.at_seat:
        lines.append("no listener - aimed at stored seat")
    else:
        lines.append("no listener - holding")
    for spk, d in st.distances.items():
        lines.append(f"{spk:<6} -> L ear {d['ear_l']:5.0f}  R ear {d['ear_r']:5.0f} mm")
    if audio is not None:
        lines.append(f"audio {audio.mode}  design {audio.design_ms:.1f} ms  xruns {audio.xruns}")
    for i, txt in enumerate(lines):
        org = (10, 22 + 19 * i)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 3, cv2.LINE_AA)
        cv2.putText(frame, txt, org, cv2.FONT_HERSHEY_SIMPLEX, 0.5, (30, 30, 30), 1, cv2.LINE_AA)

    _draw_topdown(frame, st, cfg, W, H)
    _draw_side(frame, st, cfg, W, H)


def _line(img, a, b, color, dashed=False, step=6) -> None:
    """Hard 2 px line to a seen ear; dotted line (dots every `step` px) to a virtual ear."""
    if not dashed:
        cv2.line(img, a, b, color, 2, cv2.LINE_AA)
        return
    a, b = np.array(a, float), np.array(b, float)
    n = max(int(np.linalg.norm(b - a) // step), 1)
    for i in range(n + 1):
        q = a + (b - a) * i / n
        cv2.circle(img, (int(q[0]), int(q[1])), 1, color, -1, cv2.LINE_AA)


def _draw_topdown(frame, st: State, cfg: Config, W: int, H: int) -> None:
    """Plan view (X-Z): camera bottom centre, stands, drivers, head, ears, aim rays."""
    size = 190
    x0, y0 = W - size - 10, H - size - 10
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (20, 20, 20), -1)
    s = size / 2000.0

    def P(v):
        return int(x0 + size / 2 + v[0] * s), int(y0 + size - 12 - v[2] * s)

    cv2.circle(frame, P((0, 0, 0)), 4, (200, 200, 200), -1)
    L = st.listener
    A = st.aim or L
    for i, other in enumerate(st.listeners):
        if i != st.main_index:
            cv2.circle(frame, P(other.t), max(int(other.radius_mm * s), 3), BLUE, 1)
    for i, stand in enumerate(cfg.stands):
        p = stand.position_mm
        q = P(p)
        cv2.rectangle(frame, (q[0] - 4, q[1] - 4), (q[0] + 4, q[1] + 4), ORANGE, -1)
        if A is not None:
            tgt = P(geometry.aim_point(A, stand))
            virtual = stand.aim == "near_ear" and \
                st.ear_sources.get(geometry.stand_ear(stand)) != "seen"
            _line(frame, q, tgt, ORANGE, dashed=virtual)   # dotted = aiming at a virtual ear
    if L is not None:
        cv2.circle(frame, P(L.t), max(int(L.radius_mm * s), 3), GREEN, 1)
        cv2.circle(frame, P(L.ear_l), 3, MAGENTA, -1)
        cv2.circle(frame, P(L.ear_r), 3, MAGENTA, -1)
        cv2.arrowedLine(frame, P(L.t), P(L.R @ np.array([0, 0, -250.0]) + L.t), GREEN, 1,
                        tipLength=0.3)


def _draw_side(frame, st: State, cfg: Config, W: int, H: int) -> None:
    """Side view (Z-Y), left of the plan view: shows ear height and each stand's tilt."""
    size = 190
    x0, y0 = W - 2 * size - 20, H - size - 10
    if x0 < 0:
        return
    cv2.rectangle(frame, (x0, y0), (x0 + size, y0 + size), (20, 20, 20), -1)
    s = size / 2000.0                                   # 2 m across (rear stands of 7.1)

    def P(v):                                           # Z to the right, up is -Y
        return int(x0 + 12 + v[2] * s), int(y0 + size / 2 + v[1] * s)

    cv2.line(frame, (x0, y0 + size // 2), (x0 + size, y0 + size // 2), (70, 70, 70), 1)
    cv2.circle(frame, P((0, 0, 0)), 4, (200, 200, 200), -1)
    L = st.listener
    A = st.aim or L
    for stand in cfg.stands:
        q = P(stand.position_mm)
        cv2.rectangle(frame, (q[0] - 4, q[1] - 4), (q[0] + 4, q[1] + 4), ORANGE, -1)
        if A is not None:
            virtual = stand.aim == "near_ear" and \
                st.ear_sources.get(geometry.stand_ear(stand)) != "seen"
            _line(frame, q, P(geometry.aim_point(A, stand)), ORANGE, dashed=virtual)
    if L is not None:
        cv2.circle(frame, P(L.t), max(int(L.radius_mm * s), 3), GREEN, 1)
        for ear, side in ((L.ear_l, "left"), (L.ear_r, "right")):
            seen = st.ear_sources.get(side) == "seen"
            cv2.circle(frame, P(ear), 3, GREEN if seen else MAGENTA, -1)
        cv2.arrowedLine(frame, P(L.t), P(L.R @ np.array([0, 0, -200.0]) + L.t), GREEN, 1, tipLength=0.3)
    cv2.putText(frame, "side", (x0 + 6, y0 + 16), cv2.FONT_HERSHEY_SIMPLEX, 0.45, (160, 160, 160), 1)


# ----------------------------------------------------------------- args
def parse_args(argv=None):
    ap = argparse.ArgumentParser(prog="orpheus",
                                 description="Ear-tracking pan/tilt speaker stands with HRTF rendering")
    ap.add_argument("--config", help="JSON config (see config.example.json)")
    ap.add_argument("--port", help="serial port, e.g. COM6 or /dev/ttyACM0 (omit for dry run)")
    ap.add_argument("--source", help="camera index or video file (overrides config)")
    ap.add_argument("--backend", choices=("haar", "yunet", "mediapipe"), help="landmark backend")
    ap.add_argument("--model", help="model file for yunet/mediapipe")
    ap.add_argument("--audio", choices=AUDIO_MODES, help="audio rendering mode")
    ap.add_argument("--audio-file", help="WAV/FLAC to play through the renderer")
    ap.add_argument("--audio-in", help="input device (loopback) to process live")
    ap.add_argument("--audio-out", help="output device")
    ap.add_argument("--hrtf", help="'spherical' or path to a SimpleFreeFieldHRIR .sofa file")
    ap.add_argument("--list-audio-devices", action="store_true")
    ap.add_argument("--no-display", action="store_true", help="run headless")
    ap.add_argument("--calibrate", type=float, metavar="MM",
                    help="personalise head scale + face width: eyes this far from the lens")
    ap.add_argument("--enroll", metavar="NAME",
                    help="find and measure your ears, save profiles/NAME.json (needs --distance)")
    ap.add_argument("--enroll-mode", choices=("crosshair", "sweep"), default="crosshair",
                    help="crosshair: put each ear on the centre cross and press SPACE (no clicks); "
                         "sweep: turn slowly, then click the ear in captured frames")
    ap.add_argument("--distance", type=float, metavar="MM",
                    help="lens-to-eyes distance during the first enrollment step")
    ap.add_argument("--profile", help="listener profile JSON to use (overrides config)")
    ap.add_argument("--ear-cascades", nargs=2, metavar=("LEFT_XML", "RIGHT_XML"),
                    help="optional Haar ear cascades for automatic tragion marking")
    ap.add_argument("--listeners", choices=("single", "main", "midpoint"),
                    help="one listener, track everyone and aim at the main one, or aim between them")
    ap.add_argument("--main", metavar="NAME", help="profile name to treat as the main listener")
    ap.add_argument("--faceid-model", metavar="ONNX", help="SFace model: recognise listeners, load their profiles")
    ap.add_argument("--profiles", metavar="DIR", help="folder of listener profiles for face ID")
    ap.add_argument("--add-face", metavar="NAME", help="store NAME's face in profiles/NAME.json")
    ap.add_argument("--lead-ms", type=float, help="aim this far ahead of head motion (0 = off)")
    ap.add_argument("--depth", choices=("none", "realsense", "oakd", "stereo"), help="depth camera")
    ap.add_argument("--stereo-calib", metavar="JSON", help="stereo calibration for --depth stereo")
    ap.add_argument("--calibrate-camera", metavar="COLSxROWS",
                    help="checkerboard camera calibration (inner corners, e.g. 9x6)")
    ap.add_argument("--calibrate-stereo", metavar="COLSxROWS",
                    help="calibrate two webcams (camera.index + depth.right_index) for depth")
    ap.add_argument("--square", type=float, metavar="MM", help="checkerboard square size")
    ap.add_argument("--views", type=int, default=20, help="calibration views to collect")
    ap.add_argument("--make-markers", metavar="DIR", help="write one printable ArUco marker per stand")
    ap.add_argument("--locate-stands", action="store_true",
                    help="measure stand pivot positions from their ArUco markers")
    ap.add_argument("--write-config", metavar="PATH", help="write the default config and exit")
    ap.add_argument("--layout", choices=("stereo", "5.1", "7.1"),
                    help="with --write-config: place stands for this layout around the seat; "
                         "otherwise: override audio.layout")
    ap.add_argument("--move-stand", nargs=4, action="append", metavar=("NAME", "LEFT", "FROM_CAM", "HEIGHT"),
                    help="move a stand (mm): LEFT + = listener's left, FROM_CAM = distance from the camera, "
                         "HEIGHT + = above the speaker axis; re-aims its zero direction at the seat. Repeatable.")
    ap.add_argument("--keep-facing", action="store_true",
                    help="with --move-stand: keep yaw_offset_deg instead of facing the seat")
    ap.add_argument("--move-sub", nargs=3, type=float, metavar=("LEFT", "FROM_CAM", "HEIGHT"),
                    help="move the subwoofer (for its time alignment)")
    ap.add_argument("--seat-mm", type=float, default=900.0,
                    help="with --write-config --layout: seat distance from the camera (and speaker radius)")
    ap.add_argument("-v", "--verbose", action="store_true")
    return ap.parse_args(argv)


def apply_overrides(cfg: Config, args) -> None:
    if args.port:
        cfg.link.port = args.port
    if args.backend:
        cfg.detector.backend = args.backend
    if args.model:
        cfg.detector.model_path = args.model
    if args.audio:
        cfg.audio.mode = args.audio
    if args.audio_file:
        cfg.audio.file = args.audio_file
    if args.audio_in:
        cfg.audio.input_device = args.audio_in
    if args.audio_out:
        cfg.audio.output_device = args.audio_out
    if args.hrtf:
        cfg.audio.hrtf = args.hrtf
    if args.profile:
        cfg.profile = args.profile
    if args.layout:
        cfg.audio.layout = args.layout
    if args.listeners:
        cfg.listeners.mode = args.listeners
    if args.main:
        cfg.listeners.main = args.main
    if args.faceid_model:
        cfg.faceid.model_path = args.faceid_model
        cfg.faceid.enabled = True
    if args.profiles:
        cfg.faceid.profiles_dir = args.profiles
    if args.lead_ms is not None:
        cfg.predict.lead_ms = args.lead_ms
    if args.depth:
        cfg.depth.source = args.depth
    if args.stereo_calib:
        cfg.depth.stereo_calib = args.stereo_calib
    cfg.validate()


# ----------------------------------------------------------------- main
def main(argv=None) -> int:
    args = parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(levelname)s %(message)s")

    if args.write_config:
        cfg = Config.preset(args.layout or "stereo", args.seat_mm)
        cfg.save(args.write_config)
        print(f"wrote {args.write_config} ({cfg.audio.layout}, {len(cfg.stands)} stand(s))")
        for i, s in enumerate(cfg.stands):
            x, y, z = s.position_mm
            print(f"  {s.name:<6} pins {s.pan.pin}/{s.tilt.pin}  out {s.output_channel if s.output_channel >= 0 else i}"
                  f"  x {x:+7.1f} y {y:+6.1f} z {z:+7.1f} mm  faces {s.yaw_offset_deg:+6.1f} deg")
        if cfg.audio.sub_output >= 0:
            print(f"  LFE    sub on output {cfg.audio.sub_output} at {cfg.audio.sub_position_mm} mm")
        return 0
    if args.list_audio_devices:
        import sounddevice as sd
        print(sd.query_devices())
        return 0

    cfg = Config.load(args.config) if args.config else Config()
    apply_overrides(cfg, args)

    if args.move_stand or args.move_sub:
        return move_stands(cfg, args)

    # ---- setup tools that need no camera
    if args.make_markers:
        from .markers import make_marker_sheets
        paths = make_marker_sheets(cfg, args.make_markers)
        for p, st_ in zip(paths, cfg.stands):
            print(f"  {st_.name:<6} id {st_.marker_id}  {p}")
        print(f"Print at 100 % and check the black square is {cfg.markers.size_mm:g} mm. Fix each marker "
              "to its stand's pan/tilt head, then set marker_offset_mm (marker centre -> pivot).")
        if args.config:
            cfg.save(args.config)
        return 0
    if args.calibrate_stereo:
        from .calib import parse_board, run_stereo_calibration
        if not args.square:
            raise SystemExit("--calibrate-stereo needs --square MM")
        cols, rows = parse_board(args.calibrate_stereo)
        out = cfg.depth.stereo_calib or str(Path(args.config or "stands.json").with_name("stereo.json"))
        return run_stereo_calibration(cfg, cols, rows, args.square, out, args.views, args.config)

    if args.calibrate_camera:
        from .calib import parse_board, run_camera_calibration
        if not args.square:
            raise SystemExit("--calibrate-camera needs --square MM")
        cols, rows = parse_board(args.calibrate_camera)
        return run_camera_calibration(open_capture(cfg, args.source), cfg, cols, rows, args.square,
                                      args.views, args.config)

    src = open_source(cfg, args.source)
    K = src.intrinsics or geometry.Intrinsics.from_camera(cfg.camera)
    if args.locate_stands:
        from .calib import undistorter_for
        from .markers import run_locate_stands
        und = undistorter_for(cfg.camera) if src.intrinsics is None else None

        class _Frames:                              # undistorted frames for the locator
            def read(self):
                ok, f, _ = src.read()
                return ok, (und(f) if (ok and und is not None) else f)
        return run_locate_stands(_Frames(), cfg, K, None, config_path=args.config)
    cap = _CompatCap(src)
    if args.add_face:
        return add_face(cfg, cap, args.add_face)
    if args.enroll:
        if not args.distance:
            raise SystemExit("--enroll needs --distance MM (lens to eyes, measured)")
        from . import enroll_ui
        out = Path(args.config).parent / "profiles" / f"{args.enroll}.json" if args.config \
            else Path("profiles") / f"{args.enroll}.json"
        return enroll_ui.run(cfg, cap, args.enroll, args.distance, out,
                             tuple(args.ear_cascades) if args.ear_cascades else None, args.config,
                             mode=args.enroll_mode)
    if args.calibrate:
        return calibrate(cfg, cap, args.calibrate, args.config)

    profile = None
    if cfg.profile:
        from .enroll import ListenerProfile
        profile = ListenerProfile.load(cfg.profile)
        profile.apply(cfg.head)
        log.info("profile %s: ears %.1f mm apart, sphere radius %.1f mm",
                 profile.name, profile.ear_distance_mm, profile.sphere_radius_mm)

    centers = [(s.pan.center_deg, s.tilt.center_deg) for s in cfg.stands]
    link = SerialLink(cfg.link.port, cfg.link.baud, centers) if cfg.link.port else DryRunLink()
    if isinstance(link, DryRunLink):
        log.info("dry run: no serial port given, servo commands are not sent")

    audio = None
    if cfg.audio.mode != "off" and not (cfg.audio.file or cfg.audio.input_device):
        log.info("audio %s set, but no --audio-file or --audio-in: tracking and stands only", cfg.audio.mode)
    elif cfg.audio.mode != "off":
        from .audio import AudioEngine
        from .layouts import build_routing
        audio = AudioEngine(cfg.audio, hrtf.load(cfg.audio.hrtf, cfg.head.sphere_radius_mm()),
                            routing=build_routing(cfg))
        try:
            audio.start()
        except OSError as e:                         # sounddevice without PortAudio
            raise SystemExit(f"audio: {e} - install PortAudio (Linux: libportaudio2)") from e

    identifier = None
    if cfg.faceid.enabled:
        from .faceid import FaceIdentifier, ProfileLibrary, SFaceEmbedder
        if cfg.detector.backend == "haar":
            raise SystemExit("face ID needs 5 face points: use --backend yunet or mediapipe")
        identifier = FaceIdentifier(SFaceEmbedder(cfg.faceid.model_path),
                                    ProfileLibrary(cfg.faceid.profiles_dir), cfg.faceid.threshold,
                                    cfg.faceid.confirm_frames, cfg.faceid.interval_frames)

    pipe = Pipeline(cfg, link, audio=audio, profile=profile, intrinsics=src.intrinsics,
                    identifier=identifier)
    fps, t_prev = 0.0, time.perf_counter()
    try:
        while True:
            ok, frame, depth = src.read()
            if not ok:
                log.info("video source ended")
                break
            t = time.perf_counter()
            st = pipe.step(frame, t, depth)
            fps = 0.9 * fps + 0.1 / max(t - t_prev, 1e-3)
            t_prev = t

            for line in link.poll():
                log.debug("arduino: %s", line)
            if args.no_display:
                if st.sent:
                    log.debug("send %s", ", ".join(f"({p:.1f},{q:.1f})" for p, q in st.angles))
                continue
            view = st.image.copy() if st.image is not None else frame
            draw(view, st, cfg, pipe.K, fps, audio)
            cv2.imshow("orpheus", view)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q")):
                break
            if key == ord("h"):
                link.home()
    except KeyboardInterrupt:
        pass
    finally:
        if audio is not None:
            audio.stop()
        link.close()
        src.release()
        if not args.no_display:
            cv2.destroyAllWindows()
    return 0


# ------------------------------------------------------------ calibration
def calibrate(cfg: Config, cap, distance_mm: float, config_path: str | None) -> int:
    """Personalise the head model at a known lens-to-eyes distance.

    - Landmark backends (PnP): the model scale that makes the solved eye depth
      equal the true distance. Monocular PnP only knows shape, not size, so this
      single number sets ear spacing, sphere radius (ITD) and depth together.
    - Haar: face-box width -> face_width_mm, and eye spacing -> IPD -> scale.
    """
    cfg.head.scale = 1.0
    pipe = Pipeline(cfg, DryRunLink())
    K = pipe.K
    scales, widths = [], []
    print(f"Eyes {distance_mm:.0f} mm from the lens, look straight at it, hold still. "
          "Press c to capture, q to quit.")
    capturing = False
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        st = pipe.step(frame, time.perf_counter())
        f, L = st.face, st.listener
        if capturing and f is not None and L is not None:
            if L.method == "pnp":
                eye_mid = L.R @ ((cfg.head.points()["eye_l"] + cfg.head.points()["eye_r"]) / 2) + L.t
                scales.append(distance_mm / eye_mid[2])
            else:
                if "camshift" not in f.source:
                    widths.append(f.box[2])
                if "eye_l" in f.points and "eye_r" in f.points:
                    ipd_px = float(np.hypot(*np.subtract(f.points["eye_l"], f.points["eye_r"])))
                    scales.append(ipd_px * distance_mm / K.fx / IPD_MM)
        frame = st.image.copy() if st.image is not None else frame
        draw(frame, st, cfg, K, 30.0)
        n = max(len(scales), len(widths))
        cv2.putText(frame, f"samples {n}/40", (10, frame.shape[0] - 14), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                    (0, 255, 255), 2)
        cv2.imshow("calibrate", frame)
        key = cv2.waitKey(1) & 0xFF
        if key == ord("c"):
            capturing = True
        if key in (27, ord("q")) or n >= 40:
            break
    cap.release()
    cv2.destroyAllWindows()

    if not scales and not widths:
        print("nothing captured")
        return 1
    if widths:
        cfg.detector.face_width_mm = round(
            geometry.calibrate_face_width(float(np.median(widths)), distance_mm, K), 1)
        print(f"face_width_mm = {cfg.detector.face_width_mm}")
    if scales:
        cfg.head.scale = round(float(np.median(scales)), 3)
        print(f"head scale = {cfg.head.scale}  -> ear spacing {cfg.head.scale * cfg.head.bitragion_mm:.0f} mm, "
              f"sphere radius {cfg.head.sphere_radius_mm():.1f} mm")
    if config_path:
        cfg.save(config_path)
        print(f"saved to {config_path}")
    return 0


def move_stands(cfg: Config, args) -> int:
    from .layouts import move_stand
    seat = (0.0, 0.0, args.seat_mm)
    if cfg.profile:
        from .enroll import ListenerProfile
        ref = ListenerProfile.load(cfg.profile).reference
        if ref.get("head_center_mm"):
            seat = tuple(ref["head_center_mm"])       # the enrolled listening seat
    for name, left, dist, height in args.move_stand or []:
        try:
            move_stand(cfg, name, float(left), float(dist), float(height), seat, not args.keep_facing)
        except ValueError as e:
            raise SystemExit(str(e)) from e
    if args.move_sub:
        left, dist, height = args.move_sub
        cfg.audio.sub_position_mm = (left, -height, dist)
    cfg.validate()
    for i, s in enumerate(cfg.stands):
        x, y, z = s.position_mm
        print(f"  {s.name:<6} left {x:+7.1f}  from camera {z:7.1f}  height {-y:+6.1f} mm   facing {s.yaw_offset_deg:+6.1f} deg")
    if cfg.audio.sub_output >= 0:
        x, y, z = cfg.audio.sub_position_mm
        print(f"  sub    left {x:+7.1f}  from camera {z:7.1f}  height {-y:+6.1f} mm")
    if args.config:
        cfg.save(args.config)
        print(f"saved to {args.config}")
    else:
        print("(no --config given: nothing saved)")
    return 0


class _CompatCap:
    """cv2.VideoCapture-like view of a source, for the enrollment / calibration tools."""

    def __init__(self, src) -> None:
        self.src = src

    def read(self):
        ok, frame, _ = self.src.read()
        return ok, frame

    def release(self) -> None:
        self.src.release()


def add_face(cfg: Config, cap, name: str, frames: int = 30) -> int:
    """Store a face embedding in an existing profile so face ID can load it."""
    from .enroll import ListenerProfile
    from .faceid import SFaceEmbedder, capture_embedding
    from .landmarks import make_backend

    path = Path(cfg.faceid.profiles_dir) / f"{name}.json"
    if not path.exists():
        raise SystemExit(f"{path} not found - enroll first (--enroll {name})")
    if cfg.detector.backend == "haar":
        raise SystemExit("face ID needs 5 face points: use --backend yunet or mediapipe")
    if not cfg.faceid.model_path:
        raise SystemExit("--add-face needs --faceid-model face_recognition_sface_2021dec.onnx")
    emb = SFaceEmbedder(cfg.faceid.model_path)
    backend = make_backend(cfg.detector.backend, cfg.detector, cfg.detector.model_path)
    samples = []
    print(f"Look at the camera, {name}. Capturing {frames} frames... q aborts.")
    while len(samples) < frames:
        ok, frame = cap.read()
        if not ok:
            break
        face = backend(frame, time.perf_counter())
        if face is not None and face.yunet_row() is not None:
            samples.append((frame.copy(), face))
        cv2.putText(frame, f"{len(samples)}/{frames}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                    (0, 230, 255), 2)
        cv2.imshow("add face", frame)
        if (cv2.waitKey(1) & 0xFF) in (27, ord("q")):
            return 1
    cv2.destroyAllWindows()
    vec = capture_embedding(emb, samples)
    if vec is None:
        print("not enough clear frames")
        return 1
    prof = ListenerProfile.load(path)
    prof.face_embedding = vec
    prof.save(path)
    print(f"face stored in {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
