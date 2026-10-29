"""v0.7 features: prediction, two listeners, face ID, ArUco stands, camera
calibration, depth-corrected pose, stereo depth."""

import json
import math

import cv2
import numpy as np
import pytest

from orpheus.calib import (Undistorter, board_points, calibrate, find_board, is_new_view,
                                   undistorter_for)
from orpheus.config import Config
from orpheus.depth import StereoMatcher, refine_pose_with_depth, sample_depth
from orpheus.enroll import ListenerProfile, project
from orpheus.faceid import FaceIdentifier, IdState, ProfileLibrary, unit
from orpheus.geometry import Intrinsics
from orpheus.head import HeadModel, PoseSolver, make_listener, rotation_from_ypr
from orpheus.landmarks import Face
from orpheus.link import DryRunLink
from orpheus.markers import StandLocator, dictionary
from orpheus.pipeline import Pipeline
from orpheus.predict import Predictor

K = Intrinsics(554.0, 554.0, 320.0, 240.0)
M = HeadModel()


def face_at(R, t, model=M, noise=0.0, rng=None, only=None):
    rng = rng or np.random.default_rng(0)
    pts = {}
    for k, v in model.points().items():
        if only and k not in only:
            continue
        u, w = project(R @ v + t, K)
        pts[k] = (u + rng.normal(0, noise), w + rng.normal(0, noise)) if noise else (u, w)
    us = [p[0] for p in pts.values()]
    vs = [p[1] for p in pts.values()]
    box = (int(min(us)) - 20, int(min(vs)) - 30, int(max(us) - min(us)) + 40, int(max(vs) - min(vs)) + 50)
    return Face(box=box, points=pts, source="syn")


# ---------------------------------------------------------------- prediction
def test_predictor_cancels_lag_for_steady_motion():
    p = Predictor(lead_ms=100, vel_tau_s=0.05)
    v = np.array([300.0, 0, 0])                       # 300 mm/s sideways
    out = None
    for i in range(30):
        t = i / 30
        L = make_listener(np.eye(3), np.array([0, 0, 800.0]) + v * t, M, "pnp")
        out = p(L, t)
    future = make_listener(np.eye(3), np.array([0, 0, 800.0]) + v * (29 / 30 + 0.1), M, "pnp")
    assert np.linalg.norm(out.ear_l - future.ear_l) < 3          # vs 30 mm without lead
    assert np.linalg.norm(L.ear_l - future.ear_l) == pytest.approx(30, abs=0.5)


def test_predictor_follows_turning_head_and_caps_shift():
    p = Predictor(lead_ms=100, vel_tau_s=0.05, max_shift_mm=20)
    for i in range(30):
        t = i / 30
        L = make_listener(rotation_from_ypr(90 * t), np.array([0, 0, 800.0]), M, "pnp")   # 90 deg/s
        out = p(L, t)
    assert np.linalg.norm(out.ear_l - L.ear_l) <= 20 + 1e-6
    p2 = Predictor(lead_ms=100)
    p2(L, 0.0)
    assert p2(L, 2.0) is L                              # long gap: velocity reset, no extrapolation


def test_pipeline_aims_ahead_but_reports_current():
    cfg = Config()
    for s in cfg.stands:
        s.aim = "near_ear"
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, t: None, intrinsics=K)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(20):
        t = i / 30
        tv = np.array([-150 + 450 * t, 0, 800.0])
        pipe.backend = lambda f, tt, tv=tv: face_at(np.eye(3), tv)
        st = pipe.step(frame, t)
    assert st.aim.ear_l[0] > st.listener.ear_l[0] + 20          # aim leads in the direction of motion


# ---------------------------------------------------------------- two listeners
class FakeAudio:
    def __init__(self, mode):
        self.mode_cfg, self.override, self.updates = mode, None, []

    def override_mode(self, m):
        self.override = m

    def update(self, L, drivers):
        self.updates.append(L)


def _two_face_backend(poses):
    return lambda f, t: [face_at(R, tv) for R, tv in poses]


def test_main_mode_aims_at_first_arrival_and_suspends_xtc():
    cfg = Config()
    cfg.listeners.mode = "main"
    cfg.audio.mode = "xtc"
    for s in cfg.stands:
        s.aim = "near_ear"
    audio = FakeAudio("xtc")
    a = (np.eye(3), np.array([-200.0, 0, 800]))
    b = (np.eye(3), np.array([250.0, 0, 900]))
    pipe = Pipeline(cfg, DryRunLink(), backend=_two_face_backend([a]), audio=audio, intrinsics=K)
    frame = np.zeros((480, 640, 3), np.uint8)
    pipe.step(frame, 0.0)
    assert audio.override is None
    pipe.backend = _two_face_backend([b, a])           # newcomer listed first by the detector
    st = pipe.step(frame, 0.05)
    assert len(st.listeners) == 2
    assert st.listener.t[0] == pytest.approx(-200, abs=5)   # still the first arrival
    assert audio.override == "align" and st.xtc_suspended
    pipe.backend = _two_face_backend([a])
    pipe.step(frame, 2.0)                               # b gone past hold_s
    assert audio.override is None


def test_midpoint_mode_aims_between_ears():
    cfg = Config()
    cfg.listeners.mode = "midpoint"
    for s in cfg.stands:
        s.aim = "near_ear"
    cfg.predict.lead_ms = 0
    a = (np.eye(3), np.array([-250.0, 0, 800]))
    b = (np.eye(3), np.array([250.0, 0, 800]))
    pipe = Pipeline(cfg, DryRunLink(), backend=_two_face_backend([a, b]), intrinsics=K)
    st = pipe.step(np.zeros((480, 640, 3), np.uint8), 0.0)
    la, lb = st.listeners
    assert np.allclose(st.aim.ear_l, (la.ear_l + lb.ear_l) / 2)
    assert st.aim.t[0] == pytest.approx(0, abs=3)


def test_named_main_listener_wins():
    cfg = Config()
    cfg.listeners.mode = "main"
    cfg.listeners.main = "bob"
    pipe = Pipeline(cfg, DryRunLink(), backend=_two_face_backend([
        (np.eye(3), np.array([-200.0, 0, 800])), (np.eye(3), np.array([220.0, 0, 800]))]), intrinsics=K)
    frame = np.zeros((480, 640, 3), np.uint8)
    pipe.step(frame, 0.0)
    bob = ListenerProfile(name="bob", created="", head_scale=1.0, ear_l_mm=(72.5, 0, 0),
                          ear_r_mm=(-72.5, 0, 0), ear_distance_mm=145, sphere_radius_mm=90)
    pipe.assign_profile(pipe.tracks[1], bob)
    st = pipe.step(frame, 0.03)
    assert st.names[st.main_index] == "bob" and st.listener.t[0] > 100


# ---------------------------------------------------------------- face ID
def _profile(tmp, name, vec):
    p = ListenerProfile(name=name, created="", head_scale=1.0, ear_l_mm=(80.0, 0, 0),
                        ear_r_mm=(-80.0, 0, 0), ear_distance_mm=160, sphere_radius_mm=92,
                        face_embedding=unit(vec).tolist())
    if name == "ankur":
        p.ear_l_mm, p.ear_r_mm, p.ear_distance_mm = (66.0, 0, 0), (-66.0, 0, 0), 132
    p.save(tmp / f"{name}.json")
    return p


def test_face_id_loads_profile_with_hysteresis(tmp_path):
    rng = np.random.default_rng(1)
    ankur, guest = rng.normal(size=128), rng.normal(size=128)
    _profile(tmp_path, "ankur", ankur)
    _profile(tmp_path, "sam", rng.normal(size=128))
    lib = ProfileLibrary(tmp_path)
    assert len(lib.profiles) == 2
    who = {"v": ankur}
    embed = lambda frame, face: unit(who["v"] + rng.normal(0, 0.3, 128))
    ident = FaceIdentifier(embed, lib, threshold=0.363, confirm_frames=4, interval_frames=10)

    cfg = Config()
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, t: face_at(np.eye(3), np.array([0, 0, 800.0])),
                    intrinsics=K, identifier=ident)
    frame = np.zeros((480, 640, 3), np.uint8)
    names = [pipe.step(frame, i / 30).names[0] for i in range(6)]
    assert names[:3] == ["", "", ""] and names[-1] == "ankur"     # needs 4 agreeing frames
    st = pipe.step(frame, 0.3)
    assert np.linalg.norm(st.listener.ear_l - st.listener.ear_r) == pytest.approx(132, abs=1)
    who["v"] = guest                                    # a stranger never loads anyone
    ids = IdState()
    for _ in range(10):
        assert ident.update(ids, frame, Face((0, 0, 1, 1))) is None
    assert ids.name == ""


def test_yunet_row_layout():
    f = Face(box=(10, 20, 100, 120), points={"eye_r": (40, 60), "eye_l": (80, 60), "nose": (60, 80),
                                             "mouth_r": (45, 100), "mouth_l": (75, 100)})
    row = f.yunet_row()
    assert row.shape == (1, 15) and list(row[0, 4:8]) == [40, 60, 80, 60]
    assert Face((0, 0, 1, 1), {"eye_l": (1, 1), "eye_r": (0, 0)}).yunet_row() is None


# ---------------------------------------------------------------- ArUco stands
def render_marker(R, t, size_mm, marker_id, img=None, ss=4):
    """Marker with a white quiet zone, rendered at ss x resolution then area-averaged
    like a real sensor (a plain warp aliases the edges and biases corners ~0.5 px)."""
    tex = cv2.aruco.generateImageMarker(dictionary("DICT_4X4_50"), marker_id, 200)
    tex = cv2.copyMakeBorder(tex, 50, 50, 50, 50, cv2.BORDER_CONSTANT, value=255)   # quiet zone
    s = size_mm * 300 / 200 / 2                     # half-size of the textured square incl. border
    corners3d = np.array([[-s, s, 0], [s, s, 0], [s, -s, 0], [-s, -s, 0]])
    dst = np.array([project(R @ c + t, K) for c in corners3d], np.float32) * ss + (ss - 1) / 2
    src = np.array([[-0.5, -0.5], [299.5, -0.5], [299.5, 299.5], [-0.5, 299.5]], np.float32)  # pixel edges
    H = cv2.getPerspectiveTransform(src, dst)
    big = (640 * ss, 480 * ss)
    warped = cv2.warpPerspective(tex, H, big, borderValue=0).astype(np.float32)
    mask = cv2.warpPerspective(np.full_like(tex, 255), H, big).astype(np.float32) / 255
    warped = cv2.resize(warped, (640, 480), interpolation=cv2.INTER_AREA)
    mask = cv2.resize(mask, (640, 480), interpolation=cv2.INTER_AREA)
    img = np.full((480, 640), 120.0, np.float32) if img is None else img.astype(np.float32)
    img = img * (1 - mask) + warped
    return np.clip(img, 0, 255).astype(np.uint8)


def test_locate_stands_from_markers():
    cfg = Config()
    cfg.stands[0].marker_id, cfg.stands[1].marker_id = 3, 7
    cfg.stands[0].marker_offset_mm = (0.0, 0.0, -40.0)    # pivot 40 mm behind the marker face
    cfg.markers.size_mm = 100
    true = {}
    img = None
    # marker frame (x right, y up, z toward the camera) -> camera frame, then turned about y
    facing = np.diag([1.0, -1.0, -1.0])
    for s, x, yaw in ((cfg.stands[0], 310.0, 15), (cfg.stands[1], -290.0, -10)):
        a = math.radians(yaw)
        R = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]]) @ facing
        t = np.array([x, 40.0, 1000.0])
        img = render_marker(R, t, 100, s.marker_id, img)
        true[s.name] = R @ np.array(s.marker_offset_mm) + t
    loc = StandLocator(cfg, K)
    found = loc.add(img)
    assert set(found) == {3, 7}
    res = loc.apply()
    for s in cfg.stands:
        assert np.linalg.norm(np.array(s.position_mm) - true[s.name]) < 9, (s.name, s.position_mm, true[s.name])
        assert abs(res[s.name]["facing_deg"]) < 25


# ---------------------------------------------------------------- calibration
def render_board(K_, dist, R, t, cols=9, rows=6, sq=25.0, size=(640, 480)):
    """Checkerboard with lens distortion: render ideal, then distort by remapping."""
    board_px = 40
    W, H = (cols + 3) * board_px, (rows + 3) * board_px
    tex = np.full((H, W), 255, np.uint8)
    for i in range(cols + 1):
        for j in range(rows + 1):
            if (i + j) % 2 == 0:
                x0, y0 = (i + 1) * board_px, (j + 1) * board_px
                tex[y0:y0 + board_px, x0:x0 + board_px] = 0
    # texture pixel -> board mm (inner corner (0,0) at texture (2,2)*board_px)
    A = np.array([[sq / board_px, 0, -2 * sq], [0, sq / board_px, -2 * sq], [0, 0, 1]])
    Kp = np.array(K_)
    Hm = Kp @ np.column_stack([R[:, 0], R[:, 1], t]) @ A
    ideal = cv2.warpPerspective(tex, Hm, size, borderValue=128)
    if dist is None:
        return ideal
    # for each distorted output pixel, find the ideal pixel it came from
    ys, xs = np.mgrid[0:size[1], 0:size[0]].astype(np.float32)
    pts = np.stack([xs.ravel(), ys.ravel()], 1).reshape(-1, 1, 2)
    und = cv2.undistortPoints(pts, Kp, np.array(dist), P=Kp).reshape(size[1], size[0], 2)
    return cv2.remap(ideal, und[..., 0], und[..., 1], cv2.INTER_LINEAR, borderValue=128)


def test_checkerboard_calibration_recovers_intrinsics_and_distortion():
    Kt = np.array([[600.0, 0, 330], [0, 600.0, 235], [0, 0, 1]])
    dt = [-0.18, 0.05, 0, 0, 0]
    rng = np.random.default_rng(2)
    objp = board_points(9, 6, 25.0)
    views = []
    for _ in range(40):
        ang = rng.uniform(-30, 30, 3)
        R = cv2.Rodrigues(np.radians(ang))[0]
        t = np.array([rng.uniform(-120, 0), rng.uniform(-90, 0), rng.uniform(450, 700)])
        img = render_board(Kt, dt, R, t)
        c = find_board(img, 9, 6)
        if c is not None and is_new_view(c, views, 20, 0.05):
            views.append(c)
        if len(views) >= 15:
            break
    assert len(views) >= 10
    K_, dist, rms = calibrate(objp, views, (640, 480))
    assert rms < 0.5
    assert K_[0, 0] == pytest.approx(600, rel=0.02) and K_[0, 2] == pytest.approx(330, abs=6)
    assert dist[0] == pytest.approx(-0.18, abs=0.05)


def test_intrinsics_from_calibration_scale_and_undistorter():
    cfg = Config()
    cfg.camera.matrix = [[1200.0, 0, 640], [0, 1200.0, 360], [0, 0, 1]]
    cfg.camera.calib_size = [1280, 720]
    cfg.camera.width, cfg.camera.height = 640, 360
    k = Intrinsics.from_camera(cfg.camera)
    assert (k.fx, k.cx, k.cy) == (600.0, 320.0, 180.0)
    assert undistorter_for(cfg.camera) is None              # no distortion stored
    cfg.camera.dist = [-0.2, 0.05, 0, 0, 0]
    u = undistorter_for(cfg.camera)
    assert isinstance(u, Undistorter) and u(np.zeros((360, 640, 3), np.uint8)).shape == (360, 640, 3)


# ---------------------------------------------------------------- depth
def depth_of_head(R, t, model, size=(640, 480)):
    """Depth image of the face points: each landmark gets a small disc at its true depth."""
    d = np.zeros((size[1], size[0]), np.float32)
    for v in model.points().values():
        p = R @ v + t
        u, w = project(p, K)
        cv2.circle(d, (int(round(u)), int(round(w))), 4, float(p[2]), -1)
    return d


def test_depth_fixes_distance_when_head_size_is_wrong():
    true_model = HeadModel(scale=1.12)                  # listener's head 12 % bigger than assumed
    R, t = rotation_from_ypr(20, 5), np.array([60.0, -20, 850])
    face = face_at(R, t, model=true_model)
    Rp, tp = PoseSolver(K, M).solve(face.points)
    assert abs(tp[2] - t[2]) > 60                       # monocular: ~90 mm short
    _, t2, used = refine_pose_with_depth(Rp, tp, "pnp", face.points, face.box,
                                         depth_of_head(R, t, true_model), K, M)
    assert used and abs(t2[2] - t[2]) < 15


def test_depth_gives_eyes_only_mode_the_head_turn():
    R, t = rotation_from_ypr(35, 0, 4), np.array([-50.0, 10, 750])
    face = face_at(R, t, only=("eye_l", "eye_r"))
    R2, t2, used = refine_pose_with_depth(np.eye(3), np.zeros(3), "eyes", face.points, face.box,
                                          depth_of_head(R, t, M), K, M)
    assert used
    L, L2 = make_listener(R, t, M, "x"), make_listener(R2, t2, M, "x")
    assert np.linalg.norm(L.ear_l - L2.ear_l) < 6 and np.linalg.norm(L.ear_r - L2.ear_r) < 6


def test_sample_depth_ignores_holes_and_range():
    d = np.zeros((20, 20), np.float32)
    d[8:13, 8:13] = 900
    d[10, 10] = 0
    assert sample_depth(d, 10, 10, 2) == 900
    assert sample_depth(d, 2, 2, 1) is None
    assert sample_depth(d, 10, 10, 2, lo=1000) is None


def test_stereo_matcher_measures_depth():
    f, B, Z = 500.0, 80.0, 900.0
    Kc = np.array([[f, 0, 320], [0, f, 240], [0, 0, 1]])
    calib = {"K1": Kc.tolist(), "D1": [0] * 5, "K2": Kc.tolist(), "D2": [0] * 5,
             "R": np.eye(3).tolist(), "T": [-B, 0, 0], "size": [640, 480]}
    sm = StereoMatcher(calib)
    rng = np.random.default_rng(0)
    tex = cv2.resize(rng.integers(0, 255, (120, 160)).astype(np.uint8), (640, 480),
                     interpolation=cv2.INTER_NEAREST)
    disp = f * B / Z                                     # pixels
    left = tex
    right = np.roll(tex, -int(round(disp)), axis=1)
    lr, rr = sm.rectify(cv2.cvtColor(left, cv2.COLOR_GRAY2BGR), cv2.cvtColor(right, cv2.COLOR_GRAY2BGR))
    z = sm.depth(lr, rr)
    centre = z[200:280, 260:380]
    valid = centre[centre > 0]
    expected = sm.f * B / round(disp)
    assert valid.size > 0.7 * centre.size
    assert np.median(valid) == pytest.approx(expected, rel=0.03)


def test_pipeline_uses_depth():
    true_model = HeadModel(scale=1.12)
    R, t = rotation_from_ypr(10), np.array([0.0, 0, 900])
    face = face_at(R, t, model=true_model)
    cfg = Config()
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, tt: face, intrinsics=K)
    st = pipe.step(np.zeros((480, 640, 3), np.uint8), 0.0, depth_of_head(R, t, true_model))
    assert st.depth_used
    L = make_listener(R, t, M, "x")
    assert abs(st.listener.t[2] - L.t[2]) < 15


def test_config_roundtrip_new_blocks(tmp_path):
    cfg = Config()
    cfg.listeners.mode = "midpoint"
    cfg.depth.source = "realsense"
    cfg.stands[0].marker_id = 4
    cfg.stands[0].marker_offset_mm = (0, -20, -35)
    cfg.camera.matrix = [[600, 0, 320], [0, 600, 240], [0, 0, 1]]
    p = tmp_path / "c.json"
    cfg.save(p)
    back = Config.load(p)
    assert back == cfg
    bad = json.loads(p.read_text())
    bad["stands"][1]["marker_id"] = 4
    with pytest.raises(ValueError, match="marker_id"):
        Config.from_dict(bad)
