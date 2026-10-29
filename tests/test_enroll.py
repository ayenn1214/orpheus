import math

import numpy as np
import pytest

from orpheus.enroll import (EarObservation, EnrollmentSession, ListenerProfile,
                                    ear_view_angle, project, ray_in_head, triangulate)
from orpheus.geometry import Intrinsics
from orpheus.head import HeadModel, PoseSolver, make_listener, rotation_from_ypr

K = Intrinsics(554.0, 554.0, 320.0, 240.0)
TRUE_SCALE = 1.07
TRUE_EARS = {"left": np.array([79.0, 7.0, 10.0]), "right": np.array([-75.0, 5.0, 6.0])}


def test_triangulate_exact_and_prior():
    p = np.array([10.0, -5.0, 3.0])
    origins = np.array([[0, 0, -500.0], [300, 0, -400.0], [-200, 100, -450.0]])
    dirs = (p - origins) / np.linalg.norm(p - origins, axis=1, keepdims=True)
    q, rms, base = triangulate(origins, dirs)
    assert np.allclose(q, p, atol=1e-9) and rms < 1e-9 and base > 20
    # parallel rays: the prior pins the depth
    q2, _, _ = triangulate(origins[:1].repeat(2, 0), dirs[:1].repeat(2, 0), prior=np.zeros(3))
    assert np.isfinite(q2).all()


def test_ray_in_head_passes_through_ear():
    R = rotation_from_ypr(-50, 5, 3)
    t = np.array([40.0, -20.0, 700.0])
    ear = np.array([72.0, 3.0, 4.0])
    o, d = ray_in_head(EarObservation(R, t, project(R @ ear + t, K)), K)
    assert np.linalg.norm(np.cross(ear - o, d)) < 1e-6


def test_view_angle():
    ear = np.array([72.5, 0.0, 0.0])
    t = np.array([0.0, 0.0, 700.0])
    assert ear_view_angle(rotation_from_ypr(0), t, ear) == pytest.approx(90, abs=6)
    # listener turns to their right (negative yaw) -> left ear comes round to the lens
    assert ear_view_angle(rotation_from_ypr(-60), t, ear) < 40


def _simulate_session(click_noise_px=1.5, seed=0):
    rng = np.random.default_rng(seed)
    true_model = HeadModel(scale=TRUE_SCALE)
    model = HeadModel()
    sess = EnrollmentSession(K, model, 600.0, "sim", mode="sweep")
    solver = PoseSolver(K, model)
    # eyes centred on the lens axis, 600 mm away
    t_true = np.array([0.0, TRUE_SCALE * 10.0, 600 + TRUE_SCALE * 82])
    truth = {}

    def yaw_at(i):
        if i < 60:
            return 0.0
        j = (i - 60) % 240
        side = -1 if ((i - 60) // 240) % 2 == 0 else 1
        return side * 65 * math.sin(j / 240 * math.pi)

    i = 0
    while sess.step != "mark" and i < 3000:
        R = rotation_from_ypr(yaw_at(i), rng.normal(0, 1), rng.normal(0, 1))
        truth[i] = R
        pts = {k: tuple(np.add(project(R @ v + t_true, K), rng.normal(0, 0.7, 2)))
               for k, v in true_model.points().items()}
        frame = np.full((2, 2, 3), 0, np.int32)
        frame[0, 0, 0] = i                                         # tag frame with its index
        out = solver.solve(pts)
        sess.feed(frame, *(out if out else (None, None)), pts)
        i += 1
    assert sess.step == "mark", "capture did not complete"

    for kf in sess.keyframes:                                      # simulated clicks
        R = truth[int(kf.frame[0, 0, 0])]
        ear = TRUE_EARS[kf.side]
        px = np.add(project(R @ ear + t_true, K), rng.normal(0, click_noise_px, 2))
        sess.set_pixel(kf, px, "click")
    return sess


def test_enrollment_recovers_scale_and_ears():
    sess = _simulate_session()
    assert sess.model.scale == pytest.approx(TRUE_SCALE, rel=0.01)
    prof = sess.solve(drivers={"left": np.array([-300.0, 0, 60]), "right": np.array([300.0, 0, 60])})
    for side, key in (("left", "ear_l_mm"), ("right", "ear_r_mm")):
        err = np.linalg.norm(np.array(getattr(prof, key)) - TRUE_EARS[side])
        assert err < 6.0, f"{side} ear off by {err:.1f} mm"
    assert prof.ear_distance_mm == pytest.approx(np.linalg.norm(TRUE_EARS["left"] - TRUE_EARS["right"]), abs=6)
    assert prof.quality["left"]["views"] + prof.quality["left"]["rejected"] == 12
    ref = prof.reference
    assert set(ref["speaker_to_ear_mm"]) == {"left", "right"}
    assert ref["ypr_deg"][0] == pytest.approx(0, abs=3)


def test_profile_roundtrip_and_apply(tmp_path):
    prof = _simulate_session().solve()
    p = tmp_path / "profiles" / "sim.json"
    prof.save(p)
    back = ListenerProfile.load(p)
    head = HeadModel()
    back.apply(head)
    assert head.ears_measured and head.scale == pytest.approx(prof.head_scale)
    assert head.ear_distance_mm() == pytest.approx(prof.ear_distance_mm, abs=0.1)
    assert head.sphere_radius_mm() == pytest.approx(prof.sphere_radius_mm, abs=0.1)


def test_stored_ears_beat_generic_at_runtime():
    """Same tracked pose; ears from the profile vs the generic model."""
    prof = _simulate_session().solve()
    stored, generic = HeadModel(), HeadModel(scale=prof.head_scale)
    prof.apply(stored)
    errs = {"stored": [], "generic": []}
    for yaw in (-40, -20, 0, 20, 40):
        R = rotation_from_ypr(yaw)
        t = np.array([0.0, 0.0, 800.0])
        for name, m in (("stored", stored), ("generic", generic)):
            L = make_listener(R, t, m, "pnp")
            errs[name] += [np.linalg.norm(L.ear_l - (R @ TRUE_EARS["left"] + t)),
                           np.linalg.norm(L.ear_r - (R @ TRUE_EARS["right"] + t))]
    assert max(errs["stored"]) < 6 < min(errs["generic"])


def test_solve_rejects_too_few_marks():
    sess = _simulate_session()
    for kf in sess.keyframes:
        if kf.side == "right":
            kf.pixel = None
    with pytest.raises(ValueError, match="right ear"):
        sess.solve()


def test_pipeline_uses_profile_and_returns_to_seat():
    from orpheus.config import Config
    from orpheus.landmarks import Face
    from orpheus.link import DryRunLink
    from orpheus.pipeline import Pipeline

    prof = _simulate_session().solve()
    cfg = Config()
    for s in cfg.stands:
        s.aim = "near_ear"
    cfg.filter.return_to_seat_s = 2.0
    true_model = HeadModel(scale=TRUE_SCALE)
    R_true, t_true = rotation_from_ypr(30), np.array([250.0, 0.0, 900.0])
    pts = {k: project(R_true @ v + t_true, K) for k, v in true_model.points().items()}
    seq = iter([Face((0, 0, 1, 1), pts, "syn")] * 30 + [None] * 120)
    link = DryRunLink()
    pipe = Pipeline(cfg, link, backend=lambda f, t: next(seq), profile=prof, intrinsics=K)
    frame = np.zeros((480, 640, 3), np.uint8)
    for i in range(30):
        st = pipe.step(frame, i / 30)
    # ears tracked through the stored geometry
    assert np.linalg.norm(st.listener.ear_l - (R_true @ TRUE_EARS["left"] + t_true)) < 8
    assert np.linalg.norm(st.listener.ear_r - (R_true @ TRUE_EARS["right"] + t_true)) < 8
    assert set(st.distances) == {"left", "right"}
    tracked_aim = link.last
    sent_at_seat = []
    for i in range(30, 150):
        st = pipe.step(frame, i / 30)
        if st.sent:
            sent_at_seat.append(i)
    assert len(sent_at_seat) == 1 and sent_at_seat[0] >= 30 + 60      # once, after 2 s
    assert st.at_seat and link.last != tracked_aim


def test_bad_click_is_rejected():
    sess = _simulate_session()
    good = sess.solve()
    kf = next(k for k in sess.keyframes if k.side == "left")
    kf.pixel = (kf.pixel[0] + 40, kf.pixel[1] - 30)                 # careless click
    prof = sess.solve()
    assert prof.quality["left"]["rejected"] >= 1
    assert np.linalg.norm(np.subtract(prof.ear_l_mm, good.ear_l_mm)) < 2.0


def test_manual_advance_after_min_views():
    sess = EnrollmentSession(K, HeadModel(), 600.0, "x", mode="sweep")
    sess.step = "left"
    assert not sess.can_advance()
    sess._bins["left"][sess.BINS[0]] = 3
    sess._bins["left"][sess.BINS[1]] = 3
    assert sess.can_advance()
    sess.advance()
    assert sess.step == "right"


# ------------------------------------------------------------ crosshair mode
def _crosshair_session(seed=0, align_err_px=2.0):
    """Eyes centred at 600 mm, then each ear put on the crosshair at 4 head turns."""
    rng = np.random.default_rng(seed)
    true_model = HeadModel(scale=TRUE_SCALE)
    model = HeadModel()
    sess = EnrollmentSession(K, model, 600.0, "cross")
    solver = PoseSolver(K, model)
    frame = np.zeros((2, 2, 3), np.uint8)

    def feed(R, t, n):
        for _ in range(n):
            pts = {k: tuple(np.add(project(R @ v + t, K), rng.normal(0, 0.7, 2)))
                   for k, v in true_model.points().items()}
            out = solver.solve(pts)
            sess.feed(frame, *(out if out else (None, None)), pts)

    feed(rotation_from_ypr(0), np.array([0.0, TRUE_SCALE * 10.0, 600 + TRUE_SCALE * 82]), 40)
    assert sess.step == "left"
    for side, sign in (("left", -1), ("right", 1)):
        for yaw in (20, 30, 40, 50, 60):
            R = rotation_from_ypr(sign * yaw, rng.normal(0, 2), 0)
            # listener slides until the true ear sits on the optical axis (+ alignment error)
            ear_cam = np.array([0.0, 0.0, 650.0]) + np.r_[rng.normal(0, align_err_px, 2) * 650 / K.fx, 0]
            t = ear_cam - R @ TRUE_EARS[side]
            feed(R, t, 12)
            msg = sess.capture_center()
            assert msg.startswith("Captured"), msg
    assert sess.step == "solve"
    return sess


def test_crosshair_enrollment_recovers_ears_and_ipd():
    prof = _crosshair_session().solve()
    for side, key in (("left", "ear_l_mm"), ("right", "ear_r_mm")):
        assert np.linalg.norm(np.array(getattr(prof, key)) - TRUE_EARS[side]) < 8
    assert prof.ipd_mm == pytest.approx(63 * TRUE_SCALE, rel=0.03)
    assert prof.quality["left"]["sources"] == ["crosshair"]
    el = np.array(prof.eye_mid_mm) + np.array(prof.ear_offsets_mm["left"])
    assert np.allclose(el, prof.ear_l_mm, atol=0.05)


def test_crosshair_rejects_wrong_ear_and_repeat_angle():
    sess = _crosshair_session()
    sess.step = "left"
    R = rotation_from_ypr(+40)                      # turned the wrong way for the left ear
    t = np.array([0.0, 0.0, 650.0]) - R @ TRUE_EARS["right"]
    sess._live = (np.zeros((2, 2, 3)), R, t, 1.0)
    assert "not turned" in sess.capture_center() or "does not look" in sess.capture_center()


def test_eye_anchored_ears_when_only_eyes_are_found():
    from orpheus.config import Config
    from orpheus.landmarks import Face
    from orpheus.link import DryRunLink
    from orpheus.pipeline import Pipeline

    prof = _crosshair_session().solve()
    true_model = HeadModel(scale=TRUE_SCALE)
    cfg = Config()
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, t: None, profile=prof, intrinsics=K)
    frame = np.zeros((480, 640, 3), np.uint8)
    errs = {}
    for yaw in (0, 20):
        e = []
        for x, z in ((-200, 700), (0, 900), (250, 1100)):
            R, t = rotation_from_ypr(yaw), np.array([x, 0.0, z])
            pts = {k: project(R @ v + t, K) for k, v in true_model.points().items() if k.startswith("eye")}
            pipe.backend = lambda f, tt, p=pts: Face((0, 0, 1, 1), p, "haar-frontal")
            pipe.reset()
            L = pipe.step(frame, 0.0).listener
            assert L.method == "eyes"
            e += [np.linalg.norm(L.ear_l - (R @ TRUE_EARS["left"] + t)),
                  np.linalg.norm(L.ear_r - (R @ TRUE_EARS["right"] + t))]
        errs[yaw] = max(e)
    assert errs[0] < 15                       # facing the camera: eyes + offset lands on the ears
    assert errs[20] > errs[0]                 # unobserved yaw costs accuracy (documented limit)
