"""Hidden-ear-only compensation on rendered frames: each ear is a distinct texture
drawn where it projects, only when it faces the camera."""


import cv2
import numpy as np
import pytest

from orpheus.config import Config
from orpheus.earfind import EAR_HEIGHT_MM
from orpheus.enroll import EnrollmentSession, ListenerProfile, ear_view_angle, project
from orpheus.geometry import Intrinsics
from orpheus.head import HeadModel, PoseSolver, rotation_from_ypr
from orpheus.landmarks import Face
from orpheus.link import DryRunLink
from orpheus.pipeline import Pipeline

K = Intrinsics(554.0, 554.0, 320.0, 240.0)
SCALE = 1.07
TRUE_EARS = {"left": np.array([79.0, 7.0, 10.0]), "right": np.array([-75.0, 5.0, 6.0])}
TRUE_MODEL = HeadModel(scale=SCALE)

_rng = np.random.default_rng(42)
BACKGROUND = cv2.GaussianBlur(_rng.integers(60, 200, (480, 640)).astype(np.uint8), (0, 0), 6)


def _ear_texture(seed):
    r = np.random.default_rng(seed)
    t = cv2.GaussianBlur(r.integers(0, 255, (96, 96)).astype(np.uint8), (0, 0), 3)
    t = cv2.normalize(t, None, 20, 235, cv2.NORM_MINMAX)
    mask = np.zeros((96, 96), np.uint8)
    cv2.ellipse(mask, (48, 48), (30, 44), 0, 0, 360, 255, -1)
    return t, mask


EAR_TEX = {"left": _ear_texture(1), "right": _ear_texture(2)}


def render(R, t):
    img = BACKGROUND.copy()
    for side, ear in TRUE_EARS.items():
        p = R @ ear + t
        if ear_view_angle(R, t, ear) > 75:          # edge-on or turned away: not visible
            continue
        u, v = project(p, K)
        s = int(round(K.fx * EAR_HEIGHT_MM / p[2]))   # the ear itself, ~65 mm tall
        tex, mask = EAR_TEX[side]
        tex = cv2.resize(tex, (s, s))
        mask = cv2.resize(mask, (s, s)) > 127
        x0, y0 = int(round(u - s / 2)), int(round(v - s / 2))
        if x0 < 0 or y0 < 0 or x0 + s > 640 or y0 + s > 480:
            continue
        roi = img[y0:y0 + s, x0:x0 + s]
        roi[mask] = tex[mask]
    return cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)


def landmarks(R, t, only_eyes=False, noise=0.5, rng=None):
    rng = rng or np.random.default_rng(0)
    return {k: tuple(np.add(project(R @ v + t, K), rng.normal(0, noise, 2)))
            for k, v in TRUE_MODEL.points().items() if not only_eyes or k.startswith("eye")}


def build_profile():
    rng = np.random.default_rng(5)
    model = HeadModel()
    sess = EnrollmentSession(K, model, 600.0, "tex")
    solver = PoseSolver(K, model)

    def feed(R, t, n):
        for _ in range(n):
            pts = landmarks(R, t, rng=rng)
            out = solver.solve(pts)
            sess.feed(render(R, t), *(out if out else (None, None)), pts)

    feed(rotation_from_ypr(0), np.array([0.0, SCALE * 10.0, 600 + SCALE * 82]), 40)
    for side, sign in (("left", -1), ("right", 1)):
        for yaw in (20, 30, 40, 50, 60):
            R = rotation_from_ypr(sign * yaw)
            t = np.array([0.0, 0.0, 650.0]) - R @ TRUE_EARS[side]
            feed(R, t, 12)
            assert sess.capture_center().startswith("Captured")
    return sess.solve()


@pytest.fixture(scope="module")
def profile():
    return build_profile()


def test_profile_carries_ear_templates(profile):
    assert len(profile.ear_templates["left"]) == 5 and len(profile.ear_templates["right"]) == 5
    back = ListenerProfile(**{**profile.__dict__})
    assert back.ear_templates == profile.ear_templates


def _pipeline(profile, ear_error_mm=0.0, earfind=True):
    prof = ListenerProfile(**{**profile.__dict__})
    if ear_error_mm:   # pretend enrollment was off: shift both stored ears
        prof.ear_l_mm = tuple(np.add(prof.ear_l_mm, (0, 0, ear_error_mm)))
        prof.ear_r_mm = tuple(np.add(prof.ear_r_mm, (0, 0, ear_error_mm)))
    cfg = Config()
    cfg.earfind.enabled = earfind
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, t: None, profile=prof, intrinsics=K)
    return pipe


def _run(pipe, R, t, only_eyes, frames=8):
    st = None
    for i in range(frames):
        pts = landmarks(R, t, only_eyes, rng=np.random.default_rng(i))
        pipe.backend = lambda f, tt, p=pts: Face((0, 0, 1, 1), p, "syn")
        st = pipe.step(render(R, t), i / 30)
    err = {s: float(np.linalg.norm((st.listener.ear_l if s == "left" else st.listener.ear_r)
                                   - (R @ TRUE_EARS[s] + t))) for s in ("left", "right")}
    return st, err


def test_frontal_both_ears_use_offset(profile):
    st, _ = _run(_pipeline(profile), rotation_from_ypr(0), np.array([0.0, 0, 800]), False)
    assert st.ear_sources == {"left": "offset", "right": "offset"}


def test_visible_ear_measured_hidden_ear_offset(profile):
    pipe = _pipeline(profile)
    R, t = rotation_from_ypr(-40), np.array([60.0, 0, 850])   # turned right: left ear shows
    st, err = _run(pipe, R, t, only_eyes=False)
    assert st.listener.method == "pnp"
    assert st.ear_sources == {"left": "seen", "right": "offset"}
    assert err["left"] < 8 and err["right"] < 8


@pytest.mark.parametrize("yaw,x,z,seen", [(30, -40.0, 800.0, "right"), (-35, 80.0, 950.0, "left"),
                                          (40, 60.0, 900.0, "right")])
def test_eyes_only_turned_head_seen_ear_fixes_pose(profile, yaw, x, z, seen):
    """Two eyes cannot show yaw; the seen ear is the third point that does. The hidden
    ear stays an offset placement, now under the corrected pose."""
    R, t = rotation_from_ypr(yaw), np.array([x, 0, z])
    off = _pipeline(profile, earfind=False)
    _, err_off = _run(off, R, t, only_eyes=True)
    st, err = _run(_pipeline(profile), R, t, only_eyes=True)
    hidden = "left" if seen == "right" else "right"
    assert st.listener.method == "eyes"
    assert st.ear_sources == {seen: "seen", hidden: "offset"}
    assert max(err_off.values()) > 100                      # eyes alone: badly off
    assert err[seen] < 10 and err[hidden] < 10
    assert st.listener.ypr[0] == pytest.approx(yaw, abs=3)


def test_eyes_only_facing_camera_no_ear_used(profile):
    st, err = _run(_pipeline(profile), rotation_from_ypr(0), np.array([0.0, 0, 800]), only_eyes=True)
    assert st.ear_sources == {"left": "offset", "right": "offset"}
    assert max(err.values()) < 10
