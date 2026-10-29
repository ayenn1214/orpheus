import math

import numpy as np
import pytest

from orpheus import geometry, protocol
from orpheus.config import Config, ServoConfig, StandConfig
from orpheus.filters import OneEuro
from orpheus.link import DryRunLink
from orpheus.pipeline import Pipeline
from orpheus.head import HeadModel, make_listener, rotation_from_ypr
from orpheus.landmarks import Face


# ---------------------------------------------------------------- protocol
def test_crc8_check_value():
    # CRC-8/SMBus check value for "123456789"
    assert protocol.crc8(b"123456789") == 0xF4


def test_roundtrip_full_range():
    angles = [(0.0, 180.0), (90.0, 26.5), (179.99, 0.01)]
    typ, seq, out = protocol.decode(protocol.encode_set(300, angles))
    assert typ == protocol.TYPE_SET and seq == 300 & 0xFF
    for (a, b), (c, d) in zip(angles, out):
        assert abs(a - c) < 0.006 and abs(b - d) < 0.006


def test_values_over_255_survive():
    # The 2017 link sent char(centx); anything over 255 wrapped.
    _, _, out = protocol.decode(protocol.encode_set(0, [(170.0, 3.0)]))
    assert out[0][0] == pytest.approx(170.0)


def test_corruption_detected():
    f = bytearray(protocol.encode_set(1, [(45.0, 60.0)]))
    f[6] ^= 0x01
    with pytest.raises(ValueError):
        protocol.decode(bytes(f))


def test_clamp_and_limits():
    _, _, out = protocol.decode(protocol.encode_set(0, [(-5, 999)]))
    assert out == [(0.0, 180.0)]
    with pytest.raises(ValueError):
        protocol.encode_set(0, [])
    with pytest.raises(ValueError):
        protocol.encode_set(0, [(90, 90)] * 9)


# ---------------------------------------------------------------- geometry
def _K():
    return geometry.Intrinsics(fx=500, fy=500, cx=320, cy=240)


def test_centred_face_on_axis_and_depth():
    K = _K()
    # 150 mm face spanning 125 px at f=500 -> 600 mm
    p = geometry.face_to_head_xyz((320 - 62.5, 240 - 62.5, 125, 125), K, 150)
    assert p[0] == pytest.approx(0) and p[1] == pytest.approx(0)
    assert p[2] == pytest.approx(600 + geometry.FACE_TO_HEAD_CENTER_MM)


def test_mirror_flips_x():
    K = _K()
    a = geometry.face_to_head_xyz((400, 200, 100, 100), K, 150, 640, mirror=False)
    b = geometry.face_to_head_xyz((400, 200, 100, 100), K, 150, 640, mirror=True)
    assert a[0] == pytest.approx(-b[0])


def test_calibration_inverts_depth():
    K = _K()
    fw = geometry.calibrate_face_width(125, 600, K)
    assert fw == pytest.approx(150)


def _stand(x, aim="head", yaw=0.0):
    return StandConfig("s", (x, 0, 0), ServoConfig(pin=1), ServoConfig(pin=2),
                       yaw_offset_deg=yaw, aim=aim)


def test_stands_converge_symmetrically():
    head = np.array([0.0, 0.0, 1000.0])
    pl, tl = geometry.stand_angles(head, _stand(-300))
    pr, tr = geometry.stand_angles(head, _stand(300))
    assert pl == pytest.approx(math.degrees(math.atan2(300, 1000)))
    assert pr == pytest.approx(-pl)
    assert tl == pytest.approx(0) and tr == pytest.approx(0)


def test_tilt_up_when_head_above():
    head = np.array([0.0, -200.0, 1000.0])   # -Y is up
    _, tilt = geometry.stand_angles(head, _stand(0))
    assert tilt > 0


def test_yaw_offset():
    head = np.array([0.0, 0.0, 1000.0])
    p0, _ = geometry.stand_angles(head, _stand(-300))
    p1, _ = geometry.stand_angles(head, _stand(-300, yaw=10))
    assert p1 == pytest.approx(p0 - 10)


def test_near_ear_follows_head_yaw():
    m = HeadModel()
    # Facing the camera, the listener's RIGHT ear is on camera -X (the left stand's side).
    L0 = make_listener(rotation_from_ypr(0), np.array([0, 0, 1000.0]), m, "pnp")
    ear = geometry.aim_point(L0, _stand(-300, aim="near_ear"))
    assert np.allclose(ear, L0.ear_r) and ear[0] == pytest.approx(-72.5)
    # Turn 60 deg: that ear swings back (away from the camera) and toward the centre line
    L1 = make_listener(rotation_from_ypr(60), np.array([0, 0, 1000.0]), m, "pnp")
    e1 = geometry.aim_point(L1, _stand(-300, aim="near_ear"))
    assert e1[2] < 1000 - 50 or e1[2] > 1000 + 50
    assert abs(e1[0]) < abs(ear[0])


def test_servo_mapping_sign_and_clamp():
    sv = ServoConfig(pin=1, center_deg=90, sign=-1, min_deg=20, max_deg=160)
    assert geometry.to_servo(10, sv) == 80
    assert geometry.to_servo(-100, sv) == 160


# ------------------------------------------------------------------- filter
def test_one_euro_smooths_noise_and_tracks_steps():
    f = OneEuro(0.4, 0.002)
    rng = np.random.default_rng(0)
    out = [f(np.array([rng.normal(0, 10)]), i / 30) for i in range(300)]
    assert np.std(out[100:]) < 3          # 10 mm jitter -> < 3 mm
    for i in range(300, 360):
        y = f(np.array([300.0]), i / 30)
    assert abs(y[0] - 300) < 10           # follows a 300 mm move within 2 s


# ----------------------------------------------------------------- pipeline
class FakeBackend:
    def __init__(self, boxes):
        self.boxes = iter(boxes)

    def __call__(self, frame, t):
        b = next(self.boxes)
        return None if b is None else Face(box=b, source="haar-frontal")


def test_pipeline_rate_deadband_and_hold():
    cfg = Config()
    link = DryRunLink()
    box = (270, 190, 100, 100)
    pipe = Pipeline(cfg, link, backend=FakeBackend([box, box, box, None]))
    frame = np.zeros((480, 640, 3), np.uint8)

    s1 = pipe.step(frame, 0.0)
    assert s1.sent and len(s1.angles) == 2
    s2 = pipe.step(frame, 0.01)          # inside 1/rate_hz window
    assert not s2.sent
    s3 = pipe.step(frame, 1.0)           # same box -> inside deadband
    assert not s3.sent
    s4 = pipe.step(frame, 1.1)           # lost -> hold, nothing sent
    assert s4.face is None and not s4.sent and link.last == s1.angles


def test_config_roundtrip(tmp_path):
    cfg = Config()
    cfg.stands[0].tilt.center_deg = 26
    p = tmp_path / "c.json"
    cfg.save(p)
    back = Config.load(p)
    assert back.stands[0].tilt.center_deg == 26
    assert back.stands[1].position_mm == (-300.0, 0.0, 0.0)


def test_each_stand_keeps_its_own_ear_when_head_turns():
    """The sketch case: head turned hard, one ear toward the camera, one away. The nearer
    ear for the far stand is now the *other* ear; it must still aim at its own (virtual) ear."""
    m = HeadModel()
    L = make_listener(rotation_from_ypr(-70), np.array([0, 0, 650.0]), m, "pnp")
    left, right = _stand(300, aim="near_ear"), _stand(-300, aim="near_ear")
    d = lambda s, e: np.linalg.norm(e - np.asarray(s.position_mm))
    assert d(right, L.ear_l) < d(right, L.ear_r)          # old nearest-ear rule would pick ear_l
    assert np.allclose(geometry.aim_point(L, left), L.ear_l)
    assert np.allclose(geometry.aim_point(L, right), L.ear_r)
    explicit = _stand(-300, aim="near_ear")
    explicit.ear = "left"
    assert np.allclose(geometry.aim_point(L, explicit), L.ear_l)
