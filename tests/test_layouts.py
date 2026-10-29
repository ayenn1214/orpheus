"""5.1 / 7.1: stand presets, routing, multichannel alignment, renderer, pipeline."""

import json

import numpy as np
import pytest

from orpheus import audio as A
from orpheus import geometry, protocol
from orpheus.config import Config
from orpheus.head import HeadModel, make_listener, rotation_from_ypr
from orpheus.hrtf import C_SOUND
from orpheus.layouts import CHANNELS, build_routing, speaker_channels
from orpheus.link import DryRunLink
from orpheus.pipeline import Pipeline

FS = 48000
M = HeadModel()


@pytest.mark.parametrize("layout,n", [("5.1", 5), ("7.1", 7)])
def test_preset_places_and_faces_every_stand_at_the_seat(layout, n, tmp_path):
    cfg = Config.preset(layout, seat_mm=900)
    assert len(cfg.stands) == n
    assert [s.channel for s in cfg.stands] == list(speaker_channels(layout))
    assert cfg.audio.sub_output == CHANNELS[layout].index("LFE")
    assert sorted(s.output_channel for s in cfg.stands) == sorted(
        i for i, c in enumerate(CHANNELS[layout]) if c != "LFE")
    seat = make_listener(np.eye(3), np.array([0, 0, 900.0]), M, "seat")
    for s in cfg.stands:
        pan, _ = geometry.stand_angles(seat.t, s)
        assert abs(pan) < 0.5, (s.name, pan)              # zero pan = facing the seat
        dist = np.linalg.norm(np.array(s.position_mm)[[0, 2]] - [0, 900])
        assert dist == pytest.approx(900, abs=1) or s.channel == "FC"
    p = tmp_path / "c.json"
    cfg.save(p)
    assert Config.load(p) == cfg


def test_rear_stand_pan_wraps_instead_of_jumping_360():
    cfg = Config.preset("7.1")
    bl = next(s for s in cfg.stands if s.channel == "BL")
    for x in (-60.0, 0.0, 60.0):                         # listener slides across behind-centre
        pan, _ = geometry.stand_angles(np.array([x, 0, 900.0]), bl)
        assert -90 < pan < 90


def test_centre_stand_aims_at_head_centre_and_sides_at_own_ear():
    cfg = Config.preset("5.1")
    L = make_listener(rotation_from_ypr(25), np.array([80, -40, 950.0]), M, "pnp")
    by = {s.channel: s for s in cfg.stands}
    assert np.allclose(geometry.aim_point(L, by["FC"]), L.t)
    assert np.allclose(geometry.aim_point(L, by["FL"]), L.ear_l)
    assert np.allclose(geometry.aim_point(L, by["SR"]), L.ear_r)


def test_validation_rejects_bad_multichannel_setups():
    cfg = Config.preset("5.1")
    d = cfg.to_dict()
    bad = json.loads(json.dumps(d))
    bad["stands"][1]["channel"] = "FL"
    with pytest.raises(ValueError, match="same channel"):
        Config.from_dict(bad)
    bad = json.loads(json.dumps(d))
    bad["stands"][0]["channel"] = "LFE"
    with pytest.raises(ValueError, match="LFE"):
        Config.from_dict(bad)
    bad = json.loads(json.dumps(d))
    bad["audio"]["mode"] = "xtc"
    with pytest.raises(ValueError, match="stereo layout"):
        Config.from_dict(bad)
    bad = json.loads(json.dumps(d))
    bad["stands"].append(dict(bad["stands"][0], name="extra", channel=""))
    with pytest.raises(ValueError, match="not in layout"):
        Config.from_dict(bad)
    bad = json.loads(json.dumps(d))
    bad["stands"][2]["output_channel"] = 0
    with pytest.raises(ValueError, match="output channel"):
        Config.from_dict(bad)


def test_protocol_carries_seven_stands():
    angles = [(10.0 + 20 * i, 80.0 + i) for i in range(7)]
    _, _, out = protocol.decode(protocol.encode_set(5, angles))
    assert np.allclose(out, angles)
    with pytest.raises(ValueError):
        protocol.encode_set(0, [(90, 90)] * 9)


def _arrivals(h, routing, drivers, listener, fs=FS):
    """Arrival time (s) of each input channel's impulse at its aim point."""
    out = {}
    for r, p in zip(routing.routes, drivers):
        aim = listener.t if r.ear == "center" else (listener.ear_l if r.ear == "left" else listener.ear_r)
        g = h[r.output, r.input]
        k = np.argmax(np.abs(np.fft.irfft(np.fft.rfft(g), 16 * len(g)))) / 16 / fs
        out[r.input] = (k + np.linalg.norm(p - aim) / 1000 / C_SOUND, np.linalg.norm(g) / (np.linalg.norm(p - aim) / 1000))
    return out


@pytest.mark.parametrize("layout", ["5.1", "7.1"])
def test_align_brings_every_speaker_in_together(layout):
    cfg = Config.preset(layout)
    routing = build_routing(cfg)
    assert (routing.n_in, routing.n_out) == (len(CHANNELS[layout]),) * 2
    L = make_listener(rotation_from_ypr(-20), np.array([-170, 30, 760.0]), M, "pnp")   # off-seat
    ang, drivers = geometry.solve(L, cfg.stands)
    acfg = cfg.audio
    h = A.design_multi(L, drivers, routing, acfg)
    arr = _arrivals(h, routing, drivers, L)
    times = [t for t, _ in arr.values()]
    levels = [g for _, g in arr.values()]
    assert max(times) - min(times) < 1 / FS                        # within one sample
    assert max(levels) / min(levels) == pytest.approx(1, abs=0.03)    # 1/r compensated
    # LFE -> sub only, time-aligned to the head centre
    sub, lfe = routing.sub_output, routing.lfe
    assert np.abs(h[:, lfe]).sum(axis=-1).nonzero()[0].tolist() == [sub]
    g = h[sub, lfe]
    k = np.argmax(np.abs(np.fft.irfft(np.fft.rfft(g), 16 * len(g)))) / 16 / FS
    t_sub = k + np.linalg.norm(routing.sub_position_mm - L.t) / 1000 / C_SOUND
    assert abs(t_sub - np.mean(times)) < 0.0003                    # ear vs centre path lengths differ slightly


def test_lfe_folded_into_front_pair_without_sub():
    cfg = Config.preset("5.1")
    cfg.audio.lfe_mode, cfg.audio.sub_output = "mains", -1
    routing = build_routing(cfg)
    L = make_listener(np.eye(3), np.array([0, 0, 900.0]), M, "pnp")
    _, drivers = geometry.solve(L, cfg.stands)
    h = A.design_multi(L, drivers, routing, cfg.audio)
    lfe = CHANNELS["5.1"].index("LFE")
    outs = np.abs(h[:, lfe]).sum(axis=-1).nonzero()[0].tolist()
    by = {s.channel: s.output_channel for s in cfg.stands}
    assert sorted(outs) == sorted([by["FL"], by["FR"]])


def test_off_mode_is_plain_routing():
    cfg = Config.preset("7.1")
    cfg.audio.mode = "off"
    routing = build_routing(cfg)
    L = make_listener(np.eye(3), np.array([100, 0, 800.0]), M, "pnp")
    _, drivers = geometry.solve(L, cfg.stands)
    h = A.design_multi(L, drivers, routing, cfg.audio)
    for r in routing.routes:
        assert np.argmax(np.abs(h[r.output, r.input])) == cfg.audio.taps // 2


def test_renderer_n_by_m_matches_convolution():
    rng = np.random.default_rng(0)
    taps, block, n_in, n_out = 128, 64, 8, 8
    r = A.Renderer(taps, block, n_in, n_out)
    h = rng.normal(0, 0.1, (n_out, n_in, taps))
    r.set_filters(h)
    r.Cf, r._pending = r._pending, None
    x = rng.normal(0, 1, (block * 12, n_in))
    y = np.concatenate([r.process(x[i:i + block]) for i in range(0, len(x), block)])
    ref = np.stack([sum(np.convolve(x[:, i], h[o, i])[:len(x)] for i in range(n_in)) for o in range(n_out)], 1)
    assert np.allclose(y, ref, atol=1e-9)


def test_stereo_source_into_71_plays_from_front_pair():
    x = np.arange(12, dtype=float).reshape(6, 2)
    y = A.fit_channels(x, 8)
    assert y.shape == (6, 8) and np.allclose(y[:, :2], x) and not y[:, 2:].any()
    assert A.fit_channels(np.ones((4, 1)), 6)[:, :2].all()


def test_pipeline_drives_seven_stands_and_audio_routing():
    cfg = Config.preset("7.1")
    K = geometry.Intrinsics(554.0, 554.0, 320.0, 240.0)
    from orpheus.enroll import project
    R, t = rotation_from_ypr(10), np.array([60.0, 0, 900])
    pts = {k: project(R @ v + t, K) for k, v in M.points().items()}
    from orpheus.landmarks import Face
    link = DryRunLink()
    updates = []

    class Audio:
        def override_mode(self, m):
            pass

        def update(self, L, drivers):
            updates.append(len(drivers))

    pipe = Pipeline(cfg, link, backend=lambda f, tt: Face((0, 0, 1, 1), pts, "syn"), audio=Audio(),
                    intrinsics=K)
    st = pipe.step(np.zeros((480, 640, 3), np.uint8), 0.0)
    assert len(st.angles) == 7 and len(link.last) == 7 and updates == [7]
    assert set(st.distances) == set(speaker_channels("7.1"))
    assert build_routing(Config()) is None                         # plain stereo keeps the 2x2 path


def test_move_stand_refaces_seat_and_centre_follows_listener():
    from orpheus.layouts import move_stand
    cfg = Config.preset("5.1", seat_mm=900)
    move_stand(cfg, "FC", 300, 120, 200, seat_mm=(0, 0, 900))
    fc = next(s for s in cfg.stands if s.channel == "FC")
    assert fc.position_mm == (300.0, -200.0, 120.0)
    pan0, _ = geometry.stand_angles(np.array([0, 0, 900.0]), fc)
    assert abs(pan0) < 0.1                               # zero pan still faces the seat
    pans = []
    for x in (-150.0, 0.0, 150.0):                      # the centre speaker follows the head
        L = make_listener(np.eye(3), np.array([x, 0, 900.0]), M, "pnp")
        pans.append(geometry.solve(L, [fc])[0][0][0])
    assert pans[0] < pans[1] < pans[2] and pans[2] - pans[0] > 15
    with pytest.raises(ValueError, match="no stand"):
        move_stand(cfg, "XX", 0, 0, 0)
