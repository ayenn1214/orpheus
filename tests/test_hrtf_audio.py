import math
import threading
import time

import numpy as np
import pytest

from orpheus import audio as A
from orpheus import hrtf as Hm
from orpheus.geometry import Intrinsics
from orpheus.head import (HeadModel, PoseSolver, make_listener, pose_from_box,
                                  rotation_from_ypr, ypr_from_rotation)

FS, NFFT = 48000, 2048
K = Intrinsics(554.0, 554.0, 320.0, 240.0)
M = HeadModel()
SPK = np.array([[-300.0, 0, 0], [300.0, 0, 0]])


def listener(x=0.0, z=800.0, yaw=0.0, model=M):
    return make_listener(rotation_from_ypr(yaw), np.array([x, 0.0, z]), model, "pnp")


# ------------------------------------------------------------------- head
@pytest.mark.parametrize("ypr", [(0, 0, 0), (35, -10, 5), (-60, 15, -8), (10, 25, 12)])
def test_ypr_roundtrip(ypr):
    assert np.allclose(ypr_from_rotation(rotation_from_ypr(*ypr)), ypr, atol=1e-9)


def test_pnp_recovers_pose_and_ears():
    rng = np.random.default_rng(1)
    errs, yaw_errs = [], []
    for _ in range(40):
        ypr = (rng.uniform(-60, 60), rng.uniform(-20, 20), rng.uniform(-10, 10))
        R = rotation_from_ypr(*ypr)
        t = np.array([rng.uniform(-200, 200), rng.uniform(-100, 100), rng.uniform(500, 1200)])
        pts = {}
        for k, v in M.points().items():
            p = R @ v + t
            pts[k] = (K.fx * p[0] / p[2] + K.cx + rng.normal(0, 0.5),
                      K.fy * p[1] / p[2] + K.cy + rng.normal(0, 0.5))
        Rs, ts = PoseSolver(K, M).solve(pts)
        L = make_listener(Rs, ts, M, "pnp")
        el, er = M.ears()
        errs += [np.linalg.norm(L.ear_l - (R @ el + t)), np.linalg.norm(L.ear_r - (R @ er + t))]
        yaw_errs.append(abs(L.ypr[0] - ypr[0]))
    assert np.median(errs) < 8.0 and np.percentile(errs, 95) < 25.0
    assert np.median(yaw_errs) < 2.0


def test_box_fallback_roll_and_profile():
    R, t = pose_from_box((270, 190, 100, 100), K, 150.0, M, eyes=((295, 220), (345, 230)))
    assert ypr_from_rotation(R)[2] == pytest.approx(math.degrees(math.atan2(10, 50)), abs=1e-6)
    R, _ = pose_from_box((270, 190, 100, 100), K, 150.0, M, profile_sign=-1)
    assert ypr_from_rotation(R)[0] == pytest.approx(-70)


def test_scale_personalises_radius():
    assert M.sphere_radius_mm() == pytest.approx(90.5, abs=0.2)
    big = HeadModel(scale=1.1)
    assert big.ears()[0][0] == pytest.approx(1.1 * 72.5)
    assert big.sphere_radius_mm() > M.sphere_radius_mm()


# ------------------------------------------------------------------- hrtf
def _gd(H, f):
    return -np.gradient(np.unwrap(np.angle(H)), 2 * np.pi * f)


@pytest.mark.parametrize("az", [30, 60, 90])
def test_spherical_itd_limits(az):
    h = Hm.SphericalHead(87.5)
    f = np.fft.rfftfreq(4096, 1 / FS)
    a = math.radians(az)
    L, R = h.response(np.array([math.sin(a), 0, -math.cos(a)]), FS, 4096)
    itd = _gd(R[0], f) - _gd(L[0], f)
    lf = np.median(itd[(f > 50) & (f < 300)])
    hf = np.median(itd[(f > 3000) & (f < 8000)])
    kuhn = 3 * 0.0875 / Hm.C_SOUND * math.sin(a)
    assert lf == pytest.approx(kuhn, rel=0.15)                        # low-frequency limit
    assert hf == pytest.approx(Hm.woodworth_itd(az, 0.0875), rel=0.05)  # high-frequency limit


def test_spherical_ild_and_symmetry():
    h = Hm.SphericalHead(87.5)
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    k8 = np.argmin(abs(f - 8000))
    L, R = h.response(np.array([1.0, 0, 0]), FS, NFFT)        # source at the left ear
    assert 20 * np.log10(abs(L[0, k8]) / abs(R[0, k8])) > 10
    L0, R0 = h.response(np.array([0, 0, -1.0]), FS, NFFT)     # straight ahead
    assert np.allclose(L0, R0)
    Lm, Rm = h.response(np.array([-1.0, 0, 0]), FS, NFFT)     # mirror source
    assert np.allclose(Lm, R) and np.allclose(Rm, L)


def _write_sofa(path, model, fs=44100, n=256, bulk_s=0.002):
    import h5py

    az = np.arange(0, 360, 15.0)
    el = np.array([-30.0, 0.0, 30.0])
    grid = np.array([(a, e) for e in el for a in az])
    xyz = np.stack([np.cos(np.radians(grid[:, 1])) * np.cos(np.radians(grid[:, 0])),
                    np.cos(np.radians(grid[:, 1])) * np.sin(np.radians(grid[:, 0])),
                    np.sin(np.radians(grid[:, 1]))], 1)
    L, R = model.response(Hm.sofa_to_head(xyz), fs, 4 * n)
    f = np.fft.rfftfreq(4 * n, 1 / fs)
    bulk = np.exp(-2j * np.pi * f * bulk_s)
    ir = np.stack([np.fft.irfft(L * bulk, 4 * n)[:, :n], np.fft.irfft(R * bulk, 4 * n)[:, :n]], 1)
    with h5py.File(path, "w") as h:
        h.attrs["SOFAConventions"] = "SimpleFreeFieldHRIR"
        h["Data.IR"] = ir
        h["Data.SamplingRate"] = np.array([fs], float)
        sp = h.create_dataset("SourcePosition", data=np.c_[grid, np.ones(len(grid))])
        sp.attrs["Type"] = "spherical"
    return grid


def test_sofa_load_lookup_and_itd_scaling(tmp_path):
    truth = Hm.SphericalHead(87.5)
    p = tmp_path / "synthetic.sofa"
    _write_sofa(p, truth)
    sofa = Hm.SofaHRTF(p)
    assert sofa.dataset_radius_mm == pytest.approx(87.5, rel=0.06)

    d = Hm.sofa_to_head(np.array([[math.cos(math.radians(45)), math.sin(math.radians(45)), 0]]))
    f = np.fft.rfftfreq(NFFT, 1 / FS)
    band = (f > 300) & (f < 6000)
    Ls, Rs = sofa.response(d, FS, NFFT)
    Lt, Rt = truth.response(d, FS, NFFT)
    ild_s = 20 * np.log10(abs(Ls[0, band]) / abs(Rs[0, band]))
    ild_t = 20 * np.log10(abs(Lt[0, band]) / abs(Rt[0, band]))
    assert np.median(abs(ild_s - ild_t)) < 1.0
    itd = lambda L, R: np.median((_gd(R[0], f) - _gd(L[0], f))[(f > 1500) & (f < 5000)])
    assert itd(Ls, Rs) == pytest.approx(itd(Lt, Rt), rel=0.1)

    scaled = Hm.SofaHRTF(p, radius_mm=1.2 * sofa.dataset_radius_mm)
    Lk, Rk = scaled.response(d, FS, NFFT)
    assert itd(Lk, Rk) == pytest.approx(1.2 * itd(Ls, Rs), rel=0.05)


# ------------------------------------------------------------------ audio
def _sep(C_h, G):
    Cf = np.transpose(np.fft.rfft(C_h, axis=-1), (2, 0, 1))
    E = G @ Cf
    f = np.fft.rfftfreq(C_h.shape[-1], 1 / FS)
    k = (f >= 200) & (f <= 6000)
    return float(np.median(np.concatenate(
        [20 * np.log10(abs(E[k, i, i]) / abs(E[k, 1 - i, i])) for i in (0, 1)])))


def test_xtc_separation_requires_tracking():
    h = Hm.SphericalHead(M.sphere_radius_mm())
    cfg = A.AudioConfig(mode="xtc", taps=1024)
    base = listener()
    G = lambda L: A.plant(L, SPK, h, FS, 1024)
    assert _sep(A.design(base, SPK, h, cfg), G(base)) > 40
    moved = listener(x=40)
    untracked = _sep(A.design(base, SPK, h, cfg), G(moved))
    tracked = _sep(A.design(moved, SPK, h, cfg), G(moved))
    assert untracked < 10 < 40 < tracked


def test_virtual_mode_reproduces_virtual_speakers():
    h = Hm.SphericalHead(M.sphere_radius_mm())
    cfg = A.AudioConfig(mode="virtual", taps=1024, headroom_db=0)
    L = listener(x=120, yaw=15)
    C_h = A.design(L, SPK, h, cfg)
    E = A.plant(L, SPK, h, FS, 1024) @ np.transpose(np.fft.rfft(C_h, axis=-1), (2, 0, 1))
    vL, vR = h.response(A.virtual_directions(L, 30.0), FS, 1024)
    B = np.stack([vL.T, vR.T], axis=1)
    f = np.fft.rfftfreq(1024, 1 / FS)
    k = (f >= 300) & (f <= 5000)
    rel = np.abs(np.abs(E[k]) - np.abs(B[k])) / np.abs(B[k]).max(axis=(1, 2), keepdims=True)
    assert np.median(rel) < 0.05
    # input L comes from the listener's left (ILD favours the left ear)
    k4 = np.argmin(abs(f - 4000))
    assert abs(B[k4, 0, 0]) > abs(B[k4, 1, 0])


def test_align_to_ears_times_each_speaker_to_its_own_ear():
    cfg = A.AudioConfig(mode="align", taps=1024, align_to="ears")
    L = listener(x=-150, z=700, yaw=25)
    h = A.design(L, SPK, None, cfg)
    ear = [L.ear_r, L.ear_l]                     # left stand (-X) faces the right ear
    r = [np.linalg.norm(SPK[s] - ear[s]) / 1000 for s in range(2)]
    arrive = [np.argmax(np.abs(np.fft.irfft(np.fft.rfft(h[s, A.speaker_side(L, SPK[s])]), 8 * 1024))) / 8 / FS
              + r[s] / Hm.C_SOUND for s in range(2)]
    assert abs(arrive[0] - arrive[1]) < 1 / FS
    # speaker on camera -X is on the listener's right -> plays input R
    assert A.speaker_side(L, SPK[0]) == 1 and np.abs(h[0, 0]).max() == 0


def test_align_equalises_arrival_at_head_centre():
    cfg = A.AudioConfig(mode="align", taps=1024, align_to="center")
    L = listener(x=-150, z=700)
    h = A.design(L, SPK, None, cfg)
    r = np.linalg.norm(SPK - L.t, axis=1) / 1000
    arrive = [np.argmax(np.abs(np.fft.irfft(np.fft.rfft(h[s, A.speaker_side(L, SPK[s])]), 8 * 1024))) / 8 / FS
              + r[s] / Hm.C_SOUND for s in range(2)]
    assert abs(arrive[0] - arrive[1]) < 1 / FS                         # within one sample
    g = [np.linalg.norm(h[s, A.speaker_side(L, SPK[s])]) / r[s] for s in range(2)]          # fractional delay: use energy
    assert g[0] == pytest.approx(g[1], rel=0.05)                       # equal level at the head


def test_renderer_matches_direct_convolution_and_crossfades():
    rng = np.random.default_rng(0)
    taps, block = 256, 128
    r = A.Renderer(taps, block)
    h = rng.normal(0, 0.1, (2, 2, taps))
    r.set_filters(h)
    r.Cf, r._pending = r._pending, None                                  # start on h
    x = rng.normal(0, 1, (block * 20, 2))
    y = np.concatenate([r.process(x[i:i + block]) for i in range(0, len(x), block)])
    ref = np.stack([sum(np.convolve(x[:, i], h[s, i])[:len(x)] for i in range(2)) for s in range(2)], 1)
    assert np.allclose(y, ref, atol=1e-9)

    # filter swap: output stays between the two filters' outputs (no click)
    h2 = rng.normal(0, 0.1, (2, 2, taps))
    r.set_filters(h2)
    yb = r.process(x[:block])
    assert np.isfinite(yb).all() and np.abs(yb).max() < 10 * np.abs(y).max()


def test_end_to_end_time_domain_crosstalk():
    """Noise on input L only, through renderer + simulated room plant: right ear stays quiet."""
    h = Hm.SphericalHead(M.sphere_radius_mm())
    cfg = A.AudioConfig(mode="xtc", taps=1024, block=512)
    L = listener(x=60, yaw=-10)
    r = A.Renderer(cfg.taps, cfg.block)
    r.set_filters(A.design(L, SPK, h, cfg))
    rng = np.random.default_rng(3)
    n = cfg.block * 120
    x = np.zeros((n, 2))
    x[:, 0] = rng.normal(0, 0.1, n)
    spk = np.concatenate([r.process(x[i:i + cfg.block]) for i in range(0, n, cfg.block)])
    g = np.fft.irfft(np.transpose(A.plant(L, SPK, h, FS, 4096), (1, 2, 0)), axis=-1)  # (ear,spk,t)
    ears = np.stack([sum(np.convolve(spk[:, s], g[e, s])[:n] for s in range(2)) for e in range(2)], 1)
    from scipy.signal import butter, sosfilt
    sos = butter(4, [300, 5000], "bandpass", fs=FS, output="sos")
    e = sosfilt(sos, ears[n // 2:], axis=0)
    assert 10 * np.log10(np.mean(e[:, 1] ** 2) / np.mean(e[:, 0] ** 2)) < -25


def test_engine_designer_thread_picks_up_motion():
    eng = A.AudioEngine(A.AudioConfig(mode="xtc", redesign_hz=200), Hm.SphericalHead())
    eng._thread = threading.Thread(target=eng._designer, daemon=True)
    eng._thread.start()
    try:
        eng.update(listener(), SPK)
        time.sleep(0.1)
        assert eng.renderer._pending is not None
        eng.renderer._pending = None
        eng.update(listener(x=3), SPK)                 # below 10 mm threshold
        time.sleep(0.05)
        assert eng.renderer._pending is None
        eng.update(listener(x=30), SPK)
        time.sleep(0.1)
        assert eng.renderer._pending is not None
    finally:
        eng._stop.set()


# --------------------------------------------------------------- pipeline
def test_pipeline_tracks_turning_head_with_landmarks():
    from orpheus.config import Config
    from orpheus.landmarks import Face
    from orpheus.link import DryRunLink
    from orpheus.pipeline import Pipeline

    cfg = Config()
    for s in cfg.stands:
        s.aim = "near_ear"
    pipe = Pipeline(cfg, DryRunLink(), backend=lambda f, t: None)
    Kp = pipe.K
    rng = np.random.default_rng(7)

    def truth(t):
        return rotation_from_ypr(40 * math.sin(2 * math.pi * 0.25 * t)), \
            np.array([150 * math.sin(2 * math.pi * 0.1 * t), 0.0, 800.0])

    def backend(frame, t):
        R, tv = truth(t)
        pts = {}
        for k, v in cfg.head.points().items():
            p = R @ v + tv
            pts[k] = (Kp.fx * p[0] / p[2] + Kp.cx + rng.normal(0, 0.7),
                      Kp.fy * p[1] / p[2] + Kp.cy + rng.normal(0, 0.7))
        return Face(box=(0, 0, 1, 1), points=pts, source="synthetic")

    pipe.backend = backend
    frame = np.zeros((480, 640, 3), np.uint8)
    errs = []
    for i in range(300):                       # 10 s at 30 fps
        t = i / 30
        st = pipe.step(frame, t)
        assert st.listener.method == "pnp"
        if i > 30:
            R, tv = truth(t)
            el, er = cfg.head.ears()
            errs.append(np.linalg.norm(st.listener.ear_l - (R @ el + tv)))
            errs.append(np.linalg.norm(st.listener.ear_r - (R @ er + tv)))
    assert np.median(errs) < 15 and np.percentile(errs, 95) < 35
