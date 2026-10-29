"""Listener-tracked loudspeaker rendering.

The stands point the speakers at the listener; this module corrects what
pointing alone cannot - arrival time, level, and the head's own filtering.
Three modes, each a 2x2 FIR matrix C (speaker <- input) re-designed whenever
the tracked head moves, and swapped glitch-free in the audio thread:

    align    Per-speaker delay + gain so both speakers arrive at the
             interaural centre together and equally loud. Plain stereo stays
             plain stereo; the sweet spot follows the listener.

    xtc      Crosstalk cancellation (transaural). The 2x2 acoustic plant
             G[ear, speaker] = HRTF(ear, speaker direction in head frame) x
             propagation is inverted with Tikhonov regularisation:
                 C = G^H (G G^H + beta I)^-1 e^{-j w tau_m}
             so input L reaches only the left ear and input R only the right.
             For binaural content (dummy-head recordings, game/VR audio).

    virtual  xtc preceded by binaural synthesis of a +-30 deg stereo triangle
             fixed in the room (pointing at the screen/camera): C = Cxtc B.
             Ordinary stereo gets a correct stereo image wherever the
             listener sits or turns.

Engine threading: the tracker thread calls `AudioEngine.update(listener,
speakers)`; a designer thread rebuilds filters (rate-limited, only on real
movement); the PortAudio callback picks up new filters and crossfades over
one block (overlap-save, so both filters see identical input history).
"""

from __future__ import annotations

import logging
import math
import threading
import time
from dataclasses import dataclass, replace

import numpy as np

from .hrtf import C_SOUND

log = logging.getLogger(__name__)

MODES = ("off", "align", "xtc", "virtual")


@dataclass
class AudioConfig:
    mode: str = "off"
    align_to: str = "ears"           # align mode: "ears" (each speaker -> its own-side ear) or "center"
    hrtf: str = "spherical"          # or path to a SimpleFreeFieldHRIR .sofa file
    fs: int = 48000
    block: int = 512
    taps: int = 1024                 # FIR length; latency = taps/2 samples
    beta: float = 0.005              # in-band regularisation (~ -23 dB)
    beta_edge: float = 0.5           # out-of-band regularisation
    band_hz: tuple[float, float] = (150.0, 8000.0)
    virtual_spread_deg: float = 30.0
    headroom_db: float = 12.0        # fixed trim for xtc/virtual (keeps level steady across redesigns)
    redesign_mm: float = 10.0        # redesign when the head moves this far
    redesign_deg: float = 2.0        # ...or turns this far
    redesign_hz: float = 20.0
    file: str = ""                   # play this WAV through the renderer
    loop: bool = True
    input_device: str = ""           # or route a loopback input (e.g. VB-Cable, BlackHole)
    output_device: str = ""
    layout: str = "stereo"           # "stereo", "5.1", "7.1" (layouts.py)
    lfe_mode: str = "drop"           # LFE to a "sub", folded into the front pair ("mains"), or "drop"
    lfe_gain_db: float = 0.0
    sub_output: int = -1             # subwoofer's device output channel
    sub_position_mm: tuple[float, float, float] = (0.0, 350.0, 300.0)   # for its time alignment


# ----------------------------------------------------------------- design
def _beta_curve(f: np.ndarray, cfg: AudioConfig) -> np.ndarray:
    lo, hi = cfg.band_hz
    fl = np.maximum(f, 1.0)
    out_lo = np.clip(np.log2(lo / fl), 0, None)          # octaves below band
    out_hi = np.clip(np.log2(fl / hi), 0, None)          # octaves above band
    ramp = 1 - np.exp(-(out_lo + out_hi) * 2.0)
    return cfg.beta + (cfg.beta_edge - cfg.beta) * ramp


def plant(listener, speakers_mm: np.ndarray, hrtf, fs: int, nfft: int,
          extra_delay_s: float | np.ndarray = 0.0) -> np.ndarray:
    """Acoustic plant G (F, ears=2, speakers=S): speaker signal -> ear pressure.
    Common propagation delay is removed (only relative delay matters)."""
    f = np.fft.rfftfreq(nfft, 1 / fs)
    rel = np.array([listener.to_head(p) for p in speakers_mm])        # (S,3) head frame
    r = np.linalg.norm(rel, axis=1) / 1000.0                          # m
    HL, HR = hrtf.response(rel, fs, nfft)                              # (S,F) each
    extra = np.broadcast_to(np.asarray(extra_delay_s, float), r.shape)
    delay = (r - r.min()) / C_SOUND + extra
    prop = (r.min() / r)[:, None] * np.exp(-2j * np.pi * f[None, :] * delay[:, None])
    G = np.stack([HL * prop, HR * prop], axis=0)                       # (2,S,F)
    return np.transpose(G, (2, 0, 1))


def speaker_side(listener, p) -> int:
    """0 if the speaker serves the listener's left ear, else 1. Same rule as
    geometry.stand_ear 'auto': camera +X is the listener's left as they face it."""
    return 0 if float(np.asarray(p, float)[0]) >= 0 else 1


def virtual_directions(listener, spread_deg: float) -> np.ndarray:
    """Room-fixed virtual speakers at +-spread from the listener toward the camera,
    returned as head-frame unit vectors (order: input L, input R)."""
    fwd = -listener.t.copy()
    fwd[1] = 0.0                                                       # horizontal
    fwd /= np.linalg.norm(fwd)
    out = []
    for sgn in (-1.0, 1.0):
        # Rotating fwd about camera +Y by a negative angle swings it toward camera +X,
        # which is the listener's left when they face the camera: sgn=-1 -> input L.
        a = math.radians(sgn * spread_deg)
        Ry = np.array([[math.cos(a), 0, math.sin(a)], [0, 1, 0], [-math.sin(a), 0, math.cos(a)]])
        d_cam = Ry @ fwd
        out.append(listener.R.T @ d_cam)
    return np.array(out)


def design(listener, speakers_mm, hrtf, cfg: AudioConfig) -> np.ndarray:
    """FIR matrix (speakers=2, inputs=2, taps)."""
    L, fs = cfg.taps, cfg.fs
    f = np.fft.rfftfreq(L, 1 / fs)
    tau_m = L // 2
    model_delay = np.exp(-2j * np.pi * f * tau_m / fs)

    if cfg.mode == "align":
        if cfg.align_to == "ears":
            # each speaker is timed to its own ear (seen or virtual), never the other one
            r = np.array([np.linalg.norm(np.asarray(p) - (listener.ear_l if speaker_side(listener, p) == 0
                                                         else listener.ear_r))
                          for p in speakers_mm]) / 1000.0
        else:
            r = np.linalg.norm(np.array([np.asarray(p) - listener.t for p in speakers_mm]),
                               axis=1) / 1000.0
        d = (r.max() - r) / C_SOUND                                       # extra delay per speaker
        g = r / r.max()                                                   # 1/r compensation
        Cf = np.zeros((len(f), 2, 2), complex)
        for s in range(2):
            # feed each speaker the channel for the listener's side it is on
            # (a listener facing the camera has their LEFT on camera +X)
            Cf[:, s, speaker_side(listener, speakers_mm[s])] = \
                g[s] * np.exp(-2j * np.pi * f * d[s]) * model_delay
    elif cfg.mode in ("xtc", "virtual"):
        G = plant(listener, speakers_mm, hrtf, fs, L)                     # (F,2,2)
        beta = _beta_curve(f, cfg)[:, None, None]
        Gh = np.conj(np.transpose(G, (0, 2, 1)))
        Cf = Gh @ np.linalg.inv(G @ Gh + beta * np.eye(2)[None])          # (F,S,E)
        if cfg.mode == "virtual":
            vL, vR = hrtf.response(virtual_directions(listener, cfg.virtual_spread_deg), fs, L)
            B = np.stack([vL.T, vR.T], axis=1)                             # (F, ears, inputs)
            Cf = Cf @ B
        Cf = Cf * model_delay[:, None, None] * 10 ** (-cfg.headroom_db / 20)
    else:
        Cf = np.zeros((len(f), 2, 2), complex)
        Cf[:, 0, 0] = Cf[:, 1, 1] = model_delay

    h = np.fft.irfft(np.transpose(Cf, (1, 2, 0)), n=L, axis=-1)         # (S,I,L)
    # taper both ends: the regularised inverse is two-sided around tau_m
    n = np.arange(L)
    edge = L // 8
    w = np.ones(L)
    w[:edge] = 0.5 - 0.5 * np.cos(np.pi * n[:edge] / edge)
    w[-edge:] = w[:edge][::-1]
    return h * w


def design_multi(listener, drivers_mm, routing, cfg: AudioConfig) -> np.ndarray:
    """FIR matrix (outputs, inputs, taps) for 5.1 / 7.1 / more than two stands.

    align: every speaker (and the subwoofer) is delayed and level-matched so all of
    them arrive at the listener together and equally loud - each stand to its own
    ear (align_to "ears") or to the head centre. off: plain routing."""
    L, fs = cfg.taps, cfg.fs
    f = np.fft.rfftfreq(L, 1 / fs)
    model = np.exp(-2j * np.pi * f * (L // 2) / fs)
    Cf = np.zeros((len(f), routing.n_out, routing.n_in), complex)

    def point(ear: str):
        if cfg.align_to != "ears" or ear == "center":
            return listener.t
        return listener.ear_l if ear == "left" else listener.ear_r

    paths = [(r.output, r.input, np.asarray(p, float), point(r.ear))
             for r, p in zip(routing.routes, drivers_mm)]
    lfe_paths = []
    if routing.lfe >= 0 and routing.lfe_mode == "sub" and routing.sub_output >= 0:
        lfe_paths.append((routing.sub_output, routing.lfe, routing.sub_position_mm, listener.t, routing.lfe_gain))
    elif routing.lfe >= 0 and routing.lfe_mode == "mains":
        for r, p in zip(routing.routes, drivers_mm):
            if r.input in routing.front_inputs:     # fold LFE into the front pair
                lfe_paths.append((r.output, routing.lfe, np.asarray(p, float), point(r.ear),
                                  routing.lfe_gain / 2))
    dists = [np.linalg.norm(p - q) / 1000 for _, _, p, q in paths] +             [np.linalg.norm(p - q) / 1000 for _, _, p, q, _ in lfe_paths]
    rmax = max(dists) if dists else 1.0
    align = cfg.mode == "align"

    def put(o, i, r, gain):
        if align:
            Cf[:, o, i] += gain * (r / rmax) * np.exp(-2j * np.pi * f * (rmax - r) / C_SOUND) * model
        else:
            Cf[:, o, i] += gain * model

    for (o, i, p, q), r in zip(paths, dists):
        put(o, i, r, 1.0)
    for (o, i, p, q, g), r in zip(lfe_paths, dists[len(paths):]):
        put(o, i, r, g)
    return _to_fir(Cf, L)


def _to_fir(Cf: np.ndarray, L: int) -> np.ndarray:
    h = np.fft.irfft(np.transpose(Cf, (1, 2, 0)), n=L, axis=-1)
    n = np.arange(L)
    edge = L // 8
    w = np.ones(L)
    w[:edge] = 0.5 - 0.5 * np.cos(np.pi * n[:edge] / edge)
    w[-edge:] = w[:edge][::-1]
    return h * w


# --------------------------------------------------------------- renderer
class Renderer:
    """Streaming N-in / M-out FIR matrix, overlap-save, crossfaded updates."""

    def __init__(self, taps: int, block: int, n_in: int = 2, n_out: int = 2) -> None:
        self.L, self.B = taps, block
        self.n_in, self.n_out = n_in, n_out
        self.N = 1 << int(math.ceil(math.log2(taps + block - 1)))
        self.hist = np.zeros((n_in, self.N))
        self.Cf = self._spectrum(self._identity())
        self._pending: np.ndarray | None = None
        self._fade = np.linspace(0.0, 1.0, block, endpoint=False)[None, :]

    def _identity(self) -> np.ndarray:
        h = np.zeros((self.n_out, self.n_in, self.L))
        for k in range(min(self.n_in, self.n_out)):
            h[k, k, self.L // 2] = 1.0
        return h

    def _spectrum(self, h: np.ndarray) -> np.ndarray:
        return np.fft.rfft(h, n=self.N, axis=-1)

    def set_filters(self, h: np.ndarray) -> None:
        """Thread-safe: picked up at the next block."""
        self._pending = self._spectrum(h)

    def process(self, x: np.ndarray) -> np.ndarray:
        """x: (frames, n_in) float -> (frames, n_out). frames must equal block."""
        B = self.B
        self.hist = np.roll(self.hist, -B, axis=1)
        self.hist[:, -B:] = x.T
        X = np.fft.rfft(self.hist, axis=-1)                              # (2,F)
        y = self._apply(self.Cf, X)
        pending, self._pending = self._pending, None
        if pending is not None:
            y_new = self._apply(pending, X)
            y = y * (1 - self._fade) + y_new * self._fade
            self.Cf = pending
        return y.T

    def _apply(self, Cf: np.ndarray, X: np.ndarray) -> np.ndarray:
        Y = np.einsum("sif,if->sf", Cf, X)
        return np.fft.irfft(Y, n=self.N, axis=-1)[:, -self.B:]


def soft_clip(y: np.ndarray, knee: float = 0.89) -> np.ndarray:
    """Transparent below the knee (-1 dBFS), tanh above."""
    a = np.abs(y)
    over = a > knee
    if not over.any():
        return y
    out = y.copy()
    out[over] = np.sign(y[over]) * (knee + (1 - knee) * np.tanh((a[over] - knee) / (1 - knee)))
    return out


# ----------------------------------------------------------------- engine
class AudioEngine:
    def __init__(self, cfg: AudioConfig, hrtf, routing=None) -> None:
        if cfg.mode not in MODES:
            raise ValueError(f"audio mode must be one of {MODES}")
        self.cfg = cfg
        self.hrtf = hrtf
        self.routing = routing                     # None: stereo 2x2 (align / xtc / virtual)
        self.n_in = routing.n_in if routing else 2
        self.n_out = routing.n_out if routing else 2
        self.renderer = Renderer(cfg.taps, cfg.block, self.n_in, self.n_out)
        self._latest = None
        self._designed = None
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._stream = None
        self._src: np.ndarray | None = None
        self._pos = 0
        self.design_ms = 0.0
        self.xruns = 0
        self._mode_override: str | None = None

    # ------------------------------------------------------- tracking side
    def update(self, listener, speakers_mm) -> None:
        with self._lock:
            self._latest = (listener, np.asarray(speakers_mm, float))

    def _needs_redesign(self, listener) -> bool:
        if self._designed is None:
            return True
        prev = self._designed
        moved = np.linalg.norm(listener.t - prev.t) >= self.cfg.redesign_mm
        dR = listener.R @ prev.R.T
        ang = math.degrees(math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1) / 2))))
        return moved or ang >= self.cfg.redesign_deg

    @property
    def mode(self) -> str:
        return self._mode_override or self.cfg.mode

    def effective_cfg(self) -> AudioConfig:
        return replace(self.cfg, mode=self.mode) if self._mode_override else self.cfg

    def override_mode(self, mode: str | None) -> None:
        """Temporarily render in another mode (e.g. align while two people listen:
        crosstalk cancellation is only correct for one head). None restores."""
        if mode == self._mode_override:
            return
        self._mode_override = mode
        self._designed = None                       # force a redesign
        log.info("audio mode -> %s", self.mode)

    def _designer(self) -> None:
        period = 1.0 / self.cfg.redesign_hz
        while not self._stop.is_set():
            with self._lock:
                job = self._latest
            if job is not None and self._needs_redesign(job[0]):
                t0 = time.perf_counter()
                if self.routing is None:
                    h = design(job[0], job[1], self.hrtf, self.effective_cfg())
                else:
                    h = design_multi(job[0], job[1], self.routing, self.effective_cfg())
                self.design_ms = (time.perf_counter() - t0) * 1000
                self.renderer.set_filters(h)
                self._designed = job[0]
            self._stop.wait(period)

    # ---------------------------------------------------------- audio side
    def _callback_out(self, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        x = self._next_file_block(frames)
        outdata[:] = soft_clip(self.renderer.process(x))

    def _callback_duplex(self, indata, outdata, frames, time_info, status) -> None:
        if status:
            self.xruns += 1
        outdata[:] = soft_clip(self.renderer.process(fit_channels(indata, self.n_in)))

    def _next_file_block(self, n: int) -> np.ndarray:
        src = self._src
        out = np.zeros((n, self.n_in))
        if src is None:
            return out
        got = 0
        while got < n:
            take = min(n - got, len(src) - self._pos)
            if take <= 0:
                if not self.cfg.loop:
                    break
                self._pos = 0
                continue
            out[got:got + take] = src[self._pos:self._pos + take]
            self._pos += take
            got += take
        return out

    def start(self) -> None:
        import sounddevice as sd

        self._thread = threading.Thread(target=self._designer, name="hrtf-designer", daemon=True)
        self._thread.start()
        kw = dict(samplerate=self.cfg.fs, blocksize=self.cfg.block, dtype="float32",
                  latency="low")
        if self.cfg.file:
            self._src = load_wav(self.cfg.file, self.cfg.fs, self.n_in)
            self._stream = sd.OutputStream(channels=self.n_out, callback=self._callback_out,
                                           device=self.cfg.output_device or None, **kw)
        elif self.cfg.input_device:
            self._stream = sd.Stream(channels=(self.n_in, self.n_out), callback=self._callback_duplex,
                                     device=(self.cfg.input_device, self.cfg.output_device or None),
                                     **kw)
        else:
            raise SystemExit("audio: set audio.file (WAV) or audio.input_device")
        self._stream.start()
        log.info("audio %s %s: %d in / %d out, %d Hz, block %d, %d taps (latency %.1f ms + device)",
                 self.cfg.layout, self.cfg.mode, self.n_in, self.n_out, self.cfg.fs, self.cfg.block, self.cfg.taps,
                 1000 * (self.cfg.taps / 2 + self.cfg.block) / self.cfg.fs)

    def stop(self) -> None:
        self._stop.set()
        if self._stream is not None:
            self._stream.stop()
            self._stream.close()


def fit_channels(x: np.ndarray, n: int) -> np.ndarray:
    """Match a block to the layout's input count: mono is copied to FL/FR, a stereo
    source in a 5.1/7.1 layout plays from the front pair, extra channels are dropped."""
    x = np.asarray(x, float)
    if x.shape[1] == n:
        return x
    out = np.zeros((x.shape[0], n))
    if x.shape[1] == 1:
        out[:, :min(2, n)] = x[:, :1]
    else:
        k = min(x.shape[1], n)
        out[:, :k] = x[:, :k]
    return out


def load_wav(path: str, fs: int, n_ch: int = 2) -> np.ndarray:
    """WAV/FLAC -> (n, n_ch) float at fs (channel order as the file: WAV/SMPTE)."""
    import soundfile as sf

    x, sr = sf.read(path, always_2d=True, dtype="float64")
    x = fit_channels(x, n_ch)
    if sr != fs:
        from fractions import Fraction

        from scipy.signal import resample_poly
        fr = Fraction(fs, sr).limit_denominator(1000)
        x = resample_poly(x, fr.numerator, fr.denominator, axis=0)
    return x
