"""Head-related transfer functions for a tracked listener.

Two models, same interface - `response(dirs_head, fs, nfft) -> (H_left, H_right)`:

SphericalHead  Brown & Duda (1998) structural model: a rigid sphere gives the
               head-shadow filter (one pole / one zero, incidence dependent)
               and the Woodworth ITD. No data needed; radius comes from the
               listener's anthropometry (head.HeadModel.sphere_radius_mm).
               No pinna, so no elevation cues - fine for speakers near ear level.

SofaHRTF       Measured HRIRs from an AES69 SOFA file (SimpleFreeFieldHRIR),
               e.g. MIT KEMAR, CIPIC, SADIE, or your own measurement. Includes
               pinna and torso. Delays are separated from the responses so that
               (a) neighbouring directions interpolate cleanly and (b) the ITD
               can be rescaled to the listener's head radius - the 2017
               report's 'ITD scaling factor' (Sec 2.1.3) applied to real data.

Responses are relative to a plane wave arriving at the head centre with the
head absent; propagation from the speaker is added by the caller.
Directions are unit vectors in the head frame (see head.py: +X left ear,
+Y down, +Z back of head, nose along -Z).
"""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np

C_SOUND = 343.0  # m/s at 20 C

EAR_AXES = {"left": np.array([1.0, 0.0, 0.0]), "right": np.array([-1.0, 0.0, 0.0])}


def _unit(v):
    v = np.atleast_2d(np.asarray(v, dtype=float))
    return v / np.linalg.norm(v, axis=1, keepdims=True)


def woodworth_itd(azimuth_deg: float, radius_m: float, c: float = C_SOUND) -> float:
    """Classic far-field ITD for a lateral angle (0 = front, 90 = side)."""
    th = math.radians(abs(azimuth_deg))
    return radius_m / c * (th + math.sin(th))


# ------------------------------------------------------------- spherical head
class SphericalHead:
    def __init__(self, radius_mm: float = 87.5, alpha_min: float = 0.1,
                 theta_min_deg: float = 150.0, c: float = C_SOUND) -> None:
        self.a = radius_mm / 1000.0
        self.alpha_min = alpha_min
        self.theta_min = math.radians(theta_min_deg)
        self.c = c

    def ear_response(self, cos_inc: np.ndarray, freqs: np.ndarray) -> np.ndarray:
        """cos_inc: cosine of the angle between source direction and ear axis, shape (D,)."""
        th = np.arccos(np.clip(cos_inc, -1.0, 1.0))[:, None]          # (D,1)
        w = 2 * np.pi * np.asarray(freqs)[None, :]                     # (1,F)
        w0 = self.c / self.a
        am = self.alpha_min
        alpha = (1 + am / 2) + (1 - am / 2) * np.cos(th / self.theta_min * np.pi)
        shadow = (1 + 1j * alpha * w / (2 * w0)) / (1 + 1j * w / (2 * w0))
        # Woodworth path: straight to the ear on the lit side, around the sphere behind it
        T = np.where(th < np.pi / 2, -self.a / self.c * np.cos(th),
                     self.a / self.c * (th - np.pi / 2))
        return shadow * np.exp(-1j * w * T)

    def response(self, dirs_head, fs: float, nfft: int):
        d = _unit(dirs_head)
        f = np.fft.rfftfreq(nfft, 1 / fs)
        return (self.ear_response(d @ EAR_AXES["left"], f),
                self.ear_response(d @ EAR_AXES["right"], f))


# --------------------------------------------------------------------- SOFA
def head_to_sofa(d: np.ndarray) -> np.ndarray:
    """Head frame -> SOFA listener frame (+x front, +y left, +z up)."""
    d = np.atleast_2d(d)
    return np.stack([-d[:, 2], d[:, 0], -d[:, 1]], axis=1)


def sofa_to_head(s: np.ndarray) -> np.ndarray:
    s = np.atleast_2d(s)
    return np.stack([s[:, 1], -s[:, 2], -s[:, 0]], axis=1)


def _onset(ir: np.ndarray, up: int = 8, thresh: float = 0.15) -> float:
    """Onset in samples: first crossing of thresh*peak on an 8x upsampled |IR|."""
    from scipy.signal import resample_poly

    x = np.abs(resample_poly(ir, up, 1))
    return float(np.argmax(x >= thresh * x.max())) / up


class SofaHRTF:
    def __init__(self, path: str | Path, radius_mm: float | None = None,
                 dataset_radius_mm: float | None = None, k_neighbours: int = 3) -> None:
        import h5py

        with h5py.File(path, "r") as f:
            conv = f.attrs.get("SOFAConventions", b"")
            conv = conv.decode() if isinstance(conv, bytes) else str(conv)
            if conv and conv != "SimpleFreeFieldHRIR":
                raise ValueError(f"{path}: need SimpleFreeFieldHRIR, got {conv}")
            ir = np.asarray(f["Data.IR"], dtype=float)                 # (M, R=2, N)
            self.fs = float(np.asarray(f["Data.SamplingRate"]).ravel()[0])
            pos = np.asarray(f["SourcePosition"], dtype=float)
            ptype = f["SourcePosition"].attrs.get("Type", b"spherical")
            ptype = ptype.decode() if isinstance(ptype, bytes) else str(ptype)
        if ir.ndim != 3 or ir.shape[1] != 2:
            raise ValueError(f"{path}: Data.IR must be (M, 2, N), got {ir.shape}")
        if pos.shape[0] == 1 and ir.shape[0] > 1:
            pos = np.repeat(pos, ir.shape[0], axis=0)

        if ptype.lower().startswith("cart"):
            xyz = pos[:, :3]
        else:  # azimuth (deg, CCW from front), elevation (deg), distance
            az, el = np.radians(pos[:, 0]), np.radians(pos[:, 1])
            xyz = np.stack([np.cos(el) * np.cos(az), np.cos(el) * np.sin(az), np.sin(el)], 1)
        self.dirs_head = _unit(sofa_to_head(xyz))
        self.ir = ir
        self.k = k_neighbours

        # Split each HRIR into onset delay + delay-free remainder
        on = np.array([[_onset(ir[m, e]) for e in range(2)] for m in range(ir.shape[0])])
        self.onset = on                                                    # samples
        self.tau_rel = (on - on.mean(axis=1, keepdims=True)) / self.fs     # s, per ear
        self.bulk = on.mean(axis=1) / self.fs                              # s

        itd_max = float(np.max(np.abs(self.tau_rel[:, 0] - self.tau_rel[:, 1])))
        est_a = itd_max * C_SOUND / (1 + math.pi / 2) * 1000.0
        self.dataset_radius_mm = dataset_radius_mm or est_a
        self.itd_scale = (radius_mm / self.dataset_radius_mm) if radius_mm else 1.0
        self._cache: dict[tuple[float, int], np.ndarray] = {}

    def _spectra(self, fs: float, nfft: int) -> np.ndarray:
        """Delay-free spectra (M, 2, F) on the requested grid."""
        key = (fs, nfft)
        if key not in self._cache:
            from fractions import Fraction

            from scipy.signal import resample_poly

            ir = self.ir
            if abs(fs - self.fs) > 1e-6:
                fr = Fraction(fs / self.fs).limit_denominator(1000)
                ir = resample_poly(ir, fr.numerator, fr.denominator, axis=-1)
            if ir.shape[-1] > nfft:
                raise ValueError(f"HRIR length {ir.shape[-1]} exceeds nfft {nfft}")
            H = np.fft.rfft(ir, n=nfft, axis=-1)
            f = np.fft.rfftfreq(nfft, 1 / fs)
            delay = self.onset / self.fs                                   # (M,2) s
            self._cache[key] = H * np.exp(2j * np.pi * f[None, None, :] * delay[:, :, None])
        return self._cache[key]

    def response(self, dirs_head, fs: float, nfft: int):
        d = _unit(dirs_head)
        S = self._spectra(fs, nfft)
        f = np.fft.rfftfreq(nfft, 1 / fs)
        cosang = np.clip(d @ self.dirs_head.T, -1, 1)                       # (D, M)
        idx = np.argsort(-cosang, axis=1)[:, : self.k]
        ang = np.arccos(np.take_along_axis(cosang, idx, axis=1))
        w = 1.0 / np.maximum(ang, 1e-4)
        w /= w.sum(axis=1, keepdims=True)                                   # (D, k)
        Hf = np.einsum("dk,dkef->def", w, S[idx])                          # (D, 2, F)
        tau = np.einsum("dk,dke->de", w, self.tau_rel[idx]) * self.itd_scale
        Hd = Hf * np.exp(-2j * np.pi * f[None, None, :] * tau[:, :, None])
        return Hd[:, 0], Hd[:, 1]


def load(spec: str, radius_mm: float):
    """'spherical' or a path to a .sofa file."""
    if spec == "spherical":
        return SphericalHead(radius_mm)
    return SofaHRTF(spec, radius_mm=radius_mm)
