"""Configuration: camera optics, stand geometry, servo calibration, serial link.

Coordinate frame (OpenCV camera convention), millimetres:
    +X  right (as seen by the camera)
    +Y  down
    +Z  forward, out of the lens toward the listener
"""

from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field, fields
from pathlib import Path

from .audio import MODES as AUDIO_MODES
from .audio import AudioConfig
from .earfind import EarFindConfig
from .head import HeadModel
from .layouts import CHANNELS, LAYOUTS, MAX_STANDS, PINS, placement, speaker_channels


@dataclass
class CameraConfig:
    index: int = 0
    width: int = 640
    height: int = 480
    hfov_deg: float = 62.0          # horizontal field of view; typical USB webcam 60-78 deg
    mirror: bool = False            # True if the driver delivers a mirrored image
    # from --calibrate-camera (checkerboard); when set these replace hfov_deg
    matrix: list | None = None      # 3x3 camera matrix at calib_size
    dist: list | None = None        # OpenCV distortion coefficients (k1 k2 p1 p2 k3)
    calib_size: list | None = None  # [width, height] the calibration was made at
    calib_rms_px: float | None = None


@dataclass
class DetectorConfig:
    scale_factor: float = 1.1
    min_neighbors: int = 5
    min_face_px: int = 60           # at full resolution
    detect_scale: float = 0.5       # downscale before Viola-Jones for speed
    use_profile: bool = True        # also run the profile cascade (both directions)
    backend: str = "haar"           # "haar" (no download), "yunet" or "mediapipe"
    model_path: str = ""            # .onnx (yunet) or .task (mediapipe)
    face_width_mm: float = 150.0    # real width of a Haar face box; set with --calibrate
    profile_yaw_deg: float = 70.0   # yaw assumed for a Haar profile detection
    lost_hold_s: float = 1.5        # CamShift bridges detection dropouts up to this long
    max_faces: int = 2              # faces returned per frame (two-listener mode)


@dataclass
class ServoConfig:
    pin: int
    center_deg: float = 90.0        # servo command that points the speaker at its zero angle
    sign: int = 1                   # +1 / -1 to match mounting direction
    min_deg: float = 0.0
    max_deg: float = 180.0


@dataclass
class StandConfig:
    name: str
    position_mm: tuple[float, float, float]   # pivot location in camera frame
    pan: ServoConfig
    tilt: ServoConfig
    yaw_offset_deg: float = 0.0     # toe-in of the stand's zero direction (+ = toward +X)
    aim: str = "head"               # "head" (interaural centre) or "near_ear" (this stand's ear)
    ear: str = "auto"               # which ear this stand serves: "left", "right", "center"
                                    # (aim at the head centre), or "auto": the centre channel
                                    # gets "center", others by side (camera +X = listener's left)
    channel: str = ""               # FL FR FC SL SR BL BR (see layouts.py); blank = by order
    output_channel: int = -1        # audio device output for this speaker; -1 = stand index
    driver_offset_mm: float = 60.0  # pivot -> tweeter acoustic centre, along the aim line
    marker_id: int = -1             # ArUco id stuck on this stand (--locate-stands); -1 = none
    marker_offset_mm: tuple[float, float, float] = (0.0, 0.0, 0.0)
                                    # marker centre -> pan/tilt pivot, in the marker's frame
                                    # (x right, y up, z out of the marker toward the camera)


@dataclass
class FilterConfig:
    min_cutoff_hz: float = 0.4      # One Euro filter: lower = smoother when still
    beta: float = 0.002             # higher = less lag when moving
    deadband_deg: float = 0.75      # skip servo updates smaller than this
    return_to_seat_s: float = 3.0   # listener lost this long -> aim at the stored seat (0 = hold)


@dataclass
class LinkConfig:
    port: str = ""                  # e.g. COM6 or /dev/ttyACM0; empty = dry run
    baud: int = 115200
    rate_hz: float = 30.0


@dataclass
class PredictConfig:
    """Aim ahead of the head's motion to cancel servo + tracking lag."""
    lead_ms: float = 100.0          # 0 = off
    vel_tau_s: float = 0.08         # smoothing of the velocity estimate
    max_shift_mm: float = 120.0     # cap on how far ahead the aim point may move


@dataclass
class ListenersConfig:
    mode: str = "single"            # "single", "main" (aim at the main listener), "midpoint"
    main: str = ""                  # profile name to treat as main when recognised
    hold_s: float = 0.6             # keep a listener this long after the face drops out
    xtc_with_two: bool = False      # False: xtc/virtual fall back to align with 2+ listeners


@dataclass
class FaceIdConfig:
    enabled: bool = False
    model_path: str = ""            # face_recognition_sface_2021dec.onnx (opencv_zoo)
    profiles_dir: str = "profiles"
    threshold: float = 0.363        # SFace cosine similarity for "same person"
    confirm_frames: int = 5         # consecutive matches before a profile is loaded
    interval_frames: int = 15       # re-check an identified listener this often


@dataclass
class MarkersConfig:
    dictionary: str = "DICT_4X4_50"
    size_mm: float = 100.0          # printed black-square edge length (measure it)


@dataclass
class DepthConfig:
    source: str = "none"            # "none", "realsense", "oakd", "stereo"
    stereo_calib: str = ""          # JSON from --calibrate-stereo
    right_index: int = 1            # second webcam (stereo)
    min_mm: float = 250.0
    max_mm: float = 4000.0
    window_px: int = 2              # median over (2w+1)^2 pixels per landmark


@dataclass
class Config:
    camera: CameraConfig = field(default_factory=CameraConfig)
    detector: DetectorConfig = field(default_factory=DetectorConfig)
    filter: FilterConfig = field(default_factory=FilterConfig)
    link: LinkConfig = field(default_factory=LinkConfig)
    head: HeadModel = field(default_factory=HeadModel)
    profile: str = ""               # listener profile JSON from --enroll (measured ears)
    earfind: EarFindConfig = field(default_factory=EarFindConfig)   # find the visible ear
    audio: AudioConfig = field(default_factory=AudioConfig)
    predict: PredictConfig = field(default_factory=PredictConfig)
    listeners: ListenersConfig = field(default_factory=ListenersConfig)
    faceid: FaceIdConfig = field(default_factory=FaceIdConfig)
    markers: MarkersConfig = field(default_factory=MarkersConfig)
    depth: DepthConfig = field(default_factory=DepthConfig)
    stands: list[StandConfig] = field(default_factory=lambda: [
        StandConfig(
            name="left",                       # listener's left: camera +X when they face it
            position_mm=(300.0, 0.0, 0.0),
            pan=ServoConfig(pin=6),
            tilt=ServoConfig(pin=7, center_deg=90.0),
        ),
        StandConfig(
            name="right",
            position_mm=(-300.0, 0.0, 0.0),
            pan=ServoConfig(pin=8),
            tilt=ServoConfig(pin=9, center_deg=90.0),
        ),
    ])

    @classmethod
    def preset(cls, layout: str, seat_mm: float = 900.0, radius_mm: float | None = None) -> "Config":
        """Stands placed for a layout around a seat seat_mm in front of the camera.
        Stereo keeps the classic pair beside the camera."""
        cfg = cls()
        cfg.audio.layout = layout
        if layout == "stereo":
            return cfg
        stands = []
        for i, ch in enumerate(speaker_channels(layout)):
            pos, yaw, ear = placement(layout, ch, seat_mm, radius_mm)
            pan, tilt = PINS[i]
            stands.append(StandConfig(name=ch, position_mm=pos, pan=ServoConfig(pin=pan),
                                      tilt=ServoConfig(pin=tilt), yaw_offset_deg=yaw, aim="near_ear",
                                      ear=ear, channel=ch, output_channel=CHANNELS[layout].index(ch)))
        cfg.stands = stands
        cfg.audio.mode = "align"
        # outputs in the same order as a 5.1/7.1 sound card (FL FR FC LFE ...): the sub sits on 3
        cfg.audio.lfe_mode = "sub"
        cfg.audio.sub_output = CHANNELS[layout].index("LFE")
        cfg.validate()
        return cfg

    @property
    def profile_name(self) -> str:
        return Path(self.profile).stem if self.profile else ""

    # ---------------------------------------------------------------- io
    def to_dict(self) -> dict:
        return asdict(self)

    def save(self, path: str | Path) -> None:
        Path(path).write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def from_dict(cls, d: dict) -> "Config":
        d = dict(d)
        if "head_width_mm" in d:                       # v0.1 key
            d.setdefault("head", {})["bitragion_mm"] = float(d.pop("head_width_mm"))
        cfg = _build(cls, d)
        cfg.validate()
        return cfg

    @classmethod
    def load(cls, path: str | Path) -> "Config":
        return cls.from_dict(json.loads(Path(path).read_text()))

    def validate(self) -> None:
        if not 1 <= len(self.stands) <= MAX_STANDS:
            raise ValueError(f"1-{MAX_STANDS} stands supported by the protocol/firmware")
        a = self.audio
        if a.layout not in LAYOUTS:
            raise ValueError(f"audio.layout must be one of {LAYOUTS}")
        if a.lfe_mode not in ("sub", "mains", "drop"):
            raise ValueError("audio.lfe_mode must be sub, mains or drop")
        if a.lfe_mode == "sub" and "LFE" in CHANNELS[a.layout] and a.sub_output < 0:
            raise ValueError("audio.lfe_mode 'sub' needs audio.sub_output (the subwoofer's output channel)")
        from .layouts import stand_channel
        chans = [stand_channel(self, i) for i in range(len(self.stands))]
        for s, ch in zip(self.stands, chans):
            if ch == "LFE":
                raise ValueError(f"stand {s.name}: LFE goes to a fixed subwoofer, not a stand")
            if ch not in speaker_channels(a.layout):
                raise ValueError(f"stand {s.name}: channel {ch or '(none)'} is not in layout {a.layout} "
                                 f"{speaker_channels(a.layout)} - more stands than the layout has speakers?")
        if len(set(chans)) != len(chans):
            raise ValueError(f"two stands play the same channel: {chans}")
        outs = [s.output_channel if s.output_channel >= 0 else i for i, s in enumerate(self.stands)]
        if a.sub_output >= 0:
            outs.append(a.sub_output)
        if len(set(outs)) != len(outs):
            raise ValueError(f"two speakers share an audio output channel: {outs}")
        if self.detector.backend not in ("haar", "yunet", "mediapipe"):
            raise ValueError("detector.backend must be haar, yunet or mediapipe")
        if self.audio.align_to not in ("ears", "center"):
            raise ValueError("audio.align_to must be 'ears' or 'center'")
        if self.audio.mode not in AUDIO_MODES:
            raise ValueError(f"audio.mode must be one of {AUDIO_MODES}")
        if self.audio.mode in ("xtc", "virtual") and (len(self.stands) != 2 or a.layout != "stereo"):
            raise ValueError("xtc/virtual audio needs the stereo layout on exactly 2 stands "
                             "(use align for 5.1/7.1)")
        if self.listeners.mode not in ("single", "main", "midpoint"):
            raise ValueError("listeners.mode must be single, main or midpoint")
        if self.depth.source not in ("none", "realsense", "oakd", "stereo"):
            raise ValueError("depth.source must be none, realsense, oakd or stereo")
        if self.depth.source == "stereo" and not self.depth.stereo_calib:
            raise ValueError("depth.source 'stereo' needs depth.stereo_calib (run --calibrate-stereo)")
        if self.predict.lead_ms < 0:
            raise ValueError("predict.lead_ms must be >= 0")
        ids = [s.marker_id for s in self.stands if s.marker_id >= 0]
        if len(ids) != len(set(ids)):
            raise ValueError("each stand needs its own marker_id")
        for s in self.stands:
            if s.ear not in ("auto", "left", "right", "center"):
                raise ValueError(f"stand {s.name}: ear must be auto, left, right or center")
            if s.aim not in ("head", "near_ear"):
                raise ValueError(f"stand {s.name}: aim must be 'head' or 'near_ear'")
            for sv in (s.pan, s.tilt):
                if sv.sign not in (1, -1):
                    raise ValueError(f"stand {s.name}: servo sign must be +1 or -1")
                if not 0 <= sv.min_deg < sv.max_deg <= 180:
                    raise ValueError(f"stand {s.name}: servo limits must satisfy 0 <= min < max <= 180")


def _build(tp, data):
    """Recursively build nested dataclasses from plain dicts/lists (unknown keys rejected)."""
    if not isinstance(data, dict):
        raise ValueError(f"{tp.__name__}: expected an object, got {type(data).__name__}")
    unknown = set(data) - {f.name for f in fields(tp)}
    if unknown:
        raise ValueError(f"{tp.__name__}: unknown keys {sorted(unknown)}")
    kw = {}
    for name, value in data.items():
        sub = _SUBTYPES.get((tp, name))
        if sub is not None and isinstance(value, list):
            kw[name] = [_build(sub, v) for v in value]
        elif sub is not None:
            kw[name] = _build(sub, value)
        elif name in _TUPLES and value is not None:
            kw[name] = tuple(value)
        else:
            kw[name] = value
    return tp(**kw)


_SUBTYPES = {
    (Config, "camera"): CameraConfig,
    (Config, "detector"): DetectorConfig,
    (Config, "filter"): FilterConfig,
    (Config, "link"): LinkConfig,
    (Config, "head"): HeadModel,
    (Config, "audio"): AudioConfig,
    (Config, "earfind"): EarFindConfig,
    (Config, "stands"): StandConfig,
    (Config, "predict"): PredictConfig,
    (Config, "listeners"): ListenersConfig,
    (Config, "faceid"): FaceIdConfig,
    (Config, "markers"): MarkersConfig,
    (Config, "depth"): DepthConfig,
    (StandConfig, "pan"): ServoConfig,
    (StandConfig, "tilt"): ServoConfig,
}
_TUPLES = {"position_mm", "band_hz", "ear_l_mm", "ear_r_mm", "marker_offset_mm", "sub_position_mm"}
