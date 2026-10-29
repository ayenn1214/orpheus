"""Channel layouts: stereo, 5.1 and 7.1.

Input channel order follows WAV / SMPTE / FFmpeg, so a 5.1 or 7.1 file or
loopback device plays without remapping:

    stereo  FL FR
    5.1     FL FR FC LFE SL SR        (channels 5-6 feed the surrounds whether the
                                       source calls them back or side)
    7.1     FL FR FC LFE BL BR SL SR

Every channel except LFE is a speaker on a pan/tilt stand. LFE goes to a fixed
subwoofer (time-aligned like the rest), or is folded into the front pair, or
dropped.

Placement presets follow ITU-R BS.775 / Dolby angles around the listening
seat (0 deg = straight ahead, i.e. toward the camera; + = listener's left):

    FL/FR +-30   FC 0   5.1 surrounds +-110   7.1 sides +-90, backs +-150
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

CHANNELS = {
    "stereo": ("FL", "FR"),
    "5.1": ("FL", "FR", "FC", "LFE", "SL", "SR"),
    "7.1": ("FL", "FR", "FC", "LFE", "BL", "BR", "SL", "SR"),
}
LAYOUTS = tuple(CHANNELS)

# angle around the seat (deg, + = listener's left) and which ear the stand serves
PLACEMENT = {
    "FL": (30, "left"), "FR": (-30, "right"), "FC": (0, "center"),
    "SL": (110, "left"), "SR": (-110, "right"),
}
PLACEMENT_71 = {**PLACEMENT, "SL": (90, "left"), "SR": (-90, "right"),
                "BL": (150, "left"), "BR": (-150, "right")}

# Servo pins per stand on a Mega 2560 - must match PINS in the firmware
PINS = ((6, 7), (8, 9), (10, 11), (12, 13), (22, 23), (24, 25), (26, 27), (28, 29))
MAX_STANDS = len(PINS)


def speaker_channels(layout: str) -> tuple[str, ...]:
    """Channels that get a stand (everything but LFE)."""
    return tuple(c for c in CHANNELS[layout] if c != "LFE")


def placement(layout: str, channel: str, seat_mm: float, radius_mm: float | None = None,
              center_height_mm: float = 150.0):
    """(position_mm, yaw_offset_deg, ear) for one channel's stand.
    Camera frame: +X camera right (= listener's left as they face it), +Y down, +Z away
    from the camera. The seat is at (0, 0, seat_mm); speakers sit on a circle around it.
    The centre speaker would land on the camera, so it sits center_height_mm below it.
    yaw_offset points the stand's zero direction at the seat."""
    table = PLACEMENT_71 if layout == "7.1" else PLACEMENT
    ang, ear = table[channel]
    r = radius_mm or seat_mm
    a = math.radians(ang)
    x, z = r * math.sin(a), seat_mm - r * math.cos(a)
    y = center_height_mm if channel == "FC" else 0.0
    if channel == "FC":
        z = max(z, 120.0)                              # in front of the screen, below the camera
    yaw = math.degrees(math.atan2(-x, seat_mm - z))    # facing the seat
    return (round(x, 1), round(y, 1), round(z, 1)), round(yaw, 1), ear


# ------------------------------------------------------------------ routing
@dataclass
class Route:
    input: int          # index into the layout's channels
    output: int         # audio device output channel
    ear: str            # "left", "right" or "center" - where the stand aims


@dataclass
class Routing:
    layout: str
    n_in: int
    n_out: int
    routes: list[Route]                 # one per stand, in stand order
    lfe: int = -1                       # LFE input index, -1 = none in this layout
    lfe_mode: str = "drop"              # "sub", "mains", "drop"
    sub_output: int = -1
    sub_position_mm: np.ndarray | None = None
    lfe_gain: float = 1.0
    front_inputs: tuple[int, ...] = field(default_factory=tuple)   # FL, FR for lfe_mode mains


def build_routing(cfg) -> Routing | None:
    """None for plain stereo on two stands (the 2x2 xtc/virtual path handles it)."""
    a = cfg.audio
    layout = a.layout
    if layout == "stereo" and len(cfg.stands) == 2 and a.sub_output < 0:
        return None
    names = CHANNELS[layout]
    routes = []
    for i, s in enumerate(cfg.stands):
        ch = stand_channel(cfg, i)
        out = s.output_channel if s.output_channel >= 0 else i
        routes.append(Route(input=names.index(ch), output=out, ear=stand_ear_name(s, ch)))
    lfe = names.index("LFE") if "LFE" in names else -1
    lfe_mode = a.lfe_mode if lfe >= 0 else "drop"
    n_out = max([r.output for r in routes] + [a.sub_output]) + 1
    return Routing(layout=layout, n_in=len(names), n_out=n_out, routes=routes, lfe=lfe,
                   lfe_mode=lfe_mode, sub_output=a.sub_output,
                   sub_position_mm=np.asarray(a.sub_position_mm, float) if a.sub_output >= 0 else None,
                   lfe_gain=10 ** (a.lfe_gain_db / 20),
                   front_inputs=tuple(names.index(c) for c in ("FL", "FR")))


def stand_channel(cfg, i: int) -> str:
    """A stand's channel; blank = by order of the layout's speaker channels, except plain
    stereo where it follows the stand's side (left stand plays FL)."""
    s = cfg.stands[i]
    if s.channel:
        return s.channel
    layout = cfg.audio.layout
    if layout == "stereo":
        ear = s.ear if s.ear in ("left", "right") else ("left" if s.position_mm[0] >= 0 else "right")
        return "FL" if ear == "left" else "FR"
    chans = speaker_channels(layout)
    return chans[i] if i < len(chans) else ""


def stand_ear_name(stand, channel: str) -> str:
    if stand.ear in ("left", "right", "center"):
        return stand.ear
    if channel == "FC":
        return "center"
    return "left" if stand.position_mm[0] >= 0 else "right"


def face_point(position_mm, target_mm) -> float:
    """yaw_offset_deg that points a stand's zero direction at a target (e.g. the seat)."""
    p, t = np.asarray(position_mm, float), np.asarray(target_mm, float)
    return round(math.degrees(math.atan2(t[0] - p[0], t[2] - p[2])), 1)


def move_stand(cfg, name: str, left_mm: float, from_camera_mm: float, height_mm: float,
               seat_mm=None, face_seat: bool = True) -> None:
    """Place a stand by listener-friendly numbers: left (+ = listener's left = camera
    +X), distance from the camera, height above the speaker axis (+ up)."""
    for s in cfg.stands:
        if s.name == name or s.channel == name:
            s.position_mm = (float(left_mm), -float(height_mm), float(from_camera_mm))
            if face_seat and seat_mm is not None:
                s.yaw_offset_deg = face_point(s.position_mm, seat_mm)
            return
    raise ValueError(f"no stand named {name!r}; stands: {[s.name for s in cfg.stands]}")
