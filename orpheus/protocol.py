"""Binary serial protocol shared with firmware/orpheus_stand/orpheus_stand.ino.

The 2017 link sent `char(centx)` - one unframed byte per axis, so values over
255 wrapped and a dropped byte swapped X and Y permanently. This frame is
synced, length-checked and CRC-protected.

    0xAA 0x55 | TYPE | SEQ | N | N x (pan u16, tilt u16) | CRC8
    angles: little-endian uint16, centidegrees (0..18000)
    CRC-8/SMBus (poly 0x07, init 0x00) over TYPE..payload
"""

from __future__ import annotations

import struct

SYNC = b"\xAA\x55"
TYPE_SET = 0x01     # move servos to the given angles
TYPE_HOME = 0x02    # move all servos to their centre (N = 0)
MAX_STANDS = 8                  # 7.1 needs 7 stands; must match the firmware


def crc8(data: bytes) -> int:
    crc = 0
    for b in data:
        crc ^= b
        for _ in range(8):
            crc = ((crc << 1) ^ 0x07) & 0xFF if crc & 0x80 else (crc << 1) & 0xFF
    return crc


def _cdeg(a: float) -> int:
    return int(round(min(max(a, 0.0), 180.0) * 100))


def encode_set(seq: int, angles: list[tuple[float, float]]) -> bytes:
    if not 1 <= len(angles) <= MAX_STANDS:
        raise ValueError(f"1-{MAX_STANDS} stands per frame")
    payload = b"".join(struct.pack("<HH", _cdeg(p), _cdeg(t)) for p, t in angles)
    body = bytes([TYPE_SET, seq & 0xFF, len(angles)]) + payload
    return SYNC + body + bytes([crc8(body)])


def encode_home(seq: int) -> bytes:
    body = bytes([TYPE_HOME, seq & 0xFF, 0])
    return SYNC + body + bytes([crc8(body)])


def decode(frame: bytes) -> tuple[int, int, list[tuple[float, float]]]:
    """Inverse of encode_*; used by tests and the firmware simulator."""
    if frame[:2] != SYNC:
        raise ValueError("bad sync")
    body, crc = frame[2:-1], frame[-1]
    if crc8(body) != crc:
        raise ValueError("bad crc")
    typ, seq, n = body[0], body[1], body[2]
    if len(body) != 3 + 4 * n:
        raise ValueError("bad length")
    angles = [
        (p / 100.0, t / 100.0)
        for p, t in (struct.unpack_from("<HH", body, 3 + 4 * i) for i in range(n))
    ]
    return typ, seq, angles
