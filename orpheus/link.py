"""Serial link to the Arduino, with a dry-run stand-in for bench work without hardware."""

from __future__ import annotations

import logging
import time

from . import protocol

log = logging.getLogger(__name__)


class DryRunLink:
    def __init__(self) -> None:
        self.seq = 0
        self.last: list[tuple[float, float]] | None = None

    def send(self, angles: list[tuple[float, float]]) -> None:
        protocol.encode_set(self.seq, angles)   # exercise the encoder
        self.seq = (self.seq + 1) & 0xFF
        self.last = angles

    def home(self) -> None:
        self.last = None

    def poll(self) -> list[str]:
        return []

    def close(self) -> None:
        pass


class SerialLink:
    def __init__(self, port: str, baud: int = 115200,
                 home_angles: list[tuple[float, float]] | None = None,
                 boot_wait_s: float = 2.0) -> None:
        import serial  # pyserial

        # Configured servo centres; without them HOME falls back to 90 deg on every axis.
        self.home_angles = home_angles
        self.ser = serial.Serial(port, baud, timeout=0, write_timeout=0.05)
        # Opening the port resets an Uno/Mega; wait for the bootloader to hand over.
        time.sleep(boot_wait_s)
        self.ser.reset_input_buffer()
        self.seq = 0
        self._rx = bytearray()

    def send(self, angles: list[tuple[float, float]]) -> None:
        self._write(protocol.encode_set(self.seq, angles))

    def home(self) -> None:
        if self.home_angles:
            self._write(protocol.encode_set(self.seq, self.home_angles))
        else:
            self._write(protocol.encode_home(self.seq))

    def _write(self, frame: bytes) -> None:
        import serial

        try:
            self.ser.write(frame)
        except serial.SerialTimeoutException:
            log.warning("serial write timed out; frame dropped")
        self.seq = (self.seq + 1) & 0xFF

    def poll(self) -> list[str]:
        """Return complete text lines the firmware has printed (status / errors)."""
        self._rx += self.ser.read(self.ser.in_waiting or 0)
        lines = []
        while b"\n" in self._rx:
            line, _, rest = self._rx.partition(b"\n")
            self._rx = bytearray(rest)
            lines.append(line.decode(errors="replace").strip())
        return lines

    def close(self) -> None:
        try:
            self.home()
            self.ser.flush()
        finally:
            self.ser.close()
