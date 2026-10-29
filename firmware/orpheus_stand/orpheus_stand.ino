// SPDX-License-Identifier: Apache-2.0
// Copyright 2026 Ankur Narula - part of Orpheus
// Speaker stand servo controller - receives absolute pan/tilt targets from
// orpheus (Python) and slews up to 4 stands (8 servos) toward them.
//
// Frame:  0xAA 0x55 | TYPE | SEQ | N | N x (pan u16 LE, tilt u16 LE) | CRC8
//         angles in centidegrees 0..18000, CRC-8/SMBus (poly 0x07) over TYPE..payload
// TYPE:   0x01 SET   0x02 HOME (all servos to 90 deg)
//
// Board:  Arduino Mega 2560 (up to 8 stands = 16 servos; an Uno handles <= 2 stands).
// Layouts: stereo 2 stands, 5.1 -> 5, 7.1 -> 7 (LFE is a fixed sub, no stand). Nothing to
//          set here: a stand's servos attach the first time a frame addresses it. Stand
//          order must match the config's stands list (--write-config --layout).
// Power:  servos on a separate 5-6 V supply sized ~1 A per SG90 under load
//         (7.1 = 14 servos: a 10-15 A supply, or one supply per pair of stands),
//         grounds tied to the Arduino. Do not run 4 servos off the USB 5 V pin.

#include <Servo.h>

// ---------------------------------------------------------------- config
const uint8_t MAX_STANDS = 8;
// {pan, tilt} per stand; same table as layouts.PINS in the Python package
const uint8_t PINS[MAX_STANDS][2] = {{6, 7}, {8, 9}, {10, 11}, {12, 13},
                                     {22, 23}, {24, 25}, {26, 27}, {28, 29}};

const uint32_t BAUD = 115200;
const uint16_t PULSE_MIN_US = 544;     // Servo library defaults; trim per servo if needed
const uint16_t PULSE_MAX_US = 2400;
const float    SLEW_DEG_PER_S = 120.0; // SG90 no-load ~600 deg/s; lower = smoother with a speaker on it
const uint16_t UPDATE_MS = 10;         // 100 Hz slew loop
const uint32_t STATUS_MS = 2000;

// ----------------------------------------------------------------- state
Servo servos[MAX_STANDS][2];
float current_cdeg[MAX_STANDS][2];
float target_cdeg[MAX_STANDS][2];
uint8_t num_stands = 0;                // attached so far (grows to the largest N received)

uint32_t frames_ok = 0, frames_bad = 0;
uint32_t last_frame_ms = 0, last_update_ms = 0, last_status_ms = 0;

// --------------------------------------------------------------- helpers
uint8_t crc8_update(uint8_t crc, uint8_t b) {
  crc ^= b;
  for (uint8_t i = 0; i < 8; i++) crc = (crc & 0x80) ? (uint8_t)((crc << 1) ^ 0x07) : (uint8_t)(crc << 1);
  return crc;
}

uint16_t cdeg_to_us(float cdeg) {
  return PULSE_MIN_US + (uint16_t)((cdeg / 18000.0f) * (PULSE_MAX_US - PULSE_MIN_US) + 0.5f);
}

void home_all() {
  for (uint8_t s = 0; s < num_stands; s++)
    for (uint8_t a = 0; a < 2; a++) target_cdeg[s][a] = 9000;
}

void attach_up_to(uint8_t n) {
  // unused pins stay unattached, so a 2-stand build never drives pins 10-29
  while (num_stands < n && num_stands < MAX_STANDS) {
    uint8_t s = num_stands++;
    for (uint8_t a = 0; a < 2; a++) {
      current_cdeg[s][a] = target_cdeg[s][a] = 9000;
      servos[s][a].attach(PINS[s][a], PULSE_MIN_US, PULSE_MAX_US);
      servos[s][a].writeMicroseconds(cdeg_to_us(9000));
    }
  }
}

// ---------------------------------------------------------------- parser
enum ParseState : uint8_t { WAIT_AA, WAIT_55, READ_TYPE, READ_SEQ, READ_N, READ_PAYLOAD, READ_CRC };
ParseState ps = WAIT_AA;
uint8_t f_type, f_seq, f_n, f_crc, f_idx;
uint8_t payload[MAX_STANDS * 4];

void apply_frame() {
  if (f_type == 0x02) { home_all(); return; }
  if (f_type != 0x01) { frames_bad++; return; }
  attach_up_to(f_n);
  for (uint8_t s = 0; s < f_n; s++) {
    uint16_t pan  = payload[4 * s]     | (uint16_t)payload[4 * s + 1] << 8;
    uint16_t tilt = payload[4 * s + 2] | (uint16_t)payload[4 * s + 3] << 8;
    target_cdeg[s][0] = constrain(pan, 0, 18000);
    target_cdeg[s][1] = constrain(tilt, 0, 18000);
  }
}

void parse_byte(uint8_t b) {
  switch (ps) {
    case WAIT_AA: if (b == 0xAA) ps = WAIT_55; break;
    case WAIT_55: ps = (b == 0x55) ? READ_TYPE : (b == 0xAA ? WAIT_55 : WAIT_AA); break;
    case READ_TYPE: f_type = b; f_crc = crc8_update(0, b); ps = READ_SEQ; break;
    case READ_SEQ:  f_seq = b;  f_crc = crc8_update(f_crc, b); ps = READ_N; break;
    case READ_N:
      f_n = b; f_crc = crc8_update(f_crc, b); f_idx = 0;
      if (f_n > MAX_STANDS) { frames_bad++; ps = WAIT_AA; }
      else ps = f_n ? READ_PAYLOAD : READ_CRC;
      break;
    case READ_PAYLOAD:
      payload[f_idx++] = b; f_crc = crc8_update(f_crc, b);
      if (f_idx >= 4 * f_n) ps = READ_CRC;
      break;
    case READ_CRC:
      if (b == f_crc) { apply_frame(); frames_ok++; last_frame_ms = millis(); }
      else frames_bad++;
      ps = WAIT_AA;
      break;
  }
}

// ----------------------------------------------------------------- setup
void setup() {
  Serial.begin(BAUD);
  Serial.println(F("orpheus_stand ready"));
}

// ------------------------------------------------------------------ loop
void loop() {
  while (Serial.available()) parse_byte((uint8_t)Serial.read());

  uint32_t now = millis();
  if (now - last_update_ms >= UPDATE_MS) {
    float dt = (now - last_update_ms) / 1000.0f;
    last_update_ms = now;
    float max_step = SLEW_DEG_PER_S * 100.0f * dt;   // centidegrees this tick
    for (uint8_t s = 0; s < num_stands; s++) {
      for (uint8_t a = 0; a < 2; a++) {
        float err = target_cdeg[s][a] - current_cdeg[s][a];
        if (err == 0) continue;
        current_cdeg[s][a] += constrain(err, -max_step, max_step);
        servos[s][a].writeMicroseconds(cdeg_to_us(current_cdeg[s][a]));
      }
    }
  }

  // No watchdog motion: if the host goes quiet the stands hold their last aim.
  if (now - last_status_ms >= STATUS_MS) {
    last_status_ms = now;
    Serial.print(F("ok=")); Serial.print(frames_ok);
    Serial.print(F(" bad=")); Serial.print(frames_bad);
    Serial.print(F(" stands=")); Serial.print(num_stands);
    Serial.print(F(" idle_ms=")); Serial.println(now - last_frame_ms);
  }
}
