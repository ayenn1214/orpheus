<p align="center">
  <img src="docs/logo.png" width="150" alt="Orpheus logo">
</p>

<h1 align="center">Orpheus</h1>

<p align="center">
Speakers that follow your ears.<br>
<em>Orpheus played so the world turned toward him. Here the speakers do.</em>
</p>

A webcam finds the listener's head and ears in 3D; motorised pan/tilt stands keep every speaker aimed at its ear, and the audio is re-timed and filtered for where the listener actually is. Stereo, 5.1 and 7.1.

Version 0.8.1. Python rebuild of *Real Time Facial Tracking for Speaker Stands* (A. Narula, Manipal University Dubai, 2017), extended to the two goals the original report set out but could not reach: **locating the listener's ears** and **applying HRTFs**.

```
camera ─► landmarks (haar | yunet | mediapipe)
        ─► PnP against an anthropometric head model ─► 6-DoF head pose
        ─► ear (tragion) positions in 3D ──────────────► stands aim at head or near ear
                         │
                         └─► HRTF (spherical head | SOFA) per speaker→ear path
                             ─► 2×2 FIR filters (align | xtc | virtual), redesigned on movement
                             ─► audio thread: overlap-save convolution, crossfaded swaps
```

## 5.1 and 7.1 (v0.8)

```bash
python -m orpheus --write-config stands.json --layout 7.1 --seat-mm 900
python -m orpheus --config stands.json --port COM6 --audio-file movie_71.wav
python -m orpheus --config stands.json --port COM6 --audio-in "CABLE Output" --audio-out "7.1 Speakers"
```

`--write-config --layout` places one stand per speaker channel around a seat `--seat-mm` in front of the camera, at the ITU-R BS.775 / Dolby angles, each stand's zero direction facing the seat:

| Layout | Stands | Angles (+ = listener's left) | Subwoofer |
|---|---|---|---|
| stereo | FL FR | beside the camera (unchanged) | – |
| 5.1 | FL FR FC SL SR | fronts ±30°, centre 0° (150 mm below the camera), surrounds ±110° | LFE |
| 7.1 | FL FR FC BL BR SL SR | fronts ±30°, centre 0°, sides ±90°, backs ±150° | LFE |

- **Aiming:** fronts, sides and backs aim at the ear on their side (seen or virtual); the centre aims at the head centre (`"ear": "center"`). Pan is measured from each stand's facing (`yaw_offset_deg`) and wraps, so rear stands don't spin through 360°.
- **Audio:** `align` delays and level-matches every speaker and the sub so everything arrives together at its aim point, and re-times as you move (5.1 / 7.1 only support `align` and `off`; `xtc` / `virtual` stay stereo-only). Input and output channels follow the WAV / sound-card order: 5.1 = FL FR FC LFE SL SR, 7.1 = FL FR FC LFE BL BR SL SR, so a 7.1 file or loopback plays straight to a 7.1 interface. Change `output_channel` per stand if your wiring differs. A stereo source in a 5.1/7.1 layout plays from the front pair.
- **LFE:** `audio.lfe_mode` = `sub` (fixed subwoofer on `audio.sub_output`, time-aligned from `audio.sub_position_mm`), `mains` (folded into FL/FR), or `drop`.
- **Firmware:** up to 8 stands (16 servos) on a Mega 2560, pins in `layouts.PINS` (6/7, 8/9, 10/11, 12/13, 22/23, 24/25, 26/27, 28/29). Nothing to set: a stand's servos attach the first time the host addresses it. 7.1 is 14 SG90s: size the servo supply for 10–15 A, or split it across supplies with common ground.
- **Validation:** one stand per speaker channel, no stand on LFE, unique output channels.
- **Moving speakers:** `--move-stand NAME LEFT FROM_CAM HEIGHT` (mm; LEFT + = listener's left, HEIGHT + = above the speaker axis) places a stand and re-aims its zero direction at the seat (the enrolled seat when a profile is set, else `--seat-mm`); `--keep-facing` leaves the facing alone; repeat for several stands. `--move-sub LEFT FROM_CAM HEIGHT` moves the subwoofer used for time alignment. Example: `--config stands.json --move-stand FC 0 150 -250` puts the centre under a TV. The Speaker Stand Bench page has the same editor plus drag-to-place, and "Copy as config" exports the stands block for stands.json.

Simulation (Speaker Stand Bench, 7.1, listener swaying ±180 mm): arrival-time spread across all eight outputs 0.02 ms with delays following the listener, 1.0 ms with delays fixed for the seat; level spread 0.1 dB.

## New in v0.7

| Feature | What it does | Simulated result |
|---|---|---|
| **Predict ahead** (`predict.lead_ms`, `--lead-ms`) | Aims at where each ear will be 100 ms from now, from its tracked velocity | Head sliding at 300 mm/s: aim error 30 mm → under 3 mm |
| **Two listeners** (`--listeners main\|midpoint`) | Tracks up to 2 faces, each with its own pose, filter, profile and ear finder. `main` aims at the main listener; `midpoint` aims each stand between everyone's ear on its side | xtc/virtual switch to align while two people are present, and switch back when one leaves |
| **Face ID** (`--faceid-model`) | OpenCV SFace recognises who is sitting there and loads their profile (ears, eye spacing, ear images, HRTF radius) | 4 agreeing frames before a profile loads; a stranger never loads anyone |
| **Camera calibration** (`--calibrate-camera 9x6 --square 25`) | Checkerboard → camera matrix + lens distortion; every frame is undistorted before tracking | Focal length within 2 %, distortion recovered from rendered boards |
| **ArUco stand locator** (`--make-markers`, `--locate-stands`) | One printed marker per stand; the camera measures each pivot's position | 100 mm marker on a 640 px webcam: ~4 mm per metre of distance |
| **Depth cameras** (`--depth realsense\|oakd\|stereo`) | Measured landmark depth replaces size-based distance; with eyes only, depth also gives the head turn | Head size off by up to 12 %: ear error 45 mm → 6 mm median. Eyes only + depth: 8 mm median |

### Face ID

```bash
python -m orpheus --config stands.json --backend yunet --model yunet.onnx \
    --faceid-model face_recognition_sface_2021dec.onnx --enroll ankur --distance 600   # stores the face too
python -m orpheus --config stands.json --backend yunet --model yunet.onnx \
    --faceid-model face_recognition_sface_2021dec.onnx --add-face sam                  # existing profile
python -m orpheus --config stands.json --backend yunet --model yunet.onnx \
    --faceid-model face_recognition_sface_2021dec.onnx --listeners main --main ankur
```

Profiles are read from `faceid.profiles_dir` (`profiles/`). Model: `face_recognition_sface_2021dec.onnx` from opencv_zoo. Needs 5 face points, so `yunet` or `mediapipe`.

### Two listeners

- `single` (default): one face, as before.
- `main`: everyone is tracked; the stands follow `listeners.main` when that profile is recognised, otherwise whoever arrived first. Others show as blue circles labelled with their name or "guest".
- `midpoint`: each stand aims at the average of everyone's ear on its side.
- With two or more listeners, `xtc`/`virtual` render as `align` (crosstalk cancellation only works for one pair of ears). `listeners.xtc_with_two: true` keeps it for the main listener.
- A listener who drops out of view is held for `listeners.hold_s` (0.6 s) so the aim doesn't jump.

### Camera calibration and stand markers

```bash
python -m orpheus --config stands.json --calibrate-camera 9x6 --square 25   # inner corners, mm
python -m orpheus --config stands.json --make-markers markers/              # prints, assigns ids
python -m orpheus --config stands.json --locate-stands                      # writes position_mm
```

- Calibration takes views automatically when the board is somewhere new; cover the whole frame, near and far, tilted. It stores `camera.matrix`, `camera.dist` and a measured `hfov_deg`.
- Print markers at 100 % and measure the black square; set `markers.size_mm` (default 100 mm). Fix each to its stand's pan/tilt head and set `marker_offset_mm` from the marker centre to the pivot (marker frame: x right, y up, z out toward the camera). Keep that offset short: the marker's orientation is its noisy part.
- Calibrate the camera first; the locator uses the calibrated, undistorted image.

### Depth cameras

```bash
python -m orpheus --config stands.json --depth realsense          # pip install pyrealsense2
python -m orpheus --config stands.json --depth oakd               # pip install depthai (v3)
python -m orpheus --config stands.json --calibrate-stereo 9x6 --square 25   # two webcams
python -m orpheus --config stands.json --depth stereo
```

- The depth camera's own intrinsics replace the configured ones (no checkerboard needed for RealSense/OAK-D).
- Stereo: two identical webcams side by side, 60–120 mm apart; `camera.index` is the left, `depth.right_index` the right. Images are rectified and depth comes from semi-global matching at half resolution.
- Depth per landmark is a median over a small window; readings outside `depth.min_mm`–`max_mm` are ignored. Frames without usable depth fall back to the webcam method; the preview shows `depth on` / `no reading`.

## Ear enrollment (v0.3): find the ears once, store the distances

```bash
python -m orpheus --config stands.json --backend yunet --model face_detection_yunet_2023mar.onnx \
    --enroll ankur --distance 600
```

A one-time guided session, then the stored values drive the stands and audio every run.

**Crosshair mode (default, no clicks).** The preview is mirrored and has a cross at the image centre.

1. **Eyes** — sit at the measured distance and put the midpoint of your eyes on the cross, facing the lens. Gives head scale, your measured eye spacing (IPD), and your seat.
2. **Left ear** — turn to your right, slide until your left ear (the flap in front of the canal) sits on the cross, press SPACE. Five times, turning a little further each time (about 20° to 60°). The head pose is averaged over the frames you held still.
3. **Right ear** — the same, turning left.

Each capture says "the ear is on the optical axis at this head pose"; the five rays cross at the ear. Simulation (2 px alignment, 0.7 px landmark noise, 20 runs): ear error **3.5 mm median, 6.2 mm p90**. Alignment matters: at 5 px it rises to ~10 mm, so use `--enroll-mode sweep` (click the ear in captured frames) if you can't hold the cross steadily.

The profile stores each ear as an **offset from the eye midpoint** plus the measured IPD. At run time:

- With a pose backend (`yunet`/`mediapipe`), ears ride on the full head pose.
- When only the eyes are found (the `haar` backend, or a frame where PnP fails), the ears are **placed from the eyes**: depth from the stored IPD, roll from the eye line, ear = eye midpoint + R × stored offset. The stands aim there. The preview draws the offset lines from your eyes to the placed ears.

**Hidden ear only (v0.5).** At most one ear faces the camera. That ear is found in the image; only the ear turned away uses eyes + offset. Facing the camera head-on, both ears are edge-on and both use the offset.

- The ear detector is yours: enrollment already captured each ear at a known pixel, so those patches are saved in the profile (`ear_templates`) and matched around the predicted spot, scaled for distance. No cascade or model download.
- The seen ear joins the pose solve as one more point. With eyes only (two points), it is the third point that recovers the head turn and distance (P3P), so the hidden ear's offset is applied under the right rotation.
- The preview labels each ear `seen` (green, filled) or `offset` (magenta ring).
- **Each stand keeps its own ear** (v0.5.1). The left stand always aims at the left ear and the right stand at the right ear, whether that ear is seen or virtual. Before, `near_ear` picked whichever ear was closer, and on a hard head turn both stands could swing to the same visible ear. The plan view draws a hard aim line to a seen ear and a dotted one to a virtual ear. `"ear": "left"|"right"` on a stand overrides the automatic side (camera +X = listener's left).

Simulation, eyes-only tracking, rendered frames, 16 turned-head cases from ±20° to ±50°:

| | worst ear error |
|---|---|
| eyes + offset only | 59–862 mm (grows with the turn) |
| eyes + seen ear, when the ear is found (13 of 16) | 1–6 mm (one case 38 mm) |

The 3 misses were at 20° (ear barely visible) and one 50° view; those fall back to eyes + offset. Facing the camera: both offset, within 10 mm. Cost: ~30 ms per frame with eyes only, less with a pose backend. `earfind.enabled: false` turns it off; `earfind.min_score` sets how sure a match must be.

Limit of eyes-only placement without a seen ear: two points cannot show yaw. Facing the camera it lands within ~10 mm (simulation); a 20° head turn throws it off by 30–200 mm depending on distance. Use a pose backend if you turn your head while listening.

**Sweep mode** (`--enroll-mode sweep`):

1. **Measure** — face the camera with your eyes on the cross at the measured distance. Sets your head scale (monocular PnP knows shape, not size) and records your listening seat.
2. **Left ear** — turn slowly to your right. 12 frames are captured automatically while the ear swings from 65° to 20° off the lens axis, each with its head pose. If the detector loses your face at extreme turns, press `n` once 6 are in.
3. **Right ear** — turn slowly to your left. Same again.
4. **Mark** — each frame opens zoomed on the model's guess (orange ×); click the tragion (the small flap in front of the ear canal), Enter to accept, `s` skip, `b` back. With `--ear-cascades LEFT.xml RIGHT.xml` an ear detector pre-places the mark.
5. **Solve** — each click is a ray from the camera; in the head frame the rays from different poses cross at the ear. Least-squares intersection, outlier clicks dropped.

Saved to `profiles/ankur.json` and linked from the config:

```
ear-to-ear          150.5 mm
HRTF sphere radius  93.4 mm
left  ear  101 behind / 17 below / 76 out from the eyes   (12 views, 0 rejected, 36 deg spread, ray fit 1.8 mm)
right ear   94 behind / 15 below / 74 out from the eyes   (12 views, 0 rejected, 35 deg spread, ray fit 2.8 mm)
seat: left   speaker -> L ear 746 mm, R ear 676 mm
seat: right  speaker -> L ear 671 mm, R ear 719 mm
```
(Output from a simulated session.)

At run time (`"profile"` in the config, or `--profile path`):

- The tracked pose carries **your** ears, not the generic ±72.5 mm ones. Stands with `"aim": "near_ear"` point at them.
- **Speaker-to-ear distances** are computed every frame and shown in the preview.
- `align` audio times each speaker to its own-side ear (`audio.align_to: "ears"`).
- HRTF sphere radius comes from your measured ear distance.
- If you leave for `filter.return_to_seat_s` (3 s), the stands re-aim at your stored seat instead of freezing wherever they were.

Simulation (12 runs, 0.7 px landmark noise, 1.5 px click noise, a listener whose ears sit ~10 mm off the generic model): enrolled ear position error **2.6 mm median, 4.7 mm p90**; ear-distance scatter 3.3 mm. At run time the stored ears are **2.6 mm** from the true ears versus **10.3 mm** with the generic model.

Enrollment needs head yaw, so it needs the `yunet` or `mediapipe` backend.

## Ears

Without a profile, ears are not detected — Viola-Jones cannot find them (the 2017 conclusion), and a front-facing camera rarely sees them. They are **solved**: facial landmarks give the head pose by PnP against a generic adult head model, and the tragions sit at fixed head-frame positions (±72.5 mm, scaled per listener).

| Backend | Points | Pose | Download |
|---|---|---|---|
| `haar` (default) | face box + 2 eyes | position from box width, roll from eyes, yaw 0 or ±70° on profile | none |
| `yunet` | 5 (eyes, nose, mouth corners) | full 6-DoF PnP | `face_detection_yunet_2023mar.onnx` (opencv_zoo) |
| `mediapipe` | 6 of 478 | full 6-DoF PnP | `face_landmarker.task` |

Download URLs are in `orpheus/landmarks.py` (`MODEL_URLS`). For ear tracking that follows head turns, use `yunet` or `mediapipe`; `haar` only knows where the face is.

Monocular PnP gives head shape but not size, so `--calibrate MM` (eyes at a measured distance) sets one scale factor. That factor sets ear spacing, PnP depth, and the HRTF sphere radius together — the report's anthropometric "ITD scaling factor" (§2.1.3), via the Algazi et al. (2001) head-radius regression.

Synthetic test (head turning ±40°, swaying ±150 mm, 0.7 px landmark noise, at 800 mm): ear position error **9.8 mm median, 14.8 mm p95**.

## HRTF

`hrtf.py`, one interface, two models:

- **SphericalHead** — Brown & Duda (1998) structural model: incidence-dependent head-shadow filter + Woodworth delay. No data needed; radius from the listener's anthropometry. Verified against theory: low-frequency ITD matches Kuhn's 3a/c·sinθ, high-frequency ITD matches Woodworth within 4 %. No pinna, so no elevation cues.
- **SofaHRTF** — measured HRIRs from any AES69 `SimpleFreeFieldHRIR` file (MIT KEMAR, CIPIC, SADIE, your own). Onset delays are split off so neighbouring directions interpolate cleanly, and the ITD is rescaled to the listener's head radius.

## Audio modes

`audio.py` builds a 2×2 FIR matrix C (speaker ← input) from the tracked pose and redesigns it when the head moves ≥10 mm or turns ≥2° (≈2 ms per design).

| Mode | What it does | Use for |
|---|---|---|
| `align` | Delay + gain per speaker so both arrive at the interaural centre together and equally loud | Ordinary stereo; the sweet spot follows you |
| `xtc` | Crosstalk cancellation: invert the HRTF plant, C = Gᴴ(GGᴴ+βI)⁻¹ | Binaural recordings, game/VR audio |
| `virtual` | Binaural ±30° stereo triangle fixed in the room, then `xtc` | Stereo with a correct image wherever you sit or turn |

Crosstalk cancellation is the reason tracking matters. Channel separation, 200 Hz–6 kHz, spherical head, speakers ±300 mm, listener at 800 mm:

| Condition | Separation |
|---|---|
| Filters for the exact pose (model ceiling) | 52 dB |
| Listener moves 25 mm, filters not updated | 7 dB |
| Listener moves 50 mm, filters not updated | 3 dB |
| Tracked: 10 s turning/swaying run, filters from the estimated pose | **19 dB median, 13 dB worst** |
| Head radius wrong by 10 % | 16 dB |

These are model-vs-model numbers. Real rooms (reflections, speaker response, HRTF mismatch) typically land at 10–20 dB; a measured SOFA set close to your own ears helps most.

Latency: taps/2 + block = 21.3 ms at the defaults (1024 taps, 512 block, 48 kHz). Renderer cost: 0.2 ms per 10.7 ms block.

## Install

```bash
git clone <this repo> orpheus && cd orpheus
python -m venv .venv && . .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt                  # or: pip install -e .  (adds the `orpheus` command)
python -m pytest -q                              # 88 tests, no hardware needed
```

Model files are not in the repo (licences and size): download YuNet, SFace and the MediaPipe face landmarker from the links in `orpheus/landmarks.py` / `orpheus/faceid.py` into a `models/` folder (git-ignored).

OpenCV is pinned to 4.x (OpenCV 5 removed Haar cascades from the main package). `sounddevice` needs PortAudio; the Windows/macOS wheels include it, on Linux install `libportaudio2`.

## Run

```bash
python -m orpheus --write-config stands.json
python -m orpheus --config stands.json --backend yunet --model face_detection_yunet_2023mar.onnx --calibrate 600
python -m orpheus --config stands.json --port COM6                        # stands only
python -m orpheus --config stands.json --port COM6 --audio virtual --audio-file song.wav
python -m orpheus --config stands.json --audio xtc --audio-in "CABLE Output" --audio-out "Speakers"
python -m orpheus --config stands.json --audio virtual --hrtf mit_kemar.sofa --audio-file song.wav
python -m orpheus --list-audio-devices
```

For live system audio, route it into a loopback device (VB-Cable on Windows, BlackHole on macOS) and pass that as `--audio-in`.

The preview shows landmarks (cyan), projected ears (magenta L/R), facing arrow, pose, sphere radius, and two insets: a plan view (X-Z, pan) and a side view (Z-Y, tilt and ear height), each with stands, ears and aim lines — a hard line to a seen ear, a dotted line to a virtual one. Keys: `q`/`Esc` quit, `h` home servos.

## Setup

1. **Firmware** — `firmware/orpheus_stand/orpheus_stand.ino`. Stereo uses pins 6/7 and 8/9; 5.1/7.1 add more stands on the pins in the table above.
2. **Power** — servos on a separate 5–6 V supply, ground tied to the Arduino.
3. **Geometry** — set each stand's pivot `position_mm` relative to the lens (X right, Y down, Z toward the listener), and `driver_offset_mm` from pivot to tweeter. The audio filters use these positions, so measure to ±5 mm.
4. **Servo trim** — `center_deg`, `sign`, `min_deg`/`max_deg` per axis.
5. **Camera** — `--calibrate-camera` (checkerboard), unless a depth camera supplies intrinsics.
6. **Stand positions** — `--make-markers` + `--locate-stands`, or tape-measure `position_mm`.
7. **Listener** — `--enroll NAME` (add `--faceid-model` to store the face for automatic loading).
8. **Aim** — `"aim": "near_ear"` per stand points each speaker at its own side's tragion; `"head"` points at the interaural centre. For `xtc`/`virtual`, `"head"` keeps both speakers' on-axis response similar at both ears.

## Layout

```
orpheus/
  layouts.py    stereo / 5.1 / 7.1 channels, stand placement, audio routing
  predict.py    aim-ahead (per-ear velocity, capped)
  faceid.py     SFace embeddings, profile library, identity hysteresis
  calib.py      checkerboard calibration, undistortion, stereo calibration UI
  markers.py    ArUco marker sheets, marker pose, stand locator
  depth.py      RealSense / OAK-D / stereo sources, stereo matcher, depth-corrected pose
  enroll.py     ear enrollment: capture logic, ray triangulation, listener profile
  enroll_ui.py  guided capture + zoomed tragion marking (OpenCV windows)
  earfind.py    hidden-ear-only compensation: find the visible ear, add it to the pose
  landmarks.py  haar / yunet / mediapipe backends -> named 2D landmarks
  head.py       head model, PnP pose, ears, pose smoothing, Algazi radius
  hrtf.py       spherical-head (Brown-Duda) and SOFA HRTFs, ITD scaling
  audio.py      filter design (align/xtc/virtual, multichannel align), N x M streaming renderer, engine + IO
  vision.py     Viola-Jones + CamShift face tracker (used by the haar backend)
  geometry.py   intrinsics, aiming, servo mapping
  pipeline.py   frame -> tracks (one per face) -> pose -> servos + audio updates
  config.py     nested dataclass config, JSON load/save
  protocol.py, link.py, filters.py, app.py
firmware/orpheus_stand/orpheus_stand.ino
sim/stand_bench.html   3D browser simulator (Speaker Stand Bench): open the file directly
tests/          87 tests: protocol, geometry, PnP/ears, enrollment (scale, triangulation,
                outlier clicks, profile round-trip, stored vs generic, return-to-seat),
                HRTF vs theory, SOFA round-trip, XTC separation, align timing,
                renderer exactness, end-to-end crosstalk, prediction, two listeners,
                face ID, ArUco locating, checkerboard calibration, depth pose, stereo depth,
                5.1/7.1 presets, routing, multichannel alignment, 8x8 renderer, 7-stand pipeline
```

## Limits

- Without a profile the ears are a generic adult model. With one, accuracy depends on clicking the tragion carefully; the summary's ray-fit figure flags a sloppy session (> 6 mm).
- Enrollment assumes the face landmarks stay put relative to the ears (no glasses on/off between enrollment and use, which shifts some detectors' eye points).
- The spherical HRTF has no pinna: azimuth cues are right, elevation cues are absent. Use a SOFA file for full HRTFs.
- XTC assumes free field. Treat first reflections (desk, side walls) for real-world separation.
- The `haar` backend cannot see head turns, so `xtc`/`virtual` on `haar` only tracks position.
- Face ID needs the yunet or mediapipe backend (5 face points). Twins, heavy glasses changes or very low light lower the match score; raise `faceid.confirm_frames` if a profile ever flickers.
- 5.1/7.1 re-time and re-level as you move, but don't re-pan: if you sit well off the seat, a channel still comes from that speaker's real direction. Panning between speakers to restore the intended angles is a possible next step.
- Midpoint aiming is a compromise: nobody is on-axis. With people more than ~600 mm apart, `main` usually sounds better.
- Eyes-only + depth still cannot see a nod (pitch); a seen ear or a full landmark backend fixes that.
- The stereo matcher needs texture; a plain wall behind a bald, evenly lit head gives holes, and those frames fall back to the webcam method.
- Not verified here on real hardware: webcam, YuNet/MediaPipe/SFace models, RealSense/OAK-D/stereo cameras (code written against pyrealsense2 2.59 and depthai 3.11 APIs; both fail cleanly without a device), the OpenCV calibration and click windows, PortAudio streams, or acoustic separation in a room. Everything above is from synthetic data and simulation.

## Licence

Apache License 2.0 - see [LICENSE](LICENSE) and [NOTICE](NOTICE). Copyright 2026 Ankur Narula.
