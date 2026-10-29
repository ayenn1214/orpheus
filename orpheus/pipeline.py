"""Frame -> listener(s) -> servo commands (+ audio filter updates).
Independent of camera, serial and audio hardware.

Each face is a *track* with its own pose solver, smoothing, ear finder,
predictor and profile, so two people never share filter state or ears.

    listeners.mode
      single    one face, as before
      main      track everyone, aim at the main listener (the profile named in
                listeners.main when recognised, otherwise whoever arrived first)
      midpoint  aim each stand at the midpoint of everyone's ear on its side

With two or more listeners, xtc/virtual audio falls back to align: crosstalk
cancellation is built for one pair of ears and sounds wrong to everyone else.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import cv2
import numpy as np

from . import geometry
from .calib import undistorter_for
from .config import Config
from .depth import refine_pose_with_depth
from .earfind import EarRefiner
from .enroll import ListenerProfile, speaker_ear_distances
from .faceid import IdState
from .head import (HeadModel, Listener, PoseFilter, PoseSolver, make_listener, pose_from_box,
                   pose_from_eyes)
from .landmarks import Face, detect_faces, make_backend
from .predict import Predictor


@dataclass
class Track:
    tid: int
    model: HeadModel
    profile: ListenerProfile | None
    solver: PoseSolver
    filt: PoseFilter
    refiner: EarRefiner | None
    predictor: Predictor
    box: tuple
    born: float
    seen: float
    face: Face | None = None
    listener: Listener | None = None            # current (filtered)
    aim: Listener | None = None                 # predicted lead_ms ahead
    ear_sources: dict = field(default_factory=lambda: {"left": "offset", "right": "offset"})
    ids: IdState = field(default_factory=IdState)
    depth_used: bool = False

    @property
    def name(self) -> str:
        return self.profile.name if self.profile is not None else ""


@dataclass
class State:
    face: Face | None = None
    listener: Listener | None = None            # main listener, current
    aim: Listener | None = None                 # what the stands aim at (predicted / midpoint)
    listeners: list = field(default_factory=list)   # every tracked listener (current)
    names: list = field(default_factory=list)
    main_index: int = -1
    angles: list[tuple[float, float]] = field(default_factory=list)   # servo commands
    drivers: list[np.ndarray] = field(default_factory=list)           # speaker positions, mm
    distances: dict = field(default_factory=dict)   # speaker -> {ear_l, ear_r} mm
    at_seat: bool = False                            # listener lost, aiming at the stored seat
    ear_sources: dict = field(default_factory=dict)  # side -> "seen" (image) | "offset" (eyes+offset)
    xtc_suspended: bool = False                      # 2+ listeners: audio fell back to align
    depth_used: bool = False
    image: np.ndarray | None = None                  # the frame as tracked (mirrored/undistorted)
    sent: bool = False


class Pipeline:
    def __init__(self, cfg: Config, link, backend=None, audio=None,
                 profile: ListenerProfile | None = None, intrinsics=None, identifier=None) -> None:
        self.cfg = cfg
        self.link = link
        self.audio = audio
        self.identifier = identifier
        self.base_head = HeadModel(**{**asdict(cfg.head), "ear_l_mm": None, "ear_r_mm": None})
        self.profile = profile
        if profile is not None:
            profile.apply(cfg.head)            # measured scale + ears replace the generic model
        self.K = intrinsics or geometry.Intrinsics.from_camera(cfg.camera)
        # a depth source supplies its own (rectified / factory) intrinsics: no undistortion
        self.undistort = None if intrinsics is not None else undistorter_for(cfg.camera)
        self.backend = backend or make_backend(cfg.detector.backend, cfg.detector,
                                               cfg.detector.model_path)
        self.tracks: list[Track] = []
        self._next_id = 0
        self.seat = self._seat_listener()
        self._lost_since: float | None = None
        self._at_seat = False
        self._last_sent: list[tuple[float, float]] | None = None
        self._last_send_t = -1e9
        self.state = State()

    # ------------------------------------------------------------- tracks
    def reset(self) -> None:
        self.tracks.clear()

    def _model_for(self, profile: ListenerProfile | None) -> HeadModel:
        m = HeadModel(**asdict(self.base_head))
        if profile is not None:
            profile.apply(m)
        return m

    def _new_track(self, face: Face, t: float) -> Track:
        # the configured profile goes to the first listener only; others start generic
        prof = self.profile if not any(tr.profile is self.profile for tr in self.tracks) else None
        model = self._model_for(prof)
        p = self.cfg.predict
        tr = Track(tid=self._next_id, model=model, profile=prof, solver=PoseSolver(self.K, model),
                   filt=PoseFilter(self.cfg.filter.min_cutoff_hz, self.cfg.filter.beta),
                   refiner=self._refiner_for(prof),
                   predictor=Predictor(p.lead_ms, p.vel_tau_s, p.max_shift_mm),
                   box=face.box, born=t, seen=t)
        self._next_id += 1
        return tr

    def _refiner_for(self, prof: ListenerProfile | None) -> EarRefiner | None:
        if prof is None or not prof.ear_templates:
            return None
        r = EarRefiner(prof.ear_templates, self.cfg.earfind, self.K)
        return r if r.usable else None

    def assign_profile(self, tr: Track, prof: ListenerProfile | None) -> None:
        tr.profile = prof
        tr.model = self._model_for(prof)
        tr.solver = PoseSolver(self.K, tr.model)
        tr.refiner = self._refiner_for(prof)

    def _associate(self, faces: list[Face], t: float) -> list[tuple[Track, Face]]:
        pairs, free = [], list(self.tracks)
        for face in faces:
            fx, fy, fw, fh = face.box
            fc = np.array([fx + fw / 2, fy + fh / 2])
            best, bd = None, float("inf")
            for tr in free:
                x, y, w, h = tr.box
                d = np.linalg.norm(fc - [x + w / 2, y + h / 2])
                if d < max(w, fw, 40) and d < bd:
                    best, bd = tr, d
            if best is None:
                best = self._new_track(face, t)
                self.tracks.append(best)
            else:
                free.remove(best)
            best.box, best.seen, best.face = face.box, t, face
            pairs.append((best, face))
        for tr in free:                                  # not seen this frame
            tr.face = None
        hold = self.cfg.listeners.hold_s
        self.tracks = [tr for tr in self.tracks if tr.face is not None or t - tr.seen <= hold]
        return pairs

    # --------------------------------------------------------------- step
    def step(self, frame_bgr: np.ndarray, t: float, depth_mm: np.ndarray | None = None) -> State:
        if self.cfg.camera.mirror:
            frame_bgr = cv2.flip(frame_bgr, 1)
            depth_mm = None if depth_mm is None else cv2.flip(depth_mm, 1)
        if self.undistort is not None:
            frame_bgr = self.undistort(frame_bgr)

        lmode = self.cfg.listeners.mode
        max_faces = 1 if lmode == "single" else max(self.cfg.detector.max_faces, 1)
        faces = detect_faces(self.backend, frame_bgr, t, max_faces)
        pairs = self._associate(faces, t)
        for tr, face in pairs:
            self._update_track(tr, face, frame_bgr, depth_mm, t)

        st = State(image=frame_bgr)
        if not pairs:
            for tr in self.tracks:                       # held tracks: drop warm starts
                tr.solver.reset()
            if not self.tracks:
                return self._lost(st, t)
        self._lost_since = None
        self._at_seat = False

        active = [tr for tr in self.tracks if tr.listener is not None]
        if not active:
            return self._lost(st, t)
        main = self._main(active)
        st.face = main.face
        st.listener = main.listener
        st.listeners = [tr.listener for tr in active]
        st.names = [tr.name for tr in active]
        st.main_index = active.index(main)
        st.ear_sources = dict(main.ear_sources)
        st.depth_used = main.depth_used

        if lmode == "midpoint" and len(active) >= 2:
            st.aim = _merge([tr.aim for tr in active], main.aim)
            audio_listener = _merge([tr.listener for tr in active], main.listener)
        else:
            st.aim = main.aim
            audio_listener = main.listener
        st.angles, st.drivers = geometry.solve(st.aim, self.cfg.stands)
        st.distances = self._distances(st.aim, st.drivers)

        two = len(active) >= 2 and lmode != "single"
        if self.audio is not None:
            suspend = two and not self.cfg.listeners.xtc_with_two and \
                self.cfg.audio.mode in ("xtc", "virtual")
            self.audio.override_mode("align" if suspend else None)
            st.xtc_suspended = suspend
            self.audio.update(audio_listener, st.drivers)

        min_dt = 1.0 / self.cfg.link.rate_hz
        if t - self._last_send_t >= min_dt and self._changed(st.angles):
            self.link.send(st.angles)
            self._last_sent = st.angles
            self._last_send_t = t
            st.sent = True

        self.state = st
        return st

    def _update_track(self, tr: Track, face: Face, frame_bgr, depth_mm, t: float) -> None:
        R, tvec, method = self._pose(tr, face)
        tr.depth_used = False
        if depth_mm is not None:
            d = self.cfg.depth
            R, tvec, tr.depth_used = refine_pose_with_depth(
                R, tvec, method, face.points, face.box, depth_mm, self.K, tr.model,
                d.window_px, d.min_mm, d.max_mm)
        tr.ear_sources = {"left": "offset", "right": "offset"}
        if tr.refiner is not None:
            # hidden ear only: the ear facing the camera is found in the image and joins
            # the pose solve; the turned-away ear keeps its eyes + stored offset placement
            yaw_known = method == "pnp" or (method == "eyes" and tr.depth_used)
            R, tvec, tr.ear_sources = tr.refiner.refine(frame_bgr, R, tvec, face.points,
                                                        tr.model, yaw_known=yaw_known)
        R, tvec = tr.filt(R, tvec, t)
        tr.listener = make_listener(R, tvec, tr.model, method, face.points)
        tr.aim = tr.predictor(tr.listener, t) if self.cfg.predict.lead_ms > 0 else tr.listener
        if self.identifier is not None:
            prof = self.identifier.update(tr.ids, frame_bgr, face)
            if prof is not None and prof is not tr.profile:
                self.assign_profile(tr, prof)

    def _main(self, active: list[Track]) -> Track:
        want = self.cfg.listeners.main
        if want:
            for tr in active:
                if tr.name == want:
                    return tr
        return min(active, key=lambda tr: tr.born)

    def _lost(self, st: State, t: float) -> State:
        """Hold the last aim; after return_to_seat_s, re-aim at the enrolled seat."""
        if self._lost_since is None:
            self._lost_since = t
        if self.audio is not None:
            self.audio.override_mode(None)
        wait = self.cfg.filter.return_to_seat_s
        if self.seat is not None and wait > 0 and t - self._lost_since >= wait:
            st.at_seat = True
            st.angles, st.drivers = geometry.solve(self.seat, self.cfg.stands)
            st.distances = self._distances(self.seat, st.drivers)
            if not self._at_seat:
                self.link.send(st.angles)
                self._last_sent, self._last_send_t = st.angles, t
                if self.audio is not None:
                    self.audio.update(self.seat, st.drivers)
                st.sent = self._at_seat = True
        self.state = st
        return st

    def _seat_listener(self) -> Listener | None:
        ref = self.profile.reference if self.profile else None
        if not ref or "R" not in ref:
            return None
        return make_listener(np.array(ref["R"]), np.array(ref["t_mm"]), self.cfg.head, "seat")

    def _distances(self, listener: Listener, drivers) -> dict:
        names = {s.name: d for s, d in zip(self.cfg.stands, drivers)}
        return speaker_ear_distances(names, listener.ear_l, listener.ear_r)

    def _pose(self, tr: Track, face: Face):
        pose = tr.solver.solve(face.points)
        if pose is not None:
            return pose[0], pose[1], "pnp"
        eyes = None
        if "eye_l" in face.points and "eye_r" in face.points:
            eyes = (face.points["eye_r"], face.points["eye_l"])
        ipd = tr.profile.ipd_mm if tr.profile is not None else None
        if eyes is not None and ipd:
            # ears not visible: anchor them to the eyes with the enrolled offsets
            yaw = face.profile_sign * self.cfg.detector.profile_yaw_deg
            R, tvec = pose_from_eyes(eyes[0], eyes[1], self.K, ipd, tr.model, yaw)
            return R, tvec, "eyes"
        R, tvec = pose_from_box(face.box, self.K, self.cfg.detector.face_width_mm, tr.model,
                                eyes=eyes, profile_sign=face.profile_sign,
                                profile_yaw_deg=self.cfg.detector.profile_yaw_deg)
        return R, tvec, "box"

    def _changed(self, angles: list[tuple[float, float]]) -> bool:
        if self._last_sent is None:
            return True
        db = self.cfg.filter.deadband_deg
        return any(
            abs(p - lp) >= db or abs(q - lq) >= db
            for (p, q), (lp, lq) in zip(angles, self._last_sent)
        )


def _merge(listeners: list[Listener], main: Listener) -> Listener:
    """Midpoint listener: each ear averaged across listeners, so a stand serving the
    left ear aims between everyone's left ears. Orientation and size from the main one."""
    el = np.mean([L.ear_l for L in listeners], axis=0)
    er = np.mean([L.ear_r for L in listeners], axis=0)
    return Listener(t=(el + er) / 2, R=main.R, ear_l=el, ear_r=er, radius_mm=main.radius_mm,
                    method="midpoint", landmarks={})
