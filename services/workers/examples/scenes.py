"""Procedural example memories with known ground truth.

Each Scenario describes a place: sound sources at 3D positions, diffuse ambience, a
camera path, and recording conditions (mic layout, reverb, noise, imperfect models).
`render()` turns it into exactly the files the ingest and poses stages will produce
(audio.wav, meta.json, cameras.json, points3d.npy, frame images), plus oracle
backends that stand in for CLAP / AudioSep / Grounding DINO with realistic errors:
separation leakage, tag noise, missed and false detections, jittered boxes.

Sounds are synthesized (no downloads): birdsong chirps, fountain, traffic swells,
babble-like speech, music, barking, bells, waves, wind.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import numpy as np
from scipy import signal

from memoryscapes_workers.audio.doa import SPEED_OF_SOUND, CameraPath
from memoryscapes_workers.audio.tagging import TagResult
from memoryscapes_workers.audio.vocabulary import BY_LABEL
from memoryscapes_workers.context import StageContext
from memoryscapes_workers.models import Camera
from memoryscapes_workers.storage import write_json, write_npy, write_wav
from memoryscapes_workers.vision.grounding import Detection

SR = 16_000
UP = np.array([0.0, 1.0, 0.0])

# ---------------------------------------------------------------- sound synthesis


def _bandpass(x, lo, hi, sr=SR, order=4):
    sos = signal.butter(order, [lo, min(hi, sr / 2 * 0.95)], btype="band", fs=sr, output="sos")
    return signal.sosfilt(sos, x)


def _lowpass(x, hi, sr=SR, order=4):
    return signal.sosfilt(signal.butter(order, hi, btype="low", fs=sr, output="sos"), x)


def _norm(x):
    return (x / (np.sqrt(np.mean(x**2)) + 1e-12)).astype(np.float32)


def _slow_envelope(n, rng, rate_hz, depth, sr=SR):
    k = max(2, int(n / sr * rate_hz) + 2)
    pts = 1.0 - depth * rng.random(k)
    return np.interp(np.arange(n), np.linspace(0, n, k), pts)


def birdsong(n, rng):
    out = np.zeros(n)
    t = 0
    while t < n:
        t += int(rng.uniform(0.05, 0.6) * SR)
        dur = int(rng.uniform(0.06, 0.2) * SR)
        if t + dur >= n:
            break
        tt = np.arange(dur) / SR
        f0, f1 = rng.uniform(2500, 5500), rng.uniform(2500, 6000)
        phase = 2 * np.pi * np.cumsum(np.linspace(f0, f1, dur)) / SR
        out[t : t + dur] += np.sin(phase) * np.hanning(dur) * (1 + 0.3 * np.sin(2 * np.pi * 40 * tt))
        t += dur
    return _norm(out)


def water(n, rng):
    return _norm(_bandpass(rng.standard_normal(n), 300, 6000) * _slow_envelope(n, rng, 6, 0.4))


def traffic(n, rng):
    base = _lowpass(rng.standard_normal(n), 700)
    swells = np.zeros(n)
    for c in rng.uniform(0, n / SR, size=max(1, int(n / SR / 3))):
        swells += np.exp(-(((np.arange(n) / SR) - c) ** 2) / (2 * 1.2**2))
    return _norm(base * (0.3 + swells))


def speech(n, rng):
    """Babble-like: a glottal pulse train through formant filters that change per syllable."""
    out = np.zeros(n)
    pos = 0
    while pos < n:
        syl = int(rng.uniform(0.12, 0.3) * SR)
        f0 = rng.uniform(110, 210)
        pulses = np.zeros(syl)
        pulses[:: max(1, int(SR / f0))] = 1.0
        voiced = sum(_bandpass(pulses, f * 0.85, f * 1.15, order=2) for f in (rng.uniform(300, 800), rng.uniform(900, 2200), rng.uniform(2400, 3200)))
        seg = voiced * np.hanning(syl)
        out[pos : pos + syl] += seg[: n - pos]
        pos += syl + int(rng.uniform(0.02, 0.5 if rng.random() < 0.2 else 0.08) * SR)
    return _norm(out)


def music(n, rng):
    t = np.arange(n) / SR
    out = np.zeros(n)
    note_len = int(0.5 * SR)
    for i in range(0, n, note_len):
        root = 220 * 2 ** (rng.integers(0, 12) / 12)
        seg = slice(i, min(n, i + note_len))
        for h, mult in enumerate((1, 1.25, 1.5)):
            out[seg] += np.sin(2 * np.pi * root * mult * t[seg]) / (h + 1)
        out[seg] *= np.exp(-3 * (t[seg] - t[i]))
    return _norm(out)


def dog(n, rng):
    out = np.zeros(n)
    t = int(rng.uniform(0.3, 1.0) * SR)
    while t < n:
        for _ in range(rng.integers(1, 4)):
            dur = int(0.15 * SR)
            if t + dur >= n:
                break
            tt = np.arange(dur) / SR
            f0 = rng.uniform(400, 650)
            bark = sum(np.sin(2 * np.pi * f0 * k * tt) / k for k in range(1, 6)) + 0.5 * rng.standard_normal(dur)
            out[t : t + dur] += bark * np.exp(-tt * 18)
            t += dur + int(0.12 * SR)
        t += int(rng.uniform(1.5, 4.0) * SR)
    return _norm(out)


def bells(n, rng):
    out = np.zeros(n)
    for start in range(int(0.2 * SR), n, int(1.6 * SR)):
        dur = min(n - start, int(2.5 * SR))
        tt = np.arange(dur) / SR
        f = rng.choice([392, 440, 523])
        out[start : start + dur] += sum(np.sin(2 * np.pi * f * r * tt) * np.exp(-tt * d) for r, d in ((1, 1.5), (2.0, 2.5), (2.76, 3), (5.4, 4)))
    return _norm(out)


def waves(n, rng):
    period = rng.uniform(5, 8)
    t = np.arange(n) / SR
    env = 0.2 + np.clip(np.sin(2 * np.pi * t / period), 0, None) ** 3
    return _norm(_bandpass(rng.standard_normal(n), 150, 4000) * env)


def wind(n, rng):
    return _norm(_lowpass(rng.standard_normal(n), 400) * _slow_envelope(n, rng, 0.8, 0.7))


SOUNDS: dict[str, Callable[[int, np.random.Generator], np.ndarray]] = {
    "birdsong": birdsong,
    "water": water,
    "traffic": traffic,
    "speech": speech,
    "music": music,
    "dog": dog,
    "bells": bells,
    "waves": waves,
    "wind": wind,
    "crowd": lambda n, rng: _norm(sum(speech(n, rng) for _ in range(6))),
}

# ---------------------------------------------------------------- scenario description


@dataclass
class Source:
    label: str
    position: tuple[float, float, float]
    level_db: float = -20.0  # level of the dry signal at 1 m, dBFS
    size_m: float = 1.0  # visual extent, for detection boxes and scene points


@dataclass
class Conditions:
    layout: str = "stereo"  # "stereo" | "mono"
    spacing_m: float = 0.12  # true mic spacing
    reported_spacing_m: float | None = 0.12  # what meta.json claims (None = unknown)
    camera: str = "walk_pan"  # "walk_pan" | "walk_straight" | "static"
    reverb_rt60_s: float = 0.0
    reverb_wet: float = 0.3
    noise_dbfs: float = -60.0
    leakage_db: float = -20.0  # how much of the other sources leaks into each separated stem
    tag_noise: float = 0.05
    false_tags: tuple[str, ...] = ()  # classes the tagger wrongly reports
    detector_miss_rate: float = 0.1
    detector_false_rate: float = 0.05  # false boxes per frame per prompt
    box_jitter: float = 0.1  # fraction of box size
    # how the (separately built) splat relates to the poses' world frame; anything but
    # scale 1 / offset 0 is an integration bug the debugger should flag
    splat_scale: float = 1.0
    splat_offset: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class Scenario:
    name: str
    description: str
    duration_s: float
    sources: list[Source]
    diffuse: list[tuple[str, float]] = field(default_factory=list)  # (label, level dBFS)
    conditions: Conditions = field(default_factory=Conditions)
    seed: int = 0


SCENARIOS: dict[str, Scenario] = {
    s.name: s
    for s in [
        Scenario(
            "park_walk",
            "Walking past a fountain in a park, birds in a tree, a road beyond the hedge, light wind.",
            12.0,
            [
                Source("birdsong", (3.0, 3.5, 6.0), -14, size_m=2.5),
                Source("water", (-2.5, 0.8, 4.5), -16, size_m=1.5),
                Source("traffic", (6.0, 0.7, 14.0), -10, size_m=4.0),
            ],
            diffuse=[("wind", -38)],
        ),
        Scenario(
            "street_cafe",
            "A cafe terrace: a friend talking across the table, a speaker playing music, traffic, a dog nearby.",
            12.0,
            [
                Source("speech", (1.2, 1.2, 2.0), -24, size_m=0.8),
                Source("music", (-3.0, 2.2, 4.0), -24, size_m=0.6),
                Source("traffic", (0.5, 0.7, 10.0), -12, size_m=4.0),
                Source("dog", (4.0, 0.4, 5.0), -18, size_m=0.7),
            ],
            conditions=Conditions(reverb_rt60_s=0.3, leakage_db=-15),
            seed=1,
        ),
        Scenario(
            "courtyard_static",
            "Standing still in a courtyard, panning slowly: church bells high up, a fountain close by.",
            12.0,
            [
                Source("bells", (-2.0, 12.0, 16.0), -2, size_m=3.0),
                Source("water", (2.0, 0.5, 3.5), -20, size_m=1.2),
            ],
            conditions=Conditions(camera="static", reverb_rt60_s=0.6, reverb_wet=0.35),
            seed=2,
        ),
        Scenario(
            "beach_mono",
            "A beach filmed on an old phone with a mono mic: waves ahead, a friend talking, wind and a crowd.",
            12.0,
            [
                Source("waves", (0.0, 0.0, 14.0), -8, size_m=8.0),
                Source("speech", (1.8, 1.5, 2.5), -24, size_m=0.8),
            ],
            diffuse=[("wind", -32), ("crowd", -36)],
            conditions=Conditions(layout="mono", reported_spacing_m=None),
            seed=3,
        ),
        Scenario(
            "hard_mode",
            "The cafe again with everything going wrong: heavy leakage, missed and false detections, "
            "a false tag, reverb, noise, the wrong mic spacing in the metadata, and a splat "
            "that came back at half scale.",
            12.0,
            [
                Source("speech", (1.2, 1.2, 2.0), -24, size_m=0.8),
                Source("music", (-3.0, 2.2, 4.0), -24, size_m=0.6),
                Source("traffic", (0.5, 0.7, 10.0), -12, size_m=4.0),
            ],
            conditions=Conditions(
                spacing_m=0.08,
                reported_spacing_m=0.14,
                reverb_rt60_s=0.7,
                reverb_wet=0.45,
                noise_dbfs=-38,
                leakage_db=-8,
                tag_noise=0.12,
                false_tags=("bells",),
                detector_miss_rate=0.5,
                detector_false_rate=0.3,
                box_jitter=0.3,
                splat_scale=0.5,
            ),
            seed=4,
        ),
    ]
}

# ---------------------------------------------------------------- geometry


def look_camera(position, yaw_deg, t, frame_id, width=640, height=480, f=480.0) -> Camera:
    yaw = np.radians(yaw_deg)
    fwd = np.array([np.sin(yaw), 0.0, np.cos(yaw)])
    right = np.cross(-UP, fwd)
    c2w = np.eye(4)
    c2w[:3, 0], c2w[:3, 1], c2w[:3, 2], c2w[:3, 3] = right, -UP, fwd, position
    K = np.array([[f, 0, width / 2], [0, f, height / 2], [0, 0, 1.0]])
    return Camera(frame_id, float(t), width, height, K, c2w)


def camera_path(kind: str, duration_s: float, n: int = 25) -> list[Camera]:
    cams = []
    for i, t in enumerate(np.linspace(0, duration_s, n)):
        u = t / duration_s
        if kind == "walk_pan":
            pos, yaw = (-2.5 + 5.0 * u, 1.5, 0.3 * np.sin(np.pi * u)), 45 * np.sin(2 * np.pi * u * 1.5)
        elif kind == "walk_straight":
            pos, yaw = (-2.5 + 5.0 * u, 1.5, 0.0), 0.0
        elif kind == "static":
            pos, yaw = (0.0, 1.5, 0.0), 30 * np.sin(2 * np.pi * u)
        else:
            raise ValueError(kind)
        cams.append(look_camera(np.array(pos, float), yaw, t, f"f{i:03d}"))
    return cams


def scene_points(scenario: Scenario, rng) -> np.ndarray:
    """Ground, a backdrop, and a blob of surface points on every source object."""
    xs = np.arange(-12, 12.01, 0.5)
    zs = np.arange(-4, 20.01, 0.5)
    ground = np.array([[x, 0.0, z] for x in xs for z in zs])
    backdrop = np.array([[x, y, 20.0] for x in xs for y in np.arange(0, 8.01, 0.5)])
    blobs = []
    for s in scenario.sources:
        d = rng.standard_normal((400, 3))
        d /= np.linalg.norm(d, axis=1, keepdims=True)
        blobs.append(np.array(s.position) + d * s.size_m / 2)
    return np.concatenate([ground, backdrop, *blobs]).astype(np.float32)


def project(cam: Camera, p) -> tuple[np.ndarray, float]:
    w2c = np.linalg.inv(cam.c2w)
    pc = w2c[:3, :3] @ np.asarray(p, float) + w2c[:3, 3]
    uv = cam.K @ pc
    return uv[:2] / uv[2], pc[2]


# ---------------------------------------------------------------- audio rendering


def _frac_delay(x, delay_s):
    n = len(x)
    freqs = np.fft.rfftfreq(n, 1.0 / SR)
    return np.fft.irfft(np.fft.rfft(x) * np.exp(-2j * np.pi * freqs * delay_s), n)


def _spatialize(dry, position, path: CameraPath, spacing, block_s=0.1):
    """Two mics at the camera's +/- x axis: per-block delay and 1/r attenuation."""
    n = len(dry)
    out = np.zeros((2, n))
    block = int(block_s * SR)
    pad = 256
    src = np.asarray(position, float)
    for start in range(0, n, block):
        seg = dry[start : start + block]
        pos, axis = path.at((start + len(seg) / 2) / SR)
        mics = (pos - axis * spacing / 2, pos + axis * spacing / 2)
        dists = [np.linalg.norm(src - m) for m in mics]
        mean = sum(dists) / 2
        padded = np.concatenate([np.zeros(pad), seg, np.zeros(pad)])
        for ch, d in enumerate(dists):
            y = _frac_delay(padded, (d - mean) / SPEED_OF_SOUND)[pad : pad + len(seg)]
            out[ch, start : start + len(seg)] = y / max(d, 0.5)
    return out


def _reverb(x, rt60, wet, rng):
    if rt60 <= 0:
        return x
    n_ir = int(rt60 * SR)
    t = np.arange(n_ir) / SR
    out = np.empty_like(x)
    for ch in range(x.shape[0]):
        ir = rng.standard_normal(n_ir) * np.exp(-6.9 * t / rt60)
        ir[: int(0.005 * SR)] = 0
        ir /= np.sqrt(np.sum(ir**2))
        out[ch] = (1 - wet) * x[ch] + wet * signal.fftconvolve(x[ch], ir)[: x.shape[1]]
    return out


def _db(v):
    return 10 ** (v / 20)


@dataclass
class Rendered:
    scenario: Scenario
    cameras: list[Camera]
    points: np.ndarray
    recording: np.ndarray  # (C, N) as the phone recorded it
    contributions: dict[str, np.ndarray]  # label -> (C, N) what each source contributed
    sample_rate: int = SR
    image_lookup: dict[bytes, str] = field(default_factory=dict)  # frame image bytes -> frame id


def render(scenario: Scenario) -> Rendered:
    rng = np.random.default_rng(scenario.seed)
    c = scenario.conditions
    n = int(scenario.duration_s * SR)
    cams = camera_path(c.camera, scenario.duration_s)
    path = CameraPath(cams)

    contributions: dict[str, np.ndarray] = {}
    for s in scenario.sources:
        dry = SOUNDS[s.label](n, rng) * _db(s.level_db)
        wet = _reverb(_spatialize(dry, s.position, path, c.spacing_m), c.reverb_rt60_s, c.reverb_wet, rng)
        contributions[s.label] = contributions.get(s.label, 0) + wet
    for label, level in scenario.diffuse:
        # decorrelated per channel: no single direction
        contributions[label] = np.stack([SOUNDS[label](n, rng) for _ in range(2)]) * _db(level)

    noise = rng.standard_normal((2, n)) * _db(c.noise_dbfs)
    recording = sum(contributions.values()) + noise
    if c.layout == "mono":
        recording = recording[:1]
        contributions = {k: v[:1] for k, v in contributions.items()}
    peak = np.abs(recording).max()
    if peak > 0.95:  # the phone's limiter
        scale = 0.95 / peak
        recording = recording * scale
        contributions = {k: v * scale for k, v in contributions.items()}
    return Rendered(scenario, cams, scene_points(scenario, rng), recording.astype(np.float32), contributions)


def write_inputs(ctx: StageContext, r: Rendered, images: bool = False, splat: bool = False) -> None:
    """Writes what ingest + poses will produce for this memory.

    images=True renders real PNG keyframes (slower); otherwise each frame "image" is
    just its id, which is all the oracle detector needs. splat=True also writes the
    splat pipeline's outputs (docs/SPLAT_OUTPUTS.md) and debug/truth.json.
    """
    import splat_fake

    s, c = ctx.storage, r.scenario.conditions
    write_wav(s, ctx.key("ingest", "audio.wav"), r.recording, r.sample_rate)
    mic = {"channels": int(r.recording.shape[0]), "layout": c.layout}
    if c.reported_spacing_m is not None and c.layout == "stereo":
        mic["spacing_m"] = c.reported_spacing_m
    write_json(s, ctx.key("ingest", "meta.json"), {"mic": mic, "scenario": r.scenario.name})
    if images:
        r.image_lookup = splat_fake.write_frames(ctx, r)
    else:
        for cam in r.cameras:
            cam.image_key = ctx.key("ingest", f"frames/{cam.frame_id}.jpg")
            s.write_bytes(cam.image_key, cam.frame_id.encode())  # oracle detector reads the id
    write_json(s, ctx.key("poses", "cameras.json"), [cam.to_json() for cam in r.cameras])
    write_npy(s, ctx.key("poses", "points3d.npy"), r.points)
    if splat:
        splat_fake.write_splat(ctx, r, scale=c.splat_scale, offset=c.splat_offset)
        splat_fake.write_truth(ctx, r, c.splat_scale, c.splat_offset)


# ---------------------------------------------------------------- oracle backends


class OracleTagger:
    """Scores the classes that are really present ~0.55, others ~0.03, plus noise and
    any configured false tags."""

    def __init__(self, r: Rendered):
        self.present = {s.label for s in r.scenario.sources} | {lab for lab, _ in r.scenario.diffuse}
        self.c = r.scenario.conditions
        self.rng = np.random.default_rng(r.scenario.seed + 100)

    def tag(self, mono, sr, classes):
        times = np.arange(2.5, max(2.6, len(mono) / sr - 2.4), 2.5)
        scores = {}
        for cls in classes:
            base = 0.55 if cls.label in self.present else (0.3 if cls.label in self.c.false_tags else 0.03)
            scores[cls.label] = np.clip(base + self.c.tag_noise * self.rng.standard_normal(len(times)), 0, 1)
        return TagResult(times, scores)


class OracleSeparator:
    """Returns each class's true contribution plus leakage from everything else.
    A falsely tagged class gets only leakage."""

    def __init__(self, r: Rendered):
        self.contrib = {k: v.mean(axis=0) for k, v in r.contributions.items()}
        self.leak = _db(r.scenario.conditions.leakage_db)

    def separate(self, mono, sr, classes):
        out = {}
        for cls in classes:
            own = self.contrib.get(cls.label, np.zeros_like(mono))
            others = sum((v for k, v in self.contrib.items() if k != cls.label), np.zeros_like(mono))
            out[cls.label] = (own + self.leak * others).astype(np.float32)
        return out


class OracleDetector:
    """Projects each true source's bounding sphere into the frame. Misses some, jitters
    boxes, and adds false boxes elsewhere."""

    def __init__(self, r: Rendered):
        self.r = r
        self.cams = {cam.frame_id: cam for cam in r.cameras}
        self.c = r.scenario.conditions
        self.rng = np.random.default_rng(r.scenario.seed + 200)

    def detect(self, image_bytes, prompt):
        frame_id = self.r.image_lookup.get(image_bytes) or image_bytes.decode()
        cam = self.cams[frame_id]
        out = []
        for s in self.r.scenario.sources:
            cls = BY_LABEL[s.label]
            if cls.detect_prompt != prompt or self.rng.random() < self.c.detector_miss_rate:
                continue
            uv, z = project(cam, s.position)
            if z <= 0.5:
                continue
            half = cam.K[0, 0] * s.size_m / 2 / z
            uv = uv + self.rng.standard_normal(2) * half * self.c.box_jitter
            box = (uv[0] - half, uv[1] - half, uv[0] + half, uv[1] + half)
            if box[2] < 0 or box[3] < 0 or box[0] > cam.width or box[1] > cam.height:
                continue
            box = (max(box[0], 0), max(box[1], 0), min(box[2], cam.width), min(box[3], cam.height))
            out.append(Detection(box, float(self.rng.uniform(0.45, 0.9)), cls.label))
        if self.rng.random() < self.c.detector_false_rate:
            u, v = self.rng.uniform(0, cam.width), self.rng.uniform(0, cam.height)
            out.append(Detection((u - 30, v - 30, u + 30, v + 30), float(self.rng.uniform(0.3, 0.6)), "false"))
        return out
