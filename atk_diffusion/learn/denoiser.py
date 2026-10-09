# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""B3 — the diffusion denoiser as pre-detector (plan §4.B3; DETECTION_DESIGN
§3 "Low-SNR mode", §4.2 "Clean"; ARCHITECTURE §4.4, §5).

Plan B3: *"'Erasing Noise in Signal Detection with Diffusion Model' gives
the theory: a denoising diffusion model as a detector that beats maximum
likelihood, with the optimal denoising timestep a function of SNR … A small
U-Net on spectrogram tiles from Bill's own captures, trained per profile."*

WHAT `train_denoiser` LEARNS FROM, AND WHY.
* SYNTHETIC CLEAN SIGNALS at the profile's exact rate (`learn.augment` —
  `synth.native` when installed): the clean tile or window is known exactly,
  which is what a diffusion model needs as x0. One example in four is
  EMPTY (noise only): the model has to learn that nothing is a valid answer.
  The 8-FSK `mfsk8` is deliberately left out, so the weak-burst experiment
  has a waveform the denoiser has never seen.
* BILL'S OWN RECEIVER NOISE — terminated or quiet captures (`noise_captures`,
  profile-checked), or synthetic noise through the profile's MEASURED
  impairments (`dsp.impair`; a textbook receiver when unmeasured, said in
  the card). It is used three ways, none of them needing a label (this is
  the self-supervised part): it sets the floor, it MEASURES the
  representation's noise (μ_n, σ_n — so t* comes from this receiver, not
  from a textbook), and it is the noise in the MATCHED batches.
* BILL'S OWN CAPTURES (`data`, content, unlabelled) as AMBIENT examples: a
  capture already contains noise, so it is trained only at steps at least
  ten times noisier than itself (the ambient-diffusion rule, `diffusion.
  sample_t`) — it teaches what real signals look like without teaching the
  receiver's own noise as signal.

MATCHED BATCHES. Standard diffusion training only ever shows the network
Gaussian noise. The noise it meets at t* is not Gaussian: in a dB tile it
is the log of a pooled periodogram — skewed, heteroscedastic, biased in
signal cells. So a fraction of every batch (`matched_fraction`) is the real
thing: a clean example plus the receiver's noise, put through the same
front end, at exactly the t* and scale the runtime will use; the target ε is
whatever that noise is in model units. `eps_loss` then reproduces the
runtime's x_t* exactly (x_t* = √ᾱ·y), so the single step is trained on the
input it will get.

WHAT COMES OUT. `model.onnx` (inputs "x" [B, C, …] float32 and "t" [B]
int64, output "eps" — ARCHITECTURE §5) for ATK's core, `model.pt` for the
training environment, and a card (kind "denoiser", tier INVENTED, per
profile) whose `input["schedule"]` lets numpy + onnxruntime run the
single-step denoise, and whose metrics carry the numbers that say how far to
trust it, all MEASURED after training:

* `hallucination_rate` — the fraction of noise-only tiles in which the
  denoised output yields a CFAR detection when the raw input had none
  (plan §2.1, §7: *"how often does it produce a signal … where the ground
  truth has none?"*), with the tiles tested and the false-alarm rate;
* `val_rmse_db` — error against the clean tile for raw, Wiener, median,
  wavelet and the diffusion denoiser on held-out synthetic tiles, and
  `beats_classical` (plan §7: *"A learned tool that does not beat the
  classical one is not shipped"* — the card says which it is);
* `latency_ms` per tile on the CPU through onnxruntime.

LIMITS, stated. The tiny configurations in the tests learn nothing: they
prove the path. Real numbers need Bill's GPU, his noise captures and an
hour of training; the weak-burst experiment (`experiments.weak_burst`) is
the judge. Synthetic training signals are minimal generators (right
modulation and rate, idealised framing). The log-domain noise model is
exact for noise-only cells and an approximation in signal cells — the
matched batches are there for exactly that gap.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards, profiles, provenance
from atk_diffusion.dsp import denoise_classical as _cl
from atk_diffusion.dsp import denoise_runtime as _rt
from atk_diffusion.learn import augment as _aug
from atk_diffusion.learn import diffusion as _diff

DOMAINS = _rt.DOMAINS

#: What the denoiser is trained on. `mfsk8` is NOT here on purpose: it is
#: the weak-burst experiment's unseen waveform.
DEFAULT_KINDS = ("nfm_keyup", "pocsag", "dmr", "adsb", "lora", "bpsk", "qpsk",
                 "gfsk", "ofdm", "tone", "chirp", "ook", "noise")
UNSEEN_KINDS = ("mfsk8",)
AMBIENT_FACTOR = 10.0       # capture examples train only at ≥ 10x their own noise


# ---------------------------------------------------------------------------
# Geometry and noise
# ---------------------------------------------------------------------------
def tile_geometry(profile, geometry=None, pool_mode: str = "max") -> dict:
    """The spectrogram geometry a denoiser is trained on: the profile's
    StftGeometry (or the one given), with the frames-per-row the pipeline's
    tiles use (dsp.stft.tile_layout when installed — the same arithmetic
    otherwise)."""
    prof = profile if not isinstance(profile, str) else profiles.new_profile(profile)
    fs = float(prof.sample_rate)
    g = geometry if geometry is not None else prof.stft
    if isinstance(g, dict):            # a card's stft block carries extra keys
        known = profiles.StftGeometry.__dataclass_fields__
        g = profiles.StftGeometry(**{k: v for k, v in g.items() if k in known})
    try:
        from atk_diffusion.dsp import stft as _stft
        lay = _stft.tile_layout(fs, g)
        pool, rows = int(lay.pool), int(lay.rows)
    except ImportError:
        pool = max(1, int(round(float(g.tile_seconds) * fs / int(g.hop) / int(g.tile_rows))))
        rows = int(g.tile_rows)
    return {"fft_size": int(g.fft_size), "hop": int(g.hop), "window": str(g.window),
            "pool": pool, "pool_mode": str(pool_mode), "tile_rows": rows, "fs": fs}


class NoiseSource:
    """Receiver noise at a given rate: Bill's terminated/quiet captures when
    given (each checked against the profile), else synthetic noise through
    the profile's impairments (`learn.augment.receiver_noise`)."""

    def __init__(self, profile, fs: float, captures=(), rng=None,
                 max_samples: int = 1 << 22):
        from atk_diffusion import sigmf as _sigmf
        self.profile = profile
        self.pid = profile if isinstance(profile, str) else profile.id
        self.fs = float(fs)
        self.rng = np.random.default_rng(rng) if not isinstance(rng, np.random.Generator) else rng
        self.chunks: list[np.ndarray] = []
        self.entries: list[dict] = []
        for path in captures or ():
            meta = _sigmf.read_meta(path)
            got = profiles.profile_from_meta(meta)
            if got.lower() != str(self.pid).lower():
                raise profiles.ProfileMismatch(
                    f"the noise capture {_sigmf.base_of(path).name} is "
                    f"{profiles.describe(got)}; this model is for "
                    f"{profiles.describe(self.pid)}. Profiles never mix.")
            x = _sigmf.load(path, count=int(max_samples), meta=meta)
            if x.ndim > 1:
                x = x[0]
            self.chunks.append(np.asarray(x, dtype=np.complex64))
            self.entries.append({"name": _sigmf.base_of(path).name, "kind": "noise_capture",
                                 "n": int(x.size),
                                 "sha256": provenance.sha256_path(_sigmf.data_path(path))})
        if self.chunks:
            self.words = (f"Bill's own receiver noise: {len(self.chunks)} capture(s), "
                          f"{sum(c.size for c in self.chunks)} samples")
        else:
            _x, w = _aug.receiver_noise(64, self.fs, profile, self.rng)
            self.words = "synthetic receiver noise — " + w

    def draw(self, n: int) -> np.ndarray:
        n = int(n)
        if not self.chunks:
            return _aug.receiver_noise(n, self.fs, self.profile, self.rng)[0]
        c = self.chunks[int(self.rng.integers(0, len(self.chunks)))]
        if c.size >= n:
            s = int(self.rng.integers(0, c.size - n + 1))
            return c[s:s + n].copy()
        reps = int(math.ceil(n / c.size))
        return np.tile(c, reps)[:n].copy()


def floor_from_noise(noise: NoiseSource, geom: dict, frames: int = 2048) -> np.ndarray:
    """The floor per bin: the mean unpooled power of `frames` noise frames
    (the terminated-capture measurement, plan §3.4)."""
    n = _cl.samples_for_rows(frames, geom["fft_size"], geom["hop"], 1)
    return _cl.stft_power(noise.draw(n), geom["fft_size"], geom["hop"],
                          geom["window"]).mean(axis=0)


def noise_tiles(noise: NoiseSource, geom: dict, F: np.ndarray, n: int, H: int,
                W: int | None = None, rng=None) -> np.ndarray:
    """`n` noise-only tiles [n, H, W] in dB above the floor."""
    rng = noise.rng if rng is None else rng
    fft = geom["fft_size"]
    W = int(W or fft)
    m = _cl.samples_for_rows(H, fft, geom["hop"], geom["pool"])
    out = np.empty((int(n), int(H), W), dtype=np.float32)
    for i in range(int(n)):
        P = _cl.power_tile(noise.draw(m), fft, geom["hop"], geom["pool"],
                           geom["pool_mode"], geom["window"])
        b0 = int(rng.integers(0, fft - W + 1))
        out[i] = _cl.db_above(P, F)[:H, b0:b0 + W]
    return out


# ---------------------------------------------------------------------------
# Synthetic pairs
# ---------------------------------------------------------------------------
def usable_kinds(kinds, fs: float) -> tuple[list[str], list[str]]:
    """Kinds the profile's rate can represent (ADS-B needs 2 MS/s)."""
    ok, skipped = [], []
    probe = np.random.default_rng(0)
    for k in kinds:
        try:
            _aug.signal(k, fs, 64, probe)
            ok.append(k)
        except ValueError:
            skipped.append(k)
    if not ok:
        raise ValueError("none of the requested signal kinds can be made at "
                         f"{fs:g} S/s")
    return ok, skipped


def _place(kind, fs, n, f_lo, f_hi, rng, min_frac=0.2):
    """One burst inside [f_lo, f_hi] Hz and the window's n samples."""
    want = int(round(_aug.natural_duration(kind) * fs))
    nb = int(np.clip(want, 16, n))
    if nb > n * min_frac and rng.random() < 0.5:
        nb = int(max(16, rng.uniform(min_frac, 1.0) * nb))
    s, _truth = _aug.burst_samples(kind, fs, nb, rng)
    nb = int(s.size)
    bw = _aug.occupied_bandwidth(s, fs)
    lo, hi = f_lo + bw / 2.0, f_hi - bw / 2.0
    f0 = float(rng.uniform(lo, hi)) if hi > lo else 0.5 * (f_lo + f_hi)
    start = int(rng.integers(0, n - nb + 1))
    x = np.zeros(n, dtype=np.complex64)
    x[start:start + nb] = _aug.shift(s, f0, fs)
    return x, bw, (start, nb, f0)


def spec_pair(geom: dict, F: np.ndarray, H: int, W: int, noise: NoiseSource,
              kinds, rng, snr_db=(-3.0, 15.0), max_bursts: int = 2,
              p_empty: float = 0.25):
    """(clean dB [H, W], noisy dB [H, W], bursts) — one synthetic patch.

    clean = 10·log10(1 + S/F) with S the signal's own pooled power: 0 dB
    where there is no signal. noisy = 10·log10(P/F) of signal + receiver
    noise through the same STFT and pooling. A burst's in-band SNR is set
    against the floor in its own bins: power P_b = SNR · F̄ · B/fs."""
    fs, fft, hop = geom["fs"], geom["fft_size"], geom["hop"]
    n = _cl.samples_for_rows(H, fft, hop, geom["pool"])
    b0 = int(rng.integers(0, fft - W + 1))
    bin_hz = fs / fft
    f_lo = (b0 - fft // 2) * bin_hz
    f_hi = (b0 + W - fft // 2) * bin_hz
    s = np.zeros(n, dtype=np.complex64)
    bursts = []
    nb = 0 if rng.random() < p_empty else int(rng.integers(1, max_bursts + 1))
    for _ in range(nb):
        kind = str(rng.choice(list(kinds)))
        x, bw, (st, cnt, f0) = _place(kind, fs, n, f_lo, f_hi, rng)
        snr = float(rng.uniform(*snr_db))
        kb = np.clip(((np.array([f0 - bw / 2, f0 + bw / 2]) / bin_hz) + fft // 2)
                     .astype(int), 0, fft - 1)
        fbar = float(np.mean(F[kb[0]:kb[1] + 1]))
        p_b = 10 ** (snr / 10.0) * fbar * max(bw, bin_hz) / fs
        s += (math.sqrt(p_b) * x).astype(np.complex64)
        bursts.append({"kind": kind, "snr_db": snr, "bw_hz": bw, "start": st,
                       "count": cnt, "offset_hz": f0})
    w = noise.draw(n)
    pm = geom["pool_mode"]
    S = _cl.power_tile(s, fft, hop, geom["pool"], pm, geom["window"])
    P = _cl.power_tile(s + w, fft, hop, geom["pool"], pm, geom["window"])
    clean = _cl.DB * np.log1p(S / F)
    noisy = _cl.db_above(P, F)
    return (clean[:H, b0:b0 + W].astype(np.float32),
            noisy[:H, b0:b0 + W].astype(np.float32), bursts)


def iq_pair(fs: float, L: int, noise_fn, p_n: float, kinds, rng,
            snr_db=(-10.0, 15.0), p_empty: float = 0.15):
    """(x0 unit power [L], observation y [L], SNR linear) — one IQ window.
    An empty window has x0 = 0 and is paired with a random ASSUMED SNR (the
    one a caller might pass), so the model learns that nothing is an answer
    at every step it may be run at."""
    snr = 10 ** (float(rng.uniform(*snr_db)) / 10.0)
    w = noise_fn(L)
    if rng.random() < p_empty:
        return np.zeros(L, dtype=np.complex64), w.astype(np.complex64), snr
    kind = str(rng.choice(list(kinds)))
    if rng.random() < 0.5:
        x, _t = _aug.signal(kind, fs, L, rng)
        x = _aug.shift(x, rng.uniform(-0.3, 0.3) * fs, fs)
    else:
        x, _bw, _p = _place(kind, fs, L, -0.4 * fs, 0.4 * fs, rng, min_frac=0.25)
    rms = math.sqrt(float(np.mean(np.abs(x) ** 2)) / 2.0)
    x0 = (x / max(rms, 1e-12)).astype(np.complex64)
    c = math.sqrt(snr * p_n / 2.0)
    return x0, (c * x0 + w).astype(np.complex64), snr


# ---------------------------------------------------------------------------
# Measuring
# ---------------------------------------------------------------------------
def detect_any(spec_db: np.ndarray, pfa: float, geom: dict, min_cells: int = 3,
               guard: int = 2, train: int = 16) -> bool:
    """Does the energy detector find anything in this tile? The package's
    dsp.cfar (the pipeline's own detector, exact for max-pooled Hann tiles)
    when installed, else the internal CA-CFAR."""
    s = np.asarray(spec_db, dtype=np.float32)
    try:
        from atk_diffusion.dsp import cfar as _cfar
        mask = _cfar.ca_cfar(s, pfa, guard=guard, train=train, pool=geom["pool"],
                             window=geom["window"], fft_size=geom["fft_size"],
                             hop=geom["hop"])
        comps = _cfar.components(mask)
        return any(int(c[4].sum()) >= min_cells for c in comps)
    except ImportError:
        P = np.power(10.0, s.astype(np.float64) / 10.0)
        m = _cl.ca_cfar(P, pfa, guard=guard, train=min(train, 8), k=geom["pool"],
                        mode=geom["pool_mode"])
        return any(b.cells >= min_cells for b in _cl.cfar_boxes(m))


def hallucination_rate(denoiser, noise_tiles_db, pfa: float, geom: dict,
                       min_cells: int = 3, **denoise_kw) -> dict:
    """Plan §2.1 made a number: of the noise-only tiles in which the raw
    input yields NO detection, the fraction whose denoised output yields
    one. Returns {rate, eligible, hallucinated, tested, pfa}."""
    eligible = halluc = 0
    tiles = np.asarray(noise_tiles_db)
    for tile in tiles:
        if detect_any(tile, pfa, geom, min_cells):
            continue
        eligible += 1
        out = denoiser.denoise(tile, **denoise_kw)
        if detect_any(out, pfa, geom, min_cells):
            halluc += 1
    rate = (halluc / eligible) if eligible else None
    return {"rate": rate, "eligible": eligible, "hallucinated": halluc,
            "tested": int(tiles.shape[0]), "pfa": float(pfa)}


def classical_baselines(noisy_db, noise_tiles_db, noise_mean_db: float,
                        noise_std_db: float) -> dict:
    """Wiener, median and wavelet on the same noisy tiles, each with its
    noise-only bias measured on noise tiles and removed, so all are compared
    against the clean tile on equal terms. -> {method: outputs [N, H, W]}"""
    noisy = np.asarray(noisy_db, dtype=np.float64)
    ntiles = np.asarray(noise_tiles_db, dtype=np.float64)
    noise_lin = float(np.mean(np.power(10.0, ntiles / 10.0)))
    out = {"raw": noisy - noise_mean_db}
    for m in _cl.METHODS:
        kw = {"noise_lin": noise_lin} if m == "wiener" else {}
        if m == "wavelet":
            kw["sigma_db"] = noise_std_db
        bias = float(np.mean([_cl.clean_tile(t, m, **kw).out for t in ntiles[:8]]))
        out[m] = np.stack([_cl.clean_tile(t, m, **kw).out for t in noisy]) - bias
    return out


# ---------------------------------------------------------------------------
# The torch-side denoiser
# ---------------------------------------------------------------------------
class Denoiser:
    """A trained denoiser in the training environment (PyTorch). Same
    `denoise` as `dsp.denoise_runtime.DenoiserRuntime`, the same lines of
    representation math; only the network evaluation differs."""

    def __init__(self, model, card, model_dir, device="cpu"):
        self.model, self.card, self.dir, self.device = model, card, Path(model_dir), device
        self.input = dict(card.input)
        self.schedule = _diff.Schedule.from_json(self.input["schedule"])
        self.ac = self.schedule.alphas_cumprod
        self.domain = self.input["domain"]
        self.last: dict = {}
        self._eps = _diff.torch_eps_fn(self.model)

    @classmethod
    def load(cls, model_dir, for_profile: str | None = None, device: str = "cpu"):
        import torch
        from atk_diffusion.learn import unet as _u
        card = cards.load(model_dir, expect_kind="denoiser", for_profile=for_profile)
        tw = card.input.get("torch_weights") or {}
        p = Path(model_dir) / str(tw.get("file", "model.pt"))
        if not p.exists():
            raise cards.CardRefusal(f"{Path(model_dir).name} has no PyTorch weights "
                                    f"({p.name}); use the ONNX runtime instead")
        if tw.get("sha256") and provenance.sha256_path(p) != tw["sha256"]:
            raise cards.CardRefusal(f"{p.name} is not the file the card describes "
                                    "(its hash changed). Retrain or restore it.")
        model = _u.build_unet(card.input["unet"])
        model.load_state_dict(torch.load(p, map_location=device, weights_only=True))
        return cls(model.to(device).eval(), card, model_dir, device)

    def eps_fn(self, x, t):
        return self._eps(x, t)

    def denoise(self, x, snr_db=None, noise_std=None, noise_mean_db=None,
                noise_power=None, steps: int = 1, eta: float = 0.0, seed: int = 0,
                t=None):
        out, info = _rt.run(self.eps_fn, x, self.input, self.ac, snr_db=snr_db,
                            noise_std=noise_std, noise_mean_db=noise_mean_db,
                            noise_power=noise_power, steps=steps, eta=eta,
                            seed=seed, t=t)
        info.update(model=self.card.name,
                    model_sha256=self.card.weights.get("sha256", ""),
                    hallucination_rate=(self.card.metrics or {}).get("hallucination_rate"))
        self.last = info
        return out


def load_any(model_dir, profile: str):
    """The ONNX runtime when onnxruntime is installed (what ATK runs), else
    the torch model."""
    try:
        return _rt.DenoiserRuntime(model_dir, profile)
    except RuntimeError:
        return Denoiser.load(model_dir, for_profile=profile)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _ambient_tmin(nsr: np.ndarray, ac: np.ndarray, factor: float = AMBIENT_FACTOR):
    """Smallest t whose forward noise-to-signal (1−ᾱ)/ᾱ is ≥ factor × nsr."""
    fwd = (1.0 - ac) / ac
    return np.searchsorted(fwd, factor * np.asarray(nsr, dtype=np.float64)).clip(0, ac.size - 1)


def train_denoiser(rf, profile: str, data=None, domain: str = "spectrogram", *,
                   name: str | None = None, noise_captures=(),
                   kinds=DEFAULT_KINDS, synthetic: int = 4000,
                   geometry=None, pool_mode: str = "max", patch=(64, 64),
                   window: int = 1024, canonical: str | None = None,
                   snr_db=None, unit_db: float = 10.0,
                   schedule: str = "cosine", T: int = 1000, unet: dict | None = None,
                   steps: int = 20000, batch: int = 32, lr: float = 2e-4,
                   matched_fraction: float = 0.5, ema_decay: float | None = None,
                   seed: int = 0, device: str | None = None,
                   hallucination_tiles: int = 200, validation_tiles: int = 64,
                   pfa: float | None = None, overwrite: bool = False,
                   progress=None) -> Path:
    """Train the B3 denoiser for one receiver profile; returns the model
    folder (`rf.models(profile, name)`), card included.

    data             None, or Bill's own captures (SigMF paths, unlabelled;
                     used as AMBIENT examples), or an array of clean
                     examples ([N, H, W] dB tiles / [N, L] complex windows)
    noise_captures   terminated or quiet captures of THIS profile — the real
                     noise (synthetic through dsp.impair otherwise)
    domain           "spectrogram" (tiles in dB above the floor; the
                     pipeline's low-SNR mode) or "iq" (complex windows; the
                     cut's clean step), at the profile's rate or, with
                     `canonical`, one of its canonical rates
    Everything else has a stated default; the test sizes are tiny."""
    import torch
    from atk_diffusion.learn import unet as _u
    if domain not in DOMAINS:
        raise ValueError(f"unknown denoiser domain {domain!r} — one of {', '.join(DOMAINS)}")
    say = progress or (lambda s: None)
    prof = profiles.load_profile(rf, profile)
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    sched = _diff.make_schedule(schedule, T)
    ac = sched.alphas_cumprod
    kinds_ok, kinds_skipped = usable_kinds(kinds, prof.sample_rate)
    conf = dict(unet or (_u.SMALL_2D if domain == "spectrogram" else _u.SMALL_1D))
    conf["in_ch"] = 1 if domain == "spectrogram" else 2
    conf["dims"] = 2 if domain == "spectrogram" else 1
    mult = _u.multiple(conf)
    pfa = float(pfa if pfa is not None else prof.cfar_pfa)
    notes = []
    datasets = []
    noise = NoiseSource(prof, prof.sample_rate, noise_captures, rng)
    datasets += noise.entries
    say(f"noise: {noise.words}")

    # -- build the banks --------------------------------------------------
    if domain == "spectrogram":
        geom = tile_geometry(prof, geometry, pool_mode)
        H, W = int(patch[0]), int(patch[1])
        if H % mult or W % mult or W > geom["fft_size"]:
            raise ValueError(f"the patch {H} x {W} must be multiples of {mult} and "
                             f"no wider than the {geom['fft_size']}-bin FFT")
        F = floor_from_noise(noise, geom)
        ntiles = noise_tiles(noise, geom, F, max(16, hallucination_tiles // 4), H, W)
        mu_n, sd_n = _rt.noise_stats_db(ntiles)
        snr_lin = (unit_db / sd_n) ** 2
        t_star = int(_diff.snr_matched_timestep(snr_lin, ac))
        say(f"noise in the tile: mean {mu_n:.2f} dB, spread {sd_n:.2f} dB -> matched "
            f"SNR {10 * math.log10(snr_lin):.1f} dB, t* = {t_star} of {T}")
        snr_rng = tuple(snr_db) if snr_db else (-3.0, 15.0)
        clean, noisy = [], []
        for i in range(int(synthetic)):
            c_, n_, _b = spec_pair(geom, F, H, W, noise, kinds_ok, rng, snr_rng)
            clean.append(c_)
            noisy.append(n_)
        x0_syn = (np.stack(clean) / unit_db)[:, None].astype(np.float32)
        y_syn = ((np.stack(noisy) - mu_n) / unit_db)[:, None].astype(np.float32)
        t_syn = np.full(x0_syn.shape[0], t_star, dtype=np.int64)
        amb, amb_tmin = _spec_ambient(data, prof, geom, F, H, W, mu_n, sd_n, unit_db,
                                      ac, rng, datasets, notes)
        norm = {"kind": "fixed_scale", "unit_db": float(unit_db),
                "noise_mean_db": float(mu_n), "noise_std_db": float(sd_n)}
        snr_db_min = 10 * math.log10(snr_lin) - 6.0
        shape_in = {"patch": [H, W], "stft": geom}
        rate, decim, cclass = float(prof.sample_rate), 1, ""
    else:
        rate, decim, cclass = float(prof.sample_rate), 1, ""
        if canonical:
            can = {c.cls: c for c in prof.canonical_rates()}.get(canonical)
            if can is None:
                raise ValueError(f"{profiles.describe(prof.id)} has no {canonical} "
                                 "canonical rate")
            rate, decim, cclass = float(can.rate), int(can.decimation), canonical
        L = int(window)
        if L % mult:
            raise ValueError(f"the window {L} must be a multiple of {mult}")
        kinds_ok, more = usable_kinds(kinds_ok, rate)
        kinds_skipped += more

        def noise_fn(n, _d=decim):
            if _d == 1:
                return noise.draw(n)
            from atk_diffusion.dsp import resample as _rs
            y, _f = _rs.decimate(noise.draw(n * _d + 512), _d, prof.sample_rate)
            return y[256 // _d: 256 // _d + n]
        ref = noise_fn(max(16 * L, 8192))
        p_n = float(np.mean(np.abs(ref) ** 2))
        snr_rng = tuple(snr_db) if snr_db else (-10.0, 15.0)
        x0s, ys, snrs = [], [], []
        for i in range(int(synthetic)):
            a, b, s = iq_pair(rate, L, noise_fn, p_n, kinds_ok, rng, snr_rng)
            x0s.append(a)
            ys.append(b)
            snrs.append(s)
        x0c = np.stack(x0s)
        snrs = np.asarray(snrs)
        cs = np.sqrt(snrs * p_n / 2.0)
        x0_syn = _aug.iq_to_channels(x0c)
        y_syn = _aug.iq_to_channels(np.stack(ys) / cs[:, None])
        t_syn = np.asarray(_diff.snr_matched_timestep(snrs, ac), dtype=np.int64)
        amb, amb_tmin = _iq_ambient(data, prof, rate, decim, L, p_n, ac, rng,
                                    datasets, notes)
        norm = {"kind": "signal_power", "noise_power": p_n,
                "noise_std": math.sqrt(p_n / 2.0)}
        snr_db_min = float(snr_rng[0])
        t_star = None
        shape_in = {"window": L}
        mu_n = sd_n = None
    t_max = int(_diff.snr_matched_timestep(10 ** (snr_db_min / 10.0), ac))
    # the sampler's safety rail (recorded, applied in every step): x̂0 is
    # clipped to 1.5x the largest clean training value
    x0_clip = float(1.5 * max(np.max(np.abs(x0_syn)), 1.0))
    datasets.append({"name": "synthetic (generated at the profile's rate)",
                     "kind": "synthetic", "n": int(x0_syn.shape[0]),
                     "sha256": provenance.sha256_bytes(x0_syn.tobytes()),
                     "generator": "learn.augment (synth.native when installed)"})

    # -- train --------------------------------------------------------------
    model = _u.build_unet(conf).to(dev)
    X0 = torch.as_tensor(x0_syn)
    Y = torch.as_tensor(y_syn)
    TS = torch.as_tensor(t_syn)
    A = torch.as_tensor(amb) if amb is not None else None
    ATM = torch.as_tensor(amb_tmin) if amb is not None else None
    acs = torch.as_tensor(ac, dtype=torch.float32)
    n_m = int(round(batch * float(matched_fraction)))
    n_s = int(batch) - n_m
    g = torch.Generator().manual_seed(int(seed))
    p_amb = 0.0 if A is None else min(0.5, A.shape[0] / (A.shape[0] + X0.shape[0]))

    def batch_fn(step, gen):
        xs, ts, es = [], [], []
        if n_s:
            k_amb = int(round(n_s * p_amb))
            k_syn = n_s - k_amb
            i = torch.randint(0, X0.shape[0], (k_syn,), generator=gen)
            xs.append(X0[i])
            ts.append(_diff.sample_t(k_syn, T, generator=gen))
            if k_amb:
                j = torch.randint(0, A.shape[0], (k_amb,), generator=gen)
                xs.append(A[j])
                ts.append(_diff.sample_t(k_amb, T, t_min=ATM[j], generator=gen))
            for x_ in xs:
                es.append(torch.randn(x_.shape, generator=gen))
        if n_m:
            i = torch.randint(0, X0.shape[0], (n_m,), generator=gen)
            t_m = TS[i]
            a = acs[t_m].view(-1, *([1] * (X0.dim() - 1)))
            x0m = X0[i]
            # eps_loss forms √ᾱ·x0 + √(1−ᾱ)·ε; with this ε that is √ᾱ·y — the
            # runtime's x_t*, built from the receiver's real noise
            eps_m = torch.sqrt(a) * (Y[i] - x0m) / torch.sqrt(1 - a)
            xs.append(x0m)
            ts.append(t_m)
            es.append(eps_m)
        return {"x0": torch.cat(xs).to(dev), "t": torch.cat(ts).to(dev),
                "noise": torch.cat(es).to(dev)}

    say(f"training a {conf['dims']}D U-Net ({_u.count_params(model):,} parameters) "
        f"for {steps} steps on {dev}")
    t_train = time.time()
    hist = _diff.fit(model, batch_fn, ac, steps, lr=lr, progress=say,
                     generator=g, ema_decay=ema_decay)
    train_s = time.time() - t_train
    model = model.to("cpu").eval()

    # -- save, export, verify ---------------------------------------------------
    from atk_diffusion.learn import unet as _u2
    name = name or f"denoiser_{domain}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = rf.models(prof.id, name)
    if d.exists() and any(d.iterdir()) and not overwrite:
        raise FileExistsError(f"a model named {name} already exists for this "
                              "profile; choose another name or pass overwrite=True")
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    ex_shape = tuple(shape_in["patch"]) if domain == "spectrogram" else (shape_in["window"],)
    _u2.export_onnx(model, d / "model.onnx", ex_shape)
    parity = _verify_onnx(model, d / "model.onnx", ex_shape, conf["in_ch"])
    inp = {"domain": domain, "schedule": sched.to_json(), "unet": model.config,
           "multiple": mult, "normalize": norm, "rate": rate, "decimation": decim,
           "canonical_class": cclass, "t_star": t_star, "t_max": t_max,
           "x0_clip": x0_clip,
           "snr_db_min": round(snr_db_min, 2), "matched_fraction": float(matched_fraction),
           "kinds": kinds_ok, "kinds_skipped": kinds_skipped,
           "unseen_kinds": list(UNSEEN_KINDS),
           "torch_weights": {"file": "model.pt",
                             "sha256": provenance.sha256_path(d / "model.pt")},
           "onnx": {"inputs": ["x", "t"], "outputs": ["eps"], "opset": 17,
                    "max_abs_parity_error": parity},
           **shape_in}
    card = cards.new_card(
        name, "denoiser", prof.id, input=inp,
        classes=[{"name": k, "source": "synthetic", "examples": None} for k in kinds_ok],
        datasets=datasets,
        metrics={"final_loss": float(np.mean(hist["loss"][-50:])) if hist["loss"] else None,
                 "steps": int(steps), "train_seconds": round(train_s, 1)},
        license="all rights reserved",
        trained_on=f"{dev} (torch {torch.__version__})",
        notes=[f"noise: {noise.words}",
               "outputs are INVENTED tier: a lead, never a reading (plan §2.1)",
               "the weak-burst experiment (experiments.weak_burst) is the judge"] + notes)
    cards.save(d, card, "model.onnx")

    # -- measure (through what ATK will actually run) --------------------------
    say("measuring the hallucination rate and the classical baselines")
    try:
        runner = _rt.DenoiserRuntime(d, prof.id)
    except RuntimeError:
        runner = Denoiser.load(d, for_profile=prof.id)
    metrics = dict(card.metrics)
    if domain == "spectrogram":
        hn = noise_tiles(noise, geom, F, hallucination_tiles, H, W)
        hal = hallucination_rate(runner, hn, pfa, geom, noise_std=sd_n,
                                 noise_mean_db=mu_n)
        vrng = np.random.default_rng(seed + 1)
        vc, vn = [], []
        for _ in range(int(validation_tiles)):
            c_, n_, _b = spec_pair(geom, F, H, W, noise, kinds_ok, vrng, snr_rng,
                                   p_empty=0.0)
            vc.append(c_)
            vn.append(n_)
        vc, vn = np.stack(vc), np.stack(vn)
        outs = classical_baselines(vn, ntiles, mu_n, sd_n)
        outs["diffusion"] = runner.denoise(vn, noise_std=sd_n, noise_mean_db=mu_n)
        rmse = {k: round(float(np.sqrt(np.mean((v - vc) ** 2))), 3) for k, v in outs.items()}
        t0 = time.perf_counter()
        runner.denoise(vn[0], noise_std=sd_n, noise_mean_db=mu_n)
        lat = 1000.0 * (time.perf_counter() - t0)
        metrics.update(hallucination_rate=hal["rate"], hallucination=hal,
                       hallucination_pfa=pfa, val_rmse_db=rmse,
                       beats_classical=bool(rmse["diffusion"] < min(
                           rmse[m] for m in _cl.METHODS)),
                       t_star=t_star, snr_db_matched=round(10 * math.log10(snr_lin), 2),
                       latency_ms=round(lat, 2), latency_shape=[H, W])
    else:
        hal, snr_gain = _iq_measure(runner, rate, L, noise_fn, p_n, kinds_ok,
                                    snr_db_min, hallucination_tiles, pfa, seed)
        metrics.update(hallucination_rate=hal["rate"], hallucination=hal,
                       hallucination_pfa=pfa, hallucination_at_snr_db=snr_db_min,
                       val_snr_gain_db=snr_gain,
                       beats_classical=bool(snr_gain["diffusion"] > snr_gain["wiener"]))
    if not metrics["beats_classical"]:
        card.notes.append("does NOT beat the classical denoisers on the validation "
                          "set — not to be shipped (plan §7) until it does")
    card.metrics = metrics
    cards.save(d, card, "model.onnx")
    for f in ("model.onnx", "model.pt", "card.json"):
        try:
            rf.record(d / f, "model", f"denoiser {name}")
        except Exception:                                  # noqa: BLE001
            pass
    try:
        runs = rf.runs(prof.id)
        runs.mkdir(parents=True, exist_ok=True)
        import json
        (runs / f"{name}_train.json").write_text(json.dumps(
            {"model": name, "domain": domain, "loss": hist["loss"][::max(1, len(hist["loss"]) // 500)],
             "metrics": metrics}, indent=2, default=str), encoding="utf-8")
    except Exception:                                      # noqa: BLE001
        pass
    say(f"done: {', '.join(cards.summary(card)[1:])}")
    return d


def _verify_onnx(model, path, ex_shape, in_ch) -> float | None:
    """ONNX vs torch on a random input; refuses a broken export."""
    try:
        import onnxruntime as ort
    except ImportError:
        return None
    import torch
    so = ort.SessionOptions()
    so.intra_op_num_threads = 1
    s = ort.InferenceSession(str(path), so, providers=["CPUExecutionProvider"])
    x = np.random.default_rng(0).standard_normal((2, in_ch) + tuple(ex_shape)).astype(np.float32)
    t = np.array([3, 57], dtype=np.int64)
    got = s.run(["eps"], {"x": x, "t": t})[0]
    with torch.no_grad():
        ref = model(torch.from_numpy(x), torch.from_numpy(t)).numpy()
    err = float(np.max(np.abs(got - ref)))
    if err > 1e-3 * max(1.0, float(np.max(np.abs(ref)))):
        raise RuntimeError(f"the ONNX export disagrees with the trained network "
                           f"(max error {err:.3g}); the model was not saved as usable")
    return err


def _spec_ambient(data, prof, geom, F, H, W, mu_n, sd_n, unit_db, ac, rng,
                  datasets, notes):
    """Ambient (capture) examples for the spectrogram domain, or arrays of
    clean tiles. Returns (bank [N,1,H,W] or None, t_min [N] or None)."""
    if data is None:
        return None, None
    if isinstance(data, np.ndarray):
        arr = np.asarray(data, dtype=np.float32)
        if arr.ndim != 3 or arr.shape[1:] != (H, W):
            raise ValueError(f"clean tiles must be [N, {H}, {W}] in dB above the floor")
        datasets.append({"name": "given clean tiles", "kind": "array", "n": int(arr.shape[0]),
                         "sha256": provenance.sha256_bytes(arr.tobytes())})
        return (arr / unit_db)[:, None], np.zeros(arr.shape[0], dtype=np.int64)
    from atk_diffusion import sigmf as _sigmf
    tiles = []
    for path in data:
        meta = _sigmf.read_meta(path)
        got = profiles.profile_from_meta(meta)
        if got.lower() != prof.id.lower():
            raise profiles.ProfileMismatch(
                f"the capture {_sigmf.base_of(path).name} is {profiles.describe(got)}; "
                f"this model is for {profiles.describe(prof.id)}. Profiles never mix.")
        x = _sigmf.load(path, count=1 << 22, meta=meta)
        x = x[0] if x.ndim > 1 else x
        D = _cl.db_above(_cl.power_tile(x, geom["fft_size"], geom["hop"], geom["pool"],
                                        geom["pool_mode"], geom["window"]), F)
        for _ in range(max(1, D.shape[0] // H) * max(1, D.shape[1] // W)):
            r0 = int(rng.integers(0, max(1, D.shape[0] - H + 1)))
            b0 = int(rng.integers(0, max(1, D.shape[1] - W + 1)))
            p = D[r0:r0 + H, b0:b0 + W]
            if p.shape == (H, W):
                tiles.append(p)
        datasets.append({"name": _sigmf.base_of(path).name, "kind": "capture (ambient)",
                         "n": int(x.size),
                         "sha256": provenance.sha256_path(_sigmf.data_path(path))})
    if not tiles:
        notes.append("the captures given were too short for one patch; none used")
        return None, None
    bank = ((np.stack(tiles) - mu_n) / unit_db)[:, None].astype(np.float32)
    nsr = np.full(bank.shape[0], (sd_n / unit_db) ** 2)
    notes.append(f"{bank.shape[0]} patches of Bill's captures used as AMBIENT examples "
                 f"(trained only at steps ≥ {AMBIENT_FACTOR:g}x their own noise)")
    return bank, _ambient_tmin(nsr, ac)


def _iq_ambient(data, prof, rate, decim, L, p_n, ac, rng, datasets, notes):
    if data is None:
        return None, None
    if isinstance(data, np.ndarray):
        arr = np.asarray(data)
        if arr.ndim != 2 or arr.shape[1] != L or not np.iscomplexobj(arr):
            raise ValueError(f"clean windows must be complex [N, {L}]")
        xn, _r = _aug.unit_power(arr)
        datasets.append({"name": "given clean windows", "kind": "array", "n": int(arr.shape[0]),
                         "sha256": provenance.sha256_bytes(np.ascontiguousarray(arr).tobytes())})
        return _aug.iq_to_channels(xn), np.zeros(arr.shape[0], dtype=np.int64)
    from atk_diffusion import sigmf as _sigmf
    from atk_diffusion.dsp import resample as _rs
    wins, nsrs = [], []
    for path in data:
        meta = _sigmf.read_meta(path)
        got = profiles.profile_from_meta(meta)
        if got.lower() != prof.id.lower():
            raise profiles.ProfileMismatch(
                f"the capture {_sigmf.base_of(path).name} is {profiles.describe(got)}; "
                f"this model is for {profiles.describe(prof.id)}. Profiles never mix.")
        x = _sigmf.load(path, count=1 << 22, meta=meta)
        x = x[0] if x.ndim > 1 else x
        if decim > 1:
            x, _f = _rs.decimate(x, decim, prof.sample_rate)
        for s in range(0, x.size - L + 1, L):
            w = x[s:s + L]
            snr = float(np.mean(np.abs(w) ** 2)) / p_n - 1.0
            if snr >= 10.0:                     # 10 dB and up: nearly clean
                wins.append(w)
                nsrs.append(1.0 / snr)
        datasets.append({"name": _sigmf.base_of(path).name, "kind": "capture (ambient)",
                         "n": int(x.size),
                         "sha256": provenance.sha256_path(_sigmf.data_path(path))})
    if not wins:
        notes.append("no capture window was 10 dB above the floor; none used as ambient")
        return None, None
    xn, _r = _aug.unit_power(np.stack(wins))
    notes.append(f"{len(wins)} windows of Bill's captures used as AMBIENT examples")
    return _aug.iq_to_channels(xn), _ambient_tmin(np.asarray(nsrs), ac)


def _iq_measure(runner, rate, L, noise_fn, p_n, kinds, snr_db_min, n_tiles, pfa, seed):
    """IQ domain: hallucination on noise-only windows denoised at the lowest
    SNR offered (judged by the energy detector on their spectrogram), and
    the SNR gain against the clean signal for Wiener and the model."""
    rng = np.random.default_rng(seed + 2)
    nfft = 32 if L >= 256 else 16
    geom = {"fft_size": nfft, "hop": nfft, "window": "hann", "pool": 1,
            "pool_mode": "mean", "fs": rate}
    ref = noise_fn(nfft * 512)
    F = _cl.stft_power(ref, nfft).mean(axis=0)
    elig = hall = 0
    for _ in range(int(n_tiles)):
        w = noise_fn(L)
        raw = _cl.db_above(_cl.stft_power(w, nfft), F)
        if detect_any(raw, pfa, geom):
            continue
        elig += 1
        out = runner.denoise(w, snr_db=snr_db_min, noise_power=p_n)
        if detect_any(_cl.db_above(_cl.stft_power(out, nfft), F), pfa, geom):
            hall += 1
    gains = {"wiener": [], "diffusion": []}
    for _ in range(16):
        x0, y, snr = iq_pair(rate, L, noise_fn, p_n, kinds, rng, (0.0, 10.0), p_empty=0.0)
        s = x0 * math.sqrt(snr * p_n / 2.0)
        e_in = float(np.sum(np.abs(y - s) ** 2))
        for k, est in (("wiener", _cl.wiener_iq(y, p_n, nfft=nfft)),
                       ("diffusion", runner.denoise(y, snr_db=10 * math.log10(snr),
                                                    noise_power=p_n))):
            gains[k].append(10 * math.log10(e_in / max(float(np.sum(np.abs(est - s) ** 2)), 1e-30)))
    return ({"rate": (hall / elig) if elig else None, "eligible": elig,
             "hallucinated": hall, "tested": int(n_tiles), "pfa": float(pfa)},
            {k: round(float(np.mean(v)), 2) for k, v in gains.items()})
