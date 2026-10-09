# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The B3 denoiser in ATK's CORE environment: numpy + onnxruntime, no
PyTorch (ARCHITECTURE §1, §5; plan §4.B3; DETECTION_DESIGN §3 "Low-SNR
mode", §4.2 step 3 "Clean").

This is what ATK's low-SNR mode and the signal cut's "diffusion" clean call.
It loads a model only through its card — `cards.load(expect_kind=
"denoiser", for_profile=…)` — so a model trained for another receiver or
rate is refused in the plan's own sentence, and a weights file that changed
since training is refused too.

HOW ONE DENOISE RUNS (the theory is in `learn.diffusion.
snr_matched_timestep`, arXiv 2501.07030):

1. Put the observation in MODEL UNITS, where the training signal had unit
   power, using the noise level σ (measured, per real element) and the
   SNR — the model's unit in receiver units is c = σ·√SNR.
2. Choose t* where the forward process's SNR ᾱ/(1 − ᾱ) equals the
   observation's, and rescale by √ᾱ_{t*}.
3. One ONNX evaluation gives ε̂; ŝ = c·x̂0 = y − σ·ε̂ (Tweedie). `steps` > 1
   continues with DDIM from t* (the same sampler the training environment
   uses: `learn.diffusion` runs on numpy arrays).

THE TWO DOMAINS.
* spectrogram — a tile in dB ABOVE THE FLOOR (`dsp.stft.Tile.spec`). In the
  log domain the spectral estimation noise is additive with a constant,
  measured spread σ_n (dB) and mean μ_n, whatever the signal — so the
  model's unit is a fixed `unit_db` and the matched SNR is (unit_db/σ_n)²,
  set by the noise, not by the burst. The output reads 0 dB where the model
  finds only noise and the clean level, 10·log10(1 + S/F), where it finds a
  signal.
* iq — complex windows; the model's unit is the signal's RMS, so the SNR
  of the window (signal power over the noise power the floor says) sets t*.
  When it is not given it is estimated from the window's power and the
  floor — noisy at low SNR, which is why a caller who knows it should pass
  it.

EVERYTHING OUT OF HERE IS INVENTED TIER (`provenance.tier_for(
"diffusion_denoise")`). Plan §1: *"a model good at making signals out of
noise will sometimes make a signal out of nothing … here it is a false
contact."* The card carries the hallucination rate measured at training
time; every result carries the tier, the model's hash and the "denoised"
flag a Detection from this path must wear (`detect.boxes.FLAGS`). A tile's
absolute level and mean-pooled power stay the raw ones (`Tile.with_spec`):
a measurement is never made on a reconstruction.

LIMITS. A denoiser asked about an SNR below the lowest it was measured at is
clamped to that step and says so — below it the output is mostly the
model's prior. The spectrogram geometry (FFT size, hop, window, frames per
row, pooling) must be the one the model was trained on; a different one is
refused, not adapted.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field

import numpy as np

from atk_diffusion import cards, provenance
from atk_diffusion.learn import diffusion as _diff

DOMAINS = ("spectrogram", "iq")
METHOD = "diffusion_denoise"
TIER = provenance.tier_for(METHOD)


class DenoiserRefusal(cards.CardRefusal):
    """The denoiser may not run on this input; the message says why."""


# ---------------------------------------------------------------------------
# Representation math, shared with learn.denoiser (torch) — keep it here so
# both environments run the same lines
# ---------------------------------------------------------------------------
def pad_reflect(x: np.ndarray, m: int) -> tuple[np.ndarray, tuple]:
    """Reflect-pad the trailing spatial dims of [B, C, …] to a multiple of
    m. Returns (padded, original spatial shape)."""
    spatial = tuple(int(s) for s in x.shape[2:])
    pads = [(0, 0), (0, 0)] + [(0, (-s) % int(m)) for s in spatial]
    if not any(p[1] for p in pads):
        return x, spatial
    mode = "reflect" if all(s >= 2 for s in spatial) else "edge"
    return np.pad(x, pads, mode=mode), spatial


def crop(x: np.ndarray, spatial: tuple) -> np.ndarray:
    return x[(slice(None), slice(None)) + tuple(slice(0, s) for s in spatial)]


def noise_stats_db(noise_tiles_db) -> tuple[float, float]:
    """(mean, std) in dB of noise-only tile cells — the representation's
    noise, MEASURED on this receiver's own noise."""
    a = np.asarray(noise_tiles_db, dtype=np.float64)
    return float(np.mean(a)), float(np.std(a))


def window_snr(x, noise_power: float, floor: float = 1e-3) -> np.ndarray:
    """Signal power over noise power per window: mean|x|²/P_n − 1, never
    below `floor` (a noise-only window has no SNR; the caller is told)."""
    x = np.atleast_2d(np.asarray(x))
    p = np.mean(np.abs(x) ** 2, axis=-1) / float(noise_power)
    return np.maximum(p - 1.0, floor)


@dataclass
class Prepared:
    y_model: np.ndarray            # [B, C, …] padded, model units
    t: np.ndarray                  # [B] timestep per example
    snr_db: np.ndarray             # [B] the SNR each t came from
    c: np.ndarray                  # [B] model unit in receiver units
    spatial: tuple
    batched: bool
    clamped: np.ndarray            # [B] bool
    words: list = field(default_factory=list)


def prepare(x, inp: dict, alphas_cumprod, snr_db=None, noise_std=None,
            noise_mean_db=None, noise_power=None, t=None) -> Prepared:
    """Observation -> model units, timesteps, scales (see the module
    docstring). `inp` is the card's `input` block."""
    domain = inp.get("domain")
    norm = inp.get("normalize", {})
    ac = _diff._ac(alphas_cumprod)
    m = int(inp.get("multiple", 1))
    words: list[str] = []
    if domain == "spectrogram":
        a = np.asarray(x, dtype=np.float64)
        batched = a.ndim == 3
        a = a if batched else a[None]
        if a.ndim != 3:
            raise DenoiserRefusal("a spectrogram denoiser takes a tile [rows, "
                                  "bins] (or a batch [B, rows, bins]) in dB "
                                  "above the floor")
        mu = float(norm["noise_mean_db"] if noise_mean_db is None else noise_mean_db)
        sd = float(norm["noise_std_db"] if noise_std is None else noise_std)
        unit = float(norm["unit_db"])
        if sd <= 0:
            raise DenoiserRefusal("the noise spread must be measured (positive)")
        snr = (unit / sd) ** 2 if snr_db is None else 10 ** (float(snr_db) / 10.0)
        b = a.shape[0]
        snr_v = np.full(b, snr)
        c = sd * np.sqrt(snr_v)
        y = ((a - mu) / c[:, None, None])[:, None]
    elif domain == "iq":
        z = np.asarray(x)
        if not np.iscomplexobj(z):
            raise DenoiserRefusal("an IQ denoiser takes complex samples")
        batched = z.ndim == 2
        z = z if batched else z[None]
        if noise_std is not None:
            sd = float(noise_std)
            p_n = 2.0 * sd * sd
        elif noise_power is not None:
            p_n = float(noise_power)
            sd = math.sqrt(p_n / 2.0)
        else:
            raise DenoiserRefusal("an IQ denoise needs the noise level: pass "
                                  "noise_power (per complex sample) or "
                                  "noise_std (per real element) from the floor")
        if snr_db is None:
            snr_v = window_snr(z, p_n)
            words.append("SNR estimated from the window's power and the floor "
                         "(noisy at low SNR — pass snr_db when it is known)")
        else:
            snr_v = np.full(z.shape[0], 10 ** (float(snr_db) / 10.0))
        c = sd * np.sqrt(snr_v)
        y = np.stack([z.real, z.imag], axis=1).astype(np.float64) / c[:, None, None]
    else:
        raise DenoiserRefusal(f"unknown denoiser domain {domain!r}")
    if t is None:
        tt = np.asarray(_diff.snr_matched_timestep(snr_v, ac)).reshape(-1)
    else:
        tt = np.full(y.shape[0], int(t))
    t_max = int(inp.get("t_max", ac.shape[0] - 1))
    clamped = tt > t_max
    if np.any(clamped):
        words.append(f"{int(clamped.sum())} input(s) below the lowest SNR this "
                     f"denoiser was measured at ({inp.get('snr_db_min', '?')} "
                     f"dB): clamped to step {t_max}; the output there is "
                     "mostly the model's prior")
        tt = np.minimum(tt, t_max)
    y, spatial = pad_reflect(y.astype(np.float32), m)
    return Prepared(y, tt.astype(np.int64), 10 * np.log10(snr_v), c, spatial,
                    batched, clamped, words)


def finish(x0_model: np.ndarray, prep: Prepared, inp: dict) -> np.ndarray:
    x0 = crop(np.asarray(x0_model, dtype=np.float64), prep.spatial)
    if inp.get("domain") == "spectrogram":
        out = (x0[:, 0] * prep.c[:, None, None]).astype(np.float32)
    else:
        out = ((x0[:, 0] + 1j * x0[:, 1]) * prep.c[:, None]).astype(np.complex64)
    return out if prep.batched else out[0]


def run(eps_fn, x, inp: dict, alphas_cumprod, snr_db=None, noise_std=None,
        noise_mean_db=None, noise_power=None, steps: int = 1, eta: float = 0.0,
        seed: int = 0, t=None):
    """Denoise with any eps_fn (ONNX here, torch in learn.denoiser).
    Returns (output in the input's units, info)."""
    t0 = time.perf_counter()
    prep = prepare(x, inp, alphas_cumprod, snr_db=snr_db, noise_std=noise_std,
                   noise_mean_db=noise_mean_db, noise_power=noise_power, t=t)
    ac = _diff._ac(alphas_cumprod)
    rng = np.random.default_rng(seed)
    x0 = np.empty_like(prep.y_model)
    for tv in np.unique(prep.t):
        sel = prep.t == tv
        xt = _diff.observation_to_xt(prep.y_model[sel], int(tv), ac)
        x0[sel] = _diff.ddim_sample(eps_fn, xt.astype(np.float32), int(tv), ac,
                                    steps=steps, eta=eta, rng=rng,
                                    x0_clip=inp.get("x0_clip"))
    out = finish(x0, prep, inp)
    info = {"method": METHOD, "tier": TIER, "flag": "denoised",
            "words": provenance.TIER_WORDS[TIER],
            "t": [int(v) for v in prep.t], "snr_db": [round(float(v), 2) for v in prep.snr_db],
            "steps": int(steps), "eta": float(eta), "clamped": bool(np.any(prep.clamped)),
            "notes": list(prep.words),
            "latency_ms": round(1000.0 * (time.perf_counter() - t0), 2)}
    return out, info


def geometry_words(stft: dict) -> str:
    return (f"{stft.get('fft_size')}-point {stft.get('window')} FFT, hop "
            f"{stft.get('hop')}, {stft.get('pool')} frame(s) {stft.get('pool_mode')}"
            "-pooled per row")


def check_geometry(card_stft: dict, fft_size: int, hop: int, window: str,
                   pool: int, pool_mode: str = "max", what: str = "this tile"):
    """Refuse, in words, a spectrogram of another geometry."""
    want = (int(card_stft["fft_size"]), int(card_stft["hop"]),
            str(card_stft["window"]).lower(), int(card_stft["pool"]),
            str(card_stft.get("pool_mode", "max")))
    got = (int(fft_size), int(hop), str(window).lower(), int(pool), str(pool_mode))
    if want != got:
        raise DenoiserRefusal(
            f"this denoiser was trained on tiles of {geometry_words(card_stft)}; "
            f"{what} is {got[0]}-point {got[2]}, hop {got[1]}, {got[3]} frame(s) "
            f"{got[4]}-pooled per row. A model never meets a spectrogram of a "
            "geometry it was not trained on (DETECTION_DESIGN §2).")


# ---------------------------------------------------------------------------
# The ONNX runner
# ---------------------------------------------------------------------------
class DenoiserRuntime:
    """A card-verified denoiser running on onnxruntime (CPU)."""

    def __init__(self, model_dir, profile: str, threads: int = 1):
        from atk_diffusion import capabilities
        self.card = cards.load(model_dir, expect_kind="denoiser",
                               for_profile=profile)
        ok, why = capabilities.can_infer()
        if not ok:
            raise RuntimeError(why)
        fmt = self.card.weights.get("format")
        if fmt != "onnx":
            raise DenoiserRefusal(f"{self.card.name}'s weights are {fmt!r}, not "
                                  "ONNX; the core environment runs ONNX only")
        import onnxruntime as ort
        so = ort.SessionOptions()
        so.intra_op_num_threads = max(1, int(threads))
        so.inter_op_num_threads = 1
        self.session = ort.InferenceSession(str(cards.weights_path(model_dir, self.card)),
                                            so, providers=["CPUExecutionProvider"])
        self.input = dict(self.card.input)
        self.schedule = _diff.Schedule.from_json(self.input["schedule"])
        self.ac = self.schedule.alphas_cumprod
        self.domain = self.input["domain"]
        self.profile = str(profile).lower()
        self.last: dict = {}

    def eps_fn(self, x, t):
        return self.session.run(["eps"], {"x": np.asarray(x, dtype=np.float32),
                                          "t": np.asarray(t, dtype=np.int64)})[0]

    @property
    def t_star(self) -> int | None:
        return self.input.get("t_star")

    def denoise(self, x, snr_db=None, noise_std=None, noise_mean_db=None,
                noise_power=None, steps: int = 1, eta: float = 0.0, seed: int = 0,
                t=None):
        out, info = run(self.eps_fn, x, self.input, self.ac, snr_db=snr_db,
                        noise_std=noise_std, noise_mean_db=noise_mean_db,
                        noise_power=noise_power, steps=steps, eta=eta, seed=seed, t=t)
        info.update(model=self.card.name,
                    model_sha256=self.card.weights.get("sha256", ""),
                    hallucination_rate=(self.card.metrics or {}).get("hallucination_rate"))
        self.last = info
        return out

    def denoise_tile(self, tile, noise_std=None, noise_mean_db=None,
                     steps: int = 1):
        """The pipeline's low-SNR path on a dsp.stft.Tile: geometry checked,
        `spec` replaced by the denoised one (absolute level and mean-pooled
        power stay RAW). Returns (tile, info); boxes found on it must carry
        the 'denoised' flag."""
        if self.domain != "spectrogram":
            raise DenoiserRefusal("this denoiser works on IQ, not on tiles")
        lay = tile.layout
        check_geometry(self.input["stft"], lay.fft_size, lay.hop, lay.window,
                       lay.pool, "max")
        rv = int(tile.rows_valid) if tile.rows_valid else tile.rows
        spec = np.array(tile.spec, dtype=np.float32, copy=True)
        if rv > 0:
            spec[:rv] = self.denoise(spec[:rv], noise_std=noise_std,
                                     noise_mean_db=noise_mean_db, steps=steps)
        return tile.with_spec(spec), dict(self.last)

    def summary(self) -> list[str]:
        return cards.summary(self.card)
