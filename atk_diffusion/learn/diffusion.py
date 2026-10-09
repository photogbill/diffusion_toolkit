# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""The diffusion core: noise schedules, the forward process, the
ε-prediction loss, the DDPM and DDIM samplers, RePaint inpainting,
conditional sampling, and the SNR <-> timestep map that turns a diffusion
model into a detector's front end (plan §1, §4.B3; ARCHITECTURE §4.4).

WHAT A DIFFUSION MODEL IS, FOR ATK. Plan §1: *"A diffusion model learns to
remove noise, step by step, from something corrupted."* The forward process
corrupts clean data x0 with Gaussian noise on a schedule,

    x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε,     ε ~ N(0, I),  t = 0 … T−1,

and a network ε_θ(x_t, t) learns to say which part of x_t is the noise.
Every tool in this package is that one network used differently: run it
once at the right t (the B3 denoiser), run it many times while pinning the
samples you know (RePaint, the D1 inpainter), give it another receiver's
capture as extra input channels (the A6 translator), or ask which class
label makes it best at its job (generative classification, plan §4.R).

WHY NUMPY AT MODULE SCOPE. ATK's core environment has numpy and onnxruntime
and no PyTorch (ARCHITECTURE §1). The schedule, the SNR <-> timestep map and
the DDIM / RePaint samplers are plain arithmetic, so they are written once,
here, to work on numpy arrays AND torch tensors alike: the training
environment samples with a torch model, `dsp.denoise_runtime` samples with
an ONNX model, and both run the same lines. Torch is imported only inside
the functions that train (`eps_loss`, `fit`, `torch_eps_fn`); importing this
module never imports torch.

THE HAZARD (plan §1, §2.1). *"A model good at making signals out of noise
will sometimes make a signal out of nothing … here it is a false contact."*
Nothing in this module decides that a sample is real. Every tool built on it
measures a hallucination rate on noise-only input and labels its output
INVENTED (`provenance.tier_for("diffusion_denoise")`).

CONVENTIONS.
* Timesteps are 0-based: t = 0 is the least noisy step, T−1 the most.
  ᾱ_{−1} := 1 is "the clean end", so a sampler's last step goes t -> −1.
* `alphas_cumprod` arguments accept a `Schedule` or the ᾱ array.
* An `eps_fn(x, t_vec)` takes x [B, C, …] and an int64 vector t_vec [B] in
  the same backend (numpy or torch) and returns ε̂ with x's shape.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np

SCHEDULES = ("linear", "cosine")


# ---------------------------------------------------------------------------
# Backend helpers — the samplers run on numpy arrays and torch tensors alike
# ---------------------------------------------------------------------------
def _is_torch(x) -> bool:
    return type(x).__module__.split(".")[0] == "torch"


def _torch_generator(rng):
    import torch
    if rng is None:
        return None
    if isinstance(rng, torch.Generator):
        return rng
    if isinstance(rng, (int, np.integer)):
        return torch.Generator().manual_seed(int(rng))
    raise TypeError("a torch sampler takes a torch.Generator or an int seed")


def _np_generator(rng) -> np.random.Generator:
    if isinstance(rng, np.random.Generator):
        return rng
    return np.random.default_rng(rng)


def randn_like(x, rng=None):
    """Standard normal noise shaped like `x`, in x's backend and dtype."""
    if _is_torch(x):
        import torch
        g = _torch_generator(rng)
        if g is not None and x.device.type != "cpu":
            return torch.randn(x.shape, generator=g, dtype=x.dtype).to(x.device)
        return torch.randn(x.shape, generator=g, dtype=x.dtype, device=x.device)
    a = np.asarray(x)
    dt = a.dtype if a.dtype.kind == "f" else np.float32
    return _np_generator(rng).standard_normal(a.shape).astype(dt, copy=False)


def _tvec(t: int, x):
    """An int64 vector [B] holding timestep t, in x's backend."""
    b = int(x.shape[0])
    if _is_torch(x):
        import torch
        return torch.full((b,), int(t), dtype=torch.long, device=x.device)
    return np.full((b,), int(t), dtype=np.int64)


def _to_numpy_int(t) -> np.ndarray:
    if _is_torch(t):
        return t.detach().cpu().numpy().astype(np.int64)
    return np.asarray(t, dtype=np.int64)


def _per_example(values: np.ndarray, x):
    """[B] float64 coefficients -> broadcastable over x [B, C, …], same
    backend and dtype as x."""
    shape = (values.shape[0],) + (1,) * (len(x.shape) - 1)
    if _is_torch(x):
        import torch
        return torch.as_tensor(values.reshape(shape), dtype=x.dtype,
                               device=x.device)
    return values.reshape(shape).astype(np.asarray(x).dtype, copy=False)


def _clip(x, c):
    if _is_torch(x):
        return x.clamp(-c, c)
    return np.clip(x, -c, c)


# ---------------------------------------------------------------------------
# Schedules
# ---------------------------------------------------------------------------
def linear_betas(T: int, beta_start: float = 1e-4,
                 beta_end: float = 0.02) -> np.ndarray:
    """Ho et al.'s linear schedule, scaled by 1000/T so a shorter chain
    reaches the same total noise (β_1 = 1e-4, β_T = 0.02 at T = 1000)."""
    T = int(T)
    if T < 2:
        raise ValueError("a schedule needs at least two steps")
    scale = 1000.0 / T
    b = np.linspace(scale * beta_start, scale * beta_end, T, dtype=np.float64)
    return np.clip(b, 1e-8, 0.999)


def cosine_betas(T: int, s: float = 0.008, max_beta: float = 0.999
                 ) -> np.ndarray:
    """Nichol & Dhariwal's cosine schedule: ᾱ(t) = f(t)/f(0) with
    f(t) = cos²(((t/T) + s)/(1 + s) · π/2). It spends more steps at the
    moderate noise levels where weak signals live, which is why it is the
    default here."""
    T = int(T)
    if T < 2:
        raise ValueError("a schedule needs at least two steps")
    k = np.arange(T + 1, dtype=np.float64)
    f = np.cos(((k / T) + s) / (1.0 + s) * math.pi / 2.0) ** 2
    abar = f / f[0]
    betas = 1.0 - abar[1:] / abar[:-1]
    return np.clip(betas, 1e-8, max_beta)


@dataclass
class Schedule:
    """A noise schedule. `betas` is the truth; everything else derives
    from it, so a schedule written into a model card and read back in the
    core environment is bit-identical (`to_json` stores the betas)."""
    kind: str
    betas: np.ndarray
    params: dict = field(default_factory=dict)

    @property
    def T(self) -> int:
        return int(self.betas.shape[0])

    @property
    def alphas(self) -> np.ndarray:
        return 1.0 - self.betas

    @property
    def alphas_cumprod(self) -> np.ndarray:
        return np.cumprod(1.0 - self.betas)

    def snr(self) -> np.ndarray:
        """The forward process's signal-to-noise ratio ᾱ_t / (1 − ᾱ_t)."""
        return forward_snr(self.alphas_cumprod)

    def to_json(self) -> dict:
        return {"kind": self.kind, "T": self.T, "params": dict(self.params),
                "betas": [float(b) for b in self.betas]}

    @classmethod
    def from_json(cls, d: dict) -> "Schedule":
        if d.get("betas"):
            b = np.asarray(d["betas"], dtype=np.float64)
            return cls(str(d.get("kind", "custom")), b, dict(d.get("params", {})))
        return make_schedule(d.get("kind", "cosine"), int(d["T"]),
                             **dict(d.get("params", {})))


def make_schedule(kind: str = "cosine", T: int = 1000, **params) -> Schedule:
    kind = str(kind).lower()
    if kind == "linear":
        return Schedule("linear", linear_betas(T, **params), dict(params))
    if kind == "cosine":
        return Schedule("cosine", cosine_betas(T, **params), dict(params))
    raise ValueError(f"unknown noise schedule {kind!r} — one of "
                     f"{', '.join(SCHEDULES)}")


def _ac(alphas_cumprod) -> np.ndarray:
    if isinstance(alphas_cumprod, Schedule):
        return alphas_cumprod.alphas_cumprod
    if _is_torch(alphas_cumprod):
        return alphas_cumprod.detach().cpu().numpy().astype(np.float64)
    return np.asarray(alphas_cumprod, dtype=np.float64)


def abar(alphas_cumprod, t: int) -> float:
    """ᾱ_t with ᾱ_{−1} = 1 (the clean end)."""
    ac = _ac(alphas_cumprod)
    t = int(t)
    if t < 0:
        return 1.0
    if t >= ac.shape[0]:
        raise ValueError(f"timestep {t} is past the end of a {ac.shape[0]}-step "
                         "schedule")
    return float(ac[t])


def _abar_vec(ac: np.ndarray, t) -> np.ndarray:
    t = _to_numpy_int(t).reshape(-1)
    out = np.ones(t.shape, dtype=np.float64)
    ok = t >= 0
    out[ok] = ac[np.minimum(t[ok], ac.shape[0] - 1)]
    return out


# ---------------------------------------------------------------------------
# SNR <-> timestep: the diffusion model as a detector's front end
# ---------------------------------------------------------------------------
def forward_snr(alphas_cumprod) -> np.ndarray:
    """ᾱ_t / (1 − ᾱ_t) — the signal-to-noise ratio of x_t when x0 has unit
    power, for every t."""
    ac = _ac(alphas_cumprod)
    return ac / (1.0 - ac)


def snr_matched_timestep(snr_linear, alphas_cumprod):
    """The timestep t* at which the forward process has the observation's
    SNR — "Erasing Noise in Signal Detection with Diffusion Model"
    (arXiv 2501.07030), the theory behind plan B3.

    DERIVATION, power normalisation included.

    The model was trained on x0 normalised to UNIT POWER per real element,
    E[x0²] = 1 (an IQ window is two real channels, I and Q; a spectrogram
    tile one). The forward process makes

        x_t = √ᾱ_t · x0 + √(1 − ᾱ_t) · ε,      ε ~ N(0, I),

    so the signal power in x_t is ᾱ_t and the noise power 1 − ᾱ_t: x_t has
    signal-to-noise ratio ᾱ_t / (1 − ᾱ_t) and, at every t, unit total power.

    The receiver gives y = s + n with n ~ N(0, σ²) per real element and
    SNR = P_s / σ², where P_s = E[s²]. Write s = c · x0 with c = √P_s (the
    signal's RMS: the model's unit of amplitude, in the receiver's units).
    In model units the observation is y/c = x0 + (σ/c) · n′, n′ ~ N(0, I).
    Ask for a scale k and a timestep t* that make k · y/c a draw of x_{t*}:

        signal part:  k · x0          = √ᾱ_{t*} · x0      =>  k = √ᾱ_{t*}
        noise part:   k · (σ/c) · n′  = √(1 − ᾱ_{t*}) · ε =>  ᾱ_{t*} σ²/c² = 1 − ᾱ_{t*}

    The second line is

        ᾱ_{t*} / (1 − ᾱ_{t*}) = c² / σ² = P_s / σ² = SNR.

    So t* is the step whose forward SNR equals the observation's, and the
    observation enters the network as

        x_{t*} = √ᾱ_{t*} · y / c        (the paper's √ᾱ rescale, `observation_to_xt`)
               = y · √(1 − ᾱ_{t*}) / σ  (the same thing, written with the noise
                                         level only — `noise_matched_scale`).

    Two consequences used throughout the package:

    * The rescaled observation has unit power, like every x_t the network
      saw in training: E[x_{t*}²] = ᾱ(P_s + σ²)/P_s = ᾱ(1 + 1/SNR) = 1.
    * One network evaluation is Tweedie's estimate. With ε̂ = ε_θ(x_{t*}, t*),
      x̂0 = (x_{t*} − √(1 − ᾱ) ε̂)/√ᾱ, and back in the receiver's units
      ŝ = c · x̂0 = y − σ · ε̂: the observation minus the predicted noise,
      scaled by the noise level. That is the B3 single-step denoise.

    A mismatched t is the wrong Wiener filter: at t < t* the network assumes
    less noise than there is and leaves some in; at t > t* it assumes more
    and erases signal — and invents structure in its place (the test at
    t*/2 and 2·t* checks both sides numerically).

    Discrete schedules: t* is the step nearest in log-SNR. SNR ≤ 0 (noise
    only, or an estimate that came out negative) maps to the last step,
    T − 1: pure generation, the hallucination regime — callers clamp and
    say so. Returns an int for a scalar, an int array for an array.
    """
    ac = _ac(alphas_cumprod)
    log_snr_t = np.log(ac) - np.log1p(-ac)          # decreasing in t
    snr = np.asarray(snr_linear, dtype=np.float64)
    if np.any(np.isnan(snr)):
        raise ValueError("the SNR is not a number; measure it before choosing "
                         "a timestep")
    flat = snr.reshape(-1)
    out = np.full(flat.shape, ac.shape[0] - 1, dtype=np.int64)
    pos = flat > 0
    if np.any(pos):
        target = np.log(flat[pos])
        out[pos] = np.argmin(np.abs(log_snr_t[None, :] - target[:, None]), axis=1)
    if snr.ndim == 0:
        return int(out[0])
    return out.reshape(snr.shape)


def timestep_snr_db(t: int, alphas_cumprod) -> float:
    """The forward SNR of step t in dB (the inverse of the map above)."""
    a = abar(alphas_cumprod, t)
    if a >= 1.0:
        return float("inf")
    return 10.0 * math.log10(a / (1.0 - a))


def observation_to_xt(y_model, t: int, alphas_cumprod):
    """The paper's rescale: x_t = √ᾱ_t · y, with y in MODEL units (the
    signal at unit power). Exact at t = t*(SNR); at any other t it is the
    deliberate mismatch the theory warns about."""
    return math.sqrt(abar(alphas_cumprod, t)) * y_model


def noise_matched_scale(noise_std: float, t: int, alphas_cumprod) -> float:
    """k such that k · y has the forward noise level of step t:
    k = √(1 − ᾱ_t) / σ. Equal to √ᾱ_t / c at t = t*(SNR)."""
    if not noise_std or noise_std <= 0:
        raise ValueError("the noise level must be measured (a positive "
                         "standard deviation) before a timestep can be matched")
    return math.sqrt(1.0 - abar(alphas_cumprod, t)) / float(noise_std)


def predict_x0(x_t, eps, t, alphas_cumprod):
    """x̂0 = (x_t − √(1 − ᾱ_t) ε̂) / √ᾱ_t. `t` an int or a [B] vector."""
    ac = _ac(alphas_cumprod)
    if np.ndim(_to_numpy_int(t)) == 0:
        a = abar(ac, int(_to_numpy_int(t)))
        return (x_t - math.sqrt(1.0 - a) * eps) / math.sqrt(a)
    a = _abar_vec(ac, t)
    return (x_t - _per_example(np.sqrt(1.0 - a), x_t) * eps) \
        / _per_example(np.sqrt(a), x_t)


def predict_eps(x_t, x0, t, alphas_cumprod):
    """ε = (x_t − √ᾱ_t x0) / √(1 − ᾱ_t)."""
    ac = _ac(alphas_cumprod)
    if np.ndim(_to_numpy_int(t)) == 0:
        a = abar(ac, int(_to_numpy_int(t)))
        return (x_t - math.sqrt(a) * x0) / math.sqrt(1.0 - a)
    a = _abar_vec(ac, t)
    return (x_t - _per_example(np.sqrt(a), x_t) * x0) \
        / _per_example(np.sqrt(1.0 - a), x_t)


def q_sample(x0, t, noise, alphas_cumprod):
    """The forward process: x_t = √ᾱ_t x0 + √(1 − ᾱ_t) ε. `t` is an int
    (one step for the batch) or a [B] vector (one per example)."""
    ac = _ac(alphas_cumprod)
    if np.ndim(_to_numpy_int(t)) == 0:
        a = abar(ac, int(_to_numpy_int(t)))
        return math.sqrt(a) * x0 + math.sqrt(1.0 - a) * noise
    a = _abar_vec(ac, t)
    return _per_example(np.sqrt(a), x0) * x0 \
        + _per_example(np.sqrt(1.0 - a), x0) * noise


# ---------------------------------------------------------------------------
# Samplers
# ---------------------------------------------------------------------------
def timesteps(t_start: int, steps: int) -> list[int]:
    """The descending timesteps a strided sampler visits from `t_start`,
    ending with −1 (the clean end). `steps` = network evaluations."""
    t_start = int(t_start)
    if t_start < 0:
        return [-1]
    steps = max(1, min(int(steps), t_start + 1))
    seq = np.round(np.linspace(t_start, 0, steps)).astype(np.int64)
    seq = sorted(set(seq.tolist()), reverse=True)
    return [int(v) for v in seq] + [-1]


def ddim_step(x_t, eps, t: int, t_prev: int, alphas_cumprod, eta: float = 0.0,
              rng=None, x0_clip: float | None = None):
    """One DDIM update t -> t_prev (Song et al.); η = 0 is deterministic,
    η = 1 matches DDPM's stochasticity. Returns (x_prev, x̂0)."""
    ac = _ac(alphas_cumprod)
    a_t, a_p = abar(ac, t), abar(ac, t_prev)
    x0 = (x_t - math.sqrt(1.0 - a_t) * eps) / math.sqrt(a_t)
    if x0_clip is not None:
        x0 = _clip(x0, float(x0_clip))
        eps = (x_t - math.sqrt(a_t) * x0) / math.sqrt(1.0 - a_t)
    if t_prev < 0:
        return x0, x0
    sigma = float(eta) * math.sqrt(max(0.0, (1.0 - a_p) / (1.0 - a_t))) \
        * math.sqrt(max(0.0, 1.0 - a_t / a_p))
    x_prev = math.sqrt(a_p) * x0 + math.sqrt(max(0.0, 1.0 - a_p - sigma ** 2)) * eps
    if sigma > 0:
        x_prev = x_prev + sigma * randn_like(x_t, rng)
    return x_prev, x0


def ddim_sample(eps_fn, x_start, t_start: int, alphas_cumprod, steps: int = 50,
                eta: float = 0.0, rng=None, x0_clip: float | None = None):
    """Strided DDIM from x_start (a draw of x_{t_start}) to x̂0. With
    steps = 1 it is exactly the single-step Tweedie estimate."""
    ac = _ac(alphas_cumprod)
    seq = timesteps(t_start, steps)
    x = x_start
    x0 = x
    for t, tp in zip(seq[:-1], seq[1:]):
        eps = eps_fn(x, _tvec(t, x))
        x, x0 = ddim_step(x, eps, t, tp, ac, eta=eta, rng=rng, x0_clip=x0_clip)
    return x0 if seq[:-1] else x


def ddpm_sample(eps_fn, x_start, t_start: int, alphas_cumprod, rng=None,
                x0_clip: float | None = None):
    """Ancestral DDPM (Ho et al.), every step from t_start to 0, with the
    posterior variance β̃_t = β_t (1 − ᾱ_{t−1}) / (1 − ᾱ_t)."""
    ac = _ac(alphas_cumprod)
    x = x_start
    for t in range(int(t_start), -1, -1):
        eps = eps_fn(x, _tvec(t, x))
        a_t, a_p = abar(ac, t), abar(ac, t - 1)
        beta = 1.0 - a_t / a_p
        if x0_clip is not None:
            x0 = _clip((x - math.sqrt(1.0 - a_t) * eps) / math.sqrt(a_t),
                       float(x0_clip))
            eps = (x - math.sqrt(a_t) * x0) / math.sqrt(1.0 - a_t)
        mean = (x - beta / math.sqrt(1.0 - a_t) * eps) / math.sqrt(1.0 - beta)
        if t > 0:
            var = beta * (1.0 - a_p) / (1.0 - a_t)
            x = mean + math.sqrt(var) * randn_like(x, rng)
        else:
            x = mean
    return x


def repaint(eps_fn, x_known, mask, alphas_cumprod, steps: int = 50,
            resample: int = 5, eta: float = 1.0, t_start: int | None = None,
            x_start=None, rng=None, x0_clip: float | None = None):
    """RePaint (Lugmayr et al., CVPR 2022) on a strided schedule: inpaint
    where `mask` is 0, keep x_known where it is 1.

    At every step the KNOWN region is replaced by the known samples noised
    to that step (a fresh forward draw), and the UNKNOWN region comes from
    the reverse step; then the pair is sent back one step with forward
    noise and denoised again `resample` times, so the fill has time to
    agree with what surrounds it. Network evaluations: steps × resample.

    x_known must be in model units; its values under mask = 0 are ignored.
    """
    ac = _ac(alphas_cumprod)
    T = ac.shape[0]
    t_start = T - 1 if t_start is None else int(t_start)
    seq = timesteps(t_start, steps)
    if x_start is None:
        x = randn_like(x_known, rng)
    else:
        x = x_start
    keep = mask
    resample = max(1, int(resample))
    for t, tp in zip(seq[:-1], seq[1:]):
        a_t, a_p = abar(ac, t), abar(ac, tp)
        for u in range(resample):
            eps = eps_fn(x, _tvec(t, x))
            x_unk, _x0 = ddim_step(x, eps, t, tp, ac, eta=eta, rng=rng,
                                   x0_clip=x0_clip)
            if tp >= 0:
                x_kn = math.sqrt(a_p) * x_known \
                    + math.sqrt(1.0 - a_p) * randn_like(x_known, rng)
            else:
                x_kn = x_known
            x_prev = keep * x_kn + (1 - keep) * x_unk
            if u < resample - 1 and tp >= 0:
                r = a_t / a_p
                x = math.sqrt(r) * x_prev + math.sqrt(1.0 - r) * randn_like(x_prev, rng)
            else:
                x = x_prev
                break
    return x


# ---------------------------------------------------------------------------
# The single-step and few-step denoise, in the receiver's units
# ---------------------------------------------------------------------------
def denoise(eps_fn, y_raw, noise_std: float, snr_linear: float,
            alphas_cumprod, steps: int = 1, t: int | None = None,
            eta: float = 0.0, rng=None, x0_clip: float | None = None):
    """Denoise an observation y = s + n given the noise level σ (per real
    element, receiver units) and the SNR P_s/σ². Returns (ŝ, t_used).

    t defaults to t*(SNR). The observation is put in model units with the
    signal's RMS c = σ·√SNR and rescaled by √ᾱ_t (`observation_to_xt`);
    `steps` > 1 continues with DDIM from there. With steps = 1 and t = t*,
    ŝ = y − σ·ε̂ exactly (see `snr_matched_timestep`)."""
    if snr_linear is None or not np.isfinite(snr_linear) or snr_linear <= 0:
        raise ValueError("the SNR must be a positive number to choose the "
                         "denoising step; noise-only input has no SNR and "
                         "would be pure generation")
    ac = _ac(alphas_cumprod)
    t_used = snr_matched_timestep(float(snr_linear), ac) if t is None else int(t)
    c = float(noise_std) * math.sqrt(float(snr_linear))
    x_t = observation_to_xt(y_raw / c, t_used, ac)
    x0 = ddim_sample(eps_fn, x_t, t_used, ac, steps=steps, eta=eta, rng=rng,
                     x0_clip=x0_clip)
    return c * x0, t_used


# ---------------------------------------------------------------------------
# Conditioning
# ---------------------------------------------------------------------------
def conditional(model_fn, cond=None, y=None):
    """An eps_fn for a model taking `cond` channels and/or a class label
    `y` as keywords (this package's U-Nets). The condition is fixed for the
    whole sampling run, as conditional sampling needs."""
    kw = {}
    if cond is not None:
        kw["cond"] = cond
    if y is not None:
        kw["y"] = y

    def fn(x, t):
        return model_fn(x, t, **kw)
    return fn


def concat_condition(flat_fn, cond):
    """An eps_fn for a model whose input is [x, cond] concatenated on the
    channel axis (how a conditional model is exported to ONNX: the input
    "x" carries the data channels first, the condition channels after)."""
    def fn(x, t):
        if _is_torch(x):
            import torch
            return flat_fn(torch.cat([x, cond], dim=1), t)
        return flat_fn(np.concatenate([x, cond], axis=1), t)
    return fn


# ---------------------------------------------------------------------------
# Training (torch only)
# ---------------------------------------------------------------------------
def torch_eps_fn(model, cond=None, y=None):
    """Wrap a torch model as an eps_fn, without gradients. Accepts numpy
    input too (converted to float32 tensors, returned as numpy), so the
    numpy samplers can drive a torch model."""
    import torch

    def fn(x, t):
        to_np = not _is_torch(x)
        xt = torch.as_tensor(np.asarray(x, dtype=np.float32)) if to_np else x
        tt = torch.as_tensor(_to_numpy_int(t)) if to_np else t
        kw = {}
        if cond is not None:
            kw["cond"] = torch.as_tensor(np.asarray(cond, dtype=np.float32)) \
                if not _is_torch(cond) else cond
        if y is not None:
            kw["y"] = torch.as_tensor(np.asarray(y, dtype=np.int64)) \
                if not _is_torch(y) else y
        with torch.no_grad():
            out = model(xt, tt, **kw)
        return out.cpu().numpy() if to_np else out
    return fn


def sample_t(batch: int, T: int, t_min=None, t_max=None, generator=None):
    """Uniform timesteps per example in [t_min, t_max) (torch long [B]).
    A per-example t_min is the ambient-diffusion rule: an example that
    already carries noise is only trained at steps noisier than itself."""
    import torch
    lo = torch.zeros(batch) if t_min is None else torch.as_tensor(t_min, dtype=torch.float32).expand(batch)
    hi = torch.full((batch,), float(T)) if t_max is None else torch.as_tensor(t_max, dtype=torch.float32).expand(batch)
    hi = torch.maximum(hi, lo + 1)
    u = torch.rand(batch, generator=generator)
    return torch.clamp((lo + (hi - lo) * u).floor().long(), 0, T - 1)


def eps_loss(model, x0, alphas_cumprod, t=None, noise=None, cond=None, y=None,
             t_min=None, t_max=None, generator=None):
    """The ε-prediction objective of Ho et al.: E‖ε − ε_θ(x_t, t)‖².

    `noise` may be given — the B3 denoiser passes the receiver's own noise,
    rescaled to the forward level of t*, so the single-step estimate is
    trained against the noise it will meet (see `learn.denoiser`)."""
    import torch
    ac = _ac(alphas_cumprod)
    b = x0.shape[0]
    if t is None:
        t = sample_t(b, ac.shape[0], t_min, t_max, generator)
    t = torch.as_tensor(t).to(x0.device)
    if noise is None:
        noise = torch.randn(x0.shape, generator=generator, dtype=x0.dtype)
    noise = noise.to(x0.device)
    x_t = q_sample(x0, t, noise, ac)
    kw = {}
    if cond is not None:
        kw["cond"] = cond
    if y is not None:
        kw["y"] = y
    pred = model(x_t, t, **kw)
    return torch.mean((pred - noise) ** 2)


def fit(model, batch_fn, alphas_cumprod, steps: int, lr: float = 2e-4,
        progress=None, generator=None, grad_clip: float = 1.0,
        ema_decay: float | None = None, weight_decay: float = 0.0,
        every: int = 50) -> dict:
    """A plain training loop for every diffusion tool here.

    batch_fn(step, generator) -> dict with x0 and optionally t, noise,
    cond, y, t_min — handed to `eps_loss`. Returns {"loss": [...],
    "ema": state_dict or None}. Progress lines are plain words."""
    import torch
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    ema = None
    if ema_decay:
        ema = {k: v.detach().clone() for k, v in model.state_dict().items()}
    losses = []
    model.train()
    for step in range(int(steps)):
        batch = batch_fn(step, generator)
        loss = eps_loss(model, batch["x0"], alphas_cumprod,
                        t=batch.get("t"), noise=batch.get("noise"),
                        cond=batch.get("cond"), y=batch.get("y"),
                        t_min=batch.get("t_min"), generator=generator)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        if grad_clip:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        if ema is not None:
            with torch.no_grad():
                for k, v in model.state_dict().items():
                    if v.dtype.is_floating_point:
                        ema[k].mul_(ema_decay).add_(v.detach(), alpha=1 - ema_decay)
                    else:
                        ema[k].copy_(v)
        losses.append(float(loss.detach()))
        if progress and (step % max(1, every) == 0 or step == steps - 1):
            progress(f"training step {step + 1} of {steps}: loss {losses[-1]:.4f}")
    model.eval()
    if ema is not None:
        model.load_state_dict(ema)
    return {"loss": losses, "ema": bool(ema is not None)}
