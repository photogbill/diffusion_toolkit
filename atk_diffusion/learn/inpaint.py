# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""D1 — IQ dropout repair before DSD, the learned half (plan §4.D1, §6
Phase 3; ARCHITECTURE §4.5).

Plan D1: *"USB glitches break decoder sync and the voice is lost. Inpaint
the gap in the IQ before the decoder sees it; the repaired span is marked in
the processing mark."* The classical half — finding the dropouts, restoring
the timeline, linear / AR / Janssen fills, the SigMF annotations — is
`repair.iq_dropout` (another engineer's). This module is the model its
`learned` method calls:

    inpaint(x, mask, fs, model_dir=…)  ->  y   (only masked samples change)

HOW. A diffusion model of short IQ windows of THIS profile's voice-class
cut (canonical rate, e.g. 48 kS/s for the RTL-SDR at 2.4 MS/s), trained on
DMR-like TDMA traffic in this receiver's noise, run with RePaint
(`learn.diffusion.repaint`): at every step the samples we have are pinned
(noised to that step) and the gap is generated to agree with them, with
resampling so the fill has time to fit both edges. A gap longer than the
window is filled in chained windows, each conditioned on the record and on
the fill before it — said in the result, because then later fill rests on
earlier fill.

THE SCALE IS THE FLOOR, NOT THE WINDOW. Each window is divided by the
receiver's noise level (σ per real element, times a fixed gain), never by
its own power — so a silent stretch stays silent-sized to the model, and
"fill silence with silence" is learnable. The noise level comes from the
caller (the cut's measured floor), or is measured from the capture
(`repair.iq_dropout.noise_floor_power`: the quietest 5 % of its
time–frequency cells; a local equivalent when that module is absent).

WHAT IT CAN AND CANNOT DO. Random payload symbols inside a gap are gone —
no fill recovers information the receiver never delivered, and the eval
shows every method at chance there. What a learned fill CAN give: a
continuation that does not upset the decoder's level and sync tracking
after the gap (a straight-line fill through the origin makes a phase jump
the FM discriminator turns into a frequency spike), and, where the gap
covers fixed structure — DMR's 48-bit sync word — the structure itself.
`experiments.inpaint_eval` measures both against interpolation and AR, and
the HALLUCINATION RATE: gaps in silence filled with signal. Every fill is
INVENTED tier (`provenance.tier_for("diffusion_inpaint")`); a decode across
it is a decode of a guess (`provenance.decoded_from_note`).

THE MODEM HERE. `fsk4_modulate` / `tdma_stream` make DMR-like traffic with
known symbols (4800 sym/s, ±648 / ±1944 Hz, raised-cosine frequency pulse,
the BS voice sync word in the middle of each 27.5 ms burst, one slot in two
active); `fsk4_demod` is a minimal channel filter + discriminator + slicer
with DSD-style level tracking (thresholds from the averaged peaks of the last
48 symbols), so a fill that throws the discriminator off shows up as errors
AFTER the gap, which is how a USB glitch loses voice. It is not DSD: no FEC, no
CACH, known symbol timing — stated wherever it is used.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards, profiles, provenance
from atk_diffusion.learn import diffusion as _diff

METHOD = "diffusion_inpaint"
TIER = provenance.tier_for(METHOD)

# ---------------------------------------------------------------------------
# A DMR-like modem with known symbols
# ---------------------------------------------------------------------------
DMR_SYNC_BS_VOICE = "755FD7DF75F7"
BAUD = 4800.0
DEV_HZ = 648.0                      # ±1 -> ±648 Hz, ±3 -> ±1944 Hz
LEVELS = np.array([-3, -1, 1, 3])
BURST_SYMBOLS = 132                 # 27.5 ms at 4800 sym/s
SYNC_AT = 54                        # 54 payload dibits, 24 sync, 54 payload
GUARD_SYMBOLS = 4                   # each side: carries the power ramp, so the
                                    # channel filter's transient (±3 symbols)
                                    # never lands on a counted symbol
NOMINAL_BW_HZ = 7600.0


def sync_symbols(hexstr: str = DMR_SYNC_BS_VOICE) -> np.ndarray:
    """DMR dibit mapping: 01 -> +3, 00 -> +1, 10 -> -1, 11 -> -3."""
    bits = np.array([int(b) for b in bin(int(hexstr, 16))[2:].zfill(4 * len(hexstr))])
    d = bits[0::2] * 2 + bits[1::2]
    return np.array([+1, +3, -1, -3])[d]


def burst_symbols(rng) -> np.ndarray:
    pay = LEVELS[rng.integers(0, 4, BURST_SYMBOLS - 24)]
    return np.concatenate([pay[:SYNC_AT], sync_symbols(), pay[SYNC_AT:]])


def _rc_pulse(sps: int, beta: float = 0.2, span: int = 8) -> np.ndarray:
    t = np.arange(-span * sps, span * sps + 1) / float(sps)
    with np.errstate(divide="ignore", invalid="ignore"):
        h = np.sinc(t) * np.cos(np.pi * beta * t) / (1 - (2 * beta * t) ** 2)
    sing = np.isclose(np.abs(t), 1 / (2 * beta))
    h[sing] = np.pi / 4 * np.sinc(1 / (2 * beta))
    return h


def fsk4_modulate(symbols, fs: float, baud: float = BAUD, dev_hz: float = DEV_HZ,
                  beta: float = 0.2) -> np.ndarray:
    """Continuous-phase 4FSK: the frequency follows a raised-cosine-shaped
    symbol train (zero ISI at symbol centres), so the discriminator reads
    exactly level × dev_hz at each centre. Needs an integer number of
    samples per symbol."""
    sps = fs / baud
    if abs(sps - round(sps)) > 1e-9:
        raise ValueError(f"{fs:g} S/s is not a whole number of samples per "
                         f"{baud:g} sym/s symbol; cut to a canonical rate first")
    sps = int(round(sps))
    up = np.zeros(len(symbols) * sps)
    up[sps // 2::sps] = np.asarray(symbols, dtype=np.float64) * dev_hz
    f = np.convolve(up, _rc_pulse(sps, beta), mode="full")
    lead = (len(_rc_pulse(sps, beta)) - 1) // 2
    f = f[lead: lead + up.size]
    return np.exp(2j * np.pi * np.cumsum(f) / fs).astype(np.complex64)


def tdma_stream(n_bursts: int, fs: float, rng, period_s: float = 0.06,
                lead_s: float = 0.01) -> tuple[np.ndarray, list[dict]]:
    """One active TDMA slot: a 27.5 ms burst (plus guard symbols carrying
    the power ramps) every `period_s`, silence between. Returns (clean IQ,
    bursts [{start, end, symbols}]) — `start` is the first counted symbol's
    first sample, `end` one past the last on-air sample."""
    sps = int(round(fs / BAUD))
    g = GUARD_SYMBOLS
    n_air = (BURST_SYMBOLS + 2 * g) * sps
    per = int(round(period_s * fs))
    if per < n_air:
        raise ValueError("the TDMA period is shorter than a burst")
    lead = int(round(lead_s * fs))
    x = np.zeros(lead + n_bursts * per, dtype=np.complex64)
    bursts = []
    for k in range(int(n_bursts)):
        sym = burst_symbols(rng)
        guard = np.array([1, -1] * g)[:g]
        s0 = lead + k * per
        b = fsk4_modulate(np.concatenate([guard, sym, guard]), fs)
        ramp = np.ones(n_air)
        r = sps
        ramp[:r] = 0.5 - 0.5 * np.cos(np.pi * np.arange(r) / r)
        ramp[-r:] = ramp[:r][::-1]
        x[s0:s0 + n_air] = b * ramp * np.exp(1j * rng.uniform(0, 6.28))
        bursts.append({"start": s0 + g * sps, "end": s0 + n_air, "symbols": sym})
    return x, bursts


def channel_filter(x, fs: float, bw_hz: float = 12500.0, taps: int = 63) -> np.ndarray:
    """The receiver's channel filter (12.5 kHz for DMR): linear phase,
    delay-compensated. The FM discriminator after it works at the channel's
    carrier-to-noise ratio, not the whole band's."""
    from scipy.signal import firwin
    h = firwin(int(taps) | 1, min(0.49 * fs, bw_hz / 2.0), fs=fs)
    return np.convolve(np.asarray(x, dtype=np.complex128), h, mode="full")[
        (h.size - 1) // 2: (h.size - 1) // 2 + np.asarray(x).size]


def fsk4_demod(x, fs: float, start: int, nsym: int, track: int = 48,
               baud: float = BAUD, dev_hz: float = DEV_HZ,
               filtered: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Minimal channel filter + discriminator + slicer with DSD-style level
    tracking: each symbol's value is the discriminator averaged over the
    middle 70 % of the symbol; its thresholds come from the averaged peaks
    of the previous `track` values (mean of the highest and lowest eighth —
    a single wild value still drags them, as it drags DSD's), the nominal
    ±3·dev before that. Known symbol timing. Measured (seeded, 16 bursts):
    no errors on clean traffic; about 4 % / 1 % symbol errors at 15 / 20 dB
    in-band SNR. Returns (decided symbols ±1/±3, values in Hz)."""
    x = np.asarray(x, dtype=np.complex128)
    if not filtered:
        x = channel_filter(x, fs)
    sps = int(round(fs / baud))
    d = np.angle(x[1:] * np.conj(x[:-1])) * fs / (2 * np.pi)
    d = np.concatenate([d[:1], d])
    h = max(1, int(round(0.35 * sps)))
    centres = start + sps // 2 + sps * np.arange(int(nsym))
    v = np.array([float(np.mean(d[max(0, c - h): c + h + 1])) if c < d.size else 0.0
                  for c in centres])
    out = np.empty(v.size, dtype=np.int64)
    q = max(1, int(track) // 8)
    for k in range(v.size):
        if k < track:
            hi, lo = 3 * dev_hz, -3 * dev_hz
        else:
            w = np.sort(v[k - track:k])
            hi, lo = float(np.mean(w[-q:])), float(np.mean(w[:q]))
        c = 0.5 * (hi + lo)
        step = (hi - lo) / 3.0
        if v[k] > c + step:
            out[k] = 3
        elif v[k] > c:
            out[k] = 1
        elif v[k] > c - step:
            out[k] = -1
        else:
            out[k] = -3
    return out, v


def noise_power(x) -> float:
    """The receiver's noise power per complex sample, measured from the
    signal's own quietest cells (`repair.iq_dropout.noise_floor_power` when
    installed; the same estimator otherwise)."""
    try:
        from atk_diffusion.repair import iq_dropout as _iqd
        return float(_iqd.noise_floor_power(x))
    except ImportError:
        x = np.asarray(x)
        nfft = int(min(256, max(16, x.size // 16)))
        frames = x.size // nfft
        if frames < 1:
            return float(np.mean(np.abs(x) ** 2))
        w = np.hanning(nfft + 2)[1:-1]
        X = np.fft.fft(x[: frames * nfft].reshape(frames, nfft) * w, axis=1)
        P = (np.abs(X) ** 2).ravel() / float(np.sum(w ** 2))
        P = P[P > 0]
        return float(np.percentile(P, 5) / -np.log(0.95)) if P.size else 0.0


def add_noise(x, snr_db: float, rng, bw_hz: float = NOMINAL_BW_HZ, fs: float = 48000.0,
              p_n: float | None = None) -> tuple[np.ndarray, float]:
    """Complex white noise for an IN-BAND SNR (signal power while on over
    the noise in `bw_hz`). Returns (noisy, noise power per sample)."""
    if p_n is None:
        p_n = 1.0 / (10 ** (snr_db / 10.0) * bw_hz / fs)
    w = math.sqrt(p_n / 2.0) * (rng.standard_normal(x.size) + 1j * rng.standard_normal(x.size))
    return (x + w).astype(np.complex64), float(p_n)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _canonical(prof, canonical: str | None, rate: float | None):
    if rate is not None:
        r = float(rate)
        dec = prof.sample_rate / r
        if abs(dec - round(dec)) > 1e-9:
            raise ValueError(f"{r:g} S/s is not an integer decimation of "
                             f"{profiles.describe(prof.id)}")
        return r, int(round(dec)), ""
    can = {c.cls: c for c in prof.canonical_rates()}.get(canonical or "voice")
    if can is None:
        raise ValueError(f"{profiles.describe(prof.id)} has no {canonical} canonical rate")
    return float(can.rate), int(can.decimation), can.cls


def make_windows(n: int, L: int, fs: float, rng, snr_db=(5.0, 25.0),
                 p_n: float = 1.0) -> np.ndarray:
    """`n` windows of DMR-like TDMA traffic in noise of power `p_n` (bursts,
    silences and edges as they fall), at random in-band SNRs."""
    out = np.empty((int(n), int(L)), dtype=np.complex64)
    per = int(round(0.06 * fs))
    for i in range(int(n)):
        x, _b = tdma_stream(1 + int(math.ceil(L / per)) + 1, fs, rng)
        snr = float(rng.uniform(*snr_db))
        a = math.sqrt(10 ** (snr / 10.0) * p_n * NOMINAL_BW_HZ / fs)
        y, _ = add_noise(a * x, 0.0, rng, fs=fs, p_n=p_n)
        s = int(rng.integers(0, y.size - L + 1))
        out[i] = y[s:s + L]
    return out


def train_inpainter(rf, profile: str, *, name: str | None = None,
                    canonical: str = "voice", rate: float | None = None,
                    window: int = 256, synthetic: int = 4000, snr_db=(5.0, 25.0),
                    gain_db: float = 15.0, schedule: str = "cosine", T: int = 1000,
                    unet: dict | None = None, steps: int = 20000, batch: int = 32,
                    lr: float = 2e-4, repaint_steps: int = 50, resample: int = 5,
                    seed: int = 0, device: str | None = None,
                    validation: int = 32, overwrite: bool = False,
                    progress=None) -> Path:
    """Train the D1 inpainter at one of the profile's canonical rates (voice
    by default — where DSD reads). Windows are scaled by the noise floor:
    x_model = x / (σ·g) with σ the noise per real element and
    g = √(1 + 10^(gain_db/10)), so noise alone is small and a signal near
    `gain_db` in-band is about unit power. Saved as ONNX (runs in ATK's core
    environment on onnxruntime) and as PyTorch weights. Card kind
    "inpainter", tier INVENTED; the hallucination rate on silent gaps and
    the gap SNR against linear interpolation are measured into it."""
    import torch
    from atk_diffusion.learn import unet as _u
    say = progress or (lambda s: None)
    prof = profiles.load_profile(rf, profile)
    fs, dec, cls = _canonical(prof, canonical, rate)
    if abs(fs / BAUD - round(fs / BAUD)) > 1e-9:
        raise ValueError(f"the inpainter's DMR-like training traffic needs a whole "
                         f"number of samples per 4800 sym/s symbol; {fs:g} S/s has "
                         f"{fs / BAUD:.3f}")
    rng = np.random.default_rng(seed)
    torch.manual_seed(int(seed))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    conf = dict(unet or _u.SMALL_1D)
    conf.update(dims=1, in_ch=2)
    L = int(window)
    if L % _u.multiple(conf):
        raise ValueError(f"the window {L} must be a multiple of {_u.multiple(conf)}")
    p_n = 1.0
    sigma = math.sqrt(p_n / 2.0)
    g = math.sqrt(1.0 + 10 ** (gain_db / 10.0))
    W = make_windows(synthetic, L, fs, rng, snr_db, p_n)
    X0n = np.stack([W.real, W.imag], axis=1) / (sigma * g)
    # the sampler's safety rail: x̂0 is clipped to 1.5x the largest training
    # value, so a poorly trained step cannot blow a fill up without bound
    x0_clip = float(1.5 * np.max(np.abs(X0n)))
    X0 = torch.as_tensor(X0n, dtype=torch.float32)
    sched = _diff.make_schedule(schedule, T)
    model = _u.build_unet(conf).to(dev)
    gen = torch.Generator().manual_seed(int(seed))

    def batch_fn(step, gg):
        i = torch.randint(0, X0.shape[0], (int(batch),), generator=gg)
        return {"x0": X0[i].to(dev)}
    say(f"training the D1 inpainter: {synthetic} windows of {L} at {fs:g} S/s")
    hist = _diff.fit(model, batch_fn, sched.alphas_cumprod, steps, lr=lr,
                     progress=say, generator=gen)
    model = model.to("cpu").eval()
    name = name or f"inpainter_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = rf.models(prof.id, name)
    if d.exists() and any(d.iterdir()) and not overwrite:
        raise FileExistsError(f"a model named {name} already exists for this profile")
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    _u.export_onnx(model, d / "model.onnx", (L,))
    card = cards.new_card(
        name, "inpainter", prof.id,
        input={"domain": "iq", "window": L, "rate": fs, "decimation": dec,
               "canonical_class": cls, "schedule": sched.to_json(), "unet": model.config,
               "normalize": {"kind": "noise_floor", "gain": g, "gain_db": gain_db},
               "repaint": {"steps": int(repaint_steps), "resample": int(resample)},
               "x0_clip": x0_clip,
               "torch_weights": {"file": "model.pt",
                                 "sha256": provenance.sha256_path(d / "model.pt")},
               "onnx": {"inputs": ["x", "t"], "outputs": ["eps"], "opset": 17},
               "traffic": "DMR-like 4FSK TDMA, BS voice sync, one slot active"},
        classes=[{"name": "dmr", "source": "synthetic", "examples": int(synthetic)}],
        datasets=[{"name": "synthetic DMR-like windows", "kind": "synthetic",
                   "n": int(synthetic), "sha256": provenance.sha256_bytes(W.tobytes())}],
        metrics={"final_loss": float(np.mean(hist["loss"][-50:])) if hist["loss"] else None,
                 "steps": int(steps)},
        license="all rights reserved", trained_on=f"{dev} (torch {torch.__version__})",
        notes=["fills are INVENTED tier; a decode across a fill is a decode of a guess",
               "experiments.inpaint_eval is the judge (SER, gap SNR, hallucination)"])
    cards.save(d, card, "model.onnx")
    inp = Inpainter.load(d, for_profile=prof.id)
    m = measure(inp, fs, L, rng=np.random.default_rng(seed + 1), n=validation)
    card.metrics.update(m)
    cards.save(d, card, "model.onnx")
    for f in ("model.onnx", "model.pt", "card.json"):
        try:
            rf.record(d / f, "model", f"inpainter {name}")
        except Exception:                                  # noqa: BLE001
            pass
    say(f"done: hallucination {m['hallucination_rate']}, gap SNR "
        f"{m['gap_snr_db']['diffusion']:.1f} dB (linear {m['gap_snr_db']['linear']:.1f} dB)")
    return d


#: Plan §2.1 for a fill: in a gap where the truth was ONLY NOISE, a fill
#: carrying this much more power than the noise that was really there has
#: put a signal where there was none. Judged against the truth (known in
#: synthetic trials), not against a noise-only quantile: a straight line
#: through two loud noise samples is noise-derived and must not count.
HALLUCINATION_MARGIN_DB = 6.0
HALLUCINATION_RULE = ("in a gap where the truth was only noise, the fill carries "
                      f"≥ {HALLUCINATION_MARGIN_DB:g} dB more power than that noise")


def hallucinated(fill, truth, margin_db: float = HALLUCINATION_MARGIN_DB) -> tuple[bool, float]:
    """(flag, excess dB) for a fill of a noise-only gap."""
    pf = float(np.mean(np.abs(np.asarray(fill)) ** 2))
    pt = float(np.mean(np.abs(np.asarray(truth)) ** 2))
    ex = 10.0 * math.log10(max(pf, 1e-30) / max(pt, 1e-30))
    return ex >= float(margin_db), ex


def measure(inp: "Inpainter", fs: float, L: int, rng, n: int = 32,
            gap: int | None = None) -> dict:
    """Card numbers: the hallucination rate (`HALLUCINATION_RULE`) and the
    waveform SNR in gaps inside bursts, beside straight-line interpolation."""
    gap = int(gap or max(8, L // 8))
    p_n = 1.0
    hall = elig = 0
    excess = []
    snr_d, snr_l = [], []
    for i in range(int(n)):
        if i % 2 == 0:                                   # silence
            y, _ = add_noise(np.zeros(L, dtype=np.complex64), 0.0, rng, fs=fs, p_n=p_n)
        else:
            x, b = tdma_stream(1, fs, rng, period_s=max(0.06, 2 * L / fs), lead_s=0.0)
            a = math.sqrt(10 ** (15 / 10.0) * p_n * NOMINAL_BW_HZ / fs)
            s0 = b[0]["start"]                              # inside the burst
            y, _ = add_noise(a * x[s0:s0 + L], 0.0, rng, fs=fs, p_n=p_n)
        s = int(rng.integers(L // 4, 3 * L // 4 - gap))
        mask = np.zeros(L, dtype=bool)
        mask[s:s + gap] = True
        filled, _info = inp.inpaint(y, mask, noise_power=p_n, seed=int(rng.integers(0, 1 << 30)))
        if i % 2 == 0:
            elig += 1
            flag, ex = hallucinated(filled[mask], y[mask])
            hall += int(flag)
            excess.append(ex)
        else:
            truth = y[mask]
            lin = np.linspace(y[s - 1], y[s + gap], gap + 2)[1:-1]
            e = float(np.sum(np.abs(truth) ** 2))
            snr_d.append(10 * math.log10(e / max(float(np.sum(np.abs(filled[mask] - truth) ** 2)), 1e-30)))
            snr_l.append(10 * math.log10(e / max(float(np.sum(np.abs(lin - truth) ** 2)), 1e-30)))
    return {"hallucination_rate": (hall / elig) if elig else None,
            "hallucination": {"eligible": elig, "hallucinated": hall, "gap": gap,
                              "rule": HALLUCINATION_RULE,
                              "mean_excess_db": float(np.mean(excess)) if excess else None},
            "gap_snr_db": {"diffusion": float(np.mean(snr_d)) if snr_d else None,
                           "linear": float(np.mean(snr_l)) if snr_l else None}}


# ---------------------------------------------------------------------------
# The inpainter
# ---------------------------------------------------------------------------
class Inpainter:
    """A trained D1 inpainter, loaded through its card. Runs on
    onnxruntime when it is installed (ATK's core environment — no PyTorch
    needed), on the PyTorch weights otherwise."""

    def __init__(self, eps_fn, card, model_dir, backend: str):
        self.card, self.dir, self.backend = card, Path(model_dir), backend
        self.input = dict(card.input)
        self.L = int(self.input["window"])
        self.rate = float(self.input["rate"])
        self.g = float(self.input["normalize"]["gain"])
        self.ac = _diff.Schedule.from_json(self.input["schedule"]).alphas_cumprod
        self._eps = eps_fn

    @classmethod
    def load(cls, model_dir, for_profile: str | None = None, backend: str = "auto"):
        card = cards.load(model_dir, expect_kind="inpainter", for_profile=for_profile)
        if backend in ("auto", "onnx") and card.weights.get("format") == "onnx":
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                sess = ort.InferenceSession(str(cards.weights_path(model_dir, card)), so,
                                            providers=["CPUExecutionProvider"])

                def eps_fn(x, t):
                    return sess.run(["eps"], {"x": np.asarray(x, dtype=np.float32),
                                              "t": np.asarray(t, dtype=np.int64)})[0]
                return cls(eps_fn, card, model_dir, "onnxruntime")
            except ImportError:
                if backend == "onnx":
                    raise RuntimeError("onnxruntime is not installed, so the ONNX "
                                       "inpainter cannot run here") from None
        import torch
        from atk_diffusion.learn import unet as _u
        tw = card.input.get("torch_weights") or {"file": card.weights.get("file")}
        p = Path(model_dir) / str(tw["file"])
        if tw.get("sha256") and provenance.sha256_path(p) != tw["sha256"]:
            raise cards.CardRefusal(f"{p.name} is not the file the card describes "
                                    "(its hash changed). Retrain or restore it.")
        model = _u.build_unet(card.input["unet"])
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        return cls(_diff.torch_eps_fn(model.eval()), card, model_dir, "torch")

    def _windows(self, mask: np.ndarray) -> list[int]:
        """Window starts for one round: one window per remaining gap, the
        gap's first sample a quarter-window in, never overlapping."""
        n, L = mask.size, self.L
        starts, last_end = [], -1
        i = 0
        while i < n:
            if not mask[i]:
                i += 1
                continue
            s = int(np.clip(i - L // 4, 0, max(0, n - L)))
            if s < last_end:
                break
            starts.append(s)
            last_end = s + L
            i = s + L
        return starts

    def inpaint(self, x, mask, noise_power: float | None = None, steps: int | None = None,
                resample: int | None = None, seed: int = 0):
        """Fill x where mask is True. Returns (y, info): only masked samples
        differ from x; info carries the tier, the method, how many windows
        and whether any fill was conditioned on earlier fill."""
        x = np.asarray(x, dtype=np.complex64)
        mask = np.asarray(mask, dtype=bool)
        if x.ndim != 1 or mask.shape != x.shape:
            raise ValueError("inpaint takes one complex stream and a mask of the same length")
        if x.size < self.L:
            raise ValueError(f"the stream ({x.size} samples) is shorter than the "
                             f"inpainter's {self.L}-sample window")
        known = x[~mask]
        p_n = float(noise_power) if noise_power else noise_power_of(known)
        if p_n <= 0:
            raise ValueError("the noise floor could not be measured (the stream has "
                             "no noise-only cells); pass noise_power from the cut's floor")
        scale = math.sqrt(p_n / 2.0) * self.g
        rp = self.input.get("repaint", {})
        steps = int(steps or rp.get("steps", 50))
        resample = int(resample or rp.get("resample", 5))
        rng = np.random.default_rng(seed)
        y = x.copy()
        y[mask] = 0
        todo = mask.copy()
        rounds = windows = 0
        chained = False
        t0 = time.perf_counter()
        while todo.any():
            starts = self._windows(todo)
            if not starts:
                break
            batch_x = np.stack([y[s:s + self.L] for s in starts])
            batch_m = np.stack([todo[s:s + self.L] for s in starts])
            if rounds > 0 and np.any((batch_m == False) & np.stack(  # noqa: E712
                    [mask[s:s + self.L] for s in starts])):
                chained = True
            xk = np.stack([batch_x.real, batch_x.imag], axis=1) / scale
            keep = (~batch_m)[:, None, :].astype(np.float32)
            out = _diff.repaint(self._eps, xk.astype(np.float32) * keep, keep, self.ac,
                                steps=steps, resample=resample, rng=rng,
                                x0_clip=self.input.get("x0_clip"))
            fill = (out[:, 0] + 1j * out[:, 1]) * scale
            for j, s in enumerate(starts):
                m = batch_m[j]
                seg = y[s:s + self.L]
                seg[m] = fill[j][m]
                y[s:s + self.L] = seg
                todo[s:s + self.L] = False
            rounds += 1
            windows += len(starts)
        info = {"method": METHOD, "tier": TIER, "words": provenance.TIER_WORDS[TIER],
                "model": self.card.name, "model_sha256": self.card.weights.get("sha256", ""),
                "windows": windows, "rounds": rounds, "noise_power": p_n,
                "chained": chained, "repaint": {"steps": steps, "resample": resample},
                "latency_ms": round(1000 * (time.perf_counter() - t0), 1)}
        if chained:
            info["note"] = ("a gap longer than the model's window was filled in "
                            "chained windows: later fill is conditioned on earlier "
                            "fill (invented on invented)")
        out = x.copy()
        out[mask] = y[mask]
        return out, info


def noise_power_of(x) -> float:
    return noise_power(x)


def inpaint(x, mask, fs: float | None = None, *, model=None, model_dir=None,
            profile: str | None = None, noise_power: float | None = None,
            steps: int | None = None, resample: int | None = None, seed: int = 0):
    """The repair track's hook (`repair.iq_dropout`, method "learned"):
    fill `x` where `mask` is True with the trained inpainter. Returns
    (y, info). Refuses, in words, a stream at another rate than the model's
    and a call with no model."""
    inp = model
    if inp is None:
        if model_dir is None:
            raise RuntimeError("no inpainter model was given: train one with "
                               "atk_diffusion.learn.inpaint.train_inpainter(rf, "
                               "profile) and pass model_dir=<its folder>")
        inp = Inpainter.load(model_dir, for_profile=profile)
    elif profile:
        profiles.check_match(inp.card.profile, profile, what=f"this inpainter ({inp.card.name})")
    if fs is not None and abs(float(fs) - inp.rate) > 0.5:
        raise ValueError(f"this inpainter works at {inp.rate:g} S/s (the "
                         f"{inp.input.get('canonical_class') or 'trained'} rate of "
                         f"{profiles.describe(inp.card.profile)}); this stream is at "
                         f"{float(fs):g} S/s — cut it to that rate first")
    return inp.inpaint(x, mask, noise_power=noise_power, steps=steps,
                       resample=resample, seed=seed)
