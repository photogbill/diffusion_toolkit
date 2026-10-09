# Copyright (c) 2026 William R. Duncan. All rights reserved. See LICENSE.
"""A6 — paired captures -> a receiver-to-receiver translator (plan §4.A6).

Plan A6: *"The cabled loop played into two receivers gives paired captures:
identical input, different receivers. That is the training data for a
diffusion model that translates a capture from one profile into what another
receiver would have heard. The profile rule stands — files never mix — but a
dataset collected on the bladeRF can be translated into an RTL-SDR dataset,
labeled `atk:translated_from`, and judged by the domain gap like any other
synthetic data. If it holds, every hour of collection counts for every
receiver Bill owns."*

WHAT IT IS. A CONDITIONAL diffusion model: the U-Net sees receiver A's
window as two extra input channels (`learn.unet`, condition channels) and
generates what receiver B recorded for the same input — B's floor shape,
DC spike, IQ image, quantisation, compression, phase noise, whatever is in
the pairs. Sampling is DDIM from noise conditioned on A
(`learn.diffusion.conditional`), or SDEdit-style from A itself at a middle
step (`t_start`), which keeps more of A's structure.

THE PROFILE RULE. A translator is FROM one profile TO another; a capture of
any other profile is refused in the plan's own words. The two receivers
must share a rate: pairs from receivers at different rates go through the
logged resampler first (`dsp.resample.resample_capture`), and the card
names `resampled_from` — a model of resampled data says so (plan §3.3).
The translator's card lives under the TARGET profile's `models\\`.

ALIGNMENT. The cabled loop feeds both receivers at once, but their
recordings start at different instants and their oscillators differ:
`align_pair` finds the lag (cross-correlation), and the carrier offset and
phase between them, and applies them to A. B — the truth — is never
touched. B's absolute phase is a fact of when it started recording; the
translator does not pretend to know it.

THE CLASSICAL COMPARATOR (plan §7). `fit_linear` — a least-squares
WIDELY-LINEAR FIR, B ≈ h∗A + g∗conj(A) + c: gain, frequency response, IQ
imbalance and DC are all linear in that form. What it cannot model —
quantisation, compression, phase noise — is the room a learned translator
has to earn its place, and `experiments.translator_eval` measures whether
it does.

OUTPUT. Every translated sample is INVENTED tier
(`provenance.tier_for("diffusion_translate")`); `translate_capture` writes
a SigMF file with `atk:translated_from`, `atk:receiver_profile` (the target),
`atk:tier`, `atk:method`, `atk:model_sha256`. The hallucination rate — B's
noise translated from A's noise gaining a signal — is measured into the
card.

LIMITS. Tiny models in tests learn nothing; the real translator needs the
cabled loop's paired captures (bladeRF -> RTL-SDR through a splitter, plan
§4.A6 transmit safety) and a GPU. A translation is a statistical stand-in for
receiver B, not B's recording.
"""

from __future__ import annotations

import math
import time
from pathlib import Path

import numpy as np

from atk_diffusion import cards, profiles, provenance
from atk_diffusion.learn import diffusion as _diff

METHOD = "diffusion_translate"
TIER = provenance.tier_for(METHOD)


# ---------------------------------------------------------------------------
# Pairing
# ---------------------------------------------------------------------------
def align_pair(a, b, max_lag: int | None = None) -> tuple[np.ndarray, np.ndarray, dict]:
    """Align capture A to capture B: the integer lag from the cross-
    correlation of their POWER ENVELOPES (blind to the two receivers' LO
    offset, which spins a complex correlation to nothing — 1 ppm at 433 MHz
    is most of a turn across a 4096-sample window at 2.4 MS/s), then A's
    carrier offset and phase relative to B from a·conj(b) on the overlap.
    Returns (A aligned, B cropped to the overlap, info). B's samples are
    never altered."""
    a = np.asarray(a, dtype=np.complex128)
    b = np.asarray(b, dtype=np.complex128)
    n = int(2 ** math.ceil(math.log2(a.size + b.size)))
    ea = np.abs(a) ** 2
    eb = np.abs(b) ** 2
    c = np.fft.ifft(np.fft.fft(eb - eb.mean(), n) * np.conj(np.fft.fft(ea - ea.mean(), n)))
    mag = np.real(c)
    lags = np.arange(n)
    lags[lags > n // 2] -= n
    if max_lag is not None:
        mag = np.where(np.abs(lags) <= int(max_lag), mag, 0.0)
    lag = int(lags[int(np.argmax(mag))])       # b[k + lag] ~ a[k]
    if lag >= 0:
        bb, aa = b[lag:], a
    else:
        bb, aa = b, a[-lag:]
    m = min(aa.size, bb.size)
    aa, bb = aa[:m], bb[:m]
    prod = bb * np.conj(aa)
    # carrier offset: the peak of the product's spectrum, 16x zero-padded
    # and refined by a parabola through the log-magnitudes around it
    # (estimated on at most 65536 samples — plenty for the precision)
    est = prod[: min(m, 1 << 16)]
    nf = 16 * int(2 ** math.ceil(math.log2(max(est.size, 2))))
    spec = np.abs(np.fft.fft(est, nf)) + 1e-30
    k = int(np.argmax(spec))
    y0, y1, y2 = (math.log(spec[(k - 1) % nf]), math.log(spec[k]),
                  math.log(spec[(k + 1) % nf]))
    den = y0 - 2.0 * y1 + y2
    delta = 0.5 * (y0 - y2) / den if den < 0 else 0.0
    kk = k + delta
    f_cyc = (kk if kk <= nf / 2 else kk - nf) / nf        # cycles per sample
    rot = np.exp(2j * np.pi * f_cyc * np.arange(m))
    phase = float(np.angle(np.sum(prod * np.conj(rot))))
    aa = aa * rot * np.exp(1j * phase)
    return (aa.astype(np.complex64), bb.astype(np.complex64),
            {"lag": lag, "offset_cycles_per_sample": f_cyc, "phase_rad": phase,
             "overlap": m})


def windows(x, L: int, hop: int | None = None) -> np.ndarray:
    """Non-overlapping (or `hop`-stepped) windows [N, L] of a 1-D stream."""
    x = np.asarray(x)
    hop = int(hop or L)
    n = 1 + (x.size - L) // hop if x.size >= L else 0
    if n <= 0:
        return np.zeros((0, L), dtype=x.dtype)
    idx = np.arange(L)[None, :] + hop * np.arange(n)[:, None]
    return x[idx]


# ---------------------------------------------------------------------------
# The classical comparator: a widely-linear least-squares FIR
# ---------------------------------------------------------------------------
def _design(A: np.ndarray, taps: int) -> np.ndarray:
    """Rows [A[n−k] for k] + [conj A[n−k]] + [1], per window, 'same'-aligned."""
    half = taps // 2
    rows = []
    for w in np.atleast_2d(A):
        p = np.pad(w, (half, taps - 1 - half))
        M = np.lib.stride_tricks.sliding_window_view(p, taps)[:, ::-1]
        rows.append(np.concatenate([M, np.conj(M), np.ones((M.shape[0], 1))], axis=1))
    return np.concatenate(rows, axis=0)


def fit_linear(A, B, taps: int = 15, ridge: float = 1e-6,
               max_rows: int = 200_000) -> dict:
    """Least-squares widely-linear FIR from A windows to B windows (the
    classical translator). At most `max_rows` samples are used (the design
    matrix is samples × (2·taps + 1)). Returns {h, g, c, taps}."""
    A = np.atleast_2d(np.asarray(A, dtype=np.complex128))
    B = np.atleast_2d(np.asarray(B, dtype=np.complex128))
    keep = max(1, int(max_rows) // max(1, A.shape[1]))
    A, B = A[:keep], B[:keep]
    X = _design(A, taps)
    y = B.reshape(-1)
    XtX = X.conj().T @ X
    XtX += ridge * np.trace(XtX).real / XtX.shape[0] * np.eye(XtX.shape[0])
    w = np.linalg.solve(XtX, X.conj().T @ y)
    return {"h": w[:taps], "g": w[taps:2 * taps], "c": w[-1], "taps": int(taps)}


def apply_linear(lin: dict, A) -> np.ndarray:
    A = np.atleast_2d(np.asarray(A, dtype=np.complex128))
    w = np.concatenate([lin["h"], lin["g"], [lin["c"]]])
    out = (_design(A, lin["taps"]) @ w).reshape(A.shape)
    return out.astype(np.complex64)


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
def _norm(A: np.ndarray):
    rms = np.sqrt(np.mean(np.abs(A) ** 2, axis=-1, keepdims=True) / 2.0)
    return np.maximum(rms, 1e-12)


def _ch(Z) -> np.ndarray:
    Z = np.asarray(Z)
    return np.stack([Z.real, Z.imag], axis=1).astype(np.float32)


def train_translator(rf, from_profile: str, to_profile: str, pairs_a, pairs_b, *,
                     resampled_from: str | None = None, name: str | None = None,
                     schedule: str = "cosine", T: int = 1000, unet: dict | None = None,
                     steps: int = 20000, batch: int = 32, lr: float = 2e-4,
                     sample_steps: int = 50, linear_taps: int = 15,
                     validation_fraction: float = 0.1, seed: int = 0,
                     device: str | None = None, overwrite: bool = False,
                     progress=None) -> Path:
    """Train A6 on ALIGNED pairs: complex windows [N, L] from receiver A
    (`from_profile`) and receiver B (`to_profile`) of the same input, at B's
    rate. Both are scaled per window by A's RMS (the model learns B's level
    relative to A's, which the cabled loop's arithmetic fixes). Saves
    model.onnx + model.pt + card (kind "translator", INVENTED) under the
    target profile's models folder; the validation numbers against the
    classical widely-linear comparator and the hallucination rate are
    measured into it."""
    import torch
    from atk_diffusion.learn import unet as _u
    say = progress or (lambda s: None)
    fa = profiles.parse_profile_id(from_profile)
    fb = profiles.parse_profile_id(to_profile)
    if fa.sample_rate != fb.sample_rate and not resampled_from:
        raise profiles.ProfileMismatch(
            f"{profiles.describe(from_profile)} and {profiles.describe(to_profile)} "
            "run at different rates. Resample receiver A's captures to B's rate with "
            "dsp.resample.resample_capture (a logged step) and pass resampled_from= "
            "so the card says the model was trained on resampled data.")
    A = np.asarray(pairs_a)
    B = np.asarray(pairs_b)
    if A.shape != B.shape or A.ndim != 2 or not np.iscomplexobj(A):
        raise ValueError("pairs are two complex arrays of the same shape [N, L]")
    L = int(A.shape[1])
    conf = dict(unet or _u.SMALL_1D)
    conf.update(dims=1, in_ch=2, cond_ch=2)
    if L % _u.multiple(conf):
        raise ValueError(f"the window {L} must be a multiple of {_u.multiple(conf)}")
    rng = np.random.default_rng(seed)
    order = rng.permutation(A.shape[0])
    n_val = max(1, int(round(validation_fraction * A.shape[0])))
    vi, ti = order[:n_val], order[n_val:]
    s = _norm(A)
    XA = torch.as_tensor(_ch(A / s))
    XB = torch.as_tensor(_ch(B / s))
    x0_clip = float(1.5 * np.max(np.abs(_ch(B[ti] / s[ti]))))
    torch.manual_seed(int(seed))
    dev = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model = _u.build_unet(conf).to(dev)
    sched = _diff.make_schedule(schedule, T)
    gen = torch.Generator().manual_seed(int(seed))
    tidx = torch.as_tensor(ti)

    def batch_fn(step, g):
        i = tidx[torch.randint(0, tidx.numel(), (int(batch),), generator=g)]
        return {"x0": XB[i].to(dev), "cond": XA[i].to(dev)}
    say(f"training the A6 translator {from_profile} -> {to_profile}: {ti.size} pairs "
        f"of {L} samples")
    hist = _diff.fit(model, batch_fn, sched.alphas_cumprod, steps, lr=lr,
                     progress=say, generator=gen)
    model = model.to("cpu").eval()
    name = name or f"translator_{fa.family}_to_{fb.family}_{time.strftime('%Y%m%d_%H%M%S', time.gmtime())}"
    d = rf.models(to_profile, name)
    if d.exists() and any(d.iterdir()) and not overwrite:
        raise FileExistsError(f"a model named {name} already exists")
    d.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), d / "model.pt")
    _u.export_onnx(model, d / "model.onnx", (L,))
    lin = fit_linear(A[ti], B[ti], taps=linear_taps)
    card = cards.new_card(
        name, "translator", "",
        input={"from_profile": str(from_profile).lower(), "to_profile": str(to_profile).lower(),
               "rate": float(fb.sample_rate), "window": L,
               "resampled_from": resampled_from or "",
               "normalize": "both windows by A's RMS per real element",
               "schedule": sched.to_json(), "unet": model.config,
               "sample_steps": int(sample_steps), "x0_clip": x0_clip,
               "torch_weights": {"file": "model.pt",
                                 "sha256": provenance.sha256_path(d / "model.pt")},
               "onnx": {"inputs": ["x", "t"], "outputs": ["eps"], "opset": 17,
                        "x_channels": "2 data (B, noisy) then 2 condition (A)"}},
        datasets=[{"name": "paired windows (A, B)", "kind": "paired", "n": int(A.shape[0]),
                   "sha256": provenance.sha256_bytes(np.ascontiguousarray(B).tobytes())}],
        metrics={"final_loss": float(np.mean(hist["loss"][-50:])) if hist["loss"] else None,
                 "steps": int(steps)},
        license="all rights reserved", trained_on=f"{dev} (torch {torch.__version__})",
        notes=[f"translates {profiles.describe(from_profile)} into what "
               f"{profiles.describe(to_profile)} would have heard",
               "outputs are INVENTED tier, labelled atk:translated_from",
               "experiments.translator_eval is the judge (statistics vs real B, the "
               "downstream classifier test, the domain gap)"]
        + ([f"trained on data resampled from {resampled_from} (plan §3.3)"]
           if resampled_from else []))
    card.sample_rate = float(fb.sample_rate)
    cards.save(d, card, "model.onnx")
    tr = Translator.load(d, from_profile=from_profile)
    val = tr.translate_windows(A[vi], seed=seed + 1)
    lin_v = apply_linear(lin, A[vi])
    card.metrics.update({
        "val_nmse_db": {"diffusion": nmse_db(val, B[vi]), "linear": nmse_db(lin_v, B[vi]),
                        "identity": nmse_db(A[vi], B[vi])},
        "val_psd_distance_db": {"diffusion": psd_distance_db(val, B[vi]),
                                "linear": psd_distance_db(lin_v, B[vi]),
                                "identity": psd_distance_db(A[vi], B[vi])},
        "val_pairs": int(vi.size)})
    card.metrics["beats_classical"] = bool(
        card.metrics["val_psd_distance_db"]["diffusion"]
        < card.metrics["val_psd_distance_db"]["linear"])
    cards.save(d, card, "model.onnx")
    for f in ("model.onnx", "model.pt", "card.json"):
        try:
            rf.record(d / f, "model", f"translator {name}")
        except Exception:                                  # noqa: BLE001
            pass
    return d


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------
def nmse_db(x, ref) -> float:
    """Normalised squared error of x against ref, dB (pairs aligned)."""
    x = np.asarray(x, dtype=np.complex128)
    r = np.asarray(ref, dtype=np.complex128)
    return float(10 * np.log10(max(np.sum(np.abs(x - r) ** 2), 1e-30)
                               / max(np.sum(np.abs(r) ** 2), 1e-30)))


def psd(x, nfft: int = 64) -> np.ndarray:
    """Average periodogram over windows (fft-shifted), linear."""
    x = np.atleast_2d(np.asarray(x, dtype=np.complex128))
    w = np.hanning(nfft + 1)[:-1]
    segs = []
    for row in x:
        k = row.size // nfft
        if k:
            segs.append(row[: k * nfft].reshape(k, nfft))
    S = np.concatenate(segs) if segs else np.zeros((1, nfft))
    P = np.abs(np.fft.fft(S * w, axis=1)) ** 2 / np.sum(w ** 2)
    return np.fft.fftshift(P.mean(axis=0))


def psd_distance_db(x, ref, nfft: int = 64) -> float:
    """Mean absolute difference of the two average spectra, dB — the
    receiver's floor shape, DC spike and image show up here."""
    a, b = psd(x, nfft), psd(ref, nfft)
    return float(np.mean(np.abs(10 * np.log10(np.maximum(a, 1e-30))
                                - 10 * np.log10(np.maximum(b, 1e-30)))))


def stats(x) -> dict:
    """Receiver fingerprints a translation should reproduce: DC level
    (re the total power), improperness of what is left once DC is removed
    (the IQ image: |E[z²]| / E[|z|²], 0 for a proper receiver), its
    amplitude kurtosis, and distinct I levels per RMS (quantisation)."""
    x = np.asarray(x, dtype=np.complex128).reshape(-1)
    p = float(np.mean(np.abs(x) ** 2))
    mu = complex(np.mean(x))
    dc = abs(mu) ** 2
    z = x - mu
    pz = float(np.mean(np.abs(z) ** 2))
    imp = abs(complex(np.mean(z * z))) / max(pz, 1e-30)
    a2 = np.abs(z) ** 2
    kurt = float(np.mean(a2 ** 2) / max(np.mean(a2) ** 2, 1e-30))
    lv = np.unique(np.round(x.real / max(np.sqrt(p), 1e-12), 6)).size
    return {"dc_db": float(10 * np.log10(max(dc, 1e-30) / max(p, 1e-30))),
            "improperness": float(imp), "kurtosis": kurt,
            "levels_per_rms": int(lv)}


# ---------------------------------------------------------------------------
# The translator
# ---------------------------------------------------------------------------
class Translator:
    """A trained A6 translator, loaded through its card (onnxruntime when
    installed, PyTorch otherwise)."""

    def __init__(self, eps_fn, card, model_dir, backend: str):
        self.card, self.dir, self.backend = card, Path(model_dir), backend
        self.input = dict(card.input)
        self.L = int(self.input["window"])
        self.ac = _diff.Schedule.from_json(self.input["schedule"]).alphas_cumprod
        self._flat = eps_fn

    @classmethod
    def load(cls, model_dir, from_profile: str | None = None, backend: str = "auto"):
        card = cards.load(model_dir, expect_kind="translator")
        src = card.input.get("from_profile", "")
        if from_profile:
            profiles.check_match(src, from_profile, what=f"this translator ({card.name})")
        if backend in ("auto", "onnx") and card.weights.get("format") == "onnx":
            try:
                import onnxruntime as ort
                so = ort.SessionOptions()
                so.intra_op_num_threads = 1
                sess = ort.InferenceSession(str(cards.weights_path(model_dir, card)), so,
                                            providers=["CPUExecutionProvider"])

                def flat(x, t):
                    return sess.run(["eps"], {"x": np.asarray(x, dtype=np.float32),
                                              "t": np.asarray(t, dtype=np.int64)})[0]
                return cls(flat, card, model_dir, "onnxruntime")
            except ImportError:
                if backend == "onnx":
                    raise RuntimeError("onnxruntime is not installed") from None
        import torch
        from atk_diffusion.learn import unet as _u
        tw = card.input["torch_weights"]
        p = Path(model_dir) / tw["file"]
        if provenance.sha256_path(p) != tw["sha256"]:
            raise cards.CardRefusal(f"{p.name} is not the file the card describes "
                                    "(its hash changed). Retrain or restore it.")
        model = _u.build_unet(card.input["unet"])
        model.load_state_dict(torch.load(p, map_location="cpu", weights_only=True))
        wrapped = _u._ConcatWrapper(model.eval())
        return cls(_diff.torch_eps_fn(wrapped), card, model_dir, "torch")

    def translate_windows(self, A, steps: int | None = None, t_start: int | None = None,
                          eta: float = 0.0, seed: int = 0) -> np.ndarray:
        """Translate windows [N, L] of A into B. `t_start` (default: the full
        chain) < T−1 starts from A noised to that step (SDEdit)."""
        A = np.atleast_2d(np.asarray(A))
        if A.shape[1] != self.L:
            raise ValueError(f"this translator works on {self.L}-sample windows")
        s = _norm(A)
        cond = _ch(A / s)
        fn = _diff.concat_condition(self._flat, cond)
        steps = int(steps or self.input.get("sample_steps", 50))
        T = self.ac.shape[0]
        rng = np.random.default_rng(seed)
        t0 = T - 1 if t_start is None else int(t_start)
        noise = rng.standard_normal(cond.shape).astype(np.float32)
        x_start = noise if t0 == T - 1 else _diff.q_sample(cond, t0, noise, self.ac).astype(np.float32)
        out = _diff.ddim_sample(fn, x_start, t0, self.ac, steps=steps, eta=eta, rng=rng,
                                x0_clip=self.input.get("x0_clip"))
        return ((out[:, 0] + 1j * out[:, 1]) * s).astype(np.complex64)

    def translate(self, x, steps: int | None = None, t_start: int | None = None,
                  seed: int = 0) -> tuple[np.ndarray, dict]:
        """Translate a stream: half-overlapping windows, crossfaded with
        complementary triangular weights. Returns (B estimate, info)."""
        x = np.asarray(x, dtype=np.complex64)
        L, hop = self.L, self.L // 2
        if x.size < L:
            raise ValueError(f"the stream is shorter than one {L}-sample window")
        n = 1 + int(math.ceil((x.size - L) / hop))
        pad = (n - 1) * hop + L - x.size
        xp = np.concatenate([x, np.zeros(pad, dtype=np.complex64)])
        W = windows(xp, L, hop)
        Y = self.translate_windows(W, steps=steps, t_start=t_start, seed=seed)
        w = np.bartlett(L + 2)[1:-1]
        out = np.zeros(xp.size, dtype=np.complex128)
        acc = np.zeros(xp.size)
        for i in range(n):
            out[i * hop: i * hop + L] += w * Y[i]
            acc[i * hop: i * hop + L] += w
        y = (out / np.maximum(acc, 1e-12))[: x.size].astype(np.complex64)
        info = {"method": METHOD, "tier": TIER, "words": provenance.TIER_WORDS[TIER],
                "translated_from": self.input["from_profile"],
                "receiver_profile": self.input["to_profile"],
                "model": self.card.name, "model_sha256": self.card.weights.get("sha256", ""),
                "windows": n, "backend": self.backend}
        return y, info


def translate_capture(src, model_dir, out_base, rf=None, steps: int | None = None,
                      max_samples: int = 1 << 22, seed: int = 0) -> dict:
    """Translate a SigMF capture of the translator's FROM profile into a new
    SigMF file of its TO profile, labelled: atk:translated_from,
    atk:receiver_profile (the target), atk:tier invented, atk:method,
    atk:model_sha256. The source is checked against the card (refused in
    words otherwise) and never modified."""
    from atk_diffusion import sigmf as _sigmf
    meta = _sigmf.read_meta(src)
    src_profile = profiles.profile_from_meta(meta)
    tr = Translator.load(model_dir, from_profile=src_profile)
    fs = _sigmf.sample_rate_of(meta)
    if abs(fs - float(tr.input["rate"])) > 0.5:
        raise profiles.ProfileMismatch(
            f"this capture runs at {fs:g} S/s; the translator works at "
            f"{float(tr.input['rate']):g} S/s — resample it deliberately first")
    x = _sigmf.load(src, count=int(max_samples), meta=meta)
    if x.ndim > 1:
        raise ValueError("translate one channel at a time")
    y, info = tr.translate(x, steps=steps, seed=seed)
    g = {"atk:receiver_profile": tr.input["to_profile"],
         "atk:translated_from": src_profile, "atk:tier": TIER, "atk:method": METHOD,
         "atk:method_params": {"steps": steps or tr.input.get("sample_steps"),
                               "window": tr.L, "seed": seed},
         "atk:model_sha256": tr.card.weights.get("sha256", ""),
         "atk:source_capture": _sigmf.base_of(src).name}
    dp, mp = _sigmf.write_pair(out_base, y, fs, _sigmf.center_of(meta), datatype="cf32",
                               extra_global=g,
                               description=f"INVENTED: translated from {src_profile} "
                                           f"by {tr.card.name}")
    if rf is not None:
        for f in (dp, mp):
            try:
                rf.record(f, "translated", f"from {src_profile}")
            except Exception:                              # noqa: BLE001
                pass
    return {"data": str(dp), "meta": str(mp), **info}
